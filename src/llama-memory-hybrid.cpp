#include "llama-memory-hybrid.h"

#include "llama-impl.h"
#include "llama-model.h"
#include "llama-context.h"

//
// llama_memory_hybrid
//

llama_memory_hybrid::llama_memory_hybrid(
        const llama_model & model,
                     bool   no_alloc,
                            /* attn */
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   v_trans,
                 uint32_t   kv_size,
                 uint32_t   n_swa,
           llama_swa_type   swa_type,
                            /* recurrent */
                ggml_type   type_r,
                ggml_type   type_s,
                 uint32_t   rs_size,
                            /* common */
                 uint32_t   n_seq_max,
                 uint32_t   n_rs_seq,
                     bool   offload,
                     bool   unified,
                            /* layer filters */
    const layer_filter_cb & filter_attn,
    const layer_filter_cb & filter_recr) :
    hparams(model.hparams),
    mem_attn(new llama_kv_cache(
        model,
        no_alloc,
        model.hparams,
        type_k,
        type_v,
        v_trans,
        offload,
        unified,
        kv_size,
        n_seq_max,
        n_swa,
        swa_type,
        nullptr,
        filter_attn == nullptr ?
            [&](int32_t il) { return !hparams.is_recr(il); }
            : filter_attn,
        nullptr,
        nullptr
    )),
    mem_recr(new llama_memory_recurrent(
        model,
        no_alloc,
        type_r,
        type_s,
        offload,
        rs_size,
        n_seq_max,
        n_rs_seq,
        filter_recr == nullptr ?
            [&](int32_t il) { return hparams.is_recr(il); }
            : filter_recr
    )) {}

llama_memory_context_ptr llama_memory_hybrid::init_batch(llama_batch_allocr & balloc, uint32_t n_ubatch, bool embd_all) {
    bool split_sequences = false;
    do {
        balloc.split_reset();

        // follow the recurrent pattern for creating the ubatch splits
        std::vector<llama_ubatch> ubatches;

        while (true) {
            llama_ubatch ubatch;

            if (embd_all) {
                // if all tokens are output, split by sequence
                ubatch = balloc.split_seq(n_ubatch);
            } else {
                // Use non-sequential split when KV cache is unified (needed for hellaswag/winogrande/multiple-choice)
                const bool unified = (mem_attn->get_n_stream() == 1);

                // [TAG_RECURRENT_ROLLBACK_SPLITS]
                // the trailing (1 + n_rs_seq) tokens of each seq must stay in the same ubatch
                //   so that the rollback snapshots remain valid
                const uint32_t n_rs_seq = mem_recr->n_rs_seq;

                ubatch = balloc.split_equal(n_ubatch, !unified, n_rs_seq > 0 ? n_rs_seq + 1 : 0, split_sequences ? 1 : 0);
            }

            if (ubatch.n_tokens == 0) {
                break;
            }

            ubatches.push_back(std::move(ubatch)); // NOLINT
        }

        if (balloc.get_n_used() < balloc.get_n_tokens()) {
            // failed to find a suitable split
            break;
        }

        // prepare the recurrent batches first
        if (!mem_recr->prepare(ubatches)) {
            // prepare restored the metadata without touching tensors. Retry through the same
            // batch owner with one sequence set per ubatch; keep rollback tails intact.
            if (!split_sequences && mem_recr->has_protected_graph_range()) {
                split_sequences = true;
                continue;
            }
            LLAMA_LOG_ERROR("%s: failed to prepare recurrent ubatches\n", __func__);
            return std::make_unique<llama_memory_hybrid_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }

        // prepare the attention cache
        auto heads_attn = mem_attn->prepare(ubatches);
        if (heads_attn.empty()) {
            LLAMA_LOG_ERROR("%s: failed to prepare attention ubatches\n", __func__);
            return std::make_unique<llama_memory_hybrid_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }

        return std::make_unique<llama_memory_hybrid_context>(
                this, std::move(heads_attn), std::move(ubatches));
    } while(true);

    return std::make_unique<llama_memory_hybrid_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
}

llama_memory_context_ptr llama_memory_hybrid::init_full() {
    return std::make_unique<llama_memory_hybrid_context>(this);
}

llama_memory_context_ptr llama_memory_hybrid::init_full_single_stream() {
    return std::make_unique<llama_memory_hybrid_context>(this, true);
}

llama_memory_context_ptr llama_memory_hybrid::init_update(llama_context * lctx, bool optimize) {
    return std::make_unique<llama_memory_hybrid_context>(this, lctx, optimize);
}

bool llama_memory_hybrid::get_can_shift() const {
    // Shifting is trivially supported for recurrent
    return mem_attn->get_can_shift();
}

void llama_memory_hybrid::clear(bool data) {
    mem_attn->clear(data);
    mem_recr->clear(data);
}

bool llama_memory_hybrid::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    // Try removing from the recurrent cache first since it may fail. If it does
    // fail, the cache will not have been mutated.
    if (!mem_recr->seq_rm(seq_id, p0, p1)) {
        return false;
    }
    return mem_attn->seq_rm(seq_id, p0, p1);
}

void llama_memory_hybrid::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    mem_attn->seq_cp(seq_id_src, seq_id_dst, p0, p1);
    mem_recr->seq_cp(seq_id_src, seq_id_dst, p0, p1);
}

void llama_memory_hybrid::seq_keep(llama_seq_id seq_id) {
    mem_attn->seq_keep(seq_id);
    mem_recr->seq_keep(seq_id);
}

void llama_memory_hybrid::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    mem_attn->seq_add(seq_id, p0, p1, shift);
    mem_recr->seq_add(seq_id, p0, p1, shift);
}

void llama_memory_hybrid::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    mem_attn->seq_div(seq_id, p0, p1, d);
    mem_recr->seq_div(seq_id, p0, p1, d);
}

llama_pos llama_memory_hybrid::seq_pos_min(llama_seq_id seq_id) const {
    // the min of the total cache is the max of the two caches' min values
    return std::max(mem_attn->seq_pos_min(seq_id), mem_recr->seq_pos_min(seq_id));
}

llama_pos llama_memory_hybrid::seq_pos_max(llama_seq_id seq_id) const {
    // the max of the total cache is the min of the two caches' max values
    return std::min(mem_attn->seq_pos_max(seq_id), mem_recr->seq_pos_max(seq_id));
}

std::map<ggml_backend_buffer_type_t, size_t> llama_memory_hybrid::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> mb = mem_attn->memory_breakdown();
    for (const auto & buft_size : mem_recr->memory_breakdown()) {
        mb[buft_size.first] += buft_size.second;
    }
    return mb;
}

void llama_memory_hybrid::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    mem_attn->state_write(io, seq_id, flags);
    mem_recr->state_write(io, seq_id, flags);
}

void llama_memory_hybrid::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    mem_attn->state_read(io, seq_id, flags);

    try {
        mem_recr->state_read(io, seq_id, flags);
    } catch (...) {
        // a failed restore leaves the sequence empty in both caches
        mem_attn->state_clear(seq_id);

        throw;
    }
}

llama_persona_move llama_memory_hybrid::persona_prepare(llama_persona_blocks & blocks) const {
    const llama_persona_move attn = mem_attn->persona_prepare(blocks);
    return attn != llama_persona_move::done ? attn : mem_recr->persona_prepare(blocks);
}

llama_persona_move llama_memory_hybrid::persona_check_save(llama_seq_id seq_id, llama_persona_blocks & out) const {
    const llama_persona_move attn = mem_attn->persona_check_save(seq_id, out);
    return attn != llama_persona_move::done ? attn : mem_recr->persona_check_save(seq_id, out);
}

llama_persona_move llama_memory_hybrid::persona_check_load(llama_seq_id seq_id, uint32_t stream,
                                                          const llama_persona_blocks & in, llama_persona_slot & slot) const {
    const llama_persona_move attn = mem_attn->persona_check_load(seq_id, stream, in, slot);
    return attn != llama_persona_move::done ? attn : mem_recr->persona_check_load(seq_id, stream, in, slot);
}

void llama_memory_hybrid::persona_commit_save(llama_seq_id seq_id, llama_persona_blocks & out) {
    mem_attn->persona_commit_save(seq_id, out);
    mem_recr->persona_commit_save(seq_id, out);
}

void llama_memory_hybrid::persona_commit_load(llama_seq_id seq_id, llama_persona_slot & slot,
                                              llama_persona_blocks & in) {
    mem_attn->persona_commit_load(seq_id, slot, in);
    mem_recr->persona_commit_load(seq_id, slot, in);
}

// Both halves agree on her position or the checkpoint is refused: a rewind must leave attention and recurrent state at
// the same token.
llama_persona_move llama_memory_hybrid::persona_check_checkpoint(llama_seq_id seq_id, llama_persona_checkpoint_blocks & out) const {
    const llama_persona_move attn = mem_attn->persona_check_checkpoint(seq_id, out);
    return attn != llama_persona_move::done ? attn : mem_recr->persona_check_checkpoint_rows(seq_id, out);
}

void llama_memory_hybrid::persona_commit_checkpoint(llama_seq_id seq_id, llama_persona_checkpoint_blocks & out) {
    mem_attn->persona_commit_checkpoint(seq_id, out);
    mem_recr->persona_commit_checkpoint_rows(seq_id, out);
}

llama_persona_move llama_memory_hybrid::persona_check_rewind(llama_seq_id seq_id, const llama_persona_checkpoint_blocks & in) const {
    if (in.rs.has_state && in.rs.pos != in.pos) {
        return llama_persona_move::layout_mismatch; // the halves were not at one token when it was taken
    }
    const llama_persona_move attn = mem_attn->persona_check_rewind(seq_id, in);
    return attn != llama_persona_move::done ? attn : mem_recr->persona_check_rewind_rows(seq_id, in);
}

void llama_memory_hybrid::persona_commit_rewind(llama_seq_id seq_id, const llama_persona_checkpoint_blocks & in) {
    mem_attn->persona_commit_rewind(seq_id, in);
    mem_recr->persona_commit_rewind_rows(seq_id, in);
}

llama_kv_cache * llama_memory_hybrid::get_mem_attn() const {
    return mem_attn.get();
}

llama_memory_recurrent * llama_memory_hybrid::get_mem_recr() const {
    return mem_recr.get();
}

llama_memory_hybrid_context::llama_memory_hybrid_context(llama_memory_status status) : status(status) {}

llama_memory_hybrid_context::llama_memory_hybrid_context(llama_memory_hybrid * mem, bool single_stream) :
    ctx_attn(single_stream ? mem->get_mem_attn()->init_full_single_stream() : mem->get_mem_attn()->init_full()),
    ctx_recr(mem->get_mem_recr()->init_full()),
    status(llama_memory_status_combine(ctx_attn->get_status(), ctx_recr->get_status())) {
}

llama_memory_hybrid_context::llama_memory_hybrid_context(
        llama_memory_hybrid * mem,
              llama_context * lctx,
                       bool   optimize) :
    ctx_attn(mem->get_mem_attn()->init_update(lctx, optimize)),
    ctx_recr(mem->get_mem_recr()->init_update(lctx, optimize)),
    status(llama_memory_status_combine(ctx_attn->get_status(), ctx_recr->get_status())) {
}

llama_memory_hybrid_context::llama_memory_hybrid_context(
              llama_memory_hybrid * mem,
                  slot_info_vec_t   sinfos_attn,
        std::vector<llama_ubatch>   ubatches) :
    ubatches(std::move(ubatches)),
    // note: here we copy the ubatches. not sure if this is ideal
    ctx_attn(new llama_kv_cache_context(mem->get_mem_attn(), std::move(sinfos_attn), this->ubatches)),
    ctx_recr(new llama_memory_recurrent_context(mem->get_mem_recr(), this->ubatches)),
    status(llama_memory_status_combine(ctx_attn->get_status(), ctx_recr->get_status())) {
}

bool llama_memory_hybrid_context::next() {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    ctx_attn->next();
    ctx_recr->next();

    if (++i_next >= ubatches.size()) {
        return false;
    }

    return true;
}

bool llama_memory_hybrid_context::apply() {
    assert(!llama_memory_status_is_fail(status));

    bool res = true;

    res = res & ctx_attn->apply();
    res = res & ctx_recr->apply();

    return res;
}

llama_memory_status llama_memory_hybrid_context::get_status() const {
    return status;
}

const llama_ubatch & llama_memory_hybrid_context::get_ubatch() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);
    return ubatches[i_next];
}

const llama_kv_cache_context * llama_memory_hybrid_context::get_attn() const {
    return static_cast<const llama_kv_cache_context *>(ctx_attn.get());
}

const llama_memory_recurrent_context * llama_memory_hybrid_context::get_recr() const {
    return static_cast<const llama_memory_recurrent_context *>(ctx_recr.get());
}

const char * llama_memory_hybrid::training_error(const llama_batch & batch, bool flash_attn) const {
    if (const char * why = mem_attn->training_error(batch, flash_attn)) {
        return why;
    }
    return mem_recr->training_error(batch, flash_attn);
}

std::vector<llama_memory_cells> llama_memory_hybrid::cell_usage() const {
    std::vector<llama_memory_cells> out;
    llama_memory_cells_append(out, "attn", mem_attn->cell_usage());
    llama_memory_cells_append(out, "recr", mem_recr->cell_usage());
    return out;
}
