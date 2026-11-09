// Live-sequence learning: one owner for synchronous calls and resumable serving-loop chunks.
#include "llama-context.h"
#include "llama-memory-hybrid.h"
#include "llama-memory-recurrent.h"
#include <algorithm>
#include <limits>

bool llama_context::opt_learn_begin(llama_seq_id seq, llama_pos n_cached, const llama_token * tokens, const llama_token * labels,
        int32_t n, const llama_token * restore_tokens, int32_t n_restore, ggml_opt_result_t result) {
    if (opt_reply) {
        opt_fail("a live reply already owns this context's optimizer; finish its restoration first");
        return false;
    }
    if (opt_ctx == nullptr) {
        opt_fail("llama_opt_learn needs llama_opt_init on this context first");
        return false;
    }
    if (opt_shadow && opt_shadow->restore_quarantined) {
        opt_fail("shadow generation restore is uncertain; restore a known-good generation before learning");
        return false;
    }
    // each call answers for itself: a refusal before it does not refuse this one
    opt_alloc_failed.store(false);
    opt_stop_requested.store(false);
    opt_failure.clear();
    if (n_restore <= 0 || restore_tokens == nullptr || (int64_t) n_cached + n_restore > (int64_t) cparams.n_ctx_seq) {
        opt_fail("live learn needs the exact served tail to restore within the existing window");
        return false;
    }
    if (n <= 0 || n_cached < 0 || tokens == nullptr || labels == nullptr) {
        opt_fail("llama_opt_learn: an empty reply, a negative served prefix, or no tokens / labels");
        return false;
    }
    if ((int64_t) n_cached + n > (int64_t) cparams.n_ctx_seq) {
        opt_fail("llama_opt_learn: the reply ends at " + std::to_string((int64_t) n_cached + n) + ", past the sequence's context of "
            + std::to_string(cparams.n_ctx_seq) + " positions");
        return false;
    }
    // the parts of the memory: a recurrent state cannot be cut back, attention cells can
    llama_memory_recurrent * recr = nullptr;
    llama_memory_i         * attn = memory.get();
    if (auto * hybrid = dynamic_cast<llama_memory_hybrid *>(memory.get())) {
        recr = hybrid->get_mem_recr();
        attn = hybrid->get_mem_attn();
    } else if (auto * r = dynamic_cast<llama_memory_recurrent *>(memory.get())) {
        recr = r;
        attn = nullptr;
    }
    // refusals before anything is touched
    const llama_pos held = memory->seq_pos_max(seq) + 1;
    if (held < n_cached) {
        opt_fail("llama_opt_learn: sequence " + std::to_string(seq) + " holds " + std::to_string(held)
            + " positions, fewer than the " + std::to_string(n_cached) + " served before the reply");
        return false;
    }
    if (recr != nullptr && recr->seq_pos_max(seq) + 1 != n_cached) {
        opt_fail("llama_opt_learn: the recurrent state of sequence " + std::to_string(seq) + " ends at "
            + std::to_string(recr->seq_pos_max(seq) + 1) + ", not at the served prefix " + std::to_string(n_cached)
            + ": restore the served-prefix checkpoint first");
        return false;
    }
    if (recr != nullptr && recr->rollback_pending(seq)) {
        opt_fail("llama_opt_learn: sequence " + std::to_string(seq) + " has a pending recurrent rollback: finish or drop the draft first");
        return false;
    }
    // her stable recurrent state at the served prefix, for the restore
    std::vector<uint8_t> snap;
    if (recr != nullptr) {
        synchronize();
        snap.resize(state_seq_get_size(seq, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY));
        if (state_seq_get_data(seq, snap.data(), snap.size(), LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) != snap.size()) {
            opt_fail("could not capture the recurrent prefix; live learning was not started");
            return false;
        }
    }
    if (attn != nullptr && !attn->seq_rm(seq, n_cached, -1)) {
        opt_fail("llama_opt_learn: could not drop sequence " + std::to_string(seq) + "'s cells past " + std::to_string(n_cached));
        return false;
    }
    auto state = std::make_unique<opt_live_reply>();
    state->seq = seq;
    state->prefix = n_cached;
    state->tokens.assign(tokens, tokens + n);
    state->restore_tokens.assign(restore_tokens, restore_tokens + n_restore);
    state->labels.assign(labels, labels + n);
    state->snapshot = std::move(snap);
    state->recurrent = recr;
    state->attention = attn;
    state->chunk_tokens = n_batch();
    state->batch = llama_batch_init(std::max<int32_t>(n_batch(), n), 0, 1);
    state->result = result;
    state->n_train = n;
    while (state->n_train > 0 && labels[state->n_train - 1] < 0) { --state->n_train; }
    for (int32_t i = 0; i < state->n_train; ++i) { state->n_labels += labels[i] >= 0; }
    opt_reply = std::move(state);
    return true;
}

// Revocation ends gradient admission, not sequence custody. The next step follows the
// existing stable-restoration path; no optimizer, KV, or completed update is discarded.
bool llama_context::opt_learn_drain() {
    if (!opt_reply) { return false; }
    if (opt_reply->graph_meta) {
        // Update nodes may already have run. Finish this graph before restoration;
        // its committed step remains visible rather than pretending to roll it back.
        opt_reply->drain_graph = true;
        return true;
    }
    if (!opt_reply->restoring && opt_reply->cursor < opt_reply->n_train) {
        // The full reply is one optimizer update. A revoked partial reply has only
        // accumulated gradients: clear those, preserving moments and completed updates.
        synchronize();
        if (opt_reply->labels_done > 0) { ggml_opt_reset(opt_ctx, false); }
        opt_reply->cursor = opt_reply->n_train;
    }
    return true;
}

bool llama_context::opt_learn_needs_gradient() const {
    return opt_reply && !opt_reply->restoring && opt_reply->cursor < opt_reply->n_train;
}

// One existing decode chunk, then return to the caller. The served window and the
// gradient's label denominator remain whole; chunking changes scheduling, not examples.
llama_opt_learn_status llama_context::opt_learn_step() {
    if (!opt_reply) {
        opt_fail("no live reply is active");
        return LLAMA_OPT_LEARN_FAILED;
    }
    auto & s = *opt_reply;
    if (s.graph_meta) {
        bool complete = false;
        const auto status = ggml_opt_eval_step(opt_ctx, s.result, s.graph_nodes, &complete);
        if (status != GGML_STATUS_SUCCESS) {
            // Release graph custody only after quiescence. Preserve uncertain writes,
            // quarantine the shadow, and restore the served sequence with stable weights.
            s.failure = "learning graph failed; parameter effects uncertain; shadow quarantined";
            ggml_opt_eval_quarantine(opt_ctx);
            if (opt_shadow) {
                auto * shadow = const_cast<llama_adapter_lora *>(opt_shadow);
                shadow->restore_quarantined = true;
                loras->erase(shadow);
            }
            s.graph_meta.reset();
            gf_res_prev->reset();
            s.cursor = s.n_train;
            return LLAMA_OPT_LEARN_MORE;
        }
        if (!complete) { return LLAMA_OPT_LEARN_MORE; }
        s.graph_meta.reset();
        s.cursor += s.batch.n_tokens;
        if (s.drain_graph) { s.cursor = s.n_train; }
        return LLAMA_OPT_LEARN_MORE;
    }
    if (!s.restoring && s.cursor >= s.n_train) {
        s.restoring = true;
        s.cursor = 0;
        bool restored = s.attention == nullptr || s.attention->seq_rm(s.seq, s.prefix, -1);
        if (s.recurrent != nullptr) {
            synchronize();
            restored = restored && state_seq_set_data(s.seq, s.snapshot.data(), s.snapshot.size(),
                LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == s.snapshot.size();
        }
        if (!restored) {
            opt_fail("live reply could not restore its served prefix: re-prefill it");
            opt_reply.reset();
            return LLAMA_OPT_LEARN_FAILED;
        }
    }
    const int32_t end = s.restoring ? (int32_t) s.restore_tokens.size() : s.n_train;
    const int32_t m = !s.restoring && s.graph_nodes > 0 ? end - s.cursor : std::min(s.chunk_tokens, end - s.cursor);
    auto & batch = s.batch;
    batch.n_tokens = m;
    for (int32_t i = 0; i < m; ++i) {
        batch.token[i] = s.restoring ? s.restore_tokens[s.cursor + i] : s.tokens[s.cursor + i];
        batch.pos[i] = s.prefix + s.cursor + i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = s.seq;
        batch.label[i] = s.restoring ? -1 : s.labels[s.cursor + i];
        batch.logits[i] = !s.restoring && batch.label[i] >= 0;
    }
    // Intermediate restoration batches only rebuild KV. Their vocabulary logits
    // are never consumed; publish the served tail's logits once, at the end.
    if (s.restoring && s.cursor + m == end) { batch.logits[m - 1] = true; }
    opt_result_active = s.restoring ? nullptr : s.result;
    opt_outputs_internal = !s.restoring;
    opt_learn_labels = s.restoring ? 0 : s.n_labels;
    opt_learn_labels_done = s.labels_done;
    // Build one graph for the entire reply. Serving retains its configured batches;
    // the served context window and label denominator never change.
    const bool ok = [&] {
        struct restore_batches {
            llama_cparams & params;
            uint32_t batch, ubatch;
            ~restore_batches() { params.n_batch = batch; params.n_ubatch = ubatch; }
        } restore{cparams, cparams.n_batch, cparams.n_ubatch};
        if (!s.restoring && s.graph_nodes > 0) {
            cparams.n_batch = std::max<uint32_t>(cparams.n_batch, m);
            cparams.n_ubatch = std::max<uint32_t>(cparams.n_ubatch, m);
        }
        return decode(batch) == 0;
    }();
    s.labels_done = opt_learn_labels_done;
    // No training flags leak into a neighbour's decode between calls.
    opt_result_active = nullptr;
    opt_outputs_internal = false;
    opt_learn_labels = 0;
    opt_learn_labels_done = 0;
    synchronize(); // completion and its duration belong to this chunk, not the next turn
    if (s.graph_meta) { return LLAMA_OPT_LEARN_MORE; }
    if (!ok) {
        if (s.restoring) {
            opt_fail("live reply restoration failed: re-prefill its sequence");
            opt_reply.reset();
            return LLAMA_OPT_LEARN_FAILED;
        }
        s.failure = opt_failure.empty() ? "live reply training chunk failed" : opt_failure;
        s.cursor = s.n_train; // the next call restores, even after a refused chunk
        return LLAMA_OPT_LEARN_MORE;
    }
    s.cursor += m;
    if (s.restoring && s.cursor == end) {
        const auto failure = s.failure;
        opt_reply.reset();
        if (!failure.empty()) {
            opt_fail(failure.c_str());
            return LLAMA_OPT_LEARN_FAILED;
        }
        return LLAMA_OPT_LEARN_DONE;
    }
    return LLAMA_OPT_LEARN_MORE;
}

bool llama_context::opt_learn(llama_seq_id seq, llama_pos n_cached, const llama_token * tokens,
        const llama_token * labels, int32_t n, ggml_opt_result_t result) {
    if (!opt_learn_begin(seq, n_cached, tokens, labels, n, tokens, n, result)) { return false; }
    llama_opt_learn_status status;
    do { status = opt_learn_step(); } while (status == LLAMA_OPT_LEARN_MORE);
    return status == LLAMA_OPT_LEARN_DONE;
}

bool llama_opt_learn_begin(llama_context * ctx, llama_seq_id seq, llama_pos prefix,
        const llama_token * tokens, const llama_token * labels, int32_t n,
        const llama_token * restore_tokens, int32_t n_restore, ggml_opt_result_t result) {
    return ctx->opt_learn_begin(seq, prefix, tokens, labels, n, restore_tokens, n_restore, result);
}

llama_opt_learn_status llama_opt_learn_step(llama_context * ctx) { return ctx->opt_learn_step(); }
llama_opt_learn_status llama_opt_learn_graph_step(llama_context * ctx) {
    if (ctx->opt_reply && ctx->opt_reply->graph_nodes == 0) {
        ctx->opt_reply->graph_nodes = 1;
    }
    return ctx->opt_learn_step();
}
bool llama_opt_learn_draining_graph(const llama_context * ctx) {
    return ctx->opt_reply && ctx->opt_reply->drain_graph && ctx->opt_reply->graph_meta;
}
bool llama_opt_learn_drain(llama_context * ctx) { return ctx->opt_learn_drain(); }
bool llama_opt_learn_needs_gradient(const llama_context * ctx) { return ctx->opt_learn_needs_gradient(); }

bool llama_opt_learn(
        struct llama_context * ctx,
        llama_seq_id           seq,
        llama_pos              n_cached,
        const llama_token    * tokens,
        const llama_token    * labels,
        int32_t                n,
        ggml_opt_result_t      result) {
    return ctx->opt_learn(seq, n_cached, tokens, labels, n, result);
}

// Both arenas remain charged to the same physical workspace authority. The training
// arena is retained, not a reusable discount against the neighbour's allocation.
bool llama_context::opt_admit_serving_workspace(ggml_backend_buffer_type_t buft,
        size_t required, size_t held, void * data) {
    auto & ctx = *static_cast<llama_context *>(data);
    if (!ctx.opt_workspace_admission || !ctx.opt_reply || !ctx.opt_reply->serving_sched) { return false; }
    const auto training = ctx.opt_reply->serving_sched.get(); // swapped out during serving
    size_t retained = 0;
    for (int i = 0; i < ggml_backend_sched_get_n_backends(training); ++i) {
        if (ggml_backend_sched_get_buffer_type(training, ggml_backend_sched_get_backend(training, i)) == buft) {
            const auto bytes = ggml_backend_sched_get_buffer_size(training, ggml_backend_sched_get_backend(training, i));
            if (bytes > std::numeric_limits<size_t>::max() - retained) { return false; }
            retained = bytes;
            break; // backend aliases name the same gallocr buffer
        }
    }
    if (std::max(required, held) > std::numeric_limits<size_t>::max() - retained) { return false; }
    return ctx.opt_workspace_admission(buft, retained + std::max(required, held), retained + held,
            ctx.opt_workspace_admission_data);
}

int llama_context::opt_decode_beside_graph(const llama_batch & batch) {
    if (!opt_reply || !opt_reply->graph_meta || !opt_workspace_admission || batch.n_tokens <= 0) { return -3; }
    for (int32_t i = 0; i < batch.n_tokens; ++i) {
        if (batch.label && batch.label[i] >= 0) { return -3; }
        const int count = batch.n_seq_id ? batch.n_seq_id[i] : 1;
        for (int j = 0; j < count; ++j) {
            if ((batch.seq_id ? batch.seq_id[i][j] : 0) == opt_reply->seq) { return -3; }
        }
    }
    // Updating shared KV can invalidate tensors retained by the backward graph.
    // Defer without taking the memory owner's queued shifts or copies.
    if (memory && memory->has_pending_update()) { return -3; }
    auto & state = *opt_reply;
    if (!state.serving_sched) {
        const auto nodes = graph_max_nodes(n_ubatch());
        state.serving_sched.reset(ggml_backend_sched_new(backend_ptrs.data(), backend_buft.data(),
                backend_ptrs.size(), nodes, false, cparams.op_offload));
        state.serving_graph.reset(new llm_graph_result(nodes));
        state.serving_reserve.reset(new llm_graph_result(nodes));
    }
    struct restore_owner {
        llama_context & ctx;
        opt_live_reply & state;
        llama_cparams params;
        bool reserve;
        ~restore_owner() {
            ggml_backend_sched_synchronize(ctx.sched.get());
            ctx.sched.swap(state.serving_sched);
            ctx.gf_res_prev.swap(state.serving_graph);
            ctx.gf_res_reserve.swap(state.serving_reserve);
            ctx.cparams = params;
            ctx.sched_need_reserve = reserve;
            ctx.opt_serving_continuation = false;
        }
    } guard{*this, state, cparams, sched_need_reserve};
    sched.swap(state.serving_sched);
    gf_res_prev.swap(state.serving_graph);
    gf_res_reserve.swap(state.serving_reserve);
    sched_need_reserve = false;
    opt_serving_continuation = true;
    return decode(batch);
}
