// Training: llama_context's optimizer members (opt_init, opt_epoch, the walk in
// opt_epoch_iter) and the llama_opt_* interface, apart from inference so each reads as one concern.
#include "llama-context.h"
#include "llama-walk.h"

#include "ggml.h"
#include "llama-batch.h"
#include "llama-impl.h"
#include "llama-memory-hybrid.h"
#include "llama-memory-recurrent.h"
#include "llama-kv-cache.h"
#include "llama-model.h"
#include "llama.h"

#include <algorithm>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

//
// training
//

// Returns whether the tensor was made a trainable parameter.
static bool llama_set_param(struct ggml_tensor * tensor, llama_opt_param_filter param_filter, void * userdata) {
    if (!tensor || tensor->type != GGML_TYPE_F32) {
        return false;
    }
    if (!param_filter(tensor, userdata)) {
        return false;
    }
    if (strcmp(tensor->name, "token_embd.weight") == 0) {
        return false; // FIXME
    }
    if (strcmp(tensor->name, "rope_freqs.weight") == 0) {
        return false; // FIXME
    }
    ggml_set_param(tensor);
    return true;
}

void llama_context::opt_init(struct llama_model * model, struct llama_opt_params lopt_params) {
    GGML_ASSERT(!opt_ctx);
    // The training window lives on this context, not the shared model (a serving context may
    // be running on the same weights).
    opt_n_ctx_train = lopt_params.n_ctx_train > 0 ? lopt_params.n_ctx_train : n_ctx();
    opt_walk_exact   = lopt_params.walk_exact;
    opt_walk_horizon_req = lopt_params.walk_horizon;
    opt_walk_horizon     = lopt_params.walk_horizon;
    const uint32_t n_batch     = std::min(this->n_batch(),  opt_n_ctx_train);
    const uint32_t n_ubatch    = std::min(this->n_ubatch(), n_batch);
    GGML_ASSERT(opt_n_ctx_train % n_batch  == 0);
    GGML_ASSERT(n_batch                    % n_ubatch == 0);
    // A training graph cannot backprop through the KV cache (SET_ROWS writes in place and
    // attention reads the cache as a leaf), so attention consumes this chunk's K/V directly
    // and the context before it from the cache as a constant (build_attn); opt_epoch_iter
    // walks the window decoding context and training each labelled chunk. One chunk is one
    // batch is one ubatch: a chunk's backward needs all of its own K/V in one graph.
    GGML_ASSERT(n_ubatch == n_batch && "training needs one ubatch per batch: set -ub = -b");
    // In a flash-attention context the training graph reads attention through FLASH_ATTN_EXT too
    // (build_attn_mha), whose backward is tiled over the window; flash attention also serves the
    // walk's context decodes and lets the cache hold V un-transposed.
    cparams.training = true;
    // The graph result and the scheduler were sized at context creation: before training
    // (no backward pass) and before any adapter attached since (its nodes uncounted).
    // Size both again for the training graph, from the same number, plus the nodes each
    // attention layer's tiled backward adds for this window (one count, owned by ggml).
    {
        size_t max_nodes = graph_max_nodes(n_ubatch);
        if (cparams.flash_attn) {
            max_nodes += (size_t) model->hparams.n_layer() * ggml_flash_attn_ext_back_n_nodes(opt_n_ctx_train, 0);
        }
        gf_res_prev.reset(new llm_graph_result(max_nodes));
        sched.reset(ggml_backend_sched_new(backend_ptrs.data(), backend_buft.data(), backend_ptrs.size(),
                    max_nodes, cparams.pipeline_parallel, cparams.op_offload));
    }

    ggml_opt_params opt_params = ggml_opt_default_params(sched.get(), GGML_OPT_LOSS_TYPE_CROSS_ENTROPY);
    opt_params.opt_period      = n_batch / n_ubatch;
    opt_params.get_opt_pars    = lopt_params.get_opt_pars;
    // per-layer recompute: the layer outputs are the checkpoints (llm_graph names them l_out-N)
    opt_params.checkpoint_prefix = lopt_params.recompute ? "l_out" : nullptr;
    opt_params.get_opt_pars_ud = lopt_params.get_opt_pars_ud;
    opt_params.optimizer       = lopt_params.optimizer_type;
    opt_ctx = ggml_opt_init(opt_params);
    ggml_opt_set_alloc_budget(opt_ctx, opt_memory_budget);

    llama_opt_param_filter param_filter = lopt_params.param_filter;
    void * param_filter_ud              = lopt_params.param_filter_ud;

    if (lopt_params.adapter) {
        // LoRA-only: the adapter's A/B are the parameters, the base is frozen. An adapter
        // that is not attached is not in the graph, so it would train nothing while every
        // epoch reported a loss: refuse it here instead.
        const auto it = loras->find(lopt_params.adapter);
        GGML_ASSERT(it != loras->end() && it->second != 0.0f && "the LoRA to train must be attached with a non-zero scale");
        int64_t n_params = 0;
        for (auto & [name, ab] : lopt_params.adapter->ab_map) {
            GGML_ASSERT(ab.a->type == GGML_TYPE_F32 && ab.b->type == GGML_TYPE_F32 && "a trainable LoRA holds F32 A/B tensors");
            n_params += llama_set_param(ab.a, param_filter, param_filter_ud);
            n_params += llama_set_param(ab.b, param_filter, param_filter_ud);
        }
        GGML_ASSERT(n_params > 0 && "the param filter selected none of the adapter's tensors");
        LLAMA_LOG_INFO("%s: LoRA-only training: %lld adapter tensors trainable, base frozen\n", __func__, (long long) n_params);
        return;
    }

  //llama_set_param(model->tok_embd,        param_filter, param_filter_ud); // FIXME
    llama_set_param(model->type_embd,       param_filter, param_filter_ud);
    llama_set_param(model->pos_embd,        param_filter, param_filter_ud);
    llama_set_param(model->tok_norm,        param_filter, param_filter_ud);
    llama_set_param(model->tok_norm_b,      param_filter, param_filter_ud);
    llama_set_param(model->output_norm,     param_filter, param_filter_ud);
    llama_set_param(model->output_norm_b,   param_filter, param_filter_ud);
    llama_set_param(model->output,          param_filter, param_filter_ud);
    llama_set_param(model->output_b,        param_filter, param_filter_ud);
    llama_set_param(model->output_norm_enc, param_filter, param_filter_ud);
    llama_set_param(model->cls,             param_filter, param_filter_ud);
    llama_set_param(model->cls_b,           param_filter, param_filter_ud);
    llama_set_param(model->cls_out,         param_filter, param_filter_ud);
    llama_set_param(model->cls_out_b,       param_filter, param_filter_ud);
    llama_set_param(model->cls_norm,        param_filter, param_filter_ud);

    for (struct llama_layer & layer : model->layers) {
        for (size_t i = 0; i < sizeof(layer)/sizeof(struct ggml_tensor *); ++i) {
            llama_set_param(reinterpret_cast<struct ggml_tensor **>(&layer)[i], param_filter, param_filter_ud);
        }
    }
}

// THE WALK (Fable's design, continuum 2026-10-06): one pass over a window in one context. A run
// with no label is context she did not write: a plain decode into the cache, no backward. A
// labelled run is her reply: it trains in chunks of n_batch, each attending to everything before
// it read from the cache as a constant (build_attn, cparams.training), then the reply is decoded
// into the cache under the adapter as it now is, so the next reply sees it. Memory is chunk x
// window, never window x window, so her whole lived window trains. Nothing after the last label
// is computed. One per window; a member of the context, so it runs on the context's own state.
struct llama_context::opt_walk {
    llama_context                  & lctx;
    ggml_opt_dataset_t               dataset;
    ggml_opt_result_t                result;
    const std::vector<llama_token> & tokens;
    const std::vector<llama_token> & labels;
    llama_batch                    & batch;
    ggml_opt_epoch_callback          callback;
    bool                             train;
    int64_t                          idata_in_loop;
    int64_t                          ndata_in_loop;
    int64_t                          t_loop_start;
    uint32_t                         n_ctx;
    uint32_t                         n_batch;
    llama_walk_plan                  plan;

    // a recurrent part (pure recurrent or hybrid) carries its state across chunks; a hybrid's
    // attention part removes a chunk's cells like a plain cache
    llama_memory_recurrent * recr = nullptr;
    llama_kv_cache         * attn = nullptr;

    bool first_unit = true;
    // THE EXACT WALK's chunk being trained (null in the plain walk) and its host gradients
    const llama_walk_chunk * xc = nullptr;
    llama_walk_acc           acc;

    opt_walk(llama_context & lctx, ggml_opt_dataset_t dataset, ggml_opt_result_t result,
            const std::vector<llama_token> & tokens, const std::vector<llama_token> & labels, llama_batch & batch,
            ggml_opt_epoch_callback callback, bool train, int64_t idata_in_loop, int64_t ndata_in_loop, int64_t t_loop_start,
            uint32_t n_ctx, uint32_t n_batch)
        : lctx(lctx), dataset(dataset), result(result), tokens(tokens), labels(labels), batch(batch), callback(callback),
          train(train), idata_in_loop(idata_in_loop), ndata_in_loop(ndata_in_loop), t_loop_start(t_loop_start),
          n_ctx(n_ctx), n_batch(n_batch), plan(llama_walk_plan::of(labels, n_ctx, n_batch)) {}

    bool yield_point();
    bool decode_span(uint32_t p0, uint32_t p1);
    bool train_chunk(uint32_t pos_ctx, uint32_t n_chunk);
    void run_exact();
    void run_plain();

    // the exact walk's graph flags in cparams, which build_attn and build_rs read
    void set_flags(bool exact, uint32_t grad_from, bool surrogate, bool state_surrogate) {
        lctx.cparams.walk_exact           = exact;
        lctx.cparams.walk_grad_from       = grad_from;
        lctx.cparams.walk_surrogate       = surrogate;
        lctx.cparams.walk_state_surrogate = state_surrogate;
    }
};

// THE YIELD POINT IS THE CHUNK, not the window: a turn that arrives mid-walk waits for one chunk,
// never for the whole example (measured on the 5090: one callback per example made a turn wait for
// its entire walk, 1.2-1.8 s on a 1.5B at 15k and 13.5 s on Fable's run). The first unit is the
// window's own boundary, which opt_epoch has already offered.
bool llama_context::opt_walk::yield_point() {
    if (first_unit) {
        first_unit = false;
    } else if (lctx.opt_step_callback && !lctx.opt_step_callback(train, lctx.opt_step_callback_data)) {
        lctx.opt_stop_requested.store(true);
    }
    return !lctx.opt_stop_requested.load(std::memory_order_relaxed);
}

// a plain inference decode of [p0, p1) into the cache: the training graph's attention never
// writes the cache, so context and finished replies reach it this way
bool llama_context::opt_walk::decode_span(uint32_t p0, uint32_t p1) {
    llama_cparams & cparams = lctx.cparams;
    const bool training = cparams.training;
    cparams.training = false;
    lctx.gf_res_prev->reset();
    bool ok = true;
    for (uint32_t c0 = p0; c0 < p1 && ok; c0 += n_batch) {
        if (!yield_point()) {
            cparams.training = training;
            lctx.gf_res_prev->reset();
            return false; // stopped or cancelled at a chunk boundary: not a decode failure
        }
        const uint32_t c1 = std::min(c0 + n_batch, p1);
        llama_walk_fill(batch, tokens, c0, c1 - c0);
        batch.logits[c1 - c0 - 1] = true;
        ok = lctx.decode(batch) == 0;
    }
    cparams.training = training;
    lctx.gf_res_prev->reset();
    if (!ok) {
        LLAMA_LOG_ERROR("%s: the context decode of [%u, %u) failed\n", __func__, p0, p1);
    }
    return ok;
}

// one training chunk [pos_ctx, pos_ctx + n_chunk): forward and backward over it alone, attending
// to the cached context before it
// One chunk of a walk is one llama_decode with labels: the batch loop is decode's, and the
// extras switch on in process_ubatch (llama_context::opt_ubatch). This function only fills the
// chunk's tokens and labels and says whose walk it is.
bool llama_context::opt_walk::train_chunk(uint32_t pos_ctx, uint32_t n_chunk) {
    llama_walk_fill(batch, tokens, pos_ctx, n_chunk);
    for (uint32_t pos_batch = 0; pos_batch < n_chunk; ++pos_batch) {
        // an exact walk's context chunk has no label but still trains (its K/V carry later
        // chunks' gradient): one output row, weighted zero, so the graph has its loss
        batch.logits[pos_batch] = labels[pos_ctx + pos_batch] >= 0
            || (xc != nullptr && xc->n_labels == 0 && pos_batch + 1 == n_chunk);
        batch.label[pos_batch] = labels[pos_ctx + pos_batch];
    }

    lctx.opt_walk_active      = this;
    lctx.opt_result_active    = xc != nullptr && xc->n_labels == 0 ? nullptr : result;
    lctx.opt_train_unlabelled = xc != nullptr && xc->n_labels == 0;
    const int ret = lctx.decode(batch);
    lctx.opt_walk_active      = nullptr;
    lctx.opt_result_active    = nullptr;
    lctx.opt_train_unlabelled = false;
    std::fill(batch.label, batch.label + n_chunk, -1);

    if (ret != 0) {
        // a refusal has already stopped the epoch by name (opt_fail); anything else is decode's
        if (!lctx.opt_alloc_failed.load()) {
            lctx.opt_fail("the training chunk failed to decode (llama_decode returned " + std::to_string(ret) + ")");
        }
        return false;
    }
    if (callback && !(xc != nullptr && xc->n_labels == 0)) {
        callback(train, lctx.opt_ctx, dataset, result, idata_in_loop + pos_ctx/n_batch + 1, ndata_in_loop, t_loop_start);
    }
    return true;
}

bool llama_context::opt_ubatch(llm_graph_result * res, ggml_cgraph * gf, const llama_ubatch & ubatch) {
    opt_walk * walk = opt_walk_active;
    const llama_walk_chunk * xc = walk != nullptr ? walk->xc : nullptr;

    struct ggml_context * ctx_compute_opt;
    {
        const size_t size_gf = ggml_graph_size(gf);
        const size_t size_meta = 4*size_gf*ggml_tensor_overhead() + 2*ggml_graph_overhead_custom(size_gf, /*grads = */ true);
        struct ggml_init_params params = {
            /*.mem_size   =*/ size_meta,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ctx_compute_opt = ggml_init(params);
    }
    ggml_opt_prepare_alloc(opt_ctx, ctx_compute_opt, gf, res->get_inp_tokens(), res->get_logits());
    if (xc != nullptr) {
        ggml_opt_set_next_step(opt_ctx, xc->period_end, xc->loss_scale, res->t_walk_surrogate);
        for (const auto & io : res->t_walk) {
            if (io.tk) {
                ggml_opt_accumulate_grad(opt_ctx, io.tk, io.ak);
                ggml_opt_accumulate_grad(opt_ctx, io.tv, io.av);
            }
        }
        for (const auto & ws : res->t_walk_state) {
            if (ws.tap) {
                ggml_opt_accumulate_grad(opt_ctx, ws.tap, walk->acc.state.at(ws.cache));
            }
        }
    }
    const bool train = walk == nullptr || walk->train;
    if (!ggml_opt_alloc(opt_ctx, train)) {
        ggml_free(ctx_compute_opt);
        opt_fail(ggml_opt_refusal(opt_ctx),
            std::string("the ") + (train ? "training" : "evaluation") + " graph was refused (window " + std::to_string(walk != nullptr ? walk->n_ctx : ubatch.n_tokens) + ")");
        return false;
    }

    res->set_inputs(&ubatch);
    {
        // one-hot rows for the labelled positions, in the order the ubatch emits its outputs
        // (the order the logits rows take); an unlabelled output row stays zero, so it adds no
        // LOCAL loss (its sequence's context still carries gradient to later labelled rows)
        struct ggml_tensor * labels_t = ggml_opt_labels(opt_ctx);
        GGML_ASSERT(labels_t->ne[1] == (int64_t) n_outputs);
        ggml_set_zero(labels_t);
        const float onef = 1.0f;
        uint32_t row = 0;
        for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
            if (ubatch.output[i] == 0) {
                continue;
            }
            const llama_token y = ubatch.label != nullptr ? ubatch.label[i] : -1;
            GGML_ASSERT(y < labels_t->ne[0]);
            if (y >= 0) {
                ggml_backend_tensor_set(labels_t, &onef, (row*labels_t->ne[0] + y)*sizeof(float), sizeof(float));
            }
            ++row;
        }
        GGML_ASSERT(row == n_outputs);
    }
    if (xc != nullptr) {
        // the later chunk's dL/d(entry state) is this chunk's exit gradient; then the
        // accumulator takes this chunk's own
        for (const auto & ws : res->t_walk_state) {
            ggml_tensor * a = walk->acc.state.at(ws.cache);
            if (ws.gs) {
                GGML_ASSERT(ggml_nbytes(a) == ggml_nbytes(ws.gs));
                ggml_backend_tensor_set(ws.gs, a->data, 0, ggml_nbytes(ws.gs));
            }
            memset(a->data, 0, ggml_nbytes(a));
        }
    }
    ggml_opt_eval(opt_ctx, opt_result_active);
    ggml_free(ctx_compute_opt);
    if (const char * why = ggml_opt_refusal(opt_ctx); why[0] != '\0') {
        // the backend failed the graph (ggml_opt_eval): the run fails, as a refused graph does
        opt_fail(why);
        return false;
    }
    if (xc != nullptr && !res->t_walk_state.empty()) {
        walk->acc.state_ready = true;
    }
    return true;
}

// THE EXACT WALK. The plain walk steps per chunk and stops the gradient at every chunk boundary: a
// reply learns only through its own chunk's K/V, never through what an earlier chunk computed for
// it. Here the window is decoded once under the adapter as it is, then trained in REVERSE: each
// chunk's graph attends to the cache, with its last walk_horizon prefix positions behind a
// gradient tap, and adds the surrogate <K, G> + <V, G> on its own K/V, where G is what every later
// chunk put there. Its backward then carries that gradient through itself into the adapter and
// into its own prefix's G. ONE optimizer step per window, on the gradient of the window's loss
// (exact within the horizon). Memory stays chunk x window; the cost is one decode plus one forward
// and backward per chunk, context chunks included.
// THE INVARIANT THAT MAKES THE REVERSE PASS VALID (Fable): the adapter does not change between the
// forward decode and the last chunk's backward. One optimizer step per window, taken by the chunk
// the reverse pass trains last, is what keeps each chunk's recomputed K/V (and recurrent state)
// equal to the cached values its successors attended to; a step inside the window would make
// every surrogate a gradient of a different function.
// A recurrent model's state is carried the same way, one chunk at a time: the state entering a
// chunk is tapped into a host accumulator, the state it leaves meets the next chunk's gradient
// (build_rs, build_walk_state_exit). The reverse pass restores each chunk's entry state from a
// checkpoint taken at its boundary in the forward pass.
void llama_context::opt_walk::run_exact() {
    // Each window starts from the REQUESTED horizon: a halving that fit one window's device budget
    // must not shrink every later window and epoch, which may be shorter (Fable).
    lctx.opt_walk_horizon = lctx.opt_walk_horizon_req;
    // A serving context keeps rollback slots for drafts; the walk's graphs keep one state, the live
    // one (llm_graph_context::rs_keep), so they train in that context. Only a PENDING undo reads a
    // snapshot instead of the live state: that alone is refused, by name, never asserted, since
    // training runs inside the serving process (2026-10-07 04:09Z on the 5090).
    if (recr != nullptr && recr->rollback_pending(0)) {
        lctx.opt_fail("the exact walk cannot train a sequence with a pending recurrent rollback (n_rs_seq = "
            + std::to_string(lctx.cparams.n_rs_seq) + "): finish or drop the draft first");
        return;
    }
    plan.mark_exact(recr != nullptr, lctx.opt_walk_horizon);
    const std::vector<llama_walk_chunk> & chunks = plan.chunks;
    const uint32_t n_pos = (uint32_t) plan.last_label + 1;

    // THE HOST BUDGET (llama_walk_plan::stride_for): the K/V accumulators and the recurrent state
    // checkpoints are both known before anything runs
    const size_t kv_bytes    = llama_walk_acc::kv_bytes(lctx.model.hparams, n_pos);
    const size_t state_bytes = llama_walk_acc::state_bytes(recr);
    const size_t budget      = lctx.opt_walk_host_budget;
    const size_t stride      = plan.stride_for(budget, kv_bytes, state_bytes);
    const size_t bytes       = plan.host_bytes(stride, kv_bytes, state_bytes);
    lctx.opt_walk_host_bytes = std::max(lctx.opt_walk_host_bytes, bytes);
    if (budget > 0 && bytes > budget) {
        char why[512];
        snprintf(why, sizeof(why),
            "the exact walk needs %.1f MiB of host memory for this window (%u positions, the K/V gradient alone), over its budget of %.1f MiB: "
            "the K/V gradient grows with the window; neither the horizon nor the chunk size changes it",
            bytes/1048576.0, n_pos, budget/1048576.0);
        lctx.opt_fail(why);
        return;
    }
    if (stride > 1) {
        LLAMA_LOG_INFO("%s: the exact walk checkpoints the recurrent state every %zu chunks to fit %.1f MiB of host memory (%.1f MiB)\n",
                __func__, stride, budget/1048576.0, bytes/1048576.0);
    }
    if (!acc.init(lctx.model.hparams, recr, n_pos, lctx.cparams)) {
        lctx.opt_fail("the exact walk's gradient accumulator (host memory) could not be allocated");
        return;
    }

    // the window as the adapter now reads it, chunk by chunk: a recurrent model's state is
    // checkpointed at each chunk's start through the sequence-state API (the recurrent part only,
    // PARTIAL_ONLY), after waiting for the queued decode; a restore puts back the cell's position
    // with its rows, and an empty checkpoint (before the first chunk) restores the zero state. A
    // hand-rolled per-layer copy here skipped that wait and, on Metal, carried a torn mid-graph
    // state into the reverse pass (2026-10-11).
    std::vector<std::vector<uint8_t>> snaps(recr != nullptr ? chunks.size() : 0);
    const llama_state_seq_flags state_flags = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;
    for (size_t j = 0; j < chunks.size(); ++j) {
        if (recr != nullptr && j % stride == 0) {
            lctx.synchronize();
            snaps[j].resize(lctx.state_seq_get_size(0, state_flags));
            GGML_ASSERT(lctx.state_seq_get_data(0, snaps[j].data(), snaps[j].size(), state_flags) == snaps[j].size());
        }
        if (!decode_span(chunks[j].c0, chunks[j].c0 + chunks[j].n)) {
            return;
        }
    }
    for (int64_t j = (int64_t) chunks.size() - 1; j >= plan.first; --j) {
        const llama_walk_chunk & c = chunks[j];
        if (!c.needed) {
            continue;
        }
        if (!yield_point()) {
            break;
        }
        // attention: pop the cache to [0, c0); recurrent: restore the state at the chunk's
        // checkpoint and decode forward to its start (that decode rebuilds the attention K/V in
        // between too, under the same adapter: the same values). Run again before every retry
        // below: a refused chunk graph is refused AFTER its ubatch was applied, so the chunk's
        // cells already sit in the cache and the recurrent cell's position has moved to the
        // chunk's end; retrying on top of that fails to prepare the ubatch at all (the 5090,
        // 2026-10-10: "init_batch: failed to prepare attention ubatches" after "retrying at 512",
        // and the epoch stopped instead of shrinking the horizon). Always a plain decode: the
        // walk's graph flags are off whenever it runs.
        const auto rewind = [&]() -> bool {
            const size_t   cp    = recr != nullptr ? (size_t) j - (size_t) j % stride : (size_t) j;
            const uint32_t p_pop = chunks[cp].c0;
            const bool popped = recr != nullptr ? (attn == nullptr || attn->seq_rm(0, p_pop, -1)) : lctx.memory->seq_rm(0, p_pop, -1);
            if (!popped) {
                LLAMA_LOG_ERROR("%s: could not pop the cache to [0, %u) for the reverse pass\n", __func__, p_pop);
                lctx.opt_stop_requested.store(true);
                return false;
            }
            if (recr != nullptr) {
                lctx.synchronize();
                GGML_ASSERT(lctx.state_seq_set_data(0, snaps[cp].data(), snaps[cp].size(), state_flags) == snaps[cp].size());
                if (p_pop < c.c0 && !decode_span(p_pop, c.c0)) {
                    return false;
                }
            }
            return true;
        };
        if (!rewind()) {
            break;
        }
        const bool state_surrogate = recr != nullptr && j < (int64_t) chunks.size() - 1 && acc.state_ready;
        // THE HORIZON FITS THE DEVICE. Its gradient taps are the chunk graph's largest growth with
        // the window (layers x horizon x (k + v) floats, twice with their gradient); the graph
        // preflight refuses a chunk that does not fit before allocating anything, so a memory
        // refusal halves the horizon (whole chunks) and the chunk is tried again; every later
        // chunk keeps the smaller one. The forward still attends to the whole window and the
        // recurrent state is carried exactly; only attention gradient past the horizon is dropped.
        // A refusal at a horizon of one chunk, or for a node the device cannot run, stands.
        llama_walk_chunk cc = c;
        bool ok;
        for (;;) {
            cc.grad_from = llama_walk_plan::grad_from(cc.c0, lctx.opt_walk_horizon);
            set_flags(true, cc.grad_from, c.surrogate, state_surrogate);
            xc = &cc;
            ok = train_chunk(cc.c0, cc.n);
            xc = nullptr;
            set_flags(false, 0, false, false);
            const uint32_t span = cc.c0 - cc.grad_from;
            if (ok || !lctx.opt_failure.empty() || !lctx.opt_alloc_failed.load() || span <= n_batch) {
                break;
            }
            lctx.opt_walk_horizon = std::max<uint32_t>(n_batch, (span/2)/n_batch*n_batch);
            LLAMA_LOG_WARN("%s: the chunk at %u did not fit with a gradient horizon of %u positions: retrying at %u\n",
                    __func__, cc.c0, span, lctx.opt_walk_horizon);
            lctx.opt_alloc_failed.store(false);
            lctx.opt_stop_requested.store(false);
            if (!rewind()) {
                ok = false;
                break;
            }
        }
        lctx.opt_walk_horizon_used = lctx.opt_walk_horizon;
        if (!ok) {
            break;
        }
    }
    acc.state_ready = false;
}

// THE PLAIN WALK: each labelled chunk steps on its own gradient, then joins the context
void llama_context::opt_walk::run_plain() {
    const int64_t last_label = plan.last_label;
    for (const llama_walk_chunk & c : plan.chunks) {
        if (lctx.opt_stop_requested.load(std::memory_order_relaxed)) {
            return;
        }
        const uint32_t c0 = c.c0;
        const uint32_t c1 = c.c0 + c.n;
        if (c.n_labels == 0) {
            if (!decode_span(c0, c1)) {
                return;
            }
            continue;
        }
        // The training forward advances a recurrent state in place, and a recurrent memory
        // cannot remove a partial range: snapshot the state to the scratch sequence first
        // (Fable), restore it after, then the decode below advances it from where the context
        // left it. The cache's own copy-on-write keeps it on the device; the state is per
        // sequence and small.
        const bool snapshot = recr != nullptr && c1 <= (uint32_t) last_label;
        if (!yield_point()) {
            return;
        }
        if (snapshot) {
            recr->seq_cp(0, 1, -1, -1);
        }
        if (!train_chunk(c0, c.n)) {
            return;
        }
        if (snapshot) {
            recr->seq_rm(0, -1, -1);
            recr->seq_cp(1, 0, -1, -1);
            recr->seq_rm(1, -1, -1);
        }
        // the chunk joins the context under the adapter as it now is: the training graph never
        // writes the cache, so its cells hold nothing until this decode
        if (c1 <= (uint32_t) last_label) {
            // drop the chunk's empty cells from the attention memory, then decode them for real
            const bool removed = recr != nullptr ? (attn == nullptr || attn->seq_rm(0, c0, -1)) : lctx.memory->seq_rm(0, c0, -1);
            if (!removed) {
                LLAMA_LOG_ERROR("%s: could not remove the trained chunk [%u, %u) from the cache before decoding it\n", __func__, c0, c1);
                lctx.opt_stop_requested.store(true);
                return;
            }
            if (!decode_span(c0, c1)) {
                lctx.opt_stop_requested.store(true);
                return;
            }
        }
    }
}

void llama_context::opt_epoch_iter(
        ggml_opt_dataset_t               dataset,
        ggml_opt_result_t                result,
        const std::vector<llama_token> & tokens,
        const std::vector<llama_token> & labels_sparse,
        llama_batch                    & batch,
        ggml_opt_epoch_callback          callback,
        bool                             train,
        int64_t                          idata_in_loop,
        int64_t                          ndata_in_loop,
        int64_t                          t_loop_start) {
    GGML_ASSERT(opt_ctx);
    const uint32_t n_ctx   = opt_n_ctx_train;
    const uint32_t n_batch = std::min(this->n_batch(), n_ctx);

    memory->clear(true);

    opt_walk walk(*this, dataset, result, tokens, labels_sparse, batch, callback, train,
        idata_in_loop, ndata_in_loop, t_loop_start, n_ctx, n_batch);
    if (walk.plan.last_label < 0) {
        LLAMA_LOG_ERROR("%s: a training window with no labelled position: nothing to learn from it\n", __func__);
        return;
    }
    if (auto * hybrid = dynamic_cast<llama_memory_hybrid *>(memory.get())) {
        walk.recr = hybrid->get_mem_recr();
        walk.attn = hybrid->get_mem_attn();
    } else {
        walk.recr = dynamic_cast<llama_memory_recurrent *>(memory.get());
    }
    if (train && opt_walk_exact) {
        walk.run_exact();
        return;
    }
    // the plain walk snapshots a recurrent state into a scratch sequence
    if (walk.recr != nullptr && cparams.n_seq_max < 2) {
        LLAMA_LOG_ERROR("%s: a recurrent model trains with a scratch sequence for its state snapshot: create the training context with n_seq_max >= 2\n", __func__);
        opt_stop_requested.store(true);
        return;
    }
    walk.run_plain();
}

void llama_context::opt_fail(std::string why, const std::string & what) {
    LLAMA_LOG_ERROR("%s: %s%s%s: stopping the epoch\n", __func__, what.c_str(), what.empty() ? "" : ": ",
            why.empty() ? "it does not fit in device memory" : why.c_str());
    opt_failure = std::move(why);
    opt_alloc_failed.store(true);
    opt_stop_requested.store(true);
}

void llama_context::opt_epoch(
        ggml_opt_dataset_t        dataset,
        ggml_opt_result_t         result_train,
        ggml_opt_result_t         result_eval,
        int64_t                   idata_split,
        ggml_opt_epoch_callback   callback_train,
        ggml_opt_epoch_callback   callback_eval) {
    const uint32_t n_ctx    = this->n_ctx();
    const uint32_t n_batch  = std::min(cparams.n_batch,  n_ctx);
    const uint32_t n_ubatch = std::min(cparams.n_ubatch, n_batch);
    const  int64_t ndata    = ggml_opt_dataset_ndata(dataset);

    GGML_ASSERT(idata_split >= 0);
    GGML_ASSERT(idata_split <= ndata);

    const uint32_t ubatch_per_ctx = n_ctx / n_ubatch;

    struct llama_batch batch = llama_batch_init(n_batch, 0, 1);
    std::vector<llama_token>        tokens(n_ctx);
    std::vector<llama_token> labels_sparse(n_ctx);

    // the training range [0, idata_split), then the evaluation range [idata_split, ndata)
    const struct {
        bool                    train;
        int64_t                 i0, i1;
        ggml_opt_result_t       result;
        ggml_opt_epoch_callback callback;
    } ranges[] = {
        { true,  0,           idata_split, result_train, callback_train },
        { false, idata_split, ndata,       result_eval,  callback_eval  },
    };
    for (const auto & r : ranges) {
        const int64_t t_loop_start  = ggml_time_us();
        const int64_t ndata_in_loop = (r.i1 - r.i0)*ubatch_per_ctx;
        for (int64_t idata = r.i0; idata < r.i1; ++idata) {
            if (opt_step_callback && !opt_stop_requested.load(std::memory_order_relaxed)
                    && !opt_step_callback(r.train, opt_step_callback_data)) {
                opt_stop_requested.store(true);
            }
            if (opt_stop_requested.load(std::memory_order_relaxed)) {
                break;
            }
            ggml_opt_dataset_get_batch_host(dataset, tokens.data(), n_ctx*sizeof(llama_token), labels_sparse.data(), idata);
            opt_epoch_iter(dataset, r.result, tokens, labels_sparse, batch,
                r.callback, r.train, (idata - r.i0)*ubatch_per_ctx, ndata_in_loop, t_loop_start);
        }
    }

    llama_batch_free(batch);
}

//
// training
//

bool llama_opt_param_filter_all(const struct ggml_tensor * tensor, void * userdata) {
    GGML_UNUSED(tensor);
    GGML_UNUSED(userdata);
    return true;
}

void llama_opt_init(struct llama_context * ctx, struct llama_model * model, struct llama_opt_params lopt_params) {
    ctx->opt_alloc_failed.store(false);
    ctx->opt_failure.clear();
    ctx->opt_init(model, lopt_params);
}

void llama_opt_stop(struct llama_context * ctx, bool stop) {
    ctx->opt_stop_requested.store(stop, std::memory_order_relaxed);
}

void llama_opt_set_step_callback(struct llama_context * ctx, llama_opt_step_callback callback, void * user_data) {
    ctx->opt_step_callback = callback;
    ctx->opt_step_callback_data = user_data;
}

void llama_opt_set_walk_host_budget(struct llama_context * ctx, size_t bytes) {
    ctx->opt_walk_host_budget = bytes;
}

size_t llama_opt_walk_host_bytes(struct llama_context * ctx) {
    return ctx->opt_walk_host_bytes;
}

uint32_t llama_opt_walk_horizon(struct llama_context * ctx) {
    return ctx->opt_walk_horizon_used;
}

void llama_opt_set_memory_budget(struct llama_context * ctx, size_t bytes) {
    ctx->opt_memory_budget = bytes;
}

size_t llama_context::opt_graph_bytes() const {
    return opt_ctx ? ggml_opt_peak_graph_bytes(opt_ctx) : 0;
}

size_t llama_opt_graph_bytes(struct llama_context * ctx) {
    return ctx->opt_graph_bytes();
}

size_t llama_opt_context_device_bytes(struct llama_context * ctx) {
    size_t bytes = 0;
    for (const auto & [buft, data] : ctx->memory_breakdown()) {
        // data.context only: the compute buffers are where the training graph is allocated
        // (the optimizer shares this context's scheduler, whose allocator replaces them with the
        // graph's), so the graph's measured size already covers them (Cormac on #52)
        if (!ggml_backend_buft_is_host(buft)) {
            bytes += data.context;
        }
    }
    return bytes;
}

bool llama_opt_failed(struct llama_context * ctx) {
    return ctx->opt_alloc_failed.load();
}

const char * llama_opt_failure(struct llama_context * ctx) {
    return ctx->opt_failure.c_str();
}

void llama_opt_epoch(
        struct llama_context    * ctx,
        ggml_opt_dataset_t        dataset,
        ggml_opt_result_t         result_train,
        ggml_opt_result_t         result_eval,
        int64_t                   idata_split,
        ggml_opt_epoch_callback   callback_train,
        ggml_opt_epoch_callback   callback_eval) {
    ctx->opt_epoch(
        dataset,
        result_train,
        result_eval,
        idata_split,
        callback_train,
        callback_eval);
}
