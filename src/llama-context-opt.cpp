// Training: llama_context's optimizer members (opt_init, opt_epoch, the walk in
// opt_epoch_iter) and the llama_opt_* interface, apart from inference so each reads as one concern.
#include "llama-context.h"

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

    // THE WALK (Fable's design, continuum 2026-10-06): one pass over the window in one
    // context. A run with no label is context she did not write: a plain decode into the
    // cache, no backward. A labelled run is her reply: it trains in chunks of n_batch, each
    // attending to everything before it read from the cache as a constant (build_attn,
    // cparams.training), then the reply is decoded into the cache under the adapter as it
    // now is, so the next reply sees it. Memory is chunk x window, never window x window, so
    // her whole lived window trains. Nothing after the last label is computed.
    int64_t last_label = -1;
    for (int64_t i = (int64_t) n_ctx - 1; i >= 0; --i) {
        if (labels_sparse[i] >= 0) {
            last_label = i;
            break;
        }
    }
    if (last_label < 0) {
        LLAMA_LOG_ERROR("%s: a training window with no labelled position: nothing to learn from it\n", __func__);
        return;
    }

    // THE YIELD POINT IS THE CHUNK, not the window: a turn that arrives mid-walk waits for one
    // chunk, never for the whole example (measured on the 5090: one callback per example made a
    // turn wait for its entire walk, 1.2-1.8 s on a 1.5B at 15k and 13.5 s on Fable's run). The
    // first unit is the window's own boundary, which opt_epoch has already offered.
    bool first_unit = true;
    auto yield_point = [&]() -> bool {
        if (first_unit) {
            first_unit = false;
        } else if (opt_step_callback && !opt_step_callback(train, opt_step_callback_data)) {
            opt_stop_requested.store(true);
        }
        return !opt_stop_requested.load(std::memory_order_relaxed);
    };

    // a plain inference decode of [p0, p1) into the cache: the training graph's attention
    // never writes the cache, so context and finished replies reach it this way
    auto decode_span = [&](uint32_t p0, uint32_t p1) -> bool {
        const bool training = cparams.training;
        cparams.training = false;
        gf_res_prev->reset();
        bool ok = true;
        for (uint32_t c0 = p0; c0 < p1 && ok; c0 += n_batch) {
            if (!yield_point()) {
                cparams.training = training;
                gf_res_prev->reset();
                return false; // stopped or cancelled at a chunk boundary: not a decode failure
            }
            const uint32_t c1 = std::min(c0 + n_batch, p1);
            batch.n_tokens = c1 - c0;
            for (uint32_t i = 0; i < c1 - c0; ++i) {
                batch.token   [i]    = tokens[c0 + i];
                batch.pos     [i]    = c0 + i;
                batch.n_seq_id[i]    = 1;
                batch.seq_id  [i][0] = 0;
                batch.logits  [i]    = i + 1 == c1 - c0;
            }
            const int rc = decode(batch);
            ok = rc == 0;
        }
        cparams.training = training;
        gf_res_prev->reset();
        if (!ok) {
            LLAMA_LOG_ERROR("%s: the context decode of [%u, %u) failed\n", __func__, p0, p1);
        }
        return ok;
    };

    // THE EXACT WALK's state for the chunk being trained (null in the plain walk): its place in
    // the window's one optimizer period, and the gradient accumulated on the window's cached K/V
    struct exact_chunk {
        uint32_t c0;
        uint32_t n;
        uint32_t n_labels;
        uint32_t grad_from;  // its gradient taps cover cached positions [grad_from, c0)
        bool     surrogate;  // a later chunk's gradient sits on its own K/V
        bool     needed;
        bool     period_end; // the window's optimizer step
        float    loss_scale; // its labelled positions over the window's
    };
    const exact_chunk * xc = nullptr;
    // THE EXACT WALK's gradients, in host memory, each added into in place by the backward:
    // per attention layer dL/d(cached K) and dL/d(cached V) of the window, [n_embd_*_gqa,
    // position] F32 (null for a recurrent layer), which the graph reads through cparams (never
    // outliving it, whatever way the walk returns); per recurrent state tensor (a layer's conv
    // state r_l and its recurrent state s_l) dL/d(the state entering the chunk trained last),
    // which is what the chunk before it puts on the state it leaves (state_ready once one is in)
    struct walk_acc_t {
        llama_cparams &            cp;
        std::vector<ggml_tensor *> k, v;
        std::unordered_map<const ggml_tensor *, ggml_tensor *> state;
        bool                       state_ready = false;
        ggml_context_ptr           ctx;
        ggml_backend_buffer_ptr    buf;
        ~walk_acc_t() { cp.walk_acc_k = nullptr; cp.walk_acc_v = nullptr; }
    } walk_acc { cparams, {}, {}, {}, false, nullptr, nullptr };

    // one training chunk [pos_ctx, pos_ctx + n_tokens): forward and backward over it alone,
    // attending to the cached context before it
    auto train_chunk = [&](uint32_t pos_ctx, uint32_t n_chunk) -> bool {
        batch.n_tokens = n_chunk;
        for (uint32_t pos_batch = 0; pos_batch < n_chunk; ++pos_batch) {
            batch.token   [pos_batch]    = tokens[pos_ctx + pos_batch];
            batch.pos     [pos_batch]    = pos_ctx + pos_batch;
            batch.n_seq_id[pos_batch]    = 1;
            batch.seq_id  [pos_batch][0] = 0;
            // an exact walk's context chunk has no label but still trains (its K/V carry later
            // chunks' gradient): one output row, weighted zero, so the graph has its loss
            batch.logits  [pos_batch]    = labels_sparse[pos_ctx + pos_batch] >= 0
                || (xc != nullptr && xc->n_labels == 0 && pos_batch + 1 == n_chunk);
        }

        // output_all = false: the batch's own logits flags decide which rows exist
        if (!balloc->init(batch, model.vocab, nullptr, model.hparams.n_embd_inp(), cparams.kv_unified ? LLAMA_MAX_SEQ : cparams.n_seq_max, false)) {
            LLAMA_LOG_ERROR("%s: failed to initialize batch\n", __func__);
            return false;
        }

        const uint32_t n_tokens_all = balloc->get_n_tokens();

        n_queued_tokens += n_tokens_all;

        embd_seq.clear();

        const uint32_t n_outputs_all = balloc->get_n_outputs();
        // Unreachable: the walk below trains only chunks of a labelled run. It RETURNS from the
        // whole window, since a later chunk would need this one in the cache.
        if (n_outputs_all == 0) {
            LLAMA_LOG_ERROR("%s: a training batch with no labelled position (every label masked): nothing to learn from it\n", __func__);
            return false;
        }

        auto mctx = memory->init_batch(*balloc, cparams.n_ubatch, true);
        if (!mctx || mctx->get_status() != LLAMA_MEMORY_STATUS_SUCCESS) {
            LLAMA_LOG_ERROR("%s: could not initialize batch\n", __func__);
            return false;
        }

        // reserve output buffer
        if (output_reserve(n_outputs_all) < n_outputs_all) {
            LLAMA_LOG_ERROR("%s: could not reserve space for batch with %d outputs\n", __func__, n_outputs_all);
            GGML_ABORT("TODO: handle this error");
        };

        uint32_t pos_batch = 0;
        do {
            const auto & ubatch = mctx->get_ubatch();

            // the labelled rows of this ubatch, as decode counts its outputs
            {
                uint32_t n_outputs_ubatch = 0;
                for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
                    n_outputs_ubatch += (uint32_t) (ubatch.output[i] != 0);
                }
                n_outputs = n_outputs_ubatch;
            }

            if (!mctx->apply()) {
                LLAMA_LOG_ERROR("%s: failed to update the memory context\n", __func__);
                return false;
            }

            auto * res = gf_res_prev.get();

            const auto gparams = graph_params(res, ubatch, mctx.get(), graph_type());

            res->reset();

            auto * gf = model.build_graph(gparams);

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
                        ggml_opt_accumulate_grad(opt_ctx, ws.tap, walk_acc.state.at(ws.cache));
                    }
                }
            }
            if (!ggml_opt_alloc(opt_ctx, train)) {
                const char * why = ggml_opt_refusal(opt_ctx);
                opt_failure = why;
                LLAMA_LOG_ERROR("%s: the %s graph was refused (window %u): %s: stopping the epoch\n",
                                __func__, train ? "training" : "evaluation", n_ctx,
                                why[0] ? why : "it does not fit in device memory");
                ggml_free(ctx_compute_opt);
                opt_alloc_failed.store(true);
                opt_stop_requested.store(true);
                return false;
            }

            res->set_inputs(&ubatch);
            {
                // one-hot rows for the labelled positions only, in the order the ubatch
                // emits its outputs (the same order the logits rows take)
                struct ggml_tensor * labels = ggml_opt_labels(opt_ctx);
                GGML_ASSERT(labels->ne[1] == (int64_t) n_outputs);
                ggml_set_zero(labels);
                const float onef = 1.0f;
                uint32_t row = 0;
                for (uint32_t pos_ubatch = 0; pos_ubatch < ubatch.n_tokens; ++pos_ubatch) {
                    if (ubatch.output[pos_ubatch] == 0) {
                        continue;
                    }
                    const uint32_t ilabel = pos_ctx + pos_batch + pos_ubatch;
                    // an exact walk's context chunk: its one row has no label (weighted zero)
                    GGML_ASSERT((xc != nullptr && labels_sparse[ilabel] < 0) || (labels_sparse[ilabel] >= 0 && labels_sparse[ilabel] < labels->ne[0]));
                    if (labels_sparse[ilabel] >= 0) {
                        ggml_backend_tensor_set(labels, &onef, (row*labels->ne[0] + labels_sparse[ilabel])*sizeof(float), sizeof(float));
                    }
                    ++row;
                }
                GGML_ASSERT(row == n_outputs);
            }
            if (xc != nullptr) {
                // the later chunk's dL/d(entry state) is this chunk's exit gradient; then the
                // accumulator takes this chunk's own
                for (const auto & ws : res->t_walk_state) {
                    ggml_tensor * acc = walk_acc.state.at(ws.cache);
                    if (ws.gs) {
                        GGML_ASSERT(ggml_nbytes(acc) == ggml_nbytes(ws.gs));
                        ggml_backend_tensor_set(ws.gs, acc->data, 0, ggml_nbytes(ws.gs));
                    }
                    memset(acc->data, 0, ggml_nbytes(acc));
                }
            }
            ggml_opt_eval(opt_ctx, xc != nullptr && xc->n_labels == 0 ? nullptr : result);
            if (const char * why = ggml_opt_refusal(opt_ctx); why[0] != '\0') {
                // the backend failed the graph (ggml_opt_eval): the run fails, as a refused graph does
                opt_failure = why;
                LLAMA_LOG_ERROR("%s: %s: stopping the epoch\n", __func__, why);
                ggml_free(ctx_compute_opt);
                opt_alloc_failed.store(true);
                opt_stop_requested.store(true);
                return false;
            }
            if (xc != nullptr && !res->t_walk_state.empty()) {
                walk_acc.state_ready = true;
            }
            if (callback && !(xc != nullptr && xc->n_labels == 0)) {
                callback(train, opt_ctx, dataset, result, idata_in_loop + (pos_ctx + pos_batch)/n_batch + 1, ndata_in_loop, t_loop_start);
            }
            ggml_free(ctx_compute_opt);

            pos_batch += ubatch.n_tokens;
        } while (mctx->next());
        return true;
    };

    // a recurrent part (pure recurrent or hybrid) is snapshotted around each training chunk;
    // a hybrid's attention part removes the chunk's cells like a plain cache
    llama_memory_recurrent * recr = nullptr;
    llama_kv_cache         * attn = nullptr;
    if (auto * hybrid = dynamic_cast<llama_memory_hybrid *>(memory.get())) {
        recr = hybrid->get_mem_recr();
        attn = hybrid->get_mem_attn();
    } else {
        recr = dynamic_cast<llama_memory_recurrent *>(memory.get());
    }
    // the plain walk snapshots a recurrent state into a scratch sequence; the exact walk keeps host snapshots
    if (recr != nullptr && cparams.n_seq_max < 2 && !(train && opt_walk_exact)) {
        LLAMA_LOG_ERROR("%s: a recurrent model trains with a scratch sequence for its state snapshot: create the training context with n_seq_max >= 2\n", __func__);
        opt_stop_requested.store(true);
        return;
    }

    if (train && opt_walk_exact) {
        // THE EXACT WALK. The plain walk below steps per chunk and stops the gradient at every
        // chunk boundary: a reply learns only through its own chunk's K/V, never through what
        // an earlier chunk computed for it. Here the window is decoded once under the adapter
        // as it is, then trained in REVERSE: each chunk's graph attends to the cache, with its
        // last walk_horizon prefix positions behind a GRAD leaf, and adds the surrogate
        // <K, G> + <V, G> on its own K/V, where G is what every later chunk put there. Its
        // backward then carries that gradient through itself into the adapter and into its own
        // prefix's G. ONE optimizer step per window, on the gradient of the window's loss
        // (exact within the horizon). Memory stays chunk x window; the cost is one decode plus
        // one forward and backward per chunk, context chunks included.
        // THE INVARIANT THAT MAKES THE REVERSE PASS VALID (Fable): the adapter does not change
        // between the forward decode and the last chunk's backward. One optimizer step per window,
        // taken by the chunk the reverse pass trains last, is what keeps each chunk's recomputed
        // K/V (and recurrent state) equal to the cached values its successors attended to; a step
        // inside the window would make every surrogate a gradient of a different function.
        //
        // Each window starts from the REQUESTED horizon: a halving that fit one window's device
        // budget must not shrink every later window and epoch, which may be shorter (Fable).
        opt_walk_horizon = opt_walk_horizon_req;
        // A recurrent context with rollback slots keeps several states per sequence; the walk
        // carries exactly one. Refused by name, never asserted: training runs inside the serving
        // process, and an assert there takes serving down (2026-10-07 04:09Z on the 5090).
        if (recr != nullptr && cparams.n_rs_seq > 0) {
            opt_failure = "the exact walk needs a training context with no recurrent rollback slots (n_rs_seq = 0, it has "
                + std::to_string(cparams.n_rs_seq) + "): create the training context with n_rs_seq = 0";
            LLAMA_LOG_ERROR("%s: %s\n", __func__, opt_failure.c_str());
            opt_alloc_failed.store(true);
            opt_stop_requested.store(true);
            return;
        }
        // A recurrent model's state is carried the same way, one chunk at a time: the state
        // entering a chunk is tapped into a host accumulator, the state it leaves meets the next chunk's gradient
        // (build_rs, build_walk_state_exit). The reverse pass restores each chunk's entry state
        // from a host snapshot taken at its boundary in the forward pass.
        std::vector<exact_chunk> chunks;
        uint32_t n_labels_window = 0;
        for (uint32_t p = 0; p <= (uint32_t) last_label; ) {
            const bool labelled = labels_sparse[p] >= 0;
            uint32_t end = p;
            while (end <= (uint32_t) last_label && (labels_sparse[end] >= 0) == labelled) {
                ++end;
            }
            for (uint32_t c0 = p; c0 < end; c0 += n_batch) {
                const uint32_t c1 = std::min(c0 + n_batch, end);
                chunks.push_back({ c0, c1 - c0, labelled ? c1 - c0 : 0, 0, false, false, false, 0.0f });
                n_labels_window += labelled ? c1 - c0 : 0;
            }
            p = end;
        }
        // which chunks train, and how far back each one's gradient reaches
        uint32_t reach = UINT32_MAX; // the lowest position a later trained chunk's gradient reaches
        int64_t  first = -1;         // the last chunk the reverse pass trains: the step
        for (int64_t j = (int64_t) chunks.size() - 1; j >= 0; --j) {
            exact_chunk & c = chunks[j];
            c.surrogate = reach < c.c0 + c.n;
            // a recurrent state reaches every later chunk: a chunk before a trained one trains
            c.needed    = c.n_labels > 0 || c.surrogate || (recr != nullptr && first >= 0);
            if (!c.needed) {
                continue;
            }
            c.grad_from  = opt_walk_horizon == 0 || c.c0 < opt_walk_horizon ? 0 : c.c0 - opt_walk_horizon;
            c.loss_scale = (float) c.n_labels / (float) n_labels_window;
            reach = std::min(reach, c.grad_from);
            first = j;
        }
        GGML_ASSERT(first >= 0);
        chunks[first].period_end = true;

        // THE HOST BUDGET (Fable): the accumulators are layers x positions x (k + v) floats, and a
        // recurrent state is checkpointed every `stride` chunks (one sequence's state rows each;
        // the first chunk's entry is the zero state and costs nothing). Both are known before
        // anything runs. Under a budget the stride is the smallest that fits: the reverse pass
        // rebuilds a chunk's entry state by decoding forward from its checkpoint, so a longer
        // stride trades host memory for decode, never for exactness.
        size_t stride = 1;
        {
            size_t kv_bytes = 0;
            for (uint32_t il = 0; il < model.hparams.n_layer(); ++il) {
                if (!model.hparams.is_recr(il)) {
                    kv_bytes += (size_t) (last_label + 1)*(model.hparams.n_embd_k_gqa(il) + model.hparams.n_embd_v_gqa(il))*sizeof(float);
                }
            }
            size_t state = 0;
            if (recr != nullptr) {
                for (size_t il = 0; il < recr->r_l.size(); ++il) {
                    state += recr->r_l[il] ? recr->r_l[il]->nb[1] : 0;
                    state += recr->s_l[il] ? recr->s_l[il]->nb[1] : 0;
                }
            }
            auto bytes_at = [&](size_t s) { return kv_bytes + ((chunks.size() + s - 1)/s - 1)*state; };
            if (opt_walk_host_budget > 0) {
                while (stride < chunks.size() && bytes_at(stride) > opt_walk_host_budget) {
                    ++stride;
                }
            }
            const size_t bytes = bytes_at(stride);
            opt_walk_host_bytes = std::max(opt_walk_host_bytes, bytes);
            if (opt_walk_host_budget > 0 && bytes > opt_walk_host_budget) {
                char why[512];
                snprintf(why, sizeof(why),
                    "the exact walk needs %.1f MiB of host memory for this window (%lld positions, the K/V gradient alone), over its budget of %.1f MiB: "
                    "the K/V gradient grows with the window; neither the horizon nor the chunk size changes it",
                    bytes/1048576.0, (long long) (last_label + 1), opt_walk_host_budget/1048576.0);
                opt_failure = why;
                LLAMA_LOG_ERROR("%s: %s\n", __func__, opt_failure.c_str());
                opt_alloc_failed.store(true);
                opt_stop_requested.store(true);
                return;
            }
            if (stride > 1) {
                LLAMA_LOG_INFO("%s: the exact walk checkpoints the recurrent state every %zu chunks to fit %.1f MiB of host memory (%.1f MiB)\n",
                        __func__, stride, opt_walk_host_budget/1048576.0, bytes/1048576.0);
            }
        }
        {
            const uint32_t n_layer = model.hparams.n_layer();
            walk_acc.k.assign(n_layer, nullptr);
            walk_acc.v.assign(n_layer, nullptr);
            walk_acc.ctx.reset(ggml_init({ 4*n_layer*ggml_tensor_overhead(), nullptr, /*no_alloc =*/ true }));
            walk_acc.state.clear();
            walk_acc.state_ready = false;
            for (uint32_t il = 0; il < n_layer; ++il) {
                if (model.hparams.is_recr(il)) {
                    // a recurrent layer has no K/V: its state carries the chain, one sequence's row
                    if (recr != nullptr) {
                        walk_acc.state[recr->r_l[il]] = ggml_new_tensor_2d(walk_acc.ctx.get(), GGML_TYPE_F32, model.hparams.n_embd_r(), 1);
                        walk_acc.state[recr->s_l[il]] = ggml_new_tensor_2d(walk_acc.ctx.get(), GGML_TYPE_F32, model.hparams.n_embd_s(), 1);
                    }
                    continue;
                }
                walk_acc.k[il] = ggml_new_tensor_2d(walk_acc.ctx.get(), GGML_TYPE_F32, model.hparams.n_embd_k_gqa(il), last_label + 1);
                walk_acc.v[il] = ggml_new_tensor_2d(walk_acc.ctx.get(), GGML_TYPE_F32, model.hparams.n_embd_v_gqa(il), last_label + 1);
            }
            walk_acc.buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(walk_acc.ctx.get(), ggml_backend_cpu_buffer_type()));
            if (!walk_acc.buf) {
                opt_failure = "the exact walk's gradient accumulator (host memory) could not be allocated";
                LLAMA_LOG_ERROR("%s: %s\n", __func__, opt_failure.c_str());
                opt_alloc_failed.store(true);
                opt_stop_requested.store(true);
                return;
            }
            ggml_backend_buffer_clear(walk_acc.buf.get(), 0);
            cparams.walk_acc_k = walk_acc.k.data();
            cparams.walk_acc_v = walk_acc.v.data();
        }

        // the window as the adapter now reads it, chunk by chunk: a recurrent model's state is
        // checkpointed at each chunk's start through the sequence-state API (the recurrent part
        // only, PARTIAL_ONLY), after waiting for the queued decode; a restore puts back the
        // cell's position with its rows, and an empty checkpoint (before the first chunk)
        // restores the zero state. A hand-rolled per-layer copy here skipped that wait and, on
        // Metal, carried a torn mid-graph state into the reverse pass (2026-10-11).
        std::vector<std::vector<uint8_t>> snaps(recr != nullptr ? chunks.size() : 0);
        const llama_state_seq_flags state_flags = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;
        auto save_state = [&](std::vector<uint8_t> & snap) {
            synchronize();
            snap.resize(state_seq_get_size(0, state_flags));
            GGML_ASSERT(state_seq_get_data(0, snap.data(), snap.size(), state_flags) == snap.size());
        };
        auto load_state = [&](const std::vector<uint8_t> & snap) {
            synchronize();
            GGML_ASSERT(state_seq_set_data(0, snap.data(), snap.size(), state_flags) == snap.size());
        };
        for (size_t j = 0; j < chunks.size(); ++j) {
            if (recr != nullptr && j % stride == 0) {
                save_state(snaps[j]);
            }
            if (!decode_span(chunks[j].c0, chunks[j].c0 + chunks[j].n)) {
                return;
            }
        }
        bool ok = true;
        for (int64_t j = (int64_t) chunks.size() - 1; j >= first && ok; --j) {
            const exact_chunk & c = chunks[j];
            if (!c.needed) {
                continue;
            }
            if (!yield_point()) {
                ok = false;
                break;
            }
            // attention: pop the cache to [0, c0); recurrent: restore the state at the chunk's
            // checkpoint and decode forward to its start (that decode rebuilds the attention K/V
            // in between too, under the same adapter: the same values). Run again before every
            // retry below: a refused chunk graph is refused AFTER its ubatch was applied, so the
            // chunk's cells already sit in the cache and the recurrent cell's position has moved
            // to the chunk's end; retrying on top of that fails to prepare the ubatch at all
            // (the 5090, 2026-10-10: "init_batch: failed to prepare attention ubatches" after
            // "retrying at 512", and the epoch stopped instead of shrinking the horizon).
            const auto rewind = [&]() -> bool {
                const size_t   cp    = recr != nullptr ? (size_t) j - (size_t) j % stride : (size_t) j;
                const uint32_t p_pop = chunks[cp].c0;
                const bool popped = recr != nullptr ? (attn == nullptr || attn->seq_rm(0, p_pop, -1)) : memory->seq_rm(0, p_pop, -1);
                if (!popped) {
                    LLAMA_LOG_ERROR("%s: could not pop the cache to [0, %u) for the reverse pass\n", __func__, p_pop);
                    opt_stop_requested.store(true);
                    return false;
                }
                if (recr != nullptr) {
                    load_state(snaps[cp]);
                    if (p_pop < c.c0 && !decode_span(p_pop, c.c0)) {
                        return false;
                    }
                }
                return true;
            };
            if (!rewind()) {
                ok = false;
                break;
            }
            cparams.walk_exact     = true;
            cparams.walk_surrogate = c.surrogate;
            cparams.walk_state_surrogate = recr != nullptr && j < (int64_t) chunks.size() - 1 && walk_acc.state_ready;
            // THE HORIZON FITS THE DEVICE. Its GRAD leaves are the chunk graph's largest growth
            // with the window (layers x horizon x (k + v) floats, twice with their gradient); the
            // graph preflight refuses a chunk that does not fit before allocating anything, so a
            // memory refusal halves the horizon (whole chunks) and the chunk is tried again; every
            // later chunk keeps the smaller one. The forward still attends to the whole window and
            // the recurrent state is carried exactly; only attention gradient past the horizon is
            // dropped. A refusal at a horizon of one chunk, or for a node the device cannot run,
            // stands.
            exact_chunk cc = c;
            for (;;) {
                cc.grad_from = opt_walk_horizon == 0 || cc.c0 < opt_walk_horizon ? 0 : cc.c0 - opt_walk_horizon;
                cparams.walk_grad_from = cc.grad_from;
                xc = &cc;
                ok = train_chunk(cc.c0, cc.n);
                xc = nullptr;
                const uint32_t span = cc.c0 - cc.grad_from;
                if (ok || !opt_failure.empty() || !opt_alloc_failed.load() || span <= n_batch) {
                    break;
                }
                opt_walk_horizon = std::max<uint32_t>(n_batch, (span/2)/n_batch*n_batch);
                LLAMA_LOG_WARN("%s: the chunk at %u did not fit with a gradient horizon of %u positions: retrying at %u\n",
                        __func__, cc.c0, span, opt_walk_horizon);
                opt_alloc_failed.store(false);
                opt_stop_requested.store(false);
                // the rewind's decode is a plain forward, as before the first try: no walk flags.
                // walk_grad_from is not restored here because the loop's first statements recompute
                // it from the shrunken horizon (cc.grad_from, then cparams.walk_grad_from) before the
                // retried train_chunk runs.
                const bool surrogate       = cparams.walk_surrogate;
                const bool state_surrogate = cparams.walk_state_surrogate;
                cparams.walk_exact     = false;
                cparams.walk_grad_from = 0;
                cparams.walk_surrogate = false;
                cparams.walk_state_surrogate = false;
                const bool rewound = rewind();
                cparams.walk_exact     = true;
                cparams.walk_surrogate = surrogate;
                cparams.walk_state_surrogate = state_surrogate;
                if (!rewound) {
                    ok = false;
                    break;
                }
            }
            opt_walk_horizon_used = opt_walk_horizon;
            cparams.walk_exact     = false;
            cparams.walk_grad_from = 0;
            cparams.walk_surrogate = false;
            cparams.walk_state_surrogate = false;
        }
        walk_acc.state_ready = false;
        return;
    }

    uint32_t pos = 0;
    while (pos <= (uint32_t) last_label && !opt_stop_requested.load(std::memory_order_relaxed)) {
        const bool labelled = labels_sparse[pos] >= 0;
        uint32_t end = pos;
        while (end <= (uint32_t) last_label && (labels_sparse[end] >= 0) == labelled) {
            ++end;
        }
        if (!labelled) {
            if (!decode_span(pos, end)) {
                return;
            }
        } else {
            for (uint32_t c0 = pos; c0 < end; c0 += n_batch) {
                const uint32_t c1 = std::min(c0 + n_batch, end);
                // The training forward advances a recurrent state in place, and a recurrent
                // memory cannot remove a partial range: snapshot the state to the scratch
                // sequence first (Fable), restore it after, then the decode below advances it
                // from where the context left it. The state is per sequence and small.
                const bool snapshot = recr != nullptr && c1 <= (uint32_t) last_label;
                if (!yield_point()) {
                    return;
                }
                if (snapshot) {
                    recr->seq_cp(0, 1, -1, -1);
                }
                if (!train_chunk(c0, c1 - c0)) {
                    return;
                }
                if (snapshot) {
                    recr->seq_rm(0, -1, -1);
                    recr->seq_cp(1, 0, -1, -1);
                    recr->seq_rm(1, -1, -1);
                }
                // the chunk joins the context under the adapter as it now is: the training
                // graph never writes the cache, so its cells hold nothing until this decode
                if (c1 <= (uint32_t) last_label) {
                    // the chunk's cells hold no K/V (the training graph never writes the
                    // cache): drop them from the attention memory, then decode them for real
                    const bool removed = recr != nullptr ? (attn == nullptr || attn->seq_rm(0, c0, -1)) : memory->seq_rm(0, c0, -1);
                    if (!removed) {
                        LLAMA_LOG_ERROR("%s: could not remove the trained chunk [%u, %u) from the cache before decoding it\n", __func__, c0, c1);
                        opt_stop_requested.store(true);
                        return;
                    }
                    if (!decode_span(c0, c1)) {
                        opt_stop_requested.store(true);
                        return;
                    }
                }
            }
        }
        pos = end;
    }
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

    int64_t idata = 0;

    int64_t t_loop_start = ggml_time_us();
    int64_t ndata_in_loop = idata_split*ubatch_per_ctx;
    for (; idata < idata_split; ++idata) {
        if (opt_stop_requested.load(std::memory_order_relaxed)) {
            break;
        }
        constexpr bool train = true;
        const int64_t idata_in_loop = idata*ubatch_per_ctx;

        if (opt_step_callback && !opt_step_callback(train, opt_step_callback_data)) {
            opt_stop_requested.store(true);
        }
        if (opt_stop_requested.load(std::memory_order_relaxed)) {
            break;
        }

        ggml_opt_dataset_get_batch_host(dataset, tokens.data(), n_ctx*sizeof(llama_token), labels_sparse.data(), idata);
        opt_epoch_iter(dataset, result_train, tokens, labels_sparse, batch,
            callback_train, train, idata_in_loop, ndata_in_loop, t_loop_start);
    }

    t_loop_start = ggml_time_us();
    ndata_in_loop = (ndata - idata_split)*ubatch_per_ctx;
    for (; idata < ndata; ++idata) {
        if (opt_stop_requested.load(std::memory_order_relaxed)) {
            break;
        }
        constexpr bool train = false;
        const int64_t idata_in_loop = (idata - idata_split)*ubatch_per_ctx;

        if (opt_step_callback && !opt_step_callback(train, opt_step_callback_data)) {
            opt_stop_requested.store(true);
        }
        if (opt_stop_requested.load(std::memory_order_relaxed)) {
            break;
        }

        ggml_opt_dataset_get_batch_host(dataset, tokens.data(), n_ctx*sizeof(llama_token), labels_sparse.data(), idata);
        opt_epoch_iter(dataset, result_eval, tokens, labels_sparse, batch,
            callback_eval, train, idata_in_loop, ndata_in_loop, t_loop_start);
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
