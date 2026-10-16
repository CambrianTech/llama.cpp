// Training: llama_context's optimizer members (opt_init, opt_epoch, the walk in
// opt_epoch_iter) and the llama_opt_* interface, apart from inference so each reads as one concern.
#include "llama-context.h"
#include "llama-adapter.h"
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

// whether every GPU this context computes on has a kernel for ggml_flash_attn_back, at this
// model's attention shape and this context's cache types (the CPU has one)
bool llama_context::opt_fa_back_native() const {
    const llama_hparams & hp = model.hparams;
    uint32_t il = 0;
    while (il < hp.n_layer() && hp.is_recr(il)) {
        ++il;
    }
    if (il == hp.n_layer()) {
        return false; // no attention layer
    }
    // the cache the step's attention reads in place
    const llama_kv_cache * kv = dynamic_cast<const llama_kv_cache *>(memory.get());
    if (const auto * hybrid = dynamic_cast<const llama_memory_hybrid *>(memory.get())) {
        kv = hybrid->get_mem_attn();
    }
    if (kv == nullptr) {
        return false;
    }
    ggml_init_params ip = { 8*ggml_tensor_overhead(), nullptr, /*no_alloc =*/ true };
    ggml_context_ptr ctx(ggml_init(ip));
    const int64_t H = hp.n_head(il), Hkv = hp.n_head_kv(il), D = hp.n_embd_head_k(il), Dv = hp.n_embd_head_v(il);
    ggml_tensor * q = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, D,  8,  H);
    ggml_tensor * k = ggml_new_tensor_3d(ctx.get(), kv->type_k(), D, 64, Hkv);
    ggml_tensor * v = ggml_new_tensor_3d(ctx.get(), kv->type_v(), Dv, 64, Hkv);
    ggml_tensor * m = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F16, 64, 8);
    ggml_tensor * d = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, Dv, H, 8);
    ggml_tensor * op = ggml_flash_attn_back(ctx.get(), q, k, v, m, d, 1.0f, 0);
    for (const auto & backend : backends) {
        ggml_backend_dev_t dev = ggml_backend_get_device(backend.get());
        const auto type = dev ? ggml_backend_dev_type(dev) : GGML_BACKEND_DEVICE_TYPE_CPU;
        if ((type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU) && !ggml_backend_dev_supports_op(dev, op)) {
            return false;
        }
    }
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
    // (the walk alone needs one ubatch per batch: a chunk's backward needs all of its own K/V in one
    // graph; opt_epoch refuses otherwise. llama_opt_learn trains in a serving context's ubatches.)
    // In a flash-attention context the training graph reads attention through FLASH_ATTN_EXT too
    // (build_attn_mha), whose backward is tiled over the window; flash attention also serves the
    // walk's context decodes and lets the cache hold V un-transposed.
    cparams.training = true;
    // The graph result and the scheduler were sized at context creation: before training
    // (no backward pass) and before any adapter attached since (its nodes uncounted).
    // Size both again for the training graph, from the same number, plus the nodes each
    // attention layer's tiled backward adds for this window (one count, owned by ggml).
    // the attention backward: one op where every GPU has the kernel, else composed tiles whose node
    // count grows with the window (decided first: it sizes the graph)
    cparams.fa_back_native = opt_fa_back_native();
    {
        size_t max_nodes = graph_max_nodes(n_ubatch);
        if (cparams.flash_attn && !cparams.fa_back_native) {
            max_nodes += (size_t) model->hparams.n_layer() * ggml_flash_attn_ext_back_n_nodes(opt_n_ctx_train, 0);
        }
        gf_res_prev.reset(new llm_graph_result(max_nodes));
        sched.reset(ggml_backend_sched_new(backend_ptrs.data(), backend_buft.data(), backend_ptrs.size(),
                    max_nodes, cparams.pipeline_parallel, cparams.op_offload));
    }

    LLAMA_LOG_INFO("%s: the attention backward is %s\n", __func__,
            cparams.fa_back_native ? "one op (ggml_flash_attn_back): no score is kept" : "composed from tiles (a device has no ggml_flash_attn_back kernel)");

    // one target token per row: no one-hot tensor as wide as the vocabulary beside the logits
    ggml_opt_params opt_params = ggml_opt_default_params(sched.get(), GGML_OPT_LOSS_TYPE_CROSS_ENTROPY_TARGETS);
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
        // the trained adapter is the shadow: labelled rows only, never a serving row
        opt_shadow = lopt_params.adapter;
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

// THE WALK (Fable's design, continuum 2026-10-06; objective 2026-10-14): one pass over a window
// in one context. A run with no label is context: a plain decode into the cache, as served, no
// backward. A labelled run is her reply, trained GIVEN EVERYTHING BEFORE IT AS SERVED: only its
// rows see the shadow adapter (the label mask), so the context before it carries no gradient and
// is never trained. The run trains in chunks of n_batch, each attending to everything before it
// in the cache; its chunks write their own cells under the shadow, so a later chunk of the run
// reads an earlier one as the shadow computes it. When the run is done it is RESTORED: its cells
// and the recurrent state are decoded again under her stable set, so the cache is always her
// stable self's and the next run reads this one as served. Memory is chunk x window, never
// window x window, so her whole lived window trains. Nothing after the last label is computed.
// One per window; a member of the context, so it runs on the context's own state.
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
    bool decode_span(uint32_t p0, uint32_t p1, bool shadow);
    std::vector<uint8_t> checkpoint();
    bool rewind(uint32_t p_pop, const std::vector<uint8_t> & snap, uint32_t p_to);
    bool train_chunk(uint32_t pos_ctx, uint32_t n_chunk);
    bool restore(const llama_walk_run & run, const std::vector<uint8_t> & snap);
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

// an inference decode of [p0, p1) into the cache, no loss and no backward. shadow: the labelled
// rows decode under the shadow adapter (a run's forward pass); otherwise every row decodes under
// her stable set (context, and a run restored).
bool llama_context::opt_walk::decode_span(uint32_t p0, uint32_t p1, bool shadow) {
    bool ok = true;
    for (uint32_t c0 = p0; c0 < p1 && ok; c0 += n_batch) {
        if (!yield_point()) {
            return false; // stopped or cancelled at a chunk boundary: not a decode failure
        }
        const uint32_t c1 = std::min(c0 + n_batch, p1);
        llama_walk_fill(batch, tokens, c0, c1 - c0);
        for (uint32_t i = 0; shadow && i < c1 - c0; ++i) {
            // a label marks the row the shadow sees; the allocator wants it an output row
            batch.label [i] = labels[c0 + i];
            batch.logits[i] = labels[c0 + i] >= 0;
        }
        batch.logits[c1 - c0 - 1] = true;
        lctx.opt_walk_active  = shadow ? this : nullptr;
        lctx.opt_forward_only = shadow;
        ok = lctx.decode(batch) == 0;
        lctx.opt_walk_active  = nullptr;
        lctx.opt_forward_only = false;
        std::fill(batch.label, batch.label + (c1 - c0), -1);
    }
    if (!ok) {
        LLAMA_LOG_ERROR("%s: the decode of [%u, %u) failed\n", __func__, p0, p1);
    }
    return ok;
}

// the recurrent state of the walk's sequence through the sequence-state API (the recurrent part
// only, PARTIAL_ONLY), after waiting for the queued decode; empty for a model with none. A
// hand-rolled per-layer copy skipped that wait and, on Metal, carried a torn mid-graph state into
// the reverse pass (2026-10-11).
std::vector<uint8_t> llama_context::opt_walk::checkpoint() {
    std::vector<uint8_t> snap;
    if (recr != nullptr) {
        lctx.synchronize();
        snap.resize(lctx.state_seq_get_size(0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY));
        GGML_ASSERT(lctx.state_seq_get_data(0, snap.data(), snap.size(), LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == snap.size());
    }
    return snap;
}

// back to position p_pop: the cache holds [0, p_pop), the recurrent state is `snap` (taken at
// p_pop); then forward to p_to, under the shadow for the run's labelled rows. A restore puts back
// the cell's position with its rows. Also run before every retry of a refused chunk: a refused
// chunk graph is refused AFTER its ubatch was applied, so its cells already sit in the cache and
// the recurrent cell's position has moved to the chunk's end; retrying on top of that fails to
// prepare the ubatch at all (the 5090, 2026-10-10: "init_batch: failed to prepare attention
// ubatches" after "retrying at 512").
bool llama_context::opt_walk::rewind(uint32_t p_pop, const std::vector<uint8_t> & snap, uint32_t p_to) {
    const bool popped = recr != nullptr ? (attn == nullptr || attn->seq_rm(0, p_pop, -1)) : lctx.memory->seq_rm(0, p_pop, -1);
    if (!popped) {
        LLAMA_LOG_ERROR("%s: could not pop the cache to [0, %u)\n", __func__, p_pop);
        lctx.opt_stop_requested.store(true);
        return false;
    }
    if (recr != nullptr) {
        lctx.synchronize();
        GGML_ASSERT(lctx.state_seq_set_data(0, snap.data(), snap.size(), LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == snap.size());
    }
    return p_pop >= p_to || decode_span(p_pop, p_to, /*shadow =*/ true);
}

// one training chunk [pos_ctx, pos_ctx + n_chunk) of a run: one llama_decode with labels. The
// batch loop is decode's, and the extras switch on in process_ubatch (the shadow's masked term,
// then llama_context::opt_ubatch). This function only fills the chunk's tokens and labels and
// says whose walk it is.
bool llama_context::opt_walk::train_chunk(uint32_t pos_ctx, uint32_t n_chunk) {
    llama_walk_fill(batch, tokens, pos_ctx, n_chunk);
    for (uint32_t pos_batch = 0; pos_batch < n_chunk; ++pos_batch) {
        batch.logits[pos_batch] = false;
        batch.label [pos_batch] = labels[pos_ctx + pos_batch];
    }

    lctx.opt_walk_active   = this;
    lctx.opt_result_active = result;
    const int ret = lctx.decode(batch);
    lctx.opt_walk_active   = nullptr;
    lctx.opt_result_active = nullptr;
    std::fill(batch.label, batch.label + n_chunk, -1);

    if (ret != 0) {
        // a refusal has already stopped the epoch by name (opt_fail); anything else is decode's
        if (!lctx.opt_alloc_failed.load()) {
            lctx.opt_fail("the training chunk failed to decode (llama_decode returned " + std::to_string(ret) + ")");
        }
        return false;
    }
    if (callback) {
        callback(train, lctx.opt_ctx, dataset, result, idata_in_loop + pos_ctx/n_batch + 1, ndata_in_loop, t_loop_start);
    }
    return true;
}

bool llama_context::opt_ubatch(llm_graph_result * res, ggml_cgraph * gf, const llama_ubatch & ubatch, bool prepare_restore) {
    const uint32_t n_labels = ubatch.n_labels();
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
    std::vector<int32_t> targets;
    std::vector<float>   weights; // a labelled row's weight in the loss, when the batch carries them
    targets.reserve(n_labels);
    for (int32_t row : ubatch.rows().label_rows) {
        targets.push_back(ubatch.label[row]);
        if (ubatch.label_weight != nullptr) {
            weights.push_back(ubatch.label_weight[row]);
        }
    }
    // an unweighted batch keeps the unweighted loss (every backend has its kernels)
    ggml_opt_set_next_label_weights(opt_ctx, ubatch.label_weight != nullptr);
    if (res->t_loss_slices.empty()) {
        ggml_free(ctx_compute_opt);
        opt_fail("training head was not projected into bounded label slices");
        return false;
    }
    ggml_opt_prepare_alloc_slices(opt_ctx, ctx_compute_opt, gf, res->get_inp_tokens(),
            res->t_loss_slices.data(), res->t_loss_slices.size());
    if (prepare_restore) {
        const bool ok = ggml_opt_alloc_for_restore(opt_ctx);
        ggml_free(ctx_compute_opt);
        if (!ok) {
            opt_fail(ggml_opt_refusal(opt_ctx), "fresh restore graph allocation refused; no sample evaluated");
        }
        return ok;
    }
    if (xc == nullptr && opt_learn_labels > 0) {
        // llama_opt_learn: one step per call, at the ubatch that completes its labelled rows
        const int64_t nl = ubatch.n_labels();
        opt_learn_labels_done += nl;
        ggml_opt_set_next_step(opt_ctx, opt_learn_labels_done >= opt_learn_labels, (float) nl / (float) opt_learn_labels, nullptr);
    }
    if (xc != nullptr) {
        ggml_opt_set_next_step(opt_ctx, xc->period_end, xc->loss_scale, res->t_walk_surrogate);
        for (const auto & io : res->t_walk) {
            if (io.tk) {
                // the run's cached rows before this chunk, as F32: their gradient into the accumulator
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
        // the target token of each output row, in the order the ubatch emits its outputs (the
        // order the logits rows take)
        struct ggml_tensor * labels_t = ggml_opt_labels(opt_ctx);
        GGML_ASSERT(labels_t->type == GGML_TYPE_I32 && labels_t->ne[0] == (int64_t) n_labels);

        ggml_backend_tensor_set(labels_t, targets.data(), 0, ggml_nbytes(labels_t));
        if (struct ggml_tensor * weights_t = ggml_opt_label_weights(opt_ctx)) {
            GGML_ASSERT(weights.size() == (size_t) n_labels);
            ggml_backend_tensor_set(weights_t, weights.data(), 0, ggml_nbytes(weights_t));
        }
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
// it. Here each run is decoded once under the shadow, then trained in REVERSE: each chunk's graph
// attends to the cache, with the run's positions within walk_horizon behind a gradient tap, and
// adds the surrogate <K, G> + <V, G> on its own K/V, where G is what every later chunk of the run
// put there. Its backward then carries that gradient through itself into the adapter and into the
// run's earlier positions' G. ONE optimizer step per window, on the gradient of the window's loss
// (exact within the horizon). Memory stays chunk x window; the cost is one decode of the window,
// and per run one more decode plus one forward and backward per chunk and the restore.
// THE INVARIANT THAT MAKES THE REVERSE PASS VALID (Fable): the adapter does not change between a
// run's forward decode and its last chunk's backward. One optimizer step per window, taken by the
// chunk the walk trains last, is what keeps each chunk's recomputed K/V (and recurrent state)
// equal to the cached values its successors attended to; a step inside the window would make
// every surrogate a gradient of a different function.
// A recurrent model's state is carried the same way, one chunk at a time: the state entering a
// chunk is tapped into a host accumulator, the state it leaves meets the next chunk's gradient
// (build_rs, build_walk_state_exit). The reverse pass restores each chunk's entry state from a
// checkpoint taken at its boundary in the forward pass. The state entering a run is her stable
// set's: no gradient crosses it.
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
    plan.mark_exact(lctx.opt_walk_horizon);
    const std::vector<llama_walk_chunk> & chunks = plan.chunks;
    const uint32_t n_pos = plan.longest_run();

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
            "the exact walk needs %.1f MiB of host memory for this window (its longest reply, %u positions, the K/V gradient alone), over its budget of %.1f MiB: "
            "the K/V gradient grows with the reply; neither the horizon nor the chunk size changes it",
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

    uint32_t p = 0; // the cache holds [0, p) as served
    for (const llama_walk_run & run : plan.runs) {
        const uint32_t r0 = plan.begin(run);
        const uint32_t r1 = plan.end(run);
        if (!decode_span(p, r0, /*shadow =*/ false)) {
            return;
        }
        // the run under the shadow, checkpointed every `stride` chunks from its start
        std::vector<std::vector<uint8_t>> snaps(run.j1 - run.j0);
        for (size_t j = run.j0; j < run.j1; ++j) {
            if ((j - run.j0) % stride == 0) {
                snaps[j - run.j0] = checkpoint();
            }
            if (!decode_span(chunks[j].c0, chunks[j].c0 + chunks[j].n, /*shadow =*/ true)) {
                return;
            }
        }
        // the reverse pass: back to the chunk's checkpoint, forward to its start, train it
        acc.start_run(r0);
        for (size_t j = run.j1; j-- > run.j0; ) {
            const llama_walk_chunk & c = chunks[j];
            if (!yield_point()) {
                return;
            }
            const size_t cp = recr != nullptr ? (j - run.j0) - (j - run.j0) % stride : j - run.j0;
            const auto back = [&]() { return rewind(chunks[run.j0 + cp].c0, snaps[cp], c.c0); };
            if (!back()) {
                return;
            }
            const bool state_surrogate = recr != nullptr && j + 1 < run.j1 && acc.state_ready;
            // THE HORIZON FITS THE DEVICE. Its gradient taps are the chunk graph's largest growth
            // with the run (layers x horizon x (k + v) floats, twice with their gradient); the graph
            // preflight refuses a chunk that does not fit before allocating anything, so a memory
            // refusal halves the horizon (whole chunks) and the chunk is tried again; every later
            // chunk keeps the smaller one. The forward still attends to the whole window and the
            // recurrent state is carried exactly; only attention gradient past the horizon is
            // dropped. A refusal at a horizon of one chunk, or for a node the device cannot run, stands.
            llama_walk_chunk cc = c;
            bool ok;
            for (;;) {
                cc.grad_from = std::max(r0, llama_walk_plan::grad_from(cc.c0, lctx.opt_walk_horizon));
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
                if (!back()) {
                    ok = false;
                    break;
                }
            }
            lctx.opt_walk_horizon_used = lctx.opt_walk_horizon;
            if (!ok) {
                return;
            }
        }
        acc.state_ready = false;
        // the run restored to her stable self: its cells and state as served
        if (!restore(run, snaps[0])) {
            return;
        }
        p = r1;
    }
}

// a run's cells and recurrent state decoded again under her stable set, from the checkpoint at
// its start (her stable state, since everything before the run is)
bool llama_context::opt_walk::restore(const llama_walk_run & run, const std::vector<uint8_t> & snap) {
    const uint32_t r0 = plan.begin(run);
    const uint32_t r1 = plan.end(run);
    return rewind(r0, snap, r0) && decode_span(r0, r1, /*shadow =*/ false);
}

// THE PLAIN WALK: each labelled chunk steps on its own gradient, then joins the cache under the
// shadow as it now is, so the run's next chunk reads it; the run is restored when it is done
void llama_context::opt_walk::run_plain() {
    uint32_t p = 0;
    for (const llama_walk_run & run : plan.runs) {
        if (!decode_span(p, plan.begin(run), /*shadow =*/ false)) {
            return;
        }
        const std::vector<uint8_t> at_run = checkpoint();
        for (size_t j = run.j0; j < run.j1; ++j) {
            const llama_walk_chunk & c = plan.chunks[j];
            if (!yield_point()) {
                return;
            }
            // the training forward advances a recurrent state in place and may write the chunk's
            // cells: back to the chunk's start, then decode it under the stepped shadow
            const std::vector<uint8_t> at_chunk = j + 1 < run.j1 ? checkpoint() : std::vector<uint8_t>();
            if (!train_chunk(c.c0, c.n)) {
                return;
            }
            if (j + 1 < run.j1 && !rewind(c.c0, at_chunk, c.c0 + c.n)) {
                lctx.opt_stop_requested.store(true);
                return;
            }
        }
        if (!restore(run, at_run)) {
            lctx.opt_stop_requested.store(true);
            return;
        }
        p = plan.end(run);
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
    if (this->n_ubatch() != std::min(this->n_batch(), opt_n_ctx_train)) {
        opt_fail("the walk needs one ubatch per batch (a chunk's backward needs all of its own K/V in one graph): set -ub = -b");
        return;
    }
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
    if (opt_shadow && opt_shadow->restore_quarantined) {
        opt_fail("shadow generation restore is uncertain; restore a known-good generation before learning");
        return;
    }
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

bool llama_context::opt_learn(llama_seq_id seq, llama_pos n_cached, const llama_token * tokens, const llama_token * labels,
        int32_t n, ggml_opt_result_t result) {
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
    if (attn != nullptr && !attn->seq_rm(seq, n_cached, -1)) {
        opt_fail("llama_opt_learn: could not drop sequence " + std::to_string(seq) + "'s cells past " + std::to_string(n_cached));
        return false;
    }
    // her stable recurrent state at the served prefix, for the restore
    std::vector<uint8_t> snap;
    if (recr != nullptr) {
        synchronize();
        snap.resize(state_seq_get_size(seq, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY));
        GGML_ASSERT(state_seq_get_data(seq, snap.data(), snap.size(), LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == snap.size());
    }
    const uint32_t nb = n_batch();
    llama_batch batch = llama_batch_init(nb, 0, 1);
    // [c0, c0 + m) of the reply into the batch; with labels, the trainable layer sees those rows
    auto fill = [&](int32_t c0, int32_t m, bool labelled) {
        batch.n_tokens = m;
        for (int32_t i = 0; i < m; ++i) {
            batch.token   [i]    = tokens[c0 + i];
            batch.pos     [i]    = n_cached + c0 + i;
            batch.n_seq_id[i]    = 1;
            batch.seq_id  [i][0] = seq;
            batch.label   [i]    = labelled ? labels[c0 + i] : -1;
            batch.logits  [i]    = labelled && labels[c0 + i] >= 0;
        }
        if (!labelled) {
            batch.logits[m - 1] = true;
        }
    };
    // train up to the last labelled row: rows after it cannot change the loss, and in one batch an
    // unlabelled row after the run is refused (a run ends its sequence's rows); the restore below
    // still decodes all n
    int32_t n_train = n;
    while (n_train > 0 && labels[n_train - 1] < 0) {
        --n_train;
    }
    bool ok = true;
    opt_result_active     = result;
    opt_outputs_internal  = true;
    opt_learn_labels      = 0;
    opt_learn_labels_done = 0;
    for (int32_t i = 0; i < n_train; ++i) {
        opt_learn_labels += labels[i] >= 0;
    }
    for (int32_t c0 = 0; c0 < n_train && ok; c0 += (int32_t) nb) {
        const int32_t m = std::min<int32_t>((int32_t) nb, n_train - c0);
        fill(c0, m, true);
        ok = decode(batch) == 0;
        if (!ok && !opt_alloc_failed.load()) {
            opt_fail("llama_opt_learn: the reply's training batch at " + std::to_string(n_cached + c0) + " failed to decode");
        }
    }
    opt_result_active     = nullptr;
    opt_outputs_internal  = false;
    opt_learn_labels      = 0;
    opt_learn_labels_done = 0;
    // RESTORE: the reply's cells and the recurrent state decoded again under her stable set
    bool restored = attn == nullptr || attn->seq_rm(seq, n_cached, -1);
    if (recr != nullptr) {
        synchronize();
        restored = restored && state_seq_set_data(seq, snap.data(), snap.size(), LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == snap.size();
    }
    for (int32_t c0 = 0; c0 < n && restored; c0 += (int32_t) nb) {
        const int32_t m = std::min<int32_t>((int32_t) nb, n - c0);
        fill(c0, m, false);
        restored = decode(batch) == 0;
    }
    llama_batch_free(batch);
    if (!restored) {
        opt_fail("llama_opt_learn: sequence " + std::to_string(seq) + " could not be restored after training: re-prefill it");
        return false;
    }
    return ok;
}

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

bool llama_context::opt_prepare_restore(const llama_batch & retained) {
    opt_alloc_failed.store(false);
    opt_failure.clear();
    if (!ggml_opt_can_prepare_restore(opt_ctx)) {
        opt_fail("restore preparation requires a fresh optimizer; existing graph ownership was preserved");
        return false;
    }
    if (!opt_ctx || !opt_shadow || opt_shadow->restore_quarantined || !loras->count(const_cast<llama_adapter_lora *>(opt_shadow)) ||
        opt_walk_active || opt_learn_labels != 0 || !retained.token || !retained.label || retained.n_tokens <= 0 ||
        !model.output || cparams.pooling_type != LLAMA_POOLING_TYPE_NONE || cparams.ctx_type != LLAMA_CONTEXT_TYPE_DEFAULT) {
        opt_fail("restore preparation requires an initialized attached shadow and retained labelled token experience");
        return false;
    }
    llama_batch_allocr retained_batches(model.hparams.n_pos_per_embd());
    const uint32_t max_seq = cparams.kv_unified ? LLAMA_MAX_SEQ : cparams.n_seq_max;
    // Retained positions are historical; do not infer new positions or change current cache custody.
    if (!retained_batches.init(retained, model.vocab, nullptr, model.hparams.n_embd_inp(), max_seq, false) ||
        retained_batches.get_n_labels() == 0) {
        opt_fail("restore preparation refused invalid or unlabelled retained experience");
        return false;
    }
    if (memory) {
        if (const char * why = memory->training_error(retained_batches.get_batch(), cparams.flash_attn)) {
            opt_fail(why);
            return false;
        }
    }
    llama_ubatch ubatch{};
    do {
        ubatch = retained_batches.split_seq(n_ubatch());
    } while (ubatch.n_tokens && ubatch.n_labels() == 0);
    if (!ubatch.n_tokens) {
        opt_fail("restore preparation could not form a labelled ubatch");
        return false;
    }
    // init_full is the existing read-only graph-planning view. Never call apply, decode, or
    // graph_reserve here: allocation does not insert cache cells or execute a retained sample.
    auto planning = memory ? memory->init_full() : llama_memory_context_ptr{};
    if (planning && llama_memory_status_is_fail(planning->get_status())) {
        opt_fail("restore memory planning refused");
        return false;
    }
    synchronize();
    const bool was_training = cparams.training;
    const bool was_walk = cparams.walk_active;
    const auto prior_outputs = n_outputs;
    auto * res = gf_res_prev.get();
    const auto release_graph = [&]() {
        cparams.training = was_training;
        cparams.walk_active = was_walk;
        n_outputs = prior_outputs;
        res->reset();
        ggml_backend_sched_reset(sched.get());
        sched_need_reserve = true;
    };
    bool prepared = false;
    try {
        cparams.training = true;
        cparams.walk_active = false;
        n_outputs = ubatch.n_tokens;
        res->reset();
        ggml_backend_sched_reset(sched.get());
        auto * graph = model.build_graph(graph_params(res, ubatch, planning.get(), graph_type()));
        if (graph) {
            prepared = opt_ubatch(res, graph, ubatch, true);
        } else {
            opt_fail("restore preparation could not build retained-experience graph");
        }
    } catch (...) {
        release_graph();
        throw;
    }
    release_graph();
    return prepared;
}

bool llama_opt_prepare_restore(llama_context * ctx, llama_batch retained) {
    return ctx && ctx->opt_prepare_restore(retained);
}

bool llama_context::opt_state_restore(llama_adapter_lora * shadow,
        FILE * weights, FILE * optimizer, const char * binding) {
    if (!shadow || opt_shadow != shadow || !opt_ctx) { return false; }
    // The serving set must exclude the shadow throughout restore, including its preparation.
    if (loras->count(shadow)) { return false; }
    std::unique_ptr<llama_adapter_lora_state, decltype(&llama_adapter_lora_state_free)> adapter_state(
        llama_adapter_lora_state_prepare(shadow, weights), llama_adapter_lora_state_free);
    if (!adapter_state) { return false; }
    std::unique_ptr<ggml_opt_state, decltype(&ggml_opt_state_free)> optimizer_state(
        ggml_opt_state_prepare(opt_ctx, optimizer, binding), ggml_opt_state_free);
    if (!optimizer_state) { return false; }
    synchronize();
    shadow->restore_quarantined = true;
    if (!llama_adapter_lora_state_apply(shadow, adapter_state.get()) ||
        !ggml_opt_state_apply(opt_ctx, optimizer_state.get())) { return false; }
    synchronize();
    shadow->restore_quarantined = false;
    return true;
}

bool llama_opt_state_restore(llama_context * ctx, llama_adapter_lora * shadow,
        FILE * weights, FILE * optimizer, const char * binding) {
    return ctx && ctx->opt_state_restore(shadow, weights, optimizer, binding);
}

bool llama_context::opt_state_write(FILE * optimizer, const char * binding) {
    return opt_ctx && opt_shadow && !opt_shadow->restore_quarantined &&
        ggml_opt_state_write(opt_ctx, optimizer, binding);
}

bool llama_opt_state_write(llama_context * ctx, FILE * optimizer, const char * binding) {
    return ctx && ctx->opt_state_write(optimizer, binding);
}

bool llama_context::opt_state_write_initial(FILE * optimizer, const char * binding) {
    return opt_ctx && opt_shadow && !opt_shadow->restore_quarantined && !loras->count(const_cast<llama_adapter_lora *>(opt_shadow)) &&
        ggml_opt_state_write_initial(opt_ctx, optimizer, binding);
}

bool llama_opt_state_write_initial(llama_context * ctx, FILE * optimizer, const char * binding) {
    return ctx && ctx->opt_state_write_initial(optimizer, binding);
}
