// The serving graph owner supplies the same labelled head and full-window memory
// metadata to the existing optimizer allocator. No decode, KV apply, or new trainer.
#include "llama-context.h"
#include "llama-model.h"
#include "llama-batch.h"
#include <algorithm>

bool llama_context::opt_measure_memory(llama_opt_memory_callback callback, void * user_data, ggml_backend_sched_t held) {
    if (!callback || !opt_shadow || opt_reply || !memory) {
        opt_failure = "learning preflight requires a selected shadow and an idle learning boundary";
        return false;
    }
    synchronize();
    auto mctx = memory->init_full_single_stream();
    if (!mctx || llama_memory_status_is_fail(mctx->get_status())) {
        opt_failure = "cannot describe the served memory for learning preflight";
        return false;
    }
    const auto saved_cparams = cparams;
    struct restore {
        llama_cparams & params;
        llama_cparams saved;
        ggml_backend_sched_t sched;
        ~restore() { ggml_backend_sched_reset(sched); params = saved; }
    } guard{cparams, saved_cparams, sched.get()};
    // Scope cleanup also invalidates reuse of scheduler-rewritten graph metadata.
    ggml_backend_sched_reset(sched.get());
    gf_res_prev->reset();
    cparams.training = true;
    cparams.walk_active = false;
    const uint32_t chunk = std::min(n_ubatch(), cparams.n_ctx_seq);
    size_t max_nodes = graph_max_nodes(chunk);
    if (cparams.flash_attn && !cparams.fa_back_native) {
        max_nodes += (size_t) model.hparams.n_layer() * ggml_flash_attn_ext_back_n_nodes(opt_n_ctx_train, 0);
    }
    llm_graph_result res(max_nodes);
    llama_batch_allocr balloc(model.hparams.n_pos_per_embd());
    auto ubatch = balloc.ubatch_reserve(chunk, 1);
    std::vector<llama_token> labels(chunk, 0);
    ubatch.label = labels.data();
    for (uint32_t i = 0; i < chunk; ++i) {
        ubatch.n_seq_id[i] = 1;
        ubatch.seq_id[i] = &ubatch.seq_id_unq[0];
    }
    const auto params = graph_params(&res, ubatch, mctx.get(), graph_type());
    auto * graph = model.build_graph(params);
    bool ok = false;
    size_t static_bytes = 0, host_bytes = 0;
    const int n_backends = ggml_backend_sched_get_n_backends(sched.get());
    std::vector<size_t> sizes(n_backends, 0);
    if (graph && !res.t_loss_slices.empty()) {
        const size_t count = ggml_graph_size(graph);
        ggml_context_ptr opt_meta(ggml_init({4*count*ggml_tensor_overhead() +
            2*ggml_graph_overhead_custom(count, true), nullptr, true}));
        auto probe = ggml_opt_init(opt_params_base);
        ggml_opt_prepare_alloc_slices(probe, opt_meta.get(), graph, res.get_inp_tokens(),
                res.t_loss_slices.data(), res.t_loss_slices.size());
        ok = ggml_opt_measure_alloc(probe, true, &static_bytes, &host_bytes, sizes.data());
        if (!ok) { opt_failure = ggml_opt_refusal(probe); }
        ggml_opt_free(probe);
    } else {
        opt_failure = "learning preflight could not build the served labelled head";
    }
    ggml_backend_sched_reset(sched.get());
    cparams = saved_cparams;
    if (!ok) { return false; }
    auto report = [&](ggml_backend_buffer_type_t buft, const char * kind, size_t bytes) {
        auto dev = ggml_backend_buft_get_device(buft);
        callback(opt_shadow, dev ? ggml_backend_dev_name(dev) : "host", ggml_backend_buft_name(buft),
                kind, bytes, ggml_backend_buft_is_host(buft), user_data);
    };
    report(ggml_backend_get_default_buffer_type(ggml_backend_sched_get_backend(sched.get(), 0)),
            "optimizer_static_required", static_bytes);
    report(ggml_backend_cpu_buffer_type(), "optimizer_host_required", host_bytes);
    for (int i = 0; i < n_backends; ++i) {
        auto backend = ggml_backend_sched_get_backend(sched.get(), i);
        report(backend_buft[i], "shared_compute_required", sizes[i]);
        report(backend_buft[i], "shared_compute_held", ggml_backend_sched_get_buffer_size(held ? held : sched.get(), backend));
    }
    return true;
}

bool llama_opt_measure_memory(llama_context * ctx, llama_opt_memory_callback callback, void * user_data) {
    return ctx && ctx->opt_measure_memory(callback, user_data);
}
