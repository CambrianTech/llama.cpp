#include "llama-context.h"
#include "llama-model.h"
#include <exception>

void llama_context::visit_context_memory(llama_context_memory_callback callback, void * data) const {
    for (const auto & [buft, size] : memory_breakdown()) {
        callback(ggml_backend_buft_name(buft), ggml_backend_buft_is_host(buft),
                 size.context + size.compute, data);
    }
    if (output_buft) {
        callback(ggml_backend_buft_name(output_buft), ggml_backend_buft_is_host(output_buft), output_bytes, data);
    }
}

bool llama_context_measure_memory(llama_model * model, llama_context_params params,
                                  llama_context_memory_callback callback, void * data) {
    if (!model || !callback || params.n_ctx == 0 || params.n_batch == 0 || params.n_ubatch == 0) {
        return false;
    }
    try {
        std::unique_ptr<llama_context> measured(llama_context_create(model, params, true));
        if (!measured) { return false; }
        measured->visit_context_memory(callback, data);
        return true;
    } catch (const std::exception & error) {
        LLAMA_LOG_ERROR("%s: %s\n", __func__, error.what());
        return false;
    }
}

void llama_context_visit_memory(const llama_context * ctx, llama_context_memory_callback callback, void * data) {
    if (ctx && callback) { ctx->visit_context_memory(callback, data); }
}
