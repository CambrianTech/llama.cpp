// Continuum: what a context reports to our serving governor (buffer bytes, cell usage per holder), kept beside
// llama-context.cpp so upstream's context merges without touching it.
#include "llama-context.h"

#include "llama-impl.h"
#include "llama-memory.h"

#include <stdexcept>

llama_context_buffer_bytes llama_context::buffer_bytes() const {
    llama_context_buffer_bytes result = {};
    const auto add = [&](ggml_backend_buffer_type_t buft, uint64_t bytes) {
        auto & pool = ggml_backend_buft_is_host(buft) ? result.host : result.device;
        pool += bytes;
    };
    for (const auto & [buft, bytes] : memory_breakdown()) {
        add(buft, bytes.context + bytes.compute);
    }
    if (buf_output) {
        add(ggml_backend_buffer_get_type(buf_output.get()), ggml_backend_buffer_get_size(buf_output.get()));
    }
    return result;
}

int32_t llama_memory_cell_usage(
        llama_memory_t mem,
        struct llama_memory_cells_report * out,
        int32_t n_max) {
    if (!mem) {
        return 0;
    }
    const auto holders = mem->cell_usage();
    for (int32_t i = 0; i < (int32_t) holders.size() && i < n_max; ++i) {
        const auto & h = holders[i];
        snprintf(out[i].holder, sizeof(out[i].holder), "%s", h.holder.c_str());
        out[i].used = h.used;
        out[i].size = h.size;
    }
    return (int32_t) holders.size();
}

bool llama_context_get_buffer_bytes(const llama_context * ctx, llama_context_buffer_bytes * out) {
    if (!ctx || !out) {
        return false;
    }
    try {
        *out = ctx->buffer_bytes();
        return true;
    } catch (const std::exception & e) {
        LLAMA_LOG_ERROR("%s: %s\n", __func__, e.what());
        return false;
    }
}

// Continuum: the context parameters upstream's constructor does not read (called from it, one line)
void llama_context::continuum_params(const llama_context_params & params) {
    if (params.embeddings_only && (!params.embeddings || params.pooling_type == LLAMA_POOLING_TYPE_NONE || params.pooling_type == LLAMA_POOLING_TYPE_UNSPECIFIED)) {
        throw std::runtime_error("embeddings_only requires embeddings and explicit sequence pooling");
    }
    cparams.embeddings_only = params.embeddings_only;
}
