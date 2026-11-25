// Continuum fork (UNIFIED-PAGING.md, slices 2 and 2c): see llama-persona-view.h. Kept in its own file so upstream's
// caches merge without touching it.
#include "llama-persona-view.h"

#include <algorithm>
#include <map>

namespace llama_persona_view {

namespace {

struct placement {
    ggml_backend_dev_t dev        = nullptr;
    bool               one_device = true;
    size_t             bytes      = 0;
    size_t             largest    = 0;
};

// `each(index, offset)` for every present tensor of `own`, in order; the bytes and device of the whole run
template <typename Each>
placement lay_out(const std::vector<ggml_tensor *> & own, Each && each) {
    placement out;
    for (size_t i = 0; i < own.size(); ++i) {
        const ggml_tensor * t = own[i];
        if (t == nullptr) {
            continue;
        }
        const ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(t->buffer);
        const ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
        out.one_device = out.one_device && (out.dev == nullptr || out.dev == dev);
        out.dev = dev;
        out.bytes = GGML_PAD(out.bytes, ggml_backend_buft_get_alignment(buft));
        each(i, out.bytes);
        out.bytes += ggml_nbytes(t);
        out.largest = std::max(out.largest, ggml_nbytes(t));
    }
    return out;
}

void no_placement(size_t, size_t) {}

// a context holding metadata for `n` tensors
ggml_context * metadata_for(size_t n) {
    ggml_init_params params = { /*.mem_size =*/ std::max<size_t>(n, 1) * ggml_tensor_overhead(), /*.mem_buffer =*/ nullptr, /*.no_alloc =*/ true };
    return ggml_init(params);
}

} // namespace

size_t bytes(const std::vector<ggml_tensor *> & own) {
    const placement p = lay_out(own, no_placement);
    return p.one_device && p.dev != nullptr ? p.bytes : 0;
}

bool viewable(const std::vector<ggml_tensor *> & own) {
    const placement p = lay_out(own, no_placement);
    if (!p.one_device || p.dev == nullptr) {
        return false;
    }
    ggml_backend_dev_props props;
    ggml_backend_dev_get_props(p.dev, &props);
    return props.caps.buffer_from_host_ptr;
}

llama_persona_move reserve(const std::vector<ggml_tensor *> & own, tensors & out) {
    if (bytes(own) == 0) {
        return llama_persona_move::unsupported;
    }
    tensors fresh;
    std::map<ggml_backend_buffer_type_t, ggml_context *> by_type;
    for (const ggml_tensor * t : own) {
        if (t == nullptr) {
            fresh.t.push_back(nullptr);
            continue;
        }
        const ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(t->buffer);
        auto it = by_type.find(buft);
        if (it == by_type.end()) {
            ggml_context * ctx = metadata_for(own.size());
            if (ctx == nullptr) {
                return llama_persona_move::out_of_memory;
            }
            fresh.ctxs.emplace_back(ctx);
            it = by_type.emplace(buft, ctx).first;
        }
        fresh.t.push_back(ggml_new_tensor(it->second, t->type, GGML_MAX_DIMS, t->ne));
    }
    for (const auto & [buft, ctx] : by_type) {
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
        if (buf == nullptr) {
            return llama_persona_move::out_of_memory;
        }
        ggml_backend_buffer_clear(buf, 0);
        fresh.bufs.emplace_back(buf);
    }
    out = std::move(fresh);
    return llama_persona_move::done;
}

llama_persona_move attach(const std::vector<ggml_tensor *> & own, void * base, size_t n, tensors & out) {
    if (base == nullptr) {
        return llama_persona_move::unsupported;
    }
    const placement probe = lay_out(own, no_placement);
    if (!probe.one_device || probe.dev == nullptr || !viewable(own)) {
        return llama_persona_move::unsupported;
    }
    if (n < probe.bytes) {
        return llama_persona_move::layout_mismatch;
    }
    tensors fresh;
    ggml_context * ctx = metadata_for(own.size());
    if (ctx == nullptr) {
        return llama_persona_move::out_of_memory;
    }
    fresh.ctxs.emplace_back(ctx);
    for (const ggml_tensor * t : own) {
        fresh.t.push_back(t ? ggml_new_tensor(ctx, t->type, GGML_MAX_DIMS, t->ne) : nullptr);
    }
    ggml_backend_buffer_t buf = ggml_backend_dev_buffer_from_host_ptr(probe.dev, base, n, probe.largest);
    if (buf == nullptr) {
        return llama_persona_move::out_of_memory;
    }
    fresh.bufs.emplace_back(buf);
    char * at = static_cast<char *>(base);
    bool placed = true;
    lay_out(own, [&](size_t i, size_t offset) {
        placed = placed && ggml_backend_tensor_alloc(buf, fresh.t[i], at + offset) == GGML_STATUS_SUCCESS;
    });
    if (!placed) {
        return llama_persona_move::layout_mismatch;
    }
    out = std::move(fresh);
    return llama_persona_move::done;
}

void point(const std::vector<ggml_tensor *> & own, const std::vector<ggml_tensor *> * to, std::vector<memory> & saved) {
    if (saved.empty()) {
        for (const ggml_tensor * t : own) {
            saved.push_back(t ? memory { t->data, t->buffer, t->extra } : memory {});
        }
    }
    for (size_t i = 0; i < own.size(); ++i) {
        ggml_tensor * t = own[i];
        if (t == nullptr) {
            continue;
        }
        const memory m = to ? memory { (*to)[i]->data, (*to)[i]->buffer, (*to)[i]->extra } : saved[i];
        t->data   = m.data;
        t->buffer = m.buffer;
        t->extra  = m.extra;
    }
}

size_t after(size_t bytes) {
    constexpr size_t PAGE = 64 * 1024;
    return GGML_PAD(bytes, PAGE);
}

} // namespace llama_persona_view
