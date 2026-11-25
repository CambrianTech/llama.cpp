// Continuum fork (THE-LANE-ON-SAGE.md, the pointer move): a mind's attention KV in storage of her own, moved by
// pointer. On the Core (one stream), the stream and every parked mind each own one storage of the stream's exact shape;
// a save or bind exchanges the stream's storage with hers (each layer tensor's data, buffer and extra, the stream views
// over them, and the buffers' ownership). No byte of KV is copied, at any depth.
// Kept beside llama-kv-cache.cpp so upstream's cache merges without touching it.
#include "llama-kv-cache.h"

#include <map>
#include <utility>

// A move by pointer needs only the Core's shape: one stream, so the stream is the whole layer tensor. The exchange
// swaps which buffer the device reads and the CPU never touches the data, so it holds for any buffer type (Metal shared
// or private, VRAM, host). What differs by memory is capacity: a parked mind's storage stays where it was allocated
// until a tier moves it, never copied in or out by a switch.
bool llama_kv_cache::persona_by_pointer() const {
    return n_stream == 1;
}

// Her storage: per layer a K and a V tensor of the stream's exact type and shape, in buffers of the stream's own types,
// allocated once at her prepare. Cleared once, as the cache clears its own, so no padding a kernel reads is NaN.
bool llama_kv_cache::persona_reserve_storage(llama_kv_storage & storage) const {
    if (storage.held()) {
        return true;
    }
    llama_kv_storage fresh;
    std::map<ggml_backend_buffer_type_t, ggml_context *> by_type;
    auto shaped_as = [&](const ggml_tensor * t) -> ggml_tensor * {
        if (t == nullptr) {
            return nullptr;
        }
        const ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(t->buffer);
        auto it = by_type.find(buft);
        if (it == by_type.end()) {
            ggml_init_params params = { /*.mem_size =*/ 2u * layers.size() * ggml_tensor_overhead(), /*.mem_buffer =*/ nullptr, /*.no_alloc =*/ true };
            ggml_context * ctx = ggml_init(params);
            if (ctx == nullptr) {
                return nullptr;
            }
            fresh.ctxs.emplace_back(ctx);
            it = by_type.emplace(buft, ctx).first;
        }
        return ggml_new_tensor(it->second, t->type, GGML_MAX_DIMS, t->ne);
    };
    for (const auto & layer : layers) {
        ggml_tensor * k = shaped_as(layer.k);
        ggml_tensor * v = shaped_as(layer.v);
        if ((layer.k && !k) || (layer.v && !v)) {
            return false;
        }
        fresh.k.push_back(k);
        fresh.v.push_back(v);
    }
    for (const auto & [buft, ctx] : by_type) {
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
        if (buf == nullptr) {
            return false;
        }
        ggml_backend_buffer_clear(buf, 0);
        fresh.bufs.emplace_back(buf);
    }
    storage = std::move(fresh);
    return true;
}

// The exchange: each layer tensor's storage swapped with hers, the stream views (offset 0: one stream) pointed at the
// stream's new storage, and the buffers' ownership swapped with the data (the stream's side is `persona_resident`; empty
// while it reads its own `ctxs_bufs`). The graph owner rebuilds after a move: a built graph holds the old addresses.
void llama_kv_cache::persona_exchange_storage(llama_kv_storage & storage) {
    auto exchange = [](ggml_tensor * a, ggml_tensor * b) {
        std::swap(a->data, b->data);
        std::swap(a->buffer, b->buffer);
        std::swap(a->extra, b->extra);
    };
    auto repoint = [](ggml_tensor * view, const ggml_tensor * base) {
        view->data   = static_cast<char *>(base->data) + view->view_offs;
        view->buffer = base->buffer;
    };
    for (size_t i = 0; i < layers.size(); ++i) {
        auto & layer = layers[i];
        if (layer.k) {
            exchange(layer.k, storage.k[i]);
            repoint(layer.k_stream[0], layer.k);
        }
        if (layer.v) {
            exchange(layer.v, storage.v[i]);
            repoint(layer.v_stream[0], layer.v);
        }
    }
    std::swap(storage.bufs, persona_resident.bufs);
}

// A clear of the data zeroes what the stream reads now. While that is a mind's storage (a pointer move put it there),
// the cache's own buffers belong to a parked mind and are not the cache's to clear.
bool llama_kv_cache::persona_clear_resident() {
    if (persona_resident.bufs.empty()) {
        return false;
    }
    for (const auto & layer : layers) {
        for (ggml_tensor * t : { layer.k, layer.v }) {
            if (t != nullptr) {
                ggml_backend_tensor_memset(t, 0, 0, ggml_nbytes(t));
            }
        }
    }
    return true;
}
