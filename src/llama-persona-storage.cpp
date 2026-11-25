// Continuum fork (UNIFIED-PAGING.md, slice 2): a mind's attention KV in storage of her own, for life, built over memory
// the block layer owns (her mapped file). On the Core (one stream), binding her points the stream's layer tensors at
// her storage and parking her points them back: the stream is a view, and no byte of KV is copied at any depth. A
// device that cannot wrap host memory keeps the row-copy move (the discrete adapter, slice 4).
// Kept beside llama-kv-cache.cpp so upstream's cache merges without touching it.
#include "llama-kv-cache.h"

#include <algorithm>
#include <map>
#include <utility>

// The stream can view a mind's storage on the Core's shape: one stream, so the stream is the whole layer tensor.
bool llama_kv_cache::persona_by_pointer() const {
    return n_stream == 1;
}

// The stream views her storage: each layer tensor's data, buffer and extra point at hers, and the stream views over
// them follow. `nullptr` points the stream back at its own memory, recorded before its first view. Nothing changes
// hands: her storage is hers for life, so a storage over her file is never written by anyone else. A built graph holds
// the old addresses, so the context drops its graphs after every move.
void llama_kv_cache::persona_view(llama_kv_storage * storage) {
    if (persona_own.empty()) {
        for (const auto & layer : layers) {
            for (const ggml_tensor * t : { layer.k, layer.v }) {
                persona_own.push_back(t ? persona_memory { t->data, t->buffer, t->extra } : persona_memory {});
            }
        }
    }
    auto point = [](ggml_tensor * t, const persona_memory & m) {
        t->data   = m.data;
        t->buffer = m.buffer;
        t->extra  = m.extra;
    };
    auto repoint = [](ggml_tensor * view, const ggml_tensor * base) {
        view->data   = static_cast<char *>(base->data) + view->view_offs;
        view->buffer = base->buffer;
    };
    for (size_t i = 0; i < layers.size(); ++i) {
        auto & layer = layers[i];
        if (layer.k) {
            point(layer.k, storage ? persona_memory { storage->k[i]->data, storage->k[i]->buffer, storage->k[i]->extra } : persona_own[2 * i]);
            repoint(layer.k_stream[0], layer.k);
        }
        if (layer.v) {
            point(layer.v, storage ? persona_memory { storage->v[i]->data, storage->v[i]->buffer, storage->v[i]->extra } : persona_own[2 * i + 1]);
            repoint(layer.v_stream[0], layer.v);
        }
    }
    if (persona_viewed != nullptr) {
        persona_viewed->viewer = nullptr;
    }
    if (storage != nullptr) {
        storage->viewer = this;
    }
    persona_viewed = storage;
}

// A clear of the data zeroes what the stream views. While that is a mind's storage, the stream's own memory is not
// what it reads, so the upstream clear of its own buffers would miss her and is skipped.
bool llama_kv_cache::persona_clear_view() {
    if (persona_viewed == nullptr) {
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

// One mind's storage laid out over one run of memory: each layer's K then V, each at an offset aligned for its buffer
// type, all on one device. Shared by the size a caller asks for and the attach that builds over it, so they agree.
namespace {
struct storage_layout {
    ggml_backend_dev_t dev        = nullptr;
    bool               one_device = true;
    size_t             bytes      = 0;
    size_t             largest    = 0;
};

// `each(layer, is_v, offset)` for every present tensor, in layout order
template <typename Layers, typename Each>
storage_layout lay_out(const Layers & layers, Each && each) {
    storage_layout out;
    for (size_t l = 0; l < layers.size(); ++l) {
        for (const bool is_v : { false, true }) {
            const ggml_tensor * t = is_v ? layers[l].v : layers[l].k;
            if (t == nullptr) {
                continue;
            }
            const ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(t->buffer);
            const ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
            out.one_device = out.one_device && (out.dev == nullptr || out.dev == dev);
            out.dev = dev;
            out.bytes = GGML_PAD(out.bytes, ggml_backend_buft_get_alignment(buft));
            each(l, is_v, out.bytes);
            out.bytes += ggml_nbytes(t);
            out.largest = std::max(out.largest, ggml_nbytes(t));
        }
    }
    return out;
}

void no_placement(size_t, bool, size_t) {}
} // namespace

// The bytes one mind's storage takes, or 0 when the stream cannot view caller memory here: not the Core's shape, or a
// KV device that cannot wrap host memory. The size is the capability, so a caller decides from it alone.
size_t llama_kv_cache::persona_storage_bytes() const {
    if (!persona_by_pointer()) {
        return 0;
    }
    const storage_layout layout = lay_out(layers, no_placement);
    if (!layout.one_device || layout.dev == nullptr) {
        return 0;
    }
    ggml_backend_dev_props props;
    ggml_backend_dev_get_props(layout.dev, &props);
    return props.caps.buffer_from_host_ptr ? layout.bytes : 0;
}

// Her storage built over `base..base+bytes`, memory the caller owns (a mapped file, UNIFIED-PAGING.md slice 2): the
// tensors are the stream's exact shape, placed by the same layout as persona_storage_bytes, in one device buffer that
// wraps the memory where it lies (`ggml_backend_dev_buffer_from_host_ptr`; Metal: newBufferWithBytesNoCopy, shared).
// Nothing is cleared: the memory is hers (her file's bytes, or a fresh file's zeros). Refused as unsupported where a
// KV device cannot wrap host memory (CUDA: its storage stays the fork's own until the discrete adapter, slice 4).
llama_persona_move llama_kv_cache::persona_attach_storage(llama_persona_blocks & blocks, void * base, size_t bytes) const {
    if (!persona_by_pointer() || base == nullptr) {
        return llama_persona_move::unsupported;
    }
    if (blocks.kv.storage.held()) {
        return llama_persona_move::stream_not_empty; // she already holds storage: attach once, before her first move
    }
    const storage_layout probe = lay_out(layers, no_placement);
    if (!probe.one_device || probe.dev == nullptr) {
        return llama_persona_move::unsupported;
    }
    ggml_backend_dev_props props;
    ggml_backend_dev_get_props(probe.dev, &props);
    if (!props.caps.buffer_from_host_ptr) {
        return llama_persona_move::unsupported;
    }
    if (bytes < probe.bytes) {
        return llama_persona_move::layout_mismatch;
    }
    llama_kv_storage fresh;
    ggml_init_params params = { /*.mem_size =*/ 2u * layers.size() * ggml_tensor_overhead(), /*.mem_buffer =*/ nullptr, /*.no_alloc =*/ true };
    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        return llama_persona_move::out_of_memory;
    }
    fresh.ctxs.emplace_back(ctx);
    for (const auto & layer : layers) {
        fresh.k.push_back(layer.k ? ggml_new_tensor(ctx, layer.k->type, GGML_MAX_DIMS, layer.k->ne) : nullptr);
        fresh.v.push_back(layer.v ? ggml_new_tensor(ctx, layer.v->type, GGML_MAX_DIMS, layer.v->ne) : nullptr);
    }
    ggml_backend_buffer_t buf = ggml_backend_dev_buffer_from_host_ptr(probe.dev, base, bytes, probe.largest);
    if (buf == nullptr) {
        return llama_persona_move::out_of_memory;
    }
    fresh.bufs.emplace_back(buf);
    char * at = static_cast<char *>(base);
    bool placed = true;
    lay_out(layers, [&](size_t l, bool is_v, size_t offset) {
        ggml_tensor * t = is_v ? fresh.v[l] : fresh.k[l];
        placed = placed && ggml_backend_tensor_alloc(buf, t, at + offset) == GGML_STATUS_SUCCESS;
    });
    if (!placed) {
        return llama_persona_move::layout_mismatch;
    }
    blocks.kv.storage = std::move(fresh);
    return llama_persona_move::done;
}

// A storage dropped while a stream views it points that stream back at its own memory first, and a cache destroyed
// while viewing a storage lets go of it: neither side can hold a pointer the other has freed. A storage moves only
// while no stream views it (her blocks are built once, before any bind).
llama_kv_storage::llama_kv_storage(llama_kv_storage && other) noexcept
    : ctxs(std::move(other.ctxs)), bufs(std::move(other.bufs)), k(std::move(other.k)), v(std::move(other.v)) {
    GGML_ASSERT(other.viewer == nullptr && "a viewed storage never moves");
}

llama_kv_storage & llama_kv_storage::operator=(llama_kv_storage && other) noexcept {
    GGML_ASSERT(viewer == nullptr && other.viewer == nullptr && "a viewed storage never moves");
    ctxs = std::move(other.ctxs);
    bufs = std::move(other.bufs);
    k    = std::move(other.k);
    v    = std::move(other.v);
    return *this;
}

llama_kv_storage::~llama_kv_storage() {
    if (viewer != nullptr) {
        viewer->persona_view(nullptr);
    }
}

llama_kv_cache::~llama_kv_cache() {
    if (persona_viewed != nullptr) {
        persona_view(nullptr);
    }
}
