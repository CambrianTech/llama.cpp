// Continuum fork (UNIFIED-PAGING.md, slice 2): a mind's attention KV in storage of her own, for life, built over memory
// the block layer owns (her mapped file) or reserved in the cache's own device memory. On the Core (one stream),
// binding her points the stream's layer tensors at her storage and parking her points them back: the stream is a view,
// and no byte of KV is copied at any depth. The layout, reservation, attach and repointing are llama-persona-view's,
// shared with the recurrent memory (slice 2c).
// Kept beside llama-kv-cache.cpp so upstream's cache merges without touching it.
#include "llama-kv-cache.h"
#include "llama-persona-view.h"

#include <utility>

// The stream can view a mind's storage on the Core's shape: one stream, so the stream is the whole layer tensor.
bool llama_kv_cache::persona_by_pointer() const {
    return n_stream == 1;
}

// The cache's own tensors as one list, each layer's K then V (nullptr for a layer without V): the order her storage
// is laid out and viewed in.
std::vector<ggml_tensor *> llama_kv_cache::persona_tensors() const {
    std::vector<ggml_tensor *> own;
    own.reserve(2 * layers.size());
    for (const auto & layer : layers) {
        own.push_back(layer.k);
        own.push_back(layer.v);
    }
    return own;
}

namespace {
// her storage's K and V as the cache's one list
std::vector<ggml_tensor *> listed(const llama_kv_storage & storage) {
    std::vector<ggml_tensor *> out;
    out.reserve(2 * storage.k.size());
    for (size_t i = 0; i < storage.k.size(); ++i) {
        out.push_back(storage.k[i]);
        out.push_back(storage.v[i]);
    }
    return out;
}

// a storage built by llama-persona-view, split back into her K and V
llama_kv_storage split(llama_persona_tensors && built) {
    llama_kv_storage out;
    out.ctxs = std::move(built.ctxs);
    out.bufs = std::move(built.bufs);
    for (size_t i = 0; i + 1 < built.t.size(); i += 2) {
        out.k.push_back(built.t[i]);
        out.v.push_back(built.t[i + 1]);
    }
    return out;
}
} // namespace

// The stream views her storage (`nullptr`: its own memory again), and the stream views over each layer follow. Nothing
// changes hands: her storage is hers for life.
void llama_kv_cache::persona_view(llama_kv_storage * storage) {
    const std::vector<ggml_tensor *> own = persona_tensors();
    if (storage != nullptr) {
        const std::vector<ggml_tensor *> hers = listed(*storage);
        llama_persona_view::point(own, &hers, persona_own);
    } else {
        llama_persona_view::point(own, nullptr, persona_own);
    }
    for (auto & layer : layers) {
        for (auto [base, view] : { std::pair { layer.k, layer.k_stream.empty() ? nullptr : layer.k_stream[0] },
                                   std::pair { layer.v, layer.v_stream.empty() ? nullptr : layer.v_stream[0] } }) {
            if (base != nullptr && view != nullptr) {
                view->data   = static_cast<char *>(base->data) + view->view_offs;
                view->buffer = base->buffer;
            }
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
    for (ggml_tensor * t : persona_tensors()) {
        if (t != nullptr) {
            ggml_backend_tensor_memset(t, 0, 0, ggml_nbytes(t));
        }
    }
    return true;
}

// The bytes one mind's storage takes on the Core (0: not the Core's shape, or KV on more than one device).
size_t llama_kv_cache::persona_storage_bytes() const {
    return persona_by_pointer() ? llama_persona_view::bytes(persona_tensors()) : 0;
}

// Whether her storage can be memory the caller owns (a mapped file): the KV device wraps host memory. Where it cannot
// (CUDA), the fork reserves her storage in its own device memory instead, viewed the same way.
bool llama_kv_cache::persona_storage_viewable() const {
    return persona_storage_bytes() > 0 && llama_persona_view::viewable(persona_tensors());
}

// Her storage reserved in the cache's own buffer types (VRAM on a discrete GPU), hers for life and viewed exactly as an
// attached one. out_of_memory leaves her without storage: she moves by row copy until room is made.
llama_persona_move llama_kv_cache::persona_reserve_storage(llama_persona_blocks & blocks) const {
    if (persona_storage_bytes() == 0) {
        return llama_persona_move::unsupported;
    }
    if (blocks.kv.storage.held()) {
        return llama_persona_move::stream_not_empty;
    }
    llama_persona_tensors built;
    const llama_persona_move reserved = llama_persona_view::reserve(persona_tensors(), built);
    if (reserved == llama_persona_move::done) {
        blocks.kv.storage = split(std::move(built));
    }
    return reserved;
}

// Her storage built over `base..base+bytes`, memory the caller owns (a mapped file, UNIFIED-PAGING.md slice 2).
// Refused as unsupported where a KV device cannot wrap host memory (CUDA reserves instead).
llama_persona_move llama_kv_cache::persona_attach_storage(llama_persona_blocks & blocks, void * base, size_t bytes) const {
    if (!persona_by_pointer()) {
        return llama_persona_move::unsupported;
    }
    if (blocks.kv.storage.held()) {
        return llama_persona_move::stream_not_empty; // she already holds storage: attach once, before her first move
    }
    llama_persona_tensors built;
    const llama_persona_move attached = llama_persona_view::attach(persona_tensors(), base, bytes, built);
    if (attached == llama_persona_move::done) {
        blocks.kv.storage = split(std::move(built));
        blocks.kv.storage.attached_bytes = llama_persona_view::bytes(persona_tensors());
    }
    return attached;
}


// A storage dropped while a stream views it points that stream back at its own memory first, and a cache destroyed
// while viewing a storage lets go of it: neither side can hold a pointer the other has freed. A storage moves only
// while no stream views it (her blocks are built once, before any bind).
llama_kv_storage::llama_kv_storage(llama_kv_storage && other) noexcept
    : ctxs(std::move(other.ctxs)), bufs(std::move(other.bufs)), k(std::move(other.k)), v(std::move(other.v)),
      attached_bytes(other.attached_bytes) {
    GGML_ASSERT(other.viewer == nullptr && "a viewed storage never moves");
}

llama_kv_storage & llama_kv_storage::operator=(llama_kv_storage && other) noexcept {
    GGML_ASSERT(viewer == nullptr && other.viewer == nullptr && "a viewed storage never moves");
    ctxs = std::move(other.ctxs);
    bufs = std::move(other.bufs);
    k    = std::move(other.k);
    v    = std::move(other.v);
    attached_bytes = other.attached_bytes;
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
