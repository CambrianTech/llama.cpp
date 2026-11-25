// Continuum fork (UNIFIED-PAGING.md, slice 2c): a mind's recurrent state in storage of her own, for life, viewed by
// pointer exactly as her attention KV is. On the Core the recurrent memory has one cell, so each layer's r, s and p
// tensors are exactly her rows in every snapshot group: binding her points them at her storage and parking her points
// them back. Measured before (M5, Qwen3.6-35B-A3B, 62.8 MiB of state): a switch copied it out and in, p50 3.16 ms.
// The hybrid memory lays her two parts out in one run of memory (her one mapped file, or one reservation in VRAM).
// Kept in its own file so upstream's memories merge without touching it.
#include "llama-memory-hybrid.h"
#include "llama-memory-recurrent.h"
#include "llama-persona-view.h"

#include <utility>

bool llama_memory_recurrent::persona_by_pointer() const {
    return size == 1;
}

std::vector<ggml_tensor *> llama_memory_recurrent::persona_tensors() const {
    std::vector<ggml_tensor *> own;
    own.reserve(3 * r_l.size());
    for (size_t il = 0; il < r_l.size(); ++il) {
        own.push_back(r_l[il]);
        own.push_back(s_l[il]);
        own.push_back(il < p_l.size() ? p_l[il] : nullptr);
    }
    return own;
}

void llama_memory_recurrent::persona_view(llama_rs_storage * storage) {
    llama_persona_view::point(persona_tensors(), storage ? &storage->own.t : nullptr, persona_own);
    if (persona_viewed != nullptr) {
        persona_viewed->viewer = nullptr;
    }
    if (storage != nullptr) {
        storage->viewer = this;
    }
    persona_viewed = storage;
}

// While she is viewed the memory reads her storage, so a clear zeroes hers and leaves its own buffers (which no one
// reads until it views them again); otherwise the upstream clear of its own buffers runs.
bool llama_memory_recurrent::persona_clear_view() {
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

size_t llama_memory_recurrent::persona_storage_bytes() const {
    return persona_by_pointer() ? llama_persona_view::bytes(persona_tensors()) : 0;
}

bool llama_memory_recurrent::persona_storage_viewable() const {
    return persona_storage_bytes() > 0 && llama_persona_view::viewable(persona_tensors());
}

llama_persona_move llama_memory_recurrent::persona_reserve_storage(llama_persona_blocks & blocks) const {
    if (persona_storage_bytes() == 0) {
        return llama_persona_move::unsupported;
    }
    if (blocks.rs.storage.held()) {
        return llama_persona_move::stream_not_empty;
    }
    return llama_persona_view::reserve(persona_tensors(), blocks.rs.storage.own);
}

llama_persona_move llama_memory_recurrent::persona_attach_storage(llama_persona_blocks & blocks, void * base, size_t bytes) const {
    if (!persona_by_pointer()) {
        return llama_persona_move::unsupported;
    }
    if (blocks.rs.storage.held()) {
        return llama_persona_move::stream_not_empty;
    }
    return llama_persona_view::attach(persona_tensors(), base, bytes, blocks.rs.storage.own);
}

llama_memory_recurrent::~llama_memory_recurrent() {
    if (persona_viewed != nullptr) {
        persona_view(nullptr);
    }
}

// A storage dropped while viewed points the memory back at its own first; a viewed storage never moves.
llama_rs_storage::llama_rs_storage(llama_rs_storage && other) noexcept : own(std::move(other.own)) {
    GGML_ASSERT(other.viewer == nullptr && "a viewed storage never moves");
}

llama_rs_storage & llama_rs_storage::operator=(llama_rs_storage && other) noexcept {
    GGML_ASSERT(viewer == nullptr && other.viewer == nullptr && "a viewed storage never moves");
    own = std::move(other.own);
    return *this;
}

llama_rs_storage::~llama_rs_storage() {
    if (viewer != nullptr) {
        viewer->persona_view(nullptr);
    }
}

// The hybrid: her attention storage first, her recurrent storage after it at a page boundary, in one run of memory.
// A memory whose recurrent half has no storage (not the Core's one cell) moves her recurrent rows by copy.
size_t llama_memory_hybrid::persona_storage_bytes() const {
    const size_t attn = mem_attn->persona_storage_bytes();
    const size_t recr = mem_recr->persona_storage_bytes();
    if (attn == 0) {
        return 0;
    }
    return recr == 0 ? attn : llama_persona_view::after(attn) + recr;
}

bool llama_memory_hybrid::persona_storage_viewable() const {
    return mem_attn->persona_storage_viewable() && (mem_recr->persona_storage_bytes() == 0 || mem_recr->persona_storage_viewable());
}

llama_persona_move llama_memory_hybrid::persona_reserve_storage(llama_persona_blocks & blocks) const {
    const llama_persona_move attn = mem_attn->persona_reserve_storage(blocks);
    if (attn != llama_persona_move::done || mem_recr->persona_storage_bytes() == 0) {
        return attn;
    }
    const llama_persona_move recr = mem_recr->persona_reserve_storage(blocks);
    if (recr != llama_persona_move::done) {
        blocks.kv.storage = llama_kv_storage {}; // both parts or neither: she is never half by pointer
    }
    return recr;
}

llama_persona_move llama_memory_hybrid::persona_attach_storage(llama_persona_blocks & blocks, void * base, size_t bytes) const {
    const size_t attn_bytes = mem_attn->persona_storage_bytes();
    const size_t recr_bytes = mem_recr->persona_storage_bytes();
    if (recr_bytes == 0) {
        return mem_attn->persona_attach_storage(blocks, base, bytes);
    }
    const size_t at = llama_persona_view::after(attn_bytes);
    if (base == nullptr || bytes < at + recr_bytes) {
        return base == nullptr ? llama_persona_move::unsupported : llama_persona_move::layout_mismatch;
    }
    const llama_persona_move attn = mem_attn->persona_attach_storage(blocks, base, at);
    if (attn != llama_persona_move::done) {
        return attn;
    }
    const llama_persona_move recr = mem_recr->persona_attach_storage(blocks, static_cast<char *>(base) + at, bytes - at);
    if (recr != llama_persona_move::done) {
        blocks.kv.storage = llama_kv_storage {};
        return recr;
    }
    blocks.kv.storage.attached_bytes = at + recr_bytes; // her record names the whole run: both parts are in it
    return llama_persona_move::done;
}
