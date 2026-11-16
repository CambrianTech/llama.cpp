#pragma once

// Continuum fork: THE ONE STATE MOVER between a sequence in a llama_context and host memory
// (SERVING-ENGINE.md, S2). A page is a server_prompt_cache_state: the prompt's tokens with their
// host-memory checkpoints, the {main, drft} state bytes of the target and draft contexts, and the
// adapter set they were computed under. The prompt cache (server_slot::prompt_save,
// server_prompt_cache::load), the live reconfigure (server-reconfigure.h) and the saved page's
// checkpoint sidecar (/slots?action=save|restore) all move state through this file; nothing else
// calls llama_state_seq_get_data/set_data for a slot.

#include "server-task.h"

#include "llama.h"

#include <cstddef>
#include <cstdint>
#include <list>
#include <vector>

using server_slot_page = server_prompt_cache_state;

// the bytes a sequence's state takes, in the target context and in the draft (0 without one)
struct server_slot_page_size {
    size_t main = 0;
    size_t drft = 0;

    size_t total() const { return main + drft; }
};

server_slot_page_size server_slot_page_measure(llama_context * ctx_tgt, llama_context * ctx_dft, llama_seq_id seq);

// page out: the sequence's target and draft state into `data`, whose vectors are already sized by
// server_slot_page_measure. Reading never changes the context. False if a context gave back fewer
// bytes than measured (the page is then not a copy of the sequence and must not be used).
bool server_slot_page_read(llama_context * ctx_tgt, llama_context * ctx_dft, llama_seq_id seq, server_prompt_data & data);

// page in: `data` into the sequence. False if a context could not take it (no room in its cache, a
// draft page with no draft context); the sequence then holds nothing in either context, so a
// failed page-in never leaves half a state behind.
bool server_slot_page_write(llama_context * ctx_tgt, llama_context * ctx_dft, llama_seq_id seq, const server_prompt_data & data);

// THE CHECKPOINT SIDECAR a saved page carries beside its state file (<file>.ckpt): 'CKPT', version 1,
// the count, then each checkpoint's positions, sizes and bytes. The size quote and the write share
// the writer (a counting `put` and a file `put`); the restore reads it back with the reader.
constexpr uint32_t SERVER_SLOT_PAGE_CKPT_MAGIC   = 0x434B5054u; // 'CKPT'
constexpr uint32_t SERVER_SLOT_PAGE_CKPT_VERSION = 1;

// put(const void * ptr, size_t n)
template <typename Put>
void server_slot_page_put_checkpoints(const std::vector<const common_prompt_checkpoint *> & kept, Put && put) {
    const uint32_t magic   = SERVER_SLOT_PAGE_CKPT_MAGIC;
    const uint32_t version = SERVER_SLOT_PAGE_CKPT_VERSION;
    const uint32_t n_ckpt  = (uint32_t) kept.size();
    put(&magic, sizeof(magic)); put(&version, sizeof(version)); put(&n_ckpt, sizeof(n_ckpt));
    for (const auto * c : kept) {
        const int64_t  n_tokens = c->n_tokens;
        const int32_t  pos_min  = c->pos_min;
        const int32_t  pos_max  = c->pos_max;
        const uint64_t sz_tgt   = c->data_tgt.size();
        const uint64_t sz_dft   = c->data_dft.size();
        const uint64_t sz_spec  = c->data_spec.size();
        put(&n_tokens, sizeof(n_tokens)); put(&pos_min, sizeof(pos_min)); put(&pos_max, sizeof(pos_max));
        put(&sz_tgt, sizeof(sz_tgt)); put(&sz_dft, sizeof(sz_dft)); put(&sz_spec, sizeof(sz_spec));
        put(c->data_tgt.data(), c->data_tgt.size());
        put(c->data_dft.data(), c->data_dft.size());
        put(c->data_spec.data(), c->data_spec.size());
    }
}

// get(void * ptr, size_t n) -> bool (false on a short read); on_checkpoint() after each one read.
// False at the first short read, a foreign header or an implausible size; `out` is then partial and
// the caller drops it.
template <typename Get, typename OnCheckpoint>
bool server_slot_page_get_checkpoints(Get && get, std::list<common_prompt_checkpoint> & out, OnCheckpoint && on_checkpoint) {
    uint32_t magic = 0, version = 0, n_ckpt = 0;
    bool ok = get(&magic, sizeof(magic)) && get(&version, sizeof(version)) && get(&n_ckpt, sizeof(n_ckpt));
    // a page holds at most a few hundred checkpoints; 4096 refuses a corrupt count before allocating
    ok = ok && magic == SERVER_SLOT_PAGE_CKPT_MAGIC && version == SERVER_SLOT_PAGE_CKPT_VERSION && n_ckpt <= 4096;
    for (uint32_t i = 0; ok && i < n_ckpt; ++i) {
        int64_t n_tokens = 0; int32_t pos_min = 0, pos_max = 0;
        uint64_t sz_tgt = 0, sz_dft = 0, sz_spec = 0;
        ok = get(&n_tokens, sizeof(n_tokens)) && get(&pos_min, sizeof(pos_min)) && get(&pos_max, sizeof(pos_max))
          && get(&sz_tgt, sizeof(sz_tgt)) && get(&sz_dft, sizeof(sz_dft)) && get(&sz_spec, sizeof(sz_spec));
        // a checkpoint is at most one sequence state; refuse absurd sizes
        ok = ok && sz_tgt < (1ull << 36) && sz_dft < (1ull << 36) && sz_spec < (1ull << 30);
        if (!ok) {
            break;
        }
        auto & c = out.emplace_back();
        c.update_pos(n_tokens, pos_min, pos_max);
        c.id_task = -1;
        c.data_tgt.resize(sz_tgt);
        c.data_dft.resize(sz_dft);
        c.data_spec.resize(sz_spec);
        ok = get(c.data_tgt.data(), sz_tgt) && get(c.data_dft.data(), sz_dft) && get(c.data_spec.data(), sz_spec);
        if (ok) {
            on_checkpoint();
        }
    }
    return ok;
}
