#pragma once

// Continuum fork (CONTINUUM-ENGINE 4.12.1, 4.11.10): the engine's ONE memory allocator. Every change to what the KV
// pool holds goes through it: a frame's placement (decode), a sequence's removal, shift and copy, and a sequence's
// state saved out and loaded back (park, flash, page restore, preemption). The host bytes checkpoints and the prompt
// cache hold join it, with a typed budget, when those charges migrate (no verb before it has a caller). Owners ask
// this interface, so placement and every move share one custody (Codex's singular-allocator condition).
//
// Migrated: every slot edit (server_slot::mem) and the resident minds (move, flash-out save, removal, insert).
// Still calling llama directly, each moving with the owner that redesigns it (CONTINUUM-ENGINE 4.12.1):
//   frame decode (server-context-decode.inc)                      -> frame_batcher, with 4.11.10 admission
//   learning side sequences, partial removal + checkpoint restore  -> learner, checkpoint_store
//     (server-context-learn.inc; server-context-prefill.inc draft trim)
//   page restore of the draft state (server-context-page.inc)      -> tier_store
//   prompt cache save/load (server_slot::prompt_save, server-task)  -> the typed host budget
// The list shrinks to empty; a new direct call is a defect.
//
// Outcomes are typed and exhaustive; a caller matches every one. Nothing here retries, halves, purges or falls back:
// a piece that does not fit returns no_room with nothing placed, and making room is the caller's decision
// (mind_table's coldest legal resident, or the slot is deferred).
//
// Adapters: memory_pool_llama (a target llama_context and an optional draft context sharing sequence ids).

#include "llama.h"

#include <cstddef>
#include <cstdint>

// The result of asking the pool to place one piece of a frame (one llama_decode call).
enum class placement : uint8_t {
    placed,        // the piece's tokens hold cells and were computed
    no_room,       // nothing of this piece was placed; earlier pieces of the frame stand
    aborted,       // the decode's abort callback stopped it; the pool's state for this piece is uncertain
    invalid,       // the piece was malformed (a caller defect)
    compute_error, // the backend failed while computing; the pool's state for this piece is uncertain
};

// Which of the pool's contexts an edit applies to. A slot's sequence lives in both; a learning side sequence lives in
// the target only; repairing a draft state touches the draft only. Always named, never defaulted.
enum class kv_contexts : uint8_t {
    both,
    target,
    draft,
};

// The result of a sequence edit. A partial removal can be refused (recurrent memory cannot drop a tail it did not
// checkpoint); the caller decides what a refusal means, so edits are [[nodiscard]].
enum class edit_outcome : uint8_t {
    done,
    refused, // nothing changed in the context that refused; a context listed before it in `both` may have changed
};

// The result of loading a saved sequence state back into the pool.
// llama reports a failed load as one result (no room for it, or bytes that are not this pool's state); the pool does
// not guess which.
enum class load_outcome : uint8_t {
    loaded,  // the sequence holds exactly the saved state, in both contexts
    refused, // nothing was loaded; the saved bytes are untouched and remain the caller's custody
};

// The result of saving a sequence's state out of the pool.
enum class save_outcome : uint8_t {
    saved,  // both contexts' state written in full into the caller's buffers
    failed, // a context wrote short; the buffers hold nothing usable and the pool is unchanged
};

// A sequence's state size in each context: the caller sizes its own buffers (RAM, an mmap'd page) from this, so a
// save writes straight into where the state will live.
struct seq_state_size {
    size_t tgt = 0;
    size_t dft = 0; // 0 when the pool has no draft context
    size_t total() const { return tgt + dft; }
};

// Positions a sequence holds in the target context; empty when it holds none.
struct seq_range {
    llama_pos min   = -1;
    llama_pos max   = -1;
    bool      empty = true;
};

class memory_pool {
public:
    virtual ~memory_pool() = default;

    // placement
    virtual placement place(const llama_batch & piece) = 0;

    // sequence edits, in the named contexts (the draft only when the pool has one)
    [[nodiscard]] virtual edit_outcome remove(llama_seq_id seq, llama_pos p0, llama_pos p1, kv_contexts in) = 0;
    [[nodiscard]] virtual edit_outcome shift(llama_seq_id seq, llama_pos p0, llama_pos p1, llama_pos delta, kv_contexts in) = 0;
    [[nodiscard]] virtual edit_outcome copy(llama_seq_id src, llama_seq_id dst, llama_pos p0, llama_pos p1, kv_contexts in) = 0;
    // a whole sequence moved to another id in both contexts (dst cleared first, src cleared after)
    [[nodiscard]] virtual edit_outcome move(llama_seq_id src, llama_seq_id dst) = 0;
    virtual seq_range range(llama_seq_id seq) const = 0;

    // a sequence's whole state, target and draft together, moved out of and back into the pool (park, flash, page
    // restore, preemption); buffers are the caller's, sized from state_size
    virtual seq_state_size state_size(llama_seq_id seq, llama_state_seq_flags flags) const = 0;
    virtual save_outcome   save(llama_seq_id seq, llama_state_seq_flags flags, uint8_t * tgt, uint8_t * dft) const = 0;
    virtual load_outcome   load(llama_seq_id seq, llama_state_seq_flags flags, const uint8_t * tgt, size_t n_tgt,
                                const uint8_t * dft, size_t n_dft) = 0;
};
