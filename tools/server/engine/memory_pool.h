#pragma once

// Continuum fork (CONTINUUM-ENGINE 4.12.1, 4.11.10): the engine's ONE memory allocator. Every change to what the KV
// pool holds goes through it: a frame's placement (decode), a sequence's removal, shift and copy (target and draft
// together), a sequence's state saved out and loaded back (park, flash, page restore, preemption), and the host
// bytes that checkpoints and the prompt cache hold. Nothing else in the engine calls llama_decode, llama_memory_seq_*
// or llama_state_seq_* on the serving contexts; owners ask this interface, so placement and every move share one
// custody (Codex's singular-allocator condition).
//
// Outcomes are typed and exhaustive; a caller matches every one. Nothing here retries, halves, purges or falls back:
// a piece that does not fit returns no_room with nothing placed, and making room is the caller's decision
// (mind_table's coldest legal resident, or the slot is deferred).
//
// Adapters: memory_pool_llama (a target llama_context and an optional draft context sharing sequence ids).

#include "llama.h"

#include <cstddef>
#include <cstdint>
#include <vector>

// The result of asking the pool to place one piece of a frame (one llama_decode call).
enum class placement : uint8_t {
    placed,        // the piece's tokens hold cells and were computed
    no_room,       // nothing of this piece was placed; earlier pieces of the frame stand
    invalid,       // the piece was malformed (a caller defect)
    compute_error, // the backend failed while computing; the pool's state for this piece is uncertain
};

// The result of loading a saved sequence state back into the pool.
enum class load_outcome : uint8_t {
    loaded,  // the sequence holds exactly the saved state
    no_room, // nothing was loaded; the saved bytes are untouched and remain the sequence's custody
    corrupt, // the bytes do not decode as this pool's sequence state
};

// The result of charging host bytes (checkpoints, prompt cache) to the pool's host budget.
enum class host_charge : uint8_t {
    granted,
    denied, // over budget; nothing was charged
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

    // sequence edits, applied to the target and (when present) the draft together
    virtual void      remove(llama_seq_id seq, llama_pos p0, llama_pos p1)                   = 0;
    virtual void      shift(llama_seq_id seq, llama_pos p0, llama_pos p1, llama_pos delta)   = 0;
    virtual void      copy(llama_seq_id src, llama_seq_id dst, llama_pos p0, llama_pos p1)    = 0;
    virtual seq_range range(llama_seq_id seq) const                                           = 0;

    // a sequence's whole state moved out of and back into the pool (park, flash, page restore, preemption)
    virtual size_t       state_size(llama_seq_id seq, llama_state_seq_flags flags) const                           = 0;
    virtual size_t       save(llama_seq_id seq, llama_state_seq_flags flags, std::vector<uint8_t> & out) const    = 0;
    virtual load_outcome load(llama_seq_id seq, llama_state_seq_flags flags, const uint8_t * bytes, size_t n)     = 0;

    // host bytes held for sequences outside the pool (checkpoints, prompt cache)
    virtual host_charge charge(size_t bytes)  = 0;
    virtual void        release(size_t bytes) = 0;
    virtual size_t      host_held() const     = 0;
};
