#pragma once

// Continuum fork (CONTINUUM-ENGINE 4.14): a persona's execution state, out of the memory pool, as native blocks.
// Moving a persona between the pool and host memory is raw row copies plus her small records; nothing is walked,
// serialized or recomputed. Every move is two-phase: validate (const, reserves the destination, refuses with a typed
// outcome) and commit (copies and record writes only, which cannot fail), so a refusal never leaves her split.

#include "llama.h"
#include "llama-kv-cells.h"

#include <cstdint>
#include <vector>

// her attention state: per KV layer, the rows she uses of one stream, and her cell metadata OWNED here (exchanged with
// the stream's by swap; the stream receives an already-empty object, so a move never resets or walks cells)
struct llama_kv_stream_blocks {
    uint32_t                          n_rows = 0;
    std::vector<std::vector<uint8_t>> k;     // per KV layer: n_rows rows of K
    std::vector<std::vector<uint8_t>> v;     // per KV layer: n_rows rows of V (empty for a layer without V)
    llama_kv_cells                    cells; // hers while she is out; an empty, stream-sized object while she is in
};

// her recurrent state: per layer, her row in every snapshot group (live group 0, then the rollback groups), and her
// cell's record
struct llama_rs_seq_blocks {
    bool                              has_state = false;
    llama_pos                         pos       = -1;
    uint32_t                          rs_idx    = 0;
    std::vector<std::vector<uint8_t>> r; // per layer: (1 + n_rs_seq) rows of r, group-major
    std::vector<std::vector<uint8_t>> s; // per layer: (1 + n_rs_seq) rows of s, group-major
};

struct llama_persona_blocks {
    llama_kv_stream_blocks kv;
    llama_rs_seq_blocks    rs;
};

// where a load puts her: the attention stream, and the recurrent cell reserved by validation
struct llama_persona_slot {
    uint32_t stream = 0;
    int32_t  cell   = -1;
};

enum class llama_persona_move {
    done,
    unsupported,        // this memory has no persona move
    no_such_stream,     // stream out of range, or the cache is unified (one stream holds every sequence)
    transposed_v,       // V is stored transposed (flash attention off): her V rows are not contiguous
    stream_not_empty,   // a load requires an empty stream
    cells_not_prepared, // her blocks do not hold an empty, stream-sized cells object (persona_prepare once)
    layout_mismatch,    // her blocks do not match this memory's layers, row sizes or groups
    invalid_seq,        // sequence id out of range
    seq_holds_state,    // a load requires her sequence to hold no state
    shared_cell,        // her recurrent cell is shared with another sequence: not hers alone to move
    no_free_cell,       // no free recurrent cell to load into
};

const char * llama_persona_move_name(llama_persona_move m);
