#pragma once

// Continuum fork (CONTINUUM-ENGINE 4.14): a persona's execution state, out of the memory pool, as native blocks.
// Moving a persona between the pool and host memory is raw row copies plus her small records; nothing is walked,
// serialized or recomputed. Every move is two-phase: check (decides every condition, reserves the destination and every
// byte and node the commit needs; refuses with a typed outcome) and commit (copies, swaps and reserved-node inserts:
// nothing allocates, so it cannot fail), so a refusal never leaves her split.

#include "llama.h"
#include "llama-kv-cells.h"

#include <cstdint>
#include <set>
#include <vector>

// her attention state: per KV layer, the rows she uses of one stream, and her cell metadata OWNED here (exchanged with
// the stream's by swap; the stream receives an already-empty object, so a move never resets or walks cells)
struct llama_kv_stream_blocks {
    uint32_t                          n_rows = 0;
    std::vector<std::vector<uint8_t>> k;     // per KV layer: n_rows rows of K
    std::vector<std::vector<uint8_t>> v;     // per KV layer: n_rows rows of V (empty for a layer without V)
    llama_kv_cells                    cells; // hers while she is out; an empty, stream-sized object while she is in
    // a pool's save only: her cells in the pool (row j of her blocks is pool cell rows[j]) and her metadata compacted to
    // rows 0..n_rows, both built by the check so the commit only copies rows and swaps `staged` with `cells`
    std::vector<uint32_t>             rows;
    llama_kv_cells                    staged;
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

// the byte geometry of this memory's rows, recorded once by persona_prepare from the memory's own tensors: a read of her
// file checks every length against it BEFORE allocating, so a truncated or foreign file is refused without sizing any
// buffer from what the file claims (Codex on CambrianTech/llama.cpp#155)
struct llama_persona_geometry {
    std::vector<size_t> k_row;   // per KV layer: bytes of one K row
    std::vector<size_t> v_row;   // per KV layer: bytes of one V row (0 for a layer without V)
    std::vector<size_t> r_bytes; // per recurrent layer: bytes of her r rows in every snapshot group (0 without r)
    std::vector<size_t> s_bytes; // per recurrent layer: bytes of her s rows in every snapshot group (0 without s)
};

// her identity travels with her blocks: the sequence her metadata is tagged with, and the layout of the memory they
// were prepared for (a fingerprint of layer count, stream size, row types and sizes, snapshot groups)
// Who a move is for: the persona and epoch whose token history the blocks belong to.
struct llama_persona_identity {
    uint64_t persona = 0;
    uint64_t epoch   = 0;
};

struct llama_persona_blocks {
    llama_seq_id           seq       = -1;
    // whose blocks these are: the owner of her token history when they were saved (persona + epoch), stamped by a
    // completed save and checked by every load, native or public, before anything is written; never retagged
    bool                   owned       = false;
    uint64_t               owner       = 0;
    uint64_t               owner_epoch = 0;

    bool belongs_to(const llama_persona_identity & who) const {
        return !owned || (owner == who.persona && owner_epoch == who.epoch);
    }
    uint64_t               kv_layout = 0;
    // the attention layout without the stream size (layers, row types and widths): a file restores her into any stream of
    // this memory that holds her rows, whatever window a restart's planner chose (CONTINUUM-ENGINE 4.16)
    uint64_t               kv_rows   = 0;
    uint64_t               rs_layout = 0;
    llama_persona_geometry geometry;
    llama_kv_stream_blocks kv;
    llama_rs_seq_blocks    rs;
};

// where a load puts her, reserved by the check phase: the attention stream, the recurrent cell, and the cell's
// sequence-set node (built in the check phase so the commit inserts it without allocating)
struct llama_persona_slot {
    uint32_t                          stream = 0;
    int32_t                           cell   = -1;
    std::set<llama_seq_id>::node_type node;
    // a pool's load only: her occupied rows `src` land in the pool's empty cells `dst`, and `index` is her (position,
    // cell) index over `dst`, sorted
    std::vector<uint32_t>             src;
    std::vector<uint32_t>             dst;
    llama_kv_seq_pos                  index;
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
    wrong_persona,      // her blocks belong to another sequence (her metadata is tagged with it; never retagged)
    invalid_rollback,   // her rollback index exceeds this memory's snapshot groups
    wrong_owner,        // her blocks were saved by another persona or another epoch of her
    out_of_memory,      // growing or shrinking her token history failed to allocate; nothing changed
    no_empty_stream,    // a bind found no empty stream to place her in
    no_checkpoint,      // a rewind was asked of a checkpoint never taken
    checkpoint_ahead,   // a rewind was asked to a position past what she holds: a rewind never moves forward
    checkpoint_stale,   // her cells are not the lineage the checkpoint was taken from, or were cut at or below it since
    io_error,           // her blocks' file could not be written or read whole
    file_mismatch,      // her blocks' file is not hers or not this memory's (owner, epoch, layout, sizes): nothing was read
    pool_full,          // the pool has fewer empty cells than her rows: page another mind out first
    pending_shift,      // the pool holds a position shift not yet applied to its rows: her rows cannot move as they are
};

// A copy of a bound persona at one of her positions, taken WITHOUT moving her: her recurrent rows (every snapshot
// group), her recurrent position and rollback index. Her attention KV is not copied: it stays in place, and a rewind
// cuts it back past the checkpoint's position. So a checkpoint costs her recurrent rows, never her KV (CONTINUUM-ENGINE
// 4.15: a request whose prefix matches her up to a stable boundary prefills only what follows it).
struct llama_persona_checkpoint_blocks {
    bool                taken     = false;
    llama_seq_id        seq       = -1;
    llama_pos           pos       = -1; // her last position held when taken (attention and recurrent agree)
    // whose it is (the persona and epoch it was taken for) and the lineage of the cells it was taken from: a rewind is
    // refused for another owner, or for cells that are not that lineage or were cut at or below `pos` since (a slot
    // reused by another mind, a restart, an eviction), so her rows are never spliced onto someone else's KV
    llama_persona_identity owner;
    uint64_t            lineage   = 0;
    llama_kv_cells::checkpoint_position mark;
    uint64_t            rs_layout = 0;
    llama_rs_seq_blocks rs;
};

const char * llama_persona_move_name(llama_persona_move m);
