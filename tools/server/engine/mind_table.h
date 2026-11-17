#pragma once

// Continuum fork (CONTINUUM-ENGINE 4.12.1, 4.11.9): where each resident mind lives (a lane, parked in the pool, or on
// flash) and the policy for moving her. Every move it makes is memory_pool's (the one allocator); this class decides
// which mind and where, never how bytes move. Today's implementation is the #139 resident-minds code
// (mind_table_resident); a mind is never dropped or truncated, only moved between homes.
//
// Outcomes are typed: a caller matches each. Nothing here retries in a loop on its own or purges a mind.

#include "memory_pool.h"

#include "../server-mind.h"

#include <cstdint>
#include <optional>
#include <string>

struct server_slot;

// What the table is built from: startup facts, never tuned at runtime.
struct mind_table_config {
    int32_t     n_seq      = 0;     // sequence ids the pool has (lanes + parked)
    int32_t     n_lanes    = 0;     // slots that decode
    int32_t     n_minds    = 0;     // resident minds the core asked for; 0 = the table is off
    bool        kv_unified = false; // parking needs the unified pool
    std::string page_dir;           // where flash pages live
};

class mind_table {
public:
    virtual ~mind_table() = default;

    // whether resident minds are on for this engine (n_minds > 0, unified pool, a page directory)
    virtual bool enabled() const = 0;

    // where `page` is now, starting her journey toward the pool if she is on flash (the core asks ahead of her turn)
    virtual mind_place prefetch(mind_page page) = 0;

    // put `page` on `slot` for her turn: from the slot itself, a parked sequence, or flash. false: she could not be
    // seated (the pool cannot hold her even after make_room); she stays where she was, never lost
    [[nodiscard]] virtual bool seat(server_slot & slot, mind_page page) = 0;

    // free cells for a placement that found none: the coldest parked mind goes to flash, else an idle lane's mind
    // does. false: no legal move exists (every mind is on a busy lane or pinned)
    [[nodiscard]] virtual bool make_room() = 0;

    // release a learn pin wherever her checkpoints are (a lane, parked, or on flash)
    virtual pin_release release_pin(mind_page page, uint64_t pin) = 0;

    // landed flash writes free their RAM copies; failed ones are written again
    virtual void reap() = 0;

    // parked sequence ids lent to a learning side sequence and given back
    [[nodiscard]] virtual std::optional<llama_seq_id> lend_id() = 0;
    virtual void                                      return_id(llama_seq_id id) = 0;
    virtual size_t                                    free_ids() const = 0;
};
