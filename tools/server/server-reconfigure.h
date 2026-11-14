#pragma once

// Continuum fork: RECONFIGURE, GROW OR SHRINK LIVE (SERVING-ENGINE.md, "S2 in detail"). The serving
// contexts take a new geometry (context size, sequence count) at a safe point between batched decode
// steps while the weights and their mmap stay loaded: every slot holding state is paged out through
// the one state mover (server-slot-page.h), unfinished generations included, the target and draft
// contexts are rebuilt, and every state that fits is paged back in, so each generation continues from
// its next token with its host-side sampler. Nothing is cancelled and nothing waits for a generation to
// end; new work waits in the queue. Grow and shrink are the same path.
//
// Shrink never truncates: a slot whose state the target cannot hold (an id past a smaller n_seq, or
// more tokens than a sequence of the target holds) is paged out whole and PARKED, named in the outcome;
// a later reconfigure places it back when the geometry holds it and its id is free. A KV cache TYPE
// change is not a reconfigure: it is the intentional cold start (#5188), and is refused by name.
//
// The logic lives here, against server_reconfigure_engine; the server context implements the engine
// (its context lifetimes and slots) and calls server_reconfigure from SERVER_TASK_TYPE_RECONFIGURE.

#include "server-geometry.h"
#include "server-slot-page.h"

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <variant>
#include <vector>

// a slot whose state the applied geometry cannot hold: paged out whole and parked, never cut
struct server_slot_refusal {
    int32_t slot        = -1;
    int32_t tokens_held = 0; // tokens its state holds
    int32_t tokens_fit  = 0; // tokens the applied geometry had room for in its sequence
};

struct server_reconfigure_applied {
    server_geometry live;                       // the geometry the contexts hold now
    int32_t n_ctx_seq = 0;                      // tokens one sequence of it holds
    std::vector<server_slot_refusal> paged_out; // slots this reconfigure parked
    std::vector<int32_t> paged_in;              // parked slots it placed back
    std::vector<int32_t> parked;                // every slot parked after it
    // the cost: how long serving paused, against the state that crossed the rebuild
    double   pause_ms    = 0.0;
    int64_t  tokens_held = 0;
    uint64_t bytes_paged = 0;
    bool     rebuilt     = false; // false: the geometry was already live, only parked slots moved
};

enum server_reconfigure_refusal_reason {
    SERVER_RECONFIGURE_INVALID_GEOMETRY,     // n_seq below 1, or fewer cells than sequences
    SERVER_RECONFIGURE_CACHE_TYPE_CHANGE,    // a KV type change is the intentional cold start, never a reconfigure
    SERVER_RECONFIGURE_LEARN_IN_PROGRESS,    // a learn's graph and accumulators live in the context until its restoration ends: retry after
    SERVER_RECONFIGURE_LEARNING_BOUND,       // the context holds shadow optimizers or the training workspace: they do not cross a rebuild
    SERVER_RECONFIGURE_FAN_OUT,              // a parent prompt is being copied into its child slots
    SERVER_RECONFIGURE_PAGE_OUT_FAILED,      // a slot's state could not be read; nothing changed
    SERVER_RECONFIGURE_CONTEXT_BUILD_FAILED, // the target contexts could not be created; the previous geometry was rebuilt
};

const char * server_reconfigure_refusal_name(server_reconfigure_refusal_reason reason);

struct server_reconfigure_refused {
    server_reconfigure_refusal_reason reason;
    std::string detail;
    int32_t slot = -1; // the slot it concerns, when one does
};

using server_reconfigure_outcome = std::variant<server_reconfigure_applied, server_reconfigure_refused>;

// What reconfigure needs of the server. Slots are addressed by id, 0 .. live().n_seq - 1.
struct server_reconfigure_engine {
    virtual ~server_reconfigure_engine() = default;

    virtual server_geometry live() const = 0;
    virtual int32_t n_ctx_seq()  const = 0; // tokens one sequence of the live contexts holds
    virtual int32_t n_cells()    const = 0; // cells the live target context allocated (padded, so >= n_ctx asked)
    virtual bool    kv_unified() const = 0; // every sequence draws on that one pool of cells

    // why the contexts cannot be rebuilt right now; nullopt when they can
    virtual std::optional<server_reconfigure_refused> blocked() const = 0;

    virtual int32_t tokens_held(int32_t slot) const = 0; // tokens whose state its KV holds (0: none)
    virtual bool    busy       (int32_t slot) const = 0; // it has a task, finished or not

    virtual bool page_out(int32_t slot, server_slot_page & page) = 0;
    virtual bool page_in (int32_t slot, const server_slot_page & page) = 0;

    // park: the slot's host state (prompt, checkpoints, task, sampler) leaves service and a fresh,
    // empty slot takes its id; unpark: the parked host state takes its id back
    virtual void park  (int32_t slot) = 0;
    virtual void unpark(int32_t slot) = 0;

    // the contexts' lifetime apart from the weights': destroy frees the target and draft contexts and
    // the speculative state; build creates them for `g` and sizes the slots to g.n_seq (the highest
    // ids go, and a new id comes in empty)
    virtual void destroy_contexts() = 0;
    virtual bool build_contexts(const server_geometry & g) = 0;

    // after the states are in: samplers re-attached, drafting restarted, queued work offered the free slots
    virtual void resume() = 0;
};

// a parked slot's page, with what its state held when it left
struct server_parked_slot {
    server_slot_page page;
    int32_t tokens_held = 0;
    bool    busy        = false;
};

// what reconfigure keeps between calls: the parked pages and the counters /props reports
struct server_reconfigure_state {
    std::map<int32_t, server_parked_slot> parked;
    int64_t n_applied = 0;
    int64_t n_refused = 0;
    std::optional<server_reconfigure_outcome> last;

    // THE COST, measured over every reconfigure that rebuilt the contexts, so the core can price the next
    // one: how long serving paused, against the state that crossed (server_reconfigure_cost_json)
    int64_t  n_rebuilt         = 0;
    double   pause_ms_total    = 0.0;
    double   pause_ms_max      = 0.0;
    int64_t  tokens_held_total = 0;
    uint64_t bytes_paged_total = 0;
};

// the measured cost as /props reports it (the server adds the buffers the contexts hold now)
json server_reconfigure_cost_json(const server_reconfigure_state & state);

server_reconfigure_outcome server_reconfigure(server_reconfigure_engine & engine, server_reconfigure_state & state, const server_geometry & target);

// THE HTTP EDGE (POST /reconfigure): {"n_ctx": int, "n_seq": int} and, optionally, "cache_type_k" and
// "cache_type_v", which must name the live types. Either the geometry or the refusal.
std::variant<server_geometry, server_reconfigure_refused> server_reconfigure_from_json(
        const json & body, const char * cache_type_k, const char * cache_type_v);

json server_reconfigure_to_json(const server_reconfigure_outcome & outcome);

struct server_task_result_reconfigure : server_task_result {
    server_reconfigure_outcome outcome;

    virtual json to_json() override { return server_reconfigure_to_json(outcome); }
    virtual bool is_error() override { return std::holds_alternative<server_reconfigure_refused>(outcome); }
};
