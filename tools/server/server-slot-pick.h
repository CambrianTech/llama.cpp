#pragma once

// Continuum fork: which slot a task runs on, as one pure decision over a typed view of every slot
// (Codex on #105: a reconfigure that parks a mind leaves an empty placeholder at its id, and the
// similarity and LRU choices handed that placeholder to another request, so the parked mind could not
// come back). A parked id is never chosen: a task naming it waits for its mind, an anonymous task never
// lands on it. The server context builds the view; tests/test-server-slot-pick.cpp proves the decision
// without a model.

#include <cstddef>
#include <cstdint>

struct server_slot_view {
    int32_t id          = -1;
    bool    processing  = false;
    bool    parked      = false; // a reconfigure parked this id's mind; the slot here is a placeholder
    int64_t t_last_used = -1;
    size_t  n_tokens    = 0;     // tokens the slot holds
    size_t  lcp         = 0;     // tokens it shares with the task's prompt
};

enum class server_slot_pick_kind : uint8_t {
    by_id,      // the task named this slot
    by_prefix,  // the free slot sharing the most of the task's prompt above the similarity floor
    by_lru,     // the least recently used free slot
    wait,       // nothing to run on now (busy, or the named id is parked): the task is deferred
    no_slot,    // the named id is neither live nor parked: the request is invalid
};

struct server_slot_pick {
    server_slot_pick_kind kind  = server_slot_pick_kind::wait;
    size_t                index = 0;     // into the view, for by_id / by_prefix / by_lru
    float                 f_sim = 0.0f;  // by_prefix: the shared fraction of the task's prompt
};

// `id_slot`: the slot the task names, or -1. `n_task_tokens`: the task's prompt length. `similarity`: the
// floor a prefix match must exceed (0 disables prefix choice).
inline server_slot_pick server_pick_slot(const server_slot_view * slots, size_t n_slots, int32_t id_slot, size_t n_task_tokens,
                                         float similarity) {
    if (id_slot != -1) {
        for (size_t i = 0; i < n_slots; ++i) {
            if (slots[i].id != id_slot) {
                continue;
            }
            if (slots[i].parked) {
                return { server_slot_pick_kind::wait, i, 0.0f }; // its mind comes back to it
            }
            return { server_slot_pick_kind::by_id, i, 0.0f }; // busy or not: the caller defers a busy one
        }
        return { server_slot_pick_kind::no_slot, 0, 0.0f };
    }

    const auto free = [](const server_slot_view & s) { return !s.processing && !s.parked; };

    if (similarity != 0.0f && n_task_tokens > 0) {
        float  best = 0.0f;
        size_t at   = n_slots;
        for (size_t i = 0; i < n_slots; ++i) {
            if (!free(slots[i]) || slots[i].n_tokens == 0) {
                continue;
            }
            const float f = float(slots[i].lcp) / float(n_task_tokens);
            if (f > best && f > similarity) {
                best = f;
                at   = i;
            }
        }
        if (at < n_slots) {
            return { server_slot_pick_kind::by_prefix, at, best };
        }
    }

    size_t  at     = n_slots;
    int64_t t_last = -1;
    for (size_t i = 0; i < n_slots; ++i) {
        if (!free(slots[i])) {
            continue;
        }
        if (at == n_slots || slots[i].t_last_used <= t_last) {
            t_last = slots[i].t_last_used;
            at     = i;
        }
    }
    if (at < n_slots) {
        return { server_slot_pick_kind::by_lru, at, 0.0f };
    }
    return { server_slot_pick_kind::wait, 0, 0.0f };
}
