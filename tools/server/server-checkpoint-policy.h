#pragma once

// Continuum fork (CONTINUUM-ENGINE 4.7.3, placed checkpoints): which of a slot's context checkpoints go,
// as pure rules over the slot's list (oldest first), so a native test proves them without a model.
// A checkpoint here needs `n_tokens` (its position) and `id_task` (the act that made it).

#include <cstddef>
#include <cstdint>

// The checkpoint to erase to make room for one more. With view spans (view_end >= 0) her head (the
// oldest) is her restore point when her next prompt diverges before her view end, so the oldest AFTER
// it goes first; the head goes only when it is all there is. Without view spans, the oldest goes.
template <typename List>
typename List::iterator checkpoint_to_evict(List & checkpoints, int32_t view_end) {
    auto victim = checkpoints.begin();
    if (view_end >= 0 && checkpoints.size() >= 2) {
        ++victim;
    }
    return victim;
}

// After her new view-end checkpoint `kept` (made by act `id_task`) is stored, the checkpoints an earlier
// act left on this prefix are superseded, except her head's: what stays is her head, her newest view
// end and this act's end checkpoints. Returns how many were erased; only a checkpoint AT her view end
// supersedes anything.
template <typename List, typename Checkpoint, typename OnErase>
size_t erase_superseded_checkpoints(List & checkpoints, const Checkpoint & kept, int32_t view_end, int id_task, OnErase on_erase) {
    if (view_end < 0 || kept.n_tokens != view_end || checkpoints.empty()) {
        return 0;
    }
    const auto * head = &checkpoints.front();
    size_t erased = 0;
    for (auto it = checkpoints.begin(); it != checkpoints.end();) {
        if (&*it != head && it->id_task != id_task) {
            on_erase(*it);
            it = checkpoints.erase(it);
            ++erased;
            continue;
        }
        ++it;
    }
    return erased;
}
