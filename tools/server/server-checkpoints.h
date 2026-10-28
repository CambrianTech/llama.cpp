#pragma once

// Continuum fork: context checkpoints kept within one byte budget across all slots
// (--ctx-checkpoints-bytes), in place of a per-slot count. On a hybrid 27B a checkpoint is
// ~420 MB of host RAM, which on unified memory is the GPU's RAM, so the count is the wrong unit.

#include "common.h"

#include <cstdint>
#include <functional>
#include <list>
#include <vector>

using server_checkpoint_list = std::list<common_prompt_checkpoint>;

size_t server_checkpoints_bytes(const server_checkpoint_list & list);

// Makes room for a new checkpoint of `incoming` bytes. While the lists' total plus `incoming`
// exceeds `budget`, the list holding the most bytes (the creator's on a tie) drops its oldest
// checkpoint after its first, so each slot keeps its head (the start of its first user turn) and
// its newest longest, as a saved page's subset does. `on_evict(list index, checkpoint)` runs before
// each drop. Returns false, dropping nothing, when the new checkpoint alone exceeds the budget.
bool server_checkpoints_fit(
        const std::vector<server_checkpoint_list *> & lists,
        size_t own,
        uint64_t budget,
        size_t incoming,
        const std::function<void(size_t, const common_prompt_checkpoint &)> & on_evict);
