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

// The checkpoint a list gives up first: the one select_page_checkpoints leaves out when it keeps
// all but one, so a list keeps longest exactly what a saved page keeps (its head, the newest at or
// below the slot's last raw common prefix `raw_lcp`, then the newest), never its oldest first.
size_t server_checkpoint_victim(const server_checkpoint_list & list, int32_t raw_lcp);

// Makes room for a new checkpoint of `incoming` bytes. While the lists' total plus `incoming`
// exceeds `budget`, the list holding the most bytes (the creator's on a tie) drops its
// server_checkpoint_victim under its slot's `raw_lcps` entry. `on_evict(list index, checkpoint)`
// runs before each drop. Returns false, dropping nothing, when the new checkpoint alone exceeds
// the budget.
bool server_checkpoints_fit(
        const std::vector<server_checkpoint_list *> & lists,
        const std::vector<int32_t> & raw_lcps,
        size_t own,
        uint64_t budget,
        size_t incoming,
        const std::function<void(size_t, const common_prompt_checkpoint &)> & on_evict);
