#include "server-checkpoints.h"

#include <iterator>

size_t server_checkpoints_bytes(const server_checkpoint_list & list) {
    size_t n = 0;
    for (const auto & c : list) {
        n += c.size();
    }
    return n;
}

bool server_checkpoints_fit(
        const std::vector<server_checkpoint_list *> & lists,
        size_t own,
        uint64_t budget,
        size_t incoming,
        const std::function<void(size_t, const common_prompt_checkpoint &)> & on_evict) {
    if (incoming > budget) {
        return false;
    }
    for (;;) {
        uint64_t total = incoming;
        size_t   victim = own;
        size_t   victim_bytes = 0;
        for (size_t i = 0; i < lists.size(); ++i) {
            const size_t n = server_checkpoints_bytes(*lists[i]);
            total += n;
            if (n > victim_bytes || (n == victim_bytes && i == own)) {
                victim = i;
                victim_bytes = n;
            }
        }
        if (total <= budget || victim_bytes == 0) {
            return true;
        }
        auto & list = *lists[victim];
        auto it = list.size() >= 2 ? std::next(list.begin()) : list.begin();
        on_evict(victim, *it);
        list.erase(it);
    }
}
