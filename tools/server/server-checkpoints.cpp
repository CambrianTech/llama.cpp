#include "server-checkpoints.h"

#include "server-common.h"

#include <iterator>

size_t server_checkpoints_bytes(const server_checkpoint_list & list) {
    size_t n = 0;
    for (const auto & c : list) {
        n += c.size();
    }
    return n;
}

size_t server_checkpoint_victim(const server_checkpoint_list & list, int32_t raw_lcp) {
    if (list.size() < 2) {
        return 0;
    }
    std::vector<int64_t> n_tokens;
    n_tokens.reserve(list.size());
    for (const auto & c : list) {
        n_tokens.push_back(c.n_tokens);
    }
    const auto kept = select_page_checkpoints(n_tokens, (int32_t) list.size() - 2, raw_lcp);
    size_t k = 0;
    for (size_t i = 0; i < list.size(); ++i) {
        if (k < kept.size() && kept[k] == i) {
            ++k;
        } else {
            return i;
        }
    }
    return list.size() - 1;
}

bool server_checkpoints_fit(
        const std::vector<server_checkpoint_list *> & lists,
        const std::vector<int32_t> & raw_lcps,
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
        const int32_t raw_lcp = victim < raw_lcps.size() ? raw_lcps[victim] : -1;
        auto it = std::next(list.begin(), (std::ptrdiff_t) server_checkpoint_victim(list, raw_lcp));
        on_evict(victim, *it);
        list.erase(it);
    }
}
