#pragma once

#include <cstddef>
#include <vector>

// Rotate traversal, never the slots: slot ids and their live KV stay fixed.
class server_prefill_order {
    size_t next = 0;

public:
    std::vector<size_t> order(size_t count, bool kv_unified = false) const {
        std::vector<size_t> result;
        result.reserve(count);
        if (count == 0) {
            return result;
        }
        // Shared KV has no per-prompt reservation; keep its existing admission order.
        const size_t start = kv_unified ? 0 : next % count;
        for (size_t offset = 0; offset < count; ++offset) {
            result.push_back((start + offset) % count);
        }
        return result;
    }

    // Only prompt tokens actually added to a batch spend a service opportunity.
    // The last contributor yields first place to its successor on the next batch.
    void served(size_t index, size_t before, size_t after) {
        if (after > before) {
            next = index + 1;
        }
    }
};
