#include "../tools/server/server-prefill.h"

#include <algorithm>
#include <cstdlib>
#include <iostream>

static void require(bool condition, const char * message) {
    if (!condition) {
        std::cerr << message << '\n';
        std::exit(1);
    }
}

// Model a bounded text batch using the production traversal and service accounting.
static size_t batch(server_prefill_order & order, std::vector<size_t> & remaining,
                    const std::vector<bool> & compatible, size_t capacity) {
    size_t used = 0;
    for (const size_t index : order.order(remaining.size())) {
        if (!compatible[index] || used == capacity) {
            continue;
        }
        const size_t before = used;
        const size_t take = std::min(remaining[index], capacity - used);
        remaining[index] -= take;
        used += take;
        order.served(index, before, used);
    }
    return used;
}

int main() {
    server_prefill_order order;
    require(order.order(0).empty(), "empty pool");
    std::vector<size_t> remaining = {100000, 8, 100000};
    const std::vector<bool> compatible = {true, true, true};
    require(batch(order, remaining, compatible, 32) == 32, "cold fills first batch");
    require(remaining[1] == 8, "warm initially waits");
    require(batch(order, remaining, compatible, 32) == 32, "warm and next cold share batch");
    require(remaining[1] == 0 && remaining[2] == 99976, "warm served before cold finishes");
    require(order.order(3).front() == 0, "last contributor determines successor and wraps");

    // A new low-index request must not regain first place on every iteration.
    for (int round = 0; round < 20; ++round) {
        remaining = {32, 32, 32};
        batch(order, remaining, compatible, 32);
        remaining[0] = 32;
        batch(order, remaining, compatible, 32);
        remaining[0] = 32;
        batch(order, remaining, compatible, 32);
        require(remaining[1] == 0 && remaining[2] == 0, "reused low slot starved peers");
    }

    const auto before = order.order(3);
    remaining = {0, 0, 0};
    require(batch(order, remaining, compatible, 32) == 0, "released/empty slots skipped");
    order.served(1, 7, 7);
    require(order.order(3) == before, "no contribution must not move cursor");
    // Decode-only steps do not traverse prompts or account prefill service.
    require(order.order(3) == before, "inspection must not spend service");

    remaining = {32, 32, 32};
    const std::vector<bool> incompatible = {false, true, false};
    require(batch(order, remaining, incompatible, 64) == 32, "compatibility remains authoritative");
    require(remaining[0] == 32 && remaining[2] == 32, "incompatible prompts were admitted");
    require(order.order(3).front() == 2, "skips must not override last actual contributor");
    require(order.order(1) == std::vector<size_t>{0}, "pool resize normalizes cursor");
    require(order.order(3, true) == std::vector<size_t>({0, 1, 2}), "shared KV keeps physical order after service");
    require(order.order(3, false).front() == 2, "separate KV keeps cyclic service");
    require(order.order(0, true).empty(), "empty shared KV pool");
    require(order.order(1, true) == std::vector<size_t>{0}, "shared KV pool resize");
    return 0;
}
