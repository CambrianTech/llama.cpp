#include "../tools/server/server-adapter-turn.h"

#include <cstdlib>
#include <iostream>

static void require(bool condition, const char * message) {
    if (!condition) {
        std::cerr << message << '\n';
        std::exit(1);
    }
}

// Model the batch loop with the production turn: the holder's set is served, every slot of that
// set is stamped, and the other sets wait for the next batch.
static size_t step(std::vector<server_batch_claim> & claims, const std::vector<int> & set, int64_t now) {
    const auto turn = server_adapter_turn(claims);
    require(turn.has_value(), "a waiting slot always gets a turn");
    for (size_t index = 0; index < claims.size(); ++index) {
        if (claims[index].waiting && set[index] == set[*turn]) {
            claims[index].last_served_us = now;
        }
    }
    return *turn;
}

int main() {
    require(!server_adapter_turn({}).has_value(), "no slots, no turn");
    require(!server_adapter_turn({{false, 0}, {false, 1}}).has_value(), "idle slots take no turn");

    // what this catches (the M5, 2026-11-14): slots 1 and 3 wear a gene and keep generating; slots
    // 0, 2 and 4 wear none and were launched later. Under first-generating-by-index they never
    // entered a batch; under the turn they are served on the very next batch.
    std::vector<server_batch_claim> claims = {{true, 500}, {true, 100}, {true, 520}, {true, 100}, {true, 510}};
    const std::vector<int> set = {0, 1, 0, 1, 0};
    require(step(claims, set, 1000) == 1, "the longest-unserved slot's set holds the first batch");
    require(step(claims, set, 1100) == 0, "the other set holds the next batch, oldest slot first");
    for (int64_t now = 1200; now < 1200 + 100 * 100; now += 100) {
        step(claims, set, now);
        for (const auto & claim : claims) {
            require(now - claim.last_served_us <= 100, "with two sets waiting, every slot is served within two batches");
        }
    }

    // A lone set keeps the batch: nothing else waits, so nothing alternates.
    std::vector<server_batch_claim> lone = {{true, 0}, {false, 0}, {true, 5}};
    require(step(lone, {0, 1, 0}, 10) == 0 && step(lone, {0, 1, 0}, 20) == 0, "one set waiting holds every batch");

    // A tie goes to the lowest index, so the order is total and repeatable.
    require(server_adapter_turn({{true, 7}, {true, 7}}) == std::optional<size_t>(0), "a tie goes to the lowest index");
    return 0;
}
