// what this catches: parked-sequence checkpoints (server-parked-seqs.h) without a model: park copies
// the live partial state to a free parked id and restore copies it back, both by id; the last handle
// frees the cell (seq_rm) BEFORE the id is reused, from any thread; sharing a handle (a slot's list and
// a cache entry) never parks twice; an exhausted pool refuses typed; a handle may outlive its pool.
#include "../tools/server/server-parked-seqs.h"

#include <cstdlib>
#include <iostream>
#include <map>
#include <thread>
#include <vector>

static void require(bool condition, const char * message) {
    if (!condition) {
        std::cerr << "FAILED: " << message << '\n';
        std::exit(1);
    }
}

// A stand-in memory: each sequence id holds a "state" value; copy and free act on it.
struct fake_memory {
    std::mutex             m;
    std::map<int32_t, int> state;
    std::vector<std::pair<int32_t, int32_t>> copies;
    std::vector<int32_t>   frees;

    parked_seq_ops ops() {
        return {
            [this](int32_t src, int32_t dst) { std::scoped_lock l(m); state[dst] = state[src]; copies.push_back({ src, dst }); },
            [this](int32_t seq) { std::scoped_lock l(m); state.erase(seq); frees.push_back(seq); },
        };
    }
};

static void park_and_restore_move_state_by_id() {
    fake_memory mem;
    auto pool = parked_seq_pool::make(mem.ops(), /*first=*/4, /*count=*/3);
    mem.state[0] = 111; // slot 0's live state at the checkpoint
    auto ck = pool->park(0);
    require(ck.has_value() && (*ck)->id() == 4, "parked under the first free id, past the live slots");
    require(mem.state[4] == 111, "the parked id holds the live state");
    mem.state[0] = 999; // the live sequence moves on
    (*ck)->restore_into(0);
    require(mem.state[0] == 111, "restore puts the checkpoint's state back into the live sequence");
    require(mem.copies.size() == 2, "two copies by id, no bytes anywhere else");
}

static void the_last_handle_frees_the_cell_before_the_id_is_reused() {
    fake_memory mem;
    auto pool = parked_seq_pool::make(mem.ops(), 4, 1);
    auto a = pool->park(0);
    std::vector<parked_seq_handle> slot_list{ *a };   // a slot's checkpoints
    std::vector<parked_seq_handle> cache_entry{ *a }; // a cache entry's copy: shared, never parked twice
    a = std::unexpected(parked_seq_refusal::exhausted);
    require(pool->in_use() == 1 && pool->parked() == 1, "one park, shared by reference");
    auto b = pool->park(1);
    require(!b && b.error() == parked_seq_refusal::exhausted && pool->refused() == 1, "an exhausted pool refuses typed");
    slot_list.clear();
    require(pool->in_use() == 1 && mem.frees.empty(), "still held by the cache entry");
    cache_entry.clear();
    require(pool->in_use() == 0 && mem.frees.size() == 1 && mem.frees[0] == 4, "the last handle frees the cell");
    require(mem.state.count(4) == 0, "...and the parked id holds nothing when it is handed out again");
    require(pool->park(1).has_value(), "the id is free again");
}

static void releases_from_another_thread_are_safe() {
    fake_memory mem;
    auto pool = parked_seq_pool::make(mem.ops(), 8, 64);
    std::vector<parked_seq_handle> held;
    for (int i = 0; i < 64; ++i) {
        held.push_back(*pool->park(i % 8));
    }
    require(pool->in_use() == 64, "every id in use");
    std::thread t([moved = std::move(held)]() mutable { moved.clear(); });
    t.join();
    require(pool->in_use() == 0 && mem.frees.size() == 64, "handles dropped on another thread free every cell");
}

static void a_handle_may_outlive_its_pool() {
    fake_memory mem;
    parked_seq_handle survivor;
    {
        auto pool = parked_seq_pool::make(mem.ops(), 2, 1);
        survivor = *pool->park(0);
    }
    survivor->restore_into(0); // a gone pool: a no-op, never a crash
    survivor.reset();
}

int main() {
    park_and_restore_move_state_by_id();
    the_last_handle_frees_the_cell_before_the_id_is_reused();
    releases_from_another_thread_are_safe();
    a_handle_may_outlive_its_pool();
    std::cout << "test-server-parked-seqs: ok\n";
    return 0;
}
