// what this catches: placed checkpoints (server-checkpoint-policy.h, CONTINUUM-ENGINE 4.7.3) losing her
// head checkpoint, the restore point when her next prompt diverges before her view end. Model-free: the
// rules run on a list shaped like the slot's. Codex's review of llama.cpp#121 asked for exactly this: the
// head's exact position pinned BEFORE the store fills and asserted after; the pre-fix rule shown to lose
// it; capacity exhaustion and a failed checkpoint copy covered without claiming more retention than the
// capacity allows; upstream behaviour kept without view spans.
#include "../tools/server/server-checkpoint-policy.h"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <list>

static void require(bool condition, const char * message) {
    if (!condition) {
        std::cerr << "FAILED: " << message << '\n';
        std::exit(1);
    }
}

struct ckpt {
    int64_t n_tokens;
    int     id_task;
    int     serial; // identity: a checkpoint is the same one only if its serial matches
};

using store = std::list<ckpt>;

// create_checkpoint's order: make room (evict while full), then copy; a failed copy stores nothing
static void create(store & s, size_t capacity, int32_t view_end, ckpt c, bool copy_ok = true) {
    while (s.size() >= capacity) {
        s.erase(checkpoint_to_evict(s, view_end));
    }
    if (copy_ok) {
        s.push_back(c);
    }
}

static void head_survives_a_full_store_across_many_acts() {
    const size_t capacity = 4;
    store s;
    int serial = 0;
    const int32_t view_end = 100000;
    create(s, capacity, view_end, { 9655, 1, ++serial });
    const int head_serial = s.front().serial; // pinned before the store fills
    for (int act = 2; act < 60; ++act) {
        create(s, capacity, view_end, { 9655 + act * 1000, act, ++serial });
        require(s.size() <= capacity, "never more than the capacity");
        require(s.front().serial == head_serial && s.front().n_tokens == 9655,
                "her exact head checkpoint (position 9655, the one pinned before filling) survives every act");
    }
}

static void the_pre_fix_rule_loses_the_head() {
    const size_t capacity = 4;
    store s;
    for (int i = 0; i < 4; ++i) {
        s.push_back({ 9655 + i * 1000, i, i });
    }
    const int head_serial = s.front().serial;
    // the rule before the fix: always the oldest
    s.erase(s.begin());
    s.push_back({ 20000, 9, 9 });
    require(s.front().serial != head_serial, "the pre-fix rule evicts her head (the defect this guards)");
    // the rule after
    store t;
    for (int i = 0; i < 4; ++i) {
        t.push_back({ 9655 + i * 1000, i, i });
    }
    create(t, capacity, /*view_end=*/50000, { 20000, 9, 9 });
    require(t.front().serial == head_serial, "the placed rule keeps it");
    (void) capacity;
}

static void capacity_one_keeps_only_what_fits() {
    store s;
    create(s, 1, 50000, { 9655, 1, 1 });
    create(s, 1, 50000, { 30000, 2, 2 });
    require(s.size() == 1 && s.front().serial == 2, "with room for one, the head goes when it is all there is");
}

static void a_failed_copy_keeps_the_head_and_stores_nothing() {
    store s;
    for (int i = 0; i < 4; ++i) {
        s.push_back({ 9655 + i * 1000, i, i });
    }
    create(s, 4, 50000, { 20000, 9, 9 }, /*copy_ok=*/false);
    require(s.size() == 3, "room was made, the failed copy stored nothing: one fewer, never a claim of four");
    require(s.front().serial == 0, "her head is still there");
}

static void a_view_end_checkpoint_supersedes_earlier_acts_but_the_head() {
    store s = { { 9655, 1, 1 }, { 20000, 2, 2 }, { 30000, 3, 3 }, { 40000, 7, 4 } };
    const ckpt kept = { 45000, 7, 5 };
    s.push_back(kept);
    size_t erased = erase_superseded_checkpoints(s, s.back(), /*view_end=*/45000, /*id_task=*/7, [](const ckpt &) {});
    require(erased == 2, "the two checkpoints earlier acts left are superseded");
    require(s.size() == 3 && s.front().serial == 1, "her head stays");
    for (const auto & c : s) {
        require(c.serial == 1 || c.id_task == 7, "only the head and this act's checkpoints remain");
    }
    store u = { { 9655, 1, 1 }, { 20000, 2, 2 } };
    u.push_back({ 25000, 7, 3 });
    require(erase_superseded_checkpoints(u, u.back(), /*view_end=*/30000, 7, [](const ckpt &) {}) == 0,
            "a checkpoint not AT her view end supersedes nothing");
}

static void without_view_spans_the_oldest_goes() {
    store s = { { 9655, 1, 1 }, { 20000, 2, 2 } };
    create(s, 2, /*view_end=*/-1, { 30000, 3, 3 });
    require(s.front().serial == 2, "without view spans, upstream's rule: the oldest goes");
    require(erase_superseded_checkpoints(s, s.back(), -1, 3, [](const ckpt &) {}) == 0, "and nothing is superseded");
}

// the caller's cold path (Codex on #127): a prompt filled from an empty store, with view spans, stopping
// at user turns by checkpoint_at_user_start and creating a checkpoint at each stop, must make her head at
// the FIRST user turn; later turns add none; the view end adds the next one
static void a_cold_prompt_with_view_spans_makes_her_head() {
    const int64_t user_starts[] = { 9655, 20000, 30000, 40000 };
    const int64_t view_end = 45000;
    store s;
    int serial = 0;
    for (const int64_t pos : user_starts) {
        if (checkpoint_at_user_start(/*view_spans=*/true, s.empty(), pos, /*last_user_pos=*/40000,
                                     s.empty() ? 0 : s.back().n_tokens, /*step=*/1024)) {
            create(s, 32, view_end, { pos, 1, ++serial });
        }
    }
    require(s.size() == 1 && s.front().n_tokens == 9655, "the head is made at the first user turn, and only there");
    create(s, 32, view_end, { view_end, 1, ++serial }); // the view-end stop
    require(s.size() == 2 && s.front().n_tokens == 9655 && s.back().n_tokens == view_end, "head, then her view end");
    // the rule before this fix skipped every user-start stop when the request carried view spans, so the
    // only checkpoint a cold prompt got was its view end's: no head
    store before;
    create(before, 32, view_end, { view_end, 1, 99 });
    require(before.size() == 1 && before.front().n_tokens == view_end, "the pre-fix path's store has no head (the defect this guards)");
    // without view spans, upstream: the last user message and step-spaced turns
    require(checkpoint_at_user_start(false, false, 40000, 40000, 30000, 1024), "upstream: her last user message");
    require(!checkpoint_at_user_start(false, false, 30500, 40000, 30000, 1024), "upstream: not within a step");
    require(!checkpoint_at_user_start(true, false, 40000, 40000, 30000, 1024), "with view spans, no blind step checkpoint");
}

int main() {
    a_cold_prompt_with_view_spans_makes_her_head();
    head_survives_a_full_store_across_many_acts();
    the_pre_fix_rule_loses_the_head();
    capacity_one_keeps_only_what_fits();
    a_failed_copy_keeps_the_head_and_stores_nothing();
    a_view_end_checkpoint_supersedes_earlier_acts_but_the_head();
    without_view_spans_the_oldest_goes();
    std::cout << "test-server-checkpoint-policy: ok\n";
    return 0;
}
