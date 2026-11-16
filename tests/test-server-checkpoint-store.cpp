// what this catches: the one owner of a mind's checkpoints (server-checkpoint-store.h) losing what her next act
// needs. Model-free: the store runs over a checkpoint shaped like the server's. Covers placed checkpoints
// (CONTINUUM-ENGINE 4.7.3: her head, the restore point when her next prompt diverges before her view end, kept
// while anything else may go; Codex on llama.cpp#121 and #127) and the learn pins and the single eviction order
// that replaced five separate erase sites (2026-11-16: the close-checkpoint eviction erased pins, and the host
// budget and the count evicted in two different orders).
#include "../tools/server/server-checkpoint-store.h"

#include <cstdint>
#include <cstdlib>
#include <iostream>

static void require(bool condition, const char * message) {
    if (!condition) {
        std::cerr << "FAILED: " << message << '\n';
        std::exit(1);
    }
}

struct ckpt {
    int64_t  n_tokens;
    int      id_task;
    int      serial;          // identity: a checkpoint is the same one only if its serial matches
    uint64_t pin     = 0;
    int32_t  pos_max = 0;
    size_t   bytes   = 100;
    size_t   size() const { return bytes; }
};

using store = checkpoint_store<ckpt>;
static auto quiet = [](const ckpt &) {};

static store::admission at(int id_task, size_t capacity, int32_t view_end, int64_t step = 0) {
    return { id_task, step, capacity, view_end };
}

// create_checkpoint's order: make room, copy (may fail: nothing stored), keep
static void create(store & s, const store::admission & a, ckpt c, bool copy_ok = true, uint64_t pin = 0) {
    s.make_room(a, quiet);
    if (copy_ok) {
        s.keep(std::move(c), a, pin, quiet);
    }
}

static bool holds(const store & s, int serial) {
    for (const auto & c : s) {
        if (c.serial == serial) {
            return true;
        }
    }
    return false;
}

static void head_survives_a_full_store_across_many_acts() {
    store s;
    int serial = 0;
    create(s, at(1, 4, 100000), { 9655, 1, ++serial });
    const int head = s.front().serial;
    for (int act = 2; act < 60; ++act) {
        create(s, at(act, 4, 100000), { 9655 + act * 1000, act, ++serial });
        require(s.size() <= 4, "never more than the capacity");
        require(s.front().serial == head && s.front().n_tokens == 9655, "her exact head survives every act");
    }
}

static void capacity_one_keeps_only_what_fits() {
    store s;
    create(s, at(1, 1, 50000), { 9655, 1, 1 });
    create(s, at(2, 1, 50000), { 30000, 2, 2 });
    require(s.size() == 1 && s.front().serial == 2, "with room for one, the head goes when it is all there is");
}

static void a_failed_copy_keeps_the_head_and_stores_nothing() {
    store s;
    for (int i = 0; i < 4; ++i) {
        create(s, at(i, 8, 50000), { 9655 + i * 1000, i, i });
    }
    create(s, at(9, 4, 50000), { 20000, 9, 9 }, /*copy_ok=*/false);
    require(s.size() == 3, "room was made, the failed copy stored nothing");
    require(s.front().serial == 0, "her head is still there");
}

static void a_view_end_checkpoint_supersedes_earlier_acts_but_the_head() {
    store s;
    create(s, at(1, 32, -1), { 9655, 1, 1 });
    create(s, at(2, 32, -1), { 20000, 2, 2 });
    create(s, at(3, 32, -1), { 30000, 3, 3 });
    create(s, at(7, 32, -1), { 40000, 7, 4 });
    create(s, at(7, 32, 45000), { 45000, 7, 5 });
    require(s.size() == 3 && s.front().serial == 1, "the two earlier acts' checkpoints are superseded; her head stays");
    for (const auto & c : s) {
        require(c.serial == 1 || c.id_task == 7, "only the head and this act's checkpoints remain");
    }
    store u;
    create(u, at(1, 32, -1), { 9655, 1, 1 });
    create(u, at(2, 32, -1), { 20000, 2, 2 });
    create(u, at(7, 32, 30000), { 25000, 7, 3 });
    require(u.size() == 3, "a checkpoint not AT her view end supersedes nothing");
}

static void without_view_spans_the_oldest_goes() {
    store s;
    create(s, at(1, 2, -1), { 9655, 1, 1 });
    create(s, at(2, 2, -1), { 20000, 2, 2 });
    create(s, at(3, 2, -1), { 30000, 3, 3 });
    require(s.front().serial == 2 && s.size() == 2, "without view spans, upstream's rule: the oldest goes");
}

static void a_cold_prompt_with_view_spans_makes_her_head() {
    const int64_t user_starts[] = { 9655, 20000, 30000, 40000 };
    const int32_t view_end = 45000;
    store s;
    int serial = 0;
    for (const int64_t pos : user_starts) {
        if (s.stop_at_user_start(/*view_spans=*/true, pos, /*last_user_pos=*/40000, /*step=*/1024)) {
            create(s, at(1, 32, view_end), { pos, 1, ++serial });
        }
    }
    require(s.size() == 1 && s.front().n_tokens == 9655, "the head is made at the first user turn, and only there");
    create(s, at(1, 32, view_end), { view_end, 1, ++serial });
    require(s.size() == 2 && s.front().n_tokens == 9655 && s.back().n_tokens == view_end, "head, then her view end");
    store up;
    create(up, at(1, 32, -1), { 30000, 1, 1 });
    require(up.stop_at_user_start(false, 40000, 40000, 1024), "upstream: her last user message");
    require(!up.stop_at_user_start(false, 30500, 40000, 1024), "upstream: not within a step");
    require(!up.stop_at_user_start(true, 40000, 40000, 1024), "with view spans, no blind step checkpoint");
}

// ---- the defects the single owner closes ----------------------------------------------------------------------

static void a_pin_survives_the_close_checkpoint_eviction() {
    store s;
    create(s, at(1, 32, -1, /*step=*/1000), { 3000, 1, 1 });
    create(s, at(1, 32, -1, 1000), { 3300, 1, 2 }, true, /*pin=*/7); // her pinned turn end
    create(s, at(2, 32, -1, 1000), { 3305, 2, 3 });                  // her next act, close by
    require(holds(s, 2), "the pinned checkpoint stays however close an earlier one sits (no pin survived her next turn before)");
    store control;
    create(control, at(1, 32, -1, 1000), { 3000, 1, 1 });
    create(control, at(1, 32, -1, 1000), { 3300, 1, 2 });
    create(control, at(2, 32, -1, 1000), { 3305, 2, 3 });
    require(!holds(control, 2), "the same checkpoint unpinned is evicted as too close (the rule still works)");
}

static void a_full_store_of_pins_keeps_one_more() {
    store s;
    create(s, at(1, 2, 50000), { 1000, 1, 1 }, true, 1);
    create(s, at(2, 2, 50000), { 2000, 2, 2 }, true, 2);
    create(s, at(3, 2, 50000), { 3000, 3, 3 });
    require(s.size() == 3 && holds(s, 1) && holds(s, 2), "pins are never evicted for count: one more is kept until they are released");
    require(s.release(1), "releasing pin 1 finds it");
    create(s, at(4, 2, 50000), { 4000, 4, 4 });
    require(!holds(s, 1) || !holds(s, 3), "once released, it may go");
    require(holds(s, 2), "the other pin stays");
}

static void the_budget_and_the_count_evict_in_one_order() {
    // head 1000, middle 2000, newest 3000: the middle goes first, then the head, her newest (her resume point) last
    store s;
    create(s, at(1, 32, 3000), { 1000, 1, 1 });
    create(s, at(2, 32, 3000), { 2000, 2, 2 });
    create(s, at(3, 32, -1), { 3000, 3, 3 });
    require(s.evict_one(3000, quiet) == 100 && !holds(s, 2), "budget: the middle one first");
    require(s.evict_one(3000, quiet) == 100 && !holds(s, 1), "budget: then her head");
    require(s.evict_one(3000, quiet) == 100 && s.empty(), "budget: her newest last");
    require(s.evict_one(3000, quiet) == 0, "budget: nothing left reads 0, never a false eviction");
    // the count with two checkpoints keeps her newest (it used to evict the one after her head, which was the newest)
    store t;
    create(t, at(1, 2, 5000), { 1000, 1, 1 });
    create(t, at(2, 2, 5000), { 2000, 2, 2 });
    create(t, at(3, 2, 5000), { 3000, 3, 3 });
    require(holds(t, 3) && t.size() == 2, "count: the new one is kept");
    store pinned;
    create(pinned, at(1, 32, 5000), { 1000, 1, 1 }, true, 9);
    require(pinned.evict_one(5000, quiet) == 0, "budget: a store of pins gives nothing");
}

static void a_release_is_by_id_never_by_position() {
    store s;
    create(s, at(1, 32, -1), { 3300, 1, 1 }, true, 1);
    create(s, at(2, 32, -1), { 3300, 2, 2 }, true, 2); // another turn pinned at the same position
    require(s.release(1) && !s.release(1), "pin 1 releases once");
    require(s.release(2), "pin 2 is still held after pin 1's release");
    require(!s.release(0), "0 names no pin");
}

static void a_diverged_prompt_drops_what_lies_past_it_and_a_learn_cuts_back() {
    store s;
    ckpt a = { 1000, 1, 1 }; a.pos_max = 999;
    ckpt b = { 3300, 1, 2 }; b.pos_max = 3299;
    create(s, at(1, 32, -1), a);
    create(s, at(1, 32, -1), b, true, 4);
    require(s.drop_past_position(2000, quiet) == 1 && holds(s, 1) && !holds(s, 2), "past the divergence goes, pinned or not");
    create(s, at(2, 32, -1), { 5000, 2, 3 });
    require(s.cut_back_to(1000) == 1 && s.size() == 1, "a learn's cut back keeps only what lies within it");
}

int main() {
    a_cold_prompt_with_view_spans_makes_her_head();
    head_survives_a_full_store_across_many_acts();
    capacity_one_keeps_only_what_fits();
    a_failed_copy_keeps_the_head_and_stores_nothing();
    a_view_end_checkpoint_supersedes_earlier_acts_but_the_head();
    without_view_spans_the_oldest_goes();
    a_pin_survives_the_close_checkpoint_eviction();
    a_full_store_of_pins_keeps_one_more();
    the_budget_and_the_count_evict_in_one_order();
    a_release_is_by_id_never_by_position();
    a_diverged_prompt_drops_what_lies_past_it_and_a_learn_cuts_back();
    std::cout << "test-server-checkpoint-store: ok\n";
    return 0;
}
