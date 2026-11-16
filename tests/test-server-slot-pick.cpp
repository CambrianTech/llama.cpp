// Model-free validator of the slot choice (tools/server/server-slot-pick.h).
//
// what this catches (Codex on #105): a shrink with the same n_seq parks a mind but leaves an empty
// placeholder at its in-range id; explicit and LRU/prefix admissions took that placeholder, so the parked
// mind could not be restored (or a second park of the id asserted).

#include "../tools/server/server-slot-pick.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

static void require(bool condition, const char * message) {
    if (!condition) {
        std::fprintf(stderr, "FAILED: %s\n", message);
        std::exit(1);
    }
}

// Four slots; the shrink parked id 2 (its mind's state left service, an empty placeholder holds the id).
static std::vector<server_slot_view> after_shrink() {
    return {
        { 0, true,  false, 10, 900, 0 },
        { 1, false, false, 30, 800, 0 },
        { 2, false, true,   0,   0, 0 },  // the placeholder: idle, empty, least recently used
        { 3, false, false, 20, 700, 0 },
    };
}

static void a_parked_id_is_never_handed_out() {
    const auto s = after_shrink();
    // anonymous requests, LRU: the placeholder (t_last_used 0, the oldest) must not be chosen
    auto p = server_pick_slot(s.data(), s.size(), -1, 100, 0.0f);
    require(p.kind == server_slot_pick_kind::by_lru && s[p.index].id == 3, "LRU skips the parked placeholder");
    // prefix choice: even a placeholder that somehow matched is skipped
    auto m = s;
    m[2].n_tokens = 500; m[2].lcp = 99; m[1].lcp = 60;
    p = server_pick_slot(m.data(), m.size(), -1, 100, 0.5f);
    require(p.kind == server_slot_pick_kind::by_prefix && m[p.index].id == 1, "prefix choice skips the parked placeholder");
    // the mind's own request names its id: it waits for the mind to come back, never runs on the placeholder
    p = server_pick_slot(s.data(), s.size(), 2, 100, 0.0f);
    require(p.kind == server_slot_pick_kind::wait, "a request naming a parked id waits for it");
    // an id that is neither live nor parked is invalid
    p = server_pick_slot(s.data(), s.size(), 7, 100, 0.0f);
    require(p.kind == server_slot_pick_kind::no_slot, "an unknown id names no slot");
}

static void shrink_requests_grow_restores_the_mind() {
    auto s = after_shrink();
    // requests while parked: every free non-parked slot fills, then the rest wait; none take id 2
    std::vector<int32_t> taken;
    for (int r = 0; r < 4; ++r) {
        const auto p = server_pick_slot(s.data(), s.size(), -1, 100, 0.0f);
        if (p.kind != server_slot_pick_kind::by_lru) {
            require(p.kind == server_slot_pick_kind::wait, "with no free live slot, a request waits");
            continue;
        }
        require(!s[p.index].parked, "never the parked id");
        s[p.index].processing = true;
        taken.push_back(s[p.index].id);
    }
    require(taken.size() == 2 && taken[0] != 2 && taken[1] != 2, "the two free live slots served, id 2 untouched");
    // grow: the mind is unparked into its own id, then its request runs there
    s[2].parked = false;
    s[2].n_tokens = 950;
    const auto p = server_pick_slot(s.data(), s.size(), 2, 100, 0.0f);
    require(p.kind == server_slot_pick_kind::by_id && s[p.index].id == 2, "after the grow, the mind's request runs on its own slot");
}

// what this catches (Codex on #105): a parent's fan-out handed its children free slots that included a
// parked mind's placeholder
static void fan_out_never_takes_a_parked_id() {
    const auto s = after_shrink();
    size_t out[4];
    const size_t k = server_pick_free_slots(s.data(), s.size(), 4, /*exclude parent=*/1, out);
    require(k == 1 && s[out[0]].id == 3, "the only free live slot besides the parent; never the parked id 2");
}

int main() {
    a_parked_id_is_never_handed_out();
    shrink_requests_grow_restores_the_mind();
    fan_out_never_takes_a_parked_id();
    std::printf("test-server-slot-pick: ok\n");
    return 0;
}
