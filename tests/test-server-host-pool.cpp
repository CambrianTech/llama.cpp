// what this catches: the host pool's contract (server-host-pool.h, CONTINUUM-ENGINE 4.7.6) without a
// model, in milliseconds: committed + reserved never past the grant at any allocation; bytes reserved
// before a copy and returned on failure; oversized and pinned-pressure refusals typed; eviction by one
// value unit (unmeasured first, then lowest seconds-saved per byte, then least recently used);
// pinned residents never evicted; shrinks below pinned bytes pending, then applied as pins end;
// grants keyed by instance and a strictly rising generation, idempotent on repeat; an evicted
// resident stays charged until its owner physically releases it (regression for Codex's review of
// fork 84aba863e: make_room credited bytes before the owner freed them, so a grant read applied and a
// new reservation was admitted while the old buffers still existed).
#include "../tools/server/server-host-pool.h"

#include <algorithm>
#include <cstdlib>
#include <iostream>

static void require(bool condition, const char * message) {
    if (!condition) {
        std::cerr << "FAILED: " << message << '\n';
        std::exit(1);
    }
}

using K = host_pool_kind;
using O = host_pool_outcome_kind;

static const host_pool_instance ENGINE{ 4242, 1794830000 };

// The accounting invariant, checked after every step of every scenario.
static void within(const host_pool & p, const char * where) {
    if (p.outcome() == O::applied) {
        require(p.committed_bytes() + p.reserved_bytes() <= p.target_bytes(), where);
    }
    require(p.pinned_bytes() <= p.committed_bytes(), "pinned bytes are a subset of committed");
    require(p.evicting_bytes() <= p.committed_bytes(), "evicting bytes are a subset of committed: still charged");
}

// The owner's protocol: reserve; physically free every resident the pool names and release() it;
// when refused only because those bytes were not freed yet, reserve again. The resident's id.
static uint64_t admit(host_pool & p, K kind, size_t bytes, host_pool_value value, std::vector<uint64_t> & freed) {
    auto r = p.reserve(kind, bytes);
    for (uint64_t id : r.evict) {
        p.release(id);
    }
    freed.insert(freed.end(), r.evict.begin(), r.evict.end());
    if (!r.admitted && r.admitted.error() == host_pool_refusal::awaiting_release) {
        r = p.reserve(kind, bytes);
    }
    require(r.admitted.has_value(), "the allocation is admitted once the named residents are freed");
    require(p.commit(*r.admitted, bytes, value).has_value(), "the reservation commits");
    within(p, "admit stays within the grant");
    return r.admitted->id;
}

static void eviction_order_is_one_value_unit() {
    host_pool p(ENGINE, 1000);
    std::vector<uint64_t> freed;
    const uint64_t cheap   = admit(p, K::checkpoint, 300, host_pool_value::measured(1e-6), freed);
    const uint64_t blind   = admit(p, K::checkpoint, 300, host_pool_value::unmeasured(), freed);
    const uint64_t prized  = admit(p, K::cache, 300, host_pool_value::measured(5e-6), freed);
    require(freed.empty(), "900 of 1000 bytes needs no eviction");

    admit(p, K::checkpoint, 300, host_pool_value::unmeasured(), freed);
    require(freed.size() == 1 && freed[0] == blind, "the unmeasured resident goes first");
    freed.clear();

    admit(p, K::cache, 300, host_pool_value::measured(9e-6), freed);
    require(freed.size() == 1, "one more eviction makes room");
    require(!p.holds(freed[0]) && p.holds(prized), "the higher value per byte is kept");
    require(freed[0] != cheap || !p.holds(cheap), "...and the lowest-valued measured one leaves once only unmeasured are gone");
    require(p.evicted(K::checkpoint) + p.evicted(K::cache) == 2, "each eviction is counted by kind");
}

static void reservations_bound_the_peak_and_roll_back() {
    host_pool p(ENGINE, 1000);
    auto a = p.reserve(K::checkpoint, 700);
    require(a.admitted.has_value(), "the first reservation fits");
    within(p, "a reservation counts against the grant before any copy");
    auto b = p.reserve(K::cache, 400);
    require(!b.admitted && b.admitted.error() == host_pool_refusal::pinned_pressure, "a reserved peak leaves no room: typed refusal, never an overflow");
    p.rollback(*a.admitted);
    require(p.reserved_bytes() == 0 && p.committed_bytes() == 0, "a failed copy returns every reserved byte");
    auto c = p.reserve(K::cache, 400);
    require(c.admitted.has_value(), "after the rollback the bytes are free again");
    require(!p.commit(*c.admitted, 401, host_pool_value::unmeasured()).has_value(), "a commit larger than its reservation is refused");
    require(p.commit(*c.admitted, 380, host_pool_value::unmeasured()).has_value(), "a commit at or under its reservation lands");
    require(p.committed_bytes() == 380 && p.reserved_bytes() == 0, "the actual bytes are charged, the rest returns");
    within(p, "commit");
}

static void oversized_and_pinned_are_typed_refusals() {
    host_pool p(ENGINE, 1000);
    std::vector<uint64_t> freed;
    auto big = p.reserve(K::cache, 1001);
    require(!big.admitted && big.admitted.error() == host_pool_refusal::oversized, "larger than the whole grant: oversized");
    const uint64_t held = admit(p, K::checkpoint, 900, host_pool_value::measured(1e-6), freed);
    require(p.pin(held), "a restore pins it");
    auto squeezed = p.reserve(K::cache, 200);
    require(!squeezed.admitted && squeezed.admitted.error() == host_pool_refusal::pinned_pressure, "a pinned resident is never evicted for a newcomer");
    require(p.holds(held), "the pinned resident is still held");
    require(p.refused(host_pool_refusal::oversized) == 1 && p.refused(host_pool_refusal::pinned_pressure) == 1, "refusals are counted by cause");
    within(p, "refusals leave the accounting intact");
}

static void a_shrink_below_pinned_bytes_is_pending_then_applied() {
    host_pool p(ENGINE, 1000);
    std::vector<uint64_t> freed;
    const uint64_t pinned = admit(p, K::checkpoint, 600, host_pool_value::measured(2e-6), freed);
    const uint64_t loose  = admit(p, K::cache, 300, host_pool_value::measured(9e-6), freed);
    p.pin(pinned);
    auto shrink = p.set_budget(ENGINE, 1, 200);
    require(shrink.outcome == O::pending, "pinned bytes above the new target: pending, never a false 'applied'");
    require(shrink.evict.size() == 1 && shrink.evict[0] == loose, "every unpinned resident is evicted toward the target at once");
    require(p.ceiling_bytes() == 900 && p.evicting_bytes() == 300, "the named resident is still charged until its owner frees it");
    p.release(loose);
    require(p.ceiling_bytes() == 600 && p.target_bytes() == 200, "the ceiling names what is still held above the target");
    require(p.pinned_bytes() == 600 && p.pinned_bytes() <= p.committed_bytes(), "pinned is reported as a subset of committed");
    auto after = p.unpin(pinned);
    require(after.size() == 1 && after[0] == pinned, "when custody ends the pending shrink names the last resident");
    require(p.outcome() == O::pending, "...and still reads pending until that resident is physically freed");
    p.release(pinned);
    require(p.outcome() == O::applied && p.committed_bytes() == 0, "the same generation now reads applied");
    within(p, "shrink completed");
}

static void zero_budget_is_legal() {
    host_pool p(ENGINE, 1000);
    std::vector<uint64_t> freed;
    admit(p, K::cache, 400, host_pool_value::unmeasured(), freed);
    admit(p, K::page, 400, host_pool_value::measured(1e-5), freed);
    auto zero = p.set_budget(ENGINE, 1, 0);
    require(zero.outcome == O::pending && zero.evict.size() == 2, "a zero grant names every unpinned resident, pending until freed");
    for (uint64_t id : zero.evict) {
        p.release(id);
    }
    require(p.outcome() == O::applied && p.committed_bytes() == 0, "nothing is held under a zero grant once freed");
    auto next = p.reserve(K::checkpoint, 1);
    require(!next.admitted && next.admitted.error() == host_pool_refusal::oversized, "under a zero grant every allocation is refused, typed");
}

static void grants_are_keyed_by_instance_and_generation() {
    host_pool p(ENGINE, 1000);
    const host_pool_instance other{ 4242, 1794830001 };
    auto foreign = p.set_budget(other, 1, 500);
    require(!foreign.outcome && foreign.outcome.error() == host_pool_grant_refusal::wrong_instance, "a grant for another instance is refused");
    require(p.target_bytes() == 1000 && p.generation() == 0, "a refused grant leaves the state exactly as it was");
    require(p.set_budget(ENGINE, 2, 800).outcome == O::applied, "a newer generation applies");
    auto repeat = p.set_budget(ENGINE, 2, 800);
    require(repeat.outcome == O::applied && repeat.evict.empty() && p.target_bytes() == 800, "the same generation and size is an idempotent repeat");
    auto conflict = p.set_budget(ENGINE, 2, 700);
    require(!conflict.outcome && conflict.outcome.error() == host_pool_grant_refusal::generation_conflict, "the same generation with another size is refused");
    auto stale = p.set_budget(ENGINE, 1, 900);
    require(!stale.outcome && stale.outcome.error() == host_pool_grant_refusal::stale_generation, "an older generation is refused");
    require(p.target_bytes() == 800 && p.generation() == 2, "refusals never move the grant");
}

static void a_delayed_release_keeps_the_charge() {
    host_pool p(ENGINE, 1000);
    std::vector<uint64_t> freed;
    const uint64_t old = admit(p, K::checkpoint, 800, host_pool_value::unmeasured(), freed);
    auto r = p.reserve(K::cache, 500);
    require(r.evict.size() == 1 && r.evict[0] == old, "the old resident is named for eviction");
    require(!r.admitted && r.admitted.error() == host_pool_refusal::awaiting_release,
            "its bytes still exist: refused awaiting_release, never granted on top of them");
    require(!p.holds(old) && !p.pin(old), "a resident being freed is neither held nor pinnable");
    require(p.committed_bytes() == 800 && p.evicting_bytes() == 800, "readback keeps charging the unfreed bytes");
    auto again = p.reserve(K::cache, 500);
    require(!again.admitted && again.admitted.error() == host_pool_refusal::awaiting_release && again.evict.empty(),
            "asking again before the owner frees it changes nothing and names nothing twice");
    auto shrink = p.set_budget(ENGINE, 1, 900);
    require(shrink.outcome == O::applied, "a grant still above the charged bytes applies");
    auto tighter = p.set_budget(ENGINE, 2, 500);
    require(tighter.outcome == O::pending, "a grant below the unfreed bytes reads pending, never applied");
    p.release(old); // the owner finally freed it
    require(p.outcome() == O::applied && p.committed_bytes() == 0 && p.evicting_bytes() == 0, "only the release credits the bytes");
    require(p.reserve(K::cache, 500).admitted.has_value(), "and now the reservation is admitted");
}

static void a_day_of_mixed_traffic_never_exceeds_the_grant() {
    // a deterministic stand-in for the residents' traffic: saves, restores, value updates and shrinks
    host_pool p(ENGINE, 4096);
    std::vector<uint64_t> live;
    std::vector<uint64_t> unfreed; // named for eviction, freed by the owner later (delayed custody)
    uint64_t seed = 88172645463325252ull;
    auto rnd = [&]() { seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; return seed; };
    uint64_t generation = 0;
    for (int step = 0; step < 20000; ++step) {
        switch (rnd() % 6) {
            case 0: case 1: case 2: { // a save: reserve, copy (sometimes fails), commit or roll back
                const K kind = (K) (rnd() % HOST_POOL_KINDS);
                const size_t bytes = 1 + rnd() % 1500;
                auto r = p.reserve(kind, bytes);
                for (uint64_t id : r.evict) {
                    require(!p.holds(id), "an evicted resident is no longer held");
                    live.erase(std::remove(live.begin(), live.end(), id), live.end());
                    unfreed.push_back(id);
                }
                if (r.admitted) {
                    if (rnd() % 10 == 0) {
                        p.rollback(*r.admitted);
                    } else {
                        const auto v = rnd() % 3 == 0 ? host_pool_value::unmeasured() : host_pool_value::measured((double) (rnd() % 1000) * 1e-9);
                        require(p.commit(*r.admitted, bytes - (bytes > 1 ? rnd() % (bytes / 2 + 1) : 0), v).has_value(), "commit within the reservation");
                        live.push_back(r.admitted->id);
                    }
                }
                break;
            }
            case 3: { // a restore: pin, then unpin
                if (live.empty()) break;
                const uint64_t id = live[rnd() % live.size()];
                if (p.pin(id)) {
                    for (uint64_t gone : p.unpin(id)) {
                        live.erase(std::remove(live.begin(), live.end(), gone), live.end());
                        unfreed.push_back(gone);
                    }
                }
                break;
            }
            case 4: { // the owner frees what was named (late), or a measured restore raises a value
                if (!unfreed.empty() && rnd() % 2 == 0) {
                    const size_t i = rnd() % unfreed.size();
                    p.release(unfreed[i]);
                    unfreed.erase(unfreed.begin() + (long) i);
                    break;
                }
                if (live.empty()) break;
                p.set_value(live[rnd() % live.size()], host_pool_value::measured((double) (rnd() % 1000) * 1e-8));
                break;
            }
            case 5: { // the core moves the grant
                auto g = p.set_budget(ENGINE, ++generation, 1024 + rnd() % 6000);
                for (uint64_t gone : g.evict) {
                    live.erase(std::remove(live.begin(), live.end(), gone), live.end());
                    unfreed.push_back(gone);
                }
                break;
            }
        }
        within(p, "mixed traffic stays within the grant at every step");
    }
}

int main() {
    eviction_order_is_one_value_unit();
    reservations_bound_the_peak_and_roll_back();
    oversized_and_pinned_are_typed_refusals();
    a_shrink_below_pinned_bytes_is_pending_then_applied();
    zero_budget_is_legal();
    grants_are_keyed_by_instance_and_generation();
    a_delayed_release_keeps_the_charge();
    a_day_of_mixed_traffic_never_exceeds_the_grant();
    std::cout << "test-server-host-pool: ok\n";
    return 0;
}
