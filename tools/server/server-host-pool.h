#pragma once

// Continuum fork: the engine's host memory as ONE streaming pool (CONTINUUM-ENGINE 4.7.6).
//
// Prompt-cache entries, context checkpoints and RAM pages are kinds of resident in one pool with one
// byte budget, the way a game engine's texture streamer holds one pool. Bytes are RESERVED before a
// resident is serialized or copied (its whole peak), the reservation evicts the lowest-valued unpinned
// residents until it fits, and an allocation that cannot fit after that is a typed refusal, never an
// overflow. The pool decides; it holds no native memory: it names the residents to free and their
// owner frees them on the engine thread (server-context), then calls release().
//
// Value is one unit across kinds: expected seconds of prefill saved per byte kept. A resident with no
// measured restore is UNMEASURED (explicit, never a guessed number). Eviction order: unmeasured first,
// least recently used first among them; then measured by ascending value per byte; ties by least
// recently used, then id. Pinned residents (a restore in progress, a save in flight) are never
// evicted. The budget is the core's grant, keyed by engine instance and a strictly rising generation.
//
// Pure and header-only so tests/test-server-host-pool.cpp proves it without a model.

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

enum class host_pool_kind : uint8_t { cache = 0, checkpoint = 1, page = 2 };
static constexpr size_t HOST_POOL_KINDS = 3;

inline const char * host_pool_kind_name(host_pool_kind kind) {
    switch (kind) {
        case host_pool_kind::cache:      return "cache";
        case host_pool_kind::checkpoint: return "checkpoint";
        case host_pool_kind::page:       return "page";
    }
    return "unknown";
}

// Why an allocation was not admitted.
enum class host_pool_refusal : uint8_t {
    oversized       = 0, // larger than the whole budget
    pinned_pressure = 1, // would fit the budget, but pinned and reserved bytes leave no room
};
static constexpr size_t HOST_POOL_REFUSALS = 2;

inline const char * host_pool_refusal_name(host_pool_refusal refusal) {
    switch (refusal) {
        case host_pool_refusal::oversized:       return "oversized";
        case host_pool_refusal::pinned_pressure: return "pinned_pressure";
    }
    return "unknown";
}

// Expected seconds of prefill saved per byte kept, or unmeasured.
struct host_pool_value {
    std::optional<double> s_saved_per_byte;

    static host_pool_value unmeasured() { return {}; }
    static host_pool_value measured(double s_saved_per_byte) { return { s_saved_per_byte }; }
};

struct host_pool_resident {
    host_pool_kind  kind;
    size_t          bytes    = 0;
    host_pool_value value;
    int             pins     = 0; // restores / saves holding it
    uint64_t        last_use = 0; // the pool's logical clock
};

// Bytes granted for one resident not yet committed.
struct host_pool_reservation {
    uint64_t       id    = 0;
    host_pool_kind kind  = host_pool_kind::cache;
    size_t         bytes = 0;
};

struct host_pool_reserve_result {
    std::optional<host_pool_reservation> reservation; // set when admitted
    std::optional<host_pool_refusal>     refusal;     // set when refused
    std::vector<uint64_t>                evict;       // residents the owner must free now (admitted or not)
};

// The engine instance a grant is meant for.
struct host_pool_instance {
    int64_t pid       = 0;
    int64_t started_s = 0;

    bool operator==(const host_pool_instance & o) const { return pid == o.pid && started_s == o.started_s; }
};

enum class host_pool_outcome_kind : uint8_t { applied, pending, refused };

enum class host_pool_grant_refusal : uint8_t { wrong_instance, stale_generation, generation_conflict };

inline const char * host_pool_grant_refusal_name(host_pool_grant_refusal r) {
    switch (r) {
        case host_pool_grant_refusal::wrong_instance:      return "wrong_instance";
        case host_pool_grant_refusal::stale_generation:    return "stale_generation";
        case host_pool_grant_refusal::generation_conflict: return "generation_conflict";
    }
    return "unknown";
}

struct host_pool_grant_result {
    host_pool_outcome_kind                 outcome = host_pool_outcome_kind::applied;
    std::optional<host_pool_grant_refusal> refusal;
    std::vector<uint64_t>                  evict; // residents the owner must free now to move toward the target
};

class host_pool {
public:
    host_pool(host_pool_instance instance, size_t launch_budget)
        : instance_(instance), target_(launch_budget) {}

    // ---- the core's grant -------------------------------------------------------------------
    host_pool_grant_result set_budget(host_pool_instance instance, uint64_t generation, size_t bytes) {
        host_pool_grant_result out;
        if (!(instance == instance_)) {
            out.outcome = host_pool_outcome_kind::refused;
            out.refusal = host_pool_grant_refusal::wrong_instance;
            return out;
        }
        if (generation < generation_) {
            out.outcome = host_pool_outcome_kind::refused;
            out.refusal = host_pool_grant_refusal::stale_generation;
            return out;
        }
        if (generation == generation_ && bytes != target_) {
            out.outcome = host_pool_outcome_kind::refused;
            out.refusal = host_pool_grant_refusal::generation_conflict;
            return out;
        }
        generation_ = generation;
        target_     = bytes;
        out.evict   = make_room(0);
        out.outcome = outcome();
        return out;
    }

    // ---- allocation ---------------------------------------------------------------------------
    // Reserve `bytes` for a new resident of `kind`. The owner frees every id in `evict` (whether or
    // not the reservation was admitted), then copies only inside an admitted reservation.
    host_pool_reserve_result reserve(host_pool_kind kind, size_t bytes) {
        host_pool_reserve_result out;
        if (bytes > target_) {
            out.refusal = host_pool_refusal::oversized;
            refused_[(size_t) host_pool_refusal::oversized]++;
            return out;
        }
        out.evict = make_room(bytes);
        if (charged() + bytes > target_) {
            out.refusal = host_pool_refusal::pinned_pressure;
            refused_[(size_t) host_pool_refusal::pinned_pressure]++;
            return out;
        }
        const uint64_t id = next_id_++;
        reserved_ += bytes;
        reservations_[id] = host_pool_reservation{ id, kind, bytes };
        out.reservation   = reservations_[id];
        return out;
    }

    // The copy succeeded: the reservation becomes a committed resident of its actual `bytes` (at most
    // what was reserved), returning its id. Unknown reservation: false.
    bool commit(const host_pool_reservation & r, size_t bytes, host_pool_value value) {
        auto it = reservations_.find(r.id);
        if (it == reservations_.end() || bytes > it->second.bytes) {
            return false;
        }
        reserved_ -= it->second.bytes;
        reservations_.erase(it);
        residents_[r.id] = host_pool_resident{ r.kind, bytes, value, 0, ++clock_ };
        committed_[(size_t) r.kind] += bytes;
        return true;
    }

    // The copy failed: the reservation's bytes return, nothing is committed.
    void rollback(const host_pool_reservation & r) {
        auto it = reservations_.find(r.id);
        if (it == reservations_.end()) {
            return;
        }
        reserved_ -= it->second.bytes;
        reservations_.erase(it);
    }

    // The owner freed resident `id` (after an eviction, or its own erase).
    void release(uint64_t id) {
        auto it = residents_.find(id);
        if (it == residents_.end()) {
            return;
        }
        committed_[(size_t) it->second.kind] -= it->second.bytes;
        residents_.erase(it);
    }

    // ---- use ----------------------------------------------------------------------------------
    bool pin(uint64_t id) {
        auto it = residents_.find(id);
        if (it == residents_.end()) return false;
        it->second.pins++;
        it->second.last_use = ++clock_;
        return true;
    }
    // Returns the residents to free now that a pin ended (a pending shrink proceeds).
    std::vector<uint64_t> unpin(uint64_t id) {
        auto it = residents_.find(id);
        if (it != residents_.end() && it->second.pins > 0) {
            it->second.pins--;
        }
        return make_room(0);
    }
    void touch(uint64_t id) {
        auto it = residents_.find(id);
        if (it != residents_.end()) it->second.last_use = ++clock_;
    }
    void set_value(uint64_t id, host_pool_value value) {
        auto it = residents_.find(id);
        if (it != residents_.end()) it->second.value = value;
    }

    // ---- report -------------------------------------------------------------------------------
    host_pool_outcome_kind outcome() const {
        return charged() <= target_ ? host_pool_outcome_kind::applied : host_pool_outcome_kind::pending;
    }
    host_pool_instance instance() const { return instance_; }
    uint64_t generation()       const { return generation_; }
    size_t   target_bytes()     const { return target_; }
    // What is held now above which nothing new is admitted: the target, or while pending the bytes
    // still held above it (falling as custody ends).
    size_t   ceiling_bytes()    const { return charged() > target_ ? charged() : target_; }
    size_t   reserved_bytes()   const { return reserved_; }
    size_t   committed_bytes()  const { size_t s = 0; for (size_t k : committed_) s += k; return s; }
    size_t   committed_bytes(host_pool_kind k) const { return committed_[(size_t) k]; }
    size_t   reserved_bytes(host_pool_kind k) const {
        size_t s = 0;
        for (const auto & [id, r] : reservations_) if (r.kind == k) s += r.bytes;
        return s;
    }
    size_t   pinned_bytes(std::optional<host_pool_kind> k = std::nullopt) const {
        size_t s = 0;
        for (const auto & [id, r] : residents_) if (r.pins > 0 && (!k || r.kind == *k)) s += r.bytes;
        return s;
    }
    size_t   count(host_pool_kind k) const {
        size_t n = 0;
        for (const auto & [id, r] : residents_) if (r.kind == k) n++;
        return n;
    }
    uint64_t evicted(host_pool_kind k)    const { return evicted_[(size_t) k]; }
    uint64_t refused(host_pool_refusal r) const { return refused_[(size_t) r]; }
    bool     holds(uint64_t id)           const { return residents_.count(id) > 0; }

private:
    size_t charged() const { return committed_bytes() + reserved_; }

    // True when `a` should be evicted before `b`.
    static bool evict_before(uint64_t a_id, const host_pool_resident & a, uint64_t b_id, const host_pool_resident & b) {
        const bool a_measured = a.value.s_saved_per_byte.has_value();
        const bool b_measured = b.value.s_saved_per_byte.has_value();
        if (a_measured != b_measured) {
            return !a_measured; // unmeasured first
        }
        if (a_measured && *a.value.s_saved_per_byte != *b.value.s_saved_per_byte) {
            return *a.value.s_saved_per_byte < *b.value.s_saved_per_byte;
        }
        if (a.last_use != b.last_use) {
            return a.last_use < b.last_use;
        }
        return a_id < b_id;
    }

    // Choose unpinned residents to evict until `extra` more bytes fit under the target (or nothing
    // unpinned is left). The chosen are released from the accounting here; the owner frees them.
    std::vector<uint64_t> make_room(size_t extra) {
        std::vector<uint64_t> out;
        while (charged() + extra > target_) {
            auto victim = residents_.end();
            for (auto it = residents_.begin(); it != residents_.end(); ++it) {
                if (it->second.pins > 0) continue;
                if (victim == residents_.end() || evict_before(it->first, it->second, victim->first, victim->second)) {
                    victim = it;
                }
            }
            if (victim == residents_.end()) {
                break; // only pinned residents and reservations remain
            }
            out.push_back(victim->first);
            evicted_[(size_t) victim->second.kind]++;
            committed_[(size_t) victim->second.kind] -= victim->second.bytes;
            residents_.erase(victim);
        }
        return out;
    }

    host_pool_instance instance_;
    uint64_t           generation_ = 0;
    size_t             target_     = 0;
    size_t             reserved_   = 0;
    size_t             committed_[HOST_POOL_KINDS] = {0, 0, 0};
    uint64_t           evicted_[HOST_POOL_KINDS]   = {0, 0, 0};
    uint64_t           refused_[HOST_POOL_REFUSALS] = {0, 0};
    uint64_t           next_id_ = 1;
    uint64_t           clock_   = 0;
    std::map<uint64_t, host_pool_resident>    residents_;
    std::map<uint64_t, host_pool_reservation> reservations_;
};
