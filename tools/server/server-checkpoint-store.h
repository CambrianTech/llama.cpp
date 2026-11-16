#pragma once

// Continuum fork: ONE owner of a mind's context checkpoints. Every rule that decides whether one is kept lives
// here and nowhere else: when one is made at a user turn, how close two may sit, how many fit, which goes to make
// room (by count or under the host budget), what her view end supersedes, what a diverged prompt invalidates,
// and a learn pin's lifetime. The list is private: a caller reads it or asks a verb, so a new rule cannot be
// added at a call site that forgets the others (2026-11-16: the close-checkpoint eviction erased learn pins, and
// the budget and the count evicted in two different orders).
//
// Pure over the checkpoint type `C` (it needs `n_tokens`, `id_task`, `pin`, `pos_max` and `size()`), so a native
// test proves every rule without a model (tests/test-server-checkpoint-store.cpp).

#include <cstddef>
#include <cstdint>
#include <list>
#include <utility>

template <typename C>
class checkpoint_store {
public:
    using list_type = std::list<C>;

    checkpoint_store() = default;
    explicit checkpoint_store(list_type restored) : cks(std::move(restored)) {}

    // ---- reading ----------------------------------------------------------------------------------------------
    typename list_type::const_iterator         begin()  const { return cks.begin(); }
    typename list_type::const_iterator         end()    const { return cks.end(); }
    typename list_type::const_reverse_iterator rbegin() const { return cks.rbegin(); }
    typename list_type::const_reverse_iterator rend()   const { return cks.rend(); }
    size_t     size()  const { return cks.size(); }
    bool       empty() const { return cks.empty(); }
    const C &  front() const { return cks.front(); }
    const C &  back()  const { return cks.back(); }
    const list_type & all() const { return cks; }

    size_t bytes() const {
        size_t b = 0;
        for (const auto & c : cks) {
            b += c.size();
        }
        return b;
    }

    // Whether the prompt fill stops at a user turn's start (at `pos`) so a checkpoint is made there. The first
    // boundary always gets one: with an empty store it is her head, her restore point when her next prompt
    // diverges before her view end. After it, with view spans, the view end is where her next prompt diverges,
    // so blind step checkpoints are not made (up to 32 x ~110 MB per slot thrashed the M5's host); without view
    // spans, upstream's rule: at her last user message, or a step past the newest checkpoint.
    bool stop_at_user_start(bool view_spans, int64_t pos, int64_t last_user_pos, int64_t step) const {
        if (cks.empty()) {
            return true;
        }
        if (view_spans) {
            return false;
        }
        return pos == last_user_pos || pos > cks.back().n_tokens + step;
    }

    // ---- making one -------------------------------------------------------------------------------------------
    struct admission {
        int      id_task;  // the act making it
        int64_t  step;     // checkpoints an earlier act left closer than this to the previous one go
        size_t   capacity; // the most this mind keeps
        int32_t  view_end; // her view end (-1: no view spans)
    };

    // Before a copy is made: the close ones earlier acts left go, then room by count (the one eviction order).
    // A count already full of learn pins keeps one more until they are learned or released.
    template <typename OnErase>
    void make_room(const admission & a, OnErase on_erase) {
        int64_t last = -1;
        for (auto it = cks.begin(); it != cks.end();) {
            if (it->id_task != a.id_task && it->pin == 0 && last >= 0 && it->n_tokens <= last + a.step) {
                on_erase(*it);
                it = cks.erase(it);
                continue;
            }
            last = it->n_tokens;
            ++it;
        }
        while (cks.size() >= a.capacity) {
            auto v = victim(a.view_end);
            if (v == cks.end()) {
                break;
            }
            on_erase(*v);
            cks.erase(v);
        }
    }

    // The copy made: kept, pinned when `pin` is nonzero, and her view end supersedes the checkpoints earlier acts
    // left on this prefix (her head and every pin stay). Returns the stored checkpoint.
    template <typename OnErase>
    const C & keep(C && made, const admission & a, uint64_t pin, OnErase on_erase) {
        made.pin = pin;
        cks.push_back(std::move(made));
        const C & kept = cks.back();
        if (a.view_end >= 0 && kept.n_tokens == a.view_end) {
            const C * head = &cks.front();
            for (auto it = cks.begin(); it != cks.end();) {
                if (&*it != head && &*it != &kept && it->id_task != a.id_task && it->pin == 0) {
                    on_erase(*it);
                    it = cks.erase(it);
                    continue;
                }
                ++it;
            }
        }
        return cks.back();
    }

    // ---- the host budget --------------------------------------------------------------------------------------
    // One checkpoint out, in the one eviction order; the bytes it held (0: nothing left that may go).
    template <typename OnErase>
    size_t evict_one(int32_t view_end, OnErase on_erase) {
        auto v = victim(view_end);
        if (v == cks.end()) {
            return 0;
        }
        const size_t freed = v->size();
        on_erase(*v);
        cks.erase(v);
        return freed;
    }

    // ---- her prompt moved -------------------------------------------------------------------------------------
    // A resume whose prompt diverged at `pos_next`: a checkpoint past it no longer describes her sequence, pinned
    // or not (its turn cannot be learned from this sequence any more).
    template <typename OnErase>
    size_t drop_past_position(int32_t pos_next, OnErase on_erase) {
        size_t n = 0;
        for (auto it = cks.begin(); it != cks.end();) {
            if (it->pos_max > pos_next) {
                on_erase(*it);
                it = cks.erase(it);
                ++n;
                continue;
            }
            ++it;
        }
        return n;
    }

    // Her sequence cut back to its first `n_tokens` (the slot learn's restore): what lies past it goes.
    size_t cut_back_to(int64_t n_tokens) {
        const size_t before = cks.size();
        cks.remove_if([&](const C & c) { return c.n_tokens > n_tokens; });
        return before - cks.size();
    }

    // ---- learn pins ---------------------------------------------------------------------------------------------
    // Pin `pin` released (an abandoned turn): the checkpoint stays as an ordinary one. True when one held it.
    bool release(uint64_t pin) {
        for (auto & c : cks) {
            if (pin != 0 && c.pin == pin) {
                c.pin = 0;
                return true;
            }
        }
        return false;
    }

    // The turn ending at `n_tokens` was learned: its pin is done.
    void learned_at(int64_t n_tokens) {
        for (auto & c : cks) {
            if (c.pin != 0 && c.n_tokens == n_tokens) {
                c.pin = 0;
            }
        }
    }

    // ---- whole sets (a page restored, a mind back from flash, a slot emptied) ----------------------------------
    void      replace(list_type restored) { cks = std::move(restored); }
    list_type take() { list_type out; out.swap(cks); return out; }
    void      clear() { list_type().swap(cks); }

private:
    list_type cks;

    // The one eviction order. Never a pinned checkpoint. First the oldest that is neither her head (with view
    // spans, her restore point when her next prompt diverges before her view end) nor her newest (where her next
    // act resumes: her context only appends); then her head; her newest last. end(): nothing may go.
    typename list_type::iterator victim(int32_t view_end) {
        auto newest = cks.end();
        for (auto it = cks.begin(); it != cks.end(); ++it) {
            if (newest == cks.end() || it->n_tokens > newest->n_tokens) {
                newest = it;
            }
        }
        const auto head = view_end >= 0 ? cks.begin() : cks.end();
        for (auto it = cks.begin(); it != cks.end(); ++it) {
            if (it != head && it != newest && it->pin == 0) {
                return it;
            }
        }
        if (head != cks.end() && head != newest && head->pin == 0) {
            return head;
        }
        if (newest != cks.end() && newest->pin == 0) {
            return newest;
        }
        return cks.end();
    }
};
