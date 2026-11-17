#pragma once

#include "llama.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>

// A persona's token history: one contiguous, reserved buffer of her token ids, owned with her blocks
// (CONTINUUM-ENGINE 4.14, "the persona is one contiguous structure; a request appends token ids").
//
// Her tokens are what her blocks hold, in order: what she was served and what she generated, exactly.
// A request never re-sends them; it appends only its new ids, in one validated transaction:
//   - its header names her persona, her epoch, the frontier it expects her history to end at, and the
//     request's own identity (a 128-bit id the caller mints);
//   - the owner validates all of it, then writes the delta into capacity already reserved: never an
//     allocation or a copy of what she holds (growth is the admitted owner's, measured separately);
//   - the same request again returns its receipt, unchanged; a different delta under the same identity is
//     refused; a lost response is never a reason to append twice.
// Plain arrays and memcpy: no containers, no encoding.

struct llama_persona_request_id {
    uint64_t hi = 0;
    uint64_t lo = 0;

    bool operator==(const llama_persona_request_id & o) const { return hi == o.hi && lo == o.lo; }
    bool empty() const { return hi == 0 && lo == 0; }
};

// A fixed-width header: what a request asserts about her history before its delta is appended.
struct llama_persona_append {
    uint64_t                 persona  = 0;
    uint64_t                 epoch    = 0;
    uint64_t                 frontier = 0; // where the request expects her history to end now
    llama_persona_request_id request;
};

enum class llama_persona_append_status {
    appended,      // the delta is now hers
    replayed,      // this request was already appended: its receipt, nothing written
    wrong_persona, // the header names another persona
    wrong_epoch,   // her epoch moved (a new view started): the request is stale
    moved,         // her history does not end where the request expects
    changed,       // the same request identity with a different delta: refused, never overwritten
    no_room,       // her reserved capacity cannot hold the delta: her owner grows it first
    empty_request, // a request with no identity cannot be made idempotent
};

inline const char * llama_persona_append_status_name(llama_persona_append_status s) {
    switch (s) {
        case llama_persona_append_status::appended:      return "appended";
        case llama_persona_append_status::replayed:      return "replayed";
        case llama_persona_append_status::wrong_persona: return "wrong_persona";
        case llama_persona_append_status::wrong_epoch:   return "wrong_epoch";
        case llama_persona_append_status::moved:         return "moved";
        case llama_persona_append_status::changed:       return "changed";
        case llama_persona_append_status::no_room:       return "no_room";
        case llama_persona_append_status::empty_request: return "empty_request";
    }
    return "unknown";
}

struct llama_persona_receipt {
    llama_persona_append_status status = llama_persona_append_status::empty_request;
    uint64_t from = 0; // where the delta starts in her history
    uint64_t to   = 0; // her frontier after it
};

class llama_persona_tokens {
public:
    llama_persona_tokens(uint64_t persona, uint64_t epoch, size_t capacity)
        : persona_(persona), epoch_(epoch), capacity_(capacity), ids_(new llama_token[capacity]) {}

    uint64_t persona() const { return persona_; }
    uint64_t epoch() const { return epoch_; }
    uint64_t frontier() const { return n_; }
    size_t capacity() const { return capacity_; }
    const llama_token * data() const { return ids_.get(); }

    // Her generated tokens join her history as she samples them: the same reserved tail, one id at a time.
    bool generated(llama_token id) {
        if (n_ >= capacity_) {
            return false;
        }
        ids_[n_++] = id;
        return true;
    }

    // One validated transaction: check her identity, epoch and frontier, then write the delta into the
    // reserved tail. `delta` is read in place (the caller's buffer, valid for the call).
    llama_persona_receipt append(const llama_persona_append & h, const llama_token * delta, size_t n_delta) {
        llama_persona_receipt r;
        if (h.request.empty()) {
            return r;
        }
        if (h.persona != persona_) {
            r.status = llama_persona_append_status::wrong_persona;
            return r;
        }
        if (h.epoch != epoch_) {
            r.status = llama_persona_append_status::wrong_epoch;
            return r;
        }
        // The same request again: its receipt if it carries the same delta, refused if it does not.
        if (has_last_ && h.request == last_.request) {
            const bool same = h.frontier == last_from_ && n_delta == last_.to - last_.from &&
                              std::memcmp(ids_.get() + last_.from, delta, n_delta * sizeof(llama_token)) == 0;
            r        = last_;
            r.status = same ? llama_persona_append_status::replayed : llama_persona_append_status::changed;
            return r;
        }
        if (h.frontier != n_) {
            r.status = llama_persona_append_status::moved;
            r.to     = n_;
            return r;
        }
        if (n_delta > capacity_ - n_) {
            r.status = llama_persona_append_status::no_room;
            r.to     = n_;
            return r;
        }
        std::memcpy(ids_.get() + n_, delta, n_delta * sizeof(llama_token));
        r.status   = llama_persona_append_status::appended;
        r.from     = n_;
        r.to       = n_ + n_delta;
        n_         = r.to;
        last_from_ = h.frontier;
        last_      = { r.status, r.from, r.to };
        last_.request = h.request;
        has_last_  = true;
        return r;
    }

private:
    struct last_append : llama_persona_receipt {
        llama_persona_request_id request;
        last_append() = default;
        last_append(llama_persona_append_status s, uint64_t f, uint64_t t) { status = s; from = f; to = t; }
    };

    uint64_t                       persona_;
    uint64_t                       epoch_;
    size_t                         capacity_;
    std::unique_ptr<llama_token[]> ids_;
    uint64_t                       n_         = 0;
    bool                           has_last_  = false;
    uint64_t                       last_from_ = 0;
    last_append                    last_;
};
