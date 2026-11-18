#pragma once

#include "llama.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <new>
#include <memory>

// A persona's token history: one contiguous, reserved buffer of her token ids, owned with her blocks
// (CONTINUUM-ENGINE 4.14, "the persona is one contiguous structure; a request appends token ids").
//
// Her tokens are what her blocks hold, in order: what she was served and what she generated, exactly.
// A request never re-sends them; it appends only its new ids, in one validated transaction:
//   - its header names her persona, her epoch, her request sequence (1, 2, 3, ... minted by her one
//     caller), the frontier it expects her history to end at, and the request's own 128-bit identity;
//   - the owner validates all of it, then writes the delta into capacity already reserved: never an
//     allocation or a copy of what she holds (growth is the admitted owner's, measured separately);
//   - every committed append keeps its receipt, indexed by its sequence: a retry of ANY committed
//     sequence, however delayed or reordered, gets its original receipt and writes nothing; the same
//     sequence with another identity or delta is refused; a lost response is never a reason to append twice.
// Raw contract: `delta` may be null only when `n_delta` is 0 (an empty append still commits its sequence
// and receipt); capacity is checked without overflow. Plain arrays and memcpy: no encoding.

struct llama_persona_request_id {
    uint64_t hi = 0;
    uint64_t lo = 0;

    bool operator==(const llama_persona_request_id & o) const { return hi == o.hi && lo == o.lo; }
    bool empty() const { return hi == 0 && lo == 0; }
};

// A fixed-width header: what a request asserts about her history before its delta is appended.
struct llama_persona_append_request {
    uint64_t                 persona  = 0;
    uint64_t                 epoch    = 0;
    uint64_t                 seq      = 0; // her request sequence in this epoch, from 1
    uint64_t                 frontier = 0; // where the request expects her history to end now
    llama_persona_request_id request;
};

enum class llama_persona_append_status {
    appended,      // the delta is now hers
    replayed,      // this sequence was already appended by this request: its receipt, nothing written
    wrong_persona, // the header names another persona
    wrong_epoch,   // her epoch moved (a new view started): the request is stale
    out_of_order,  // not her next sequence: a gap, or a sequence she never committed
    moved,         // her next sequence, but her history does not end where the request expects
    changed,       // a committed sequence with another identity or delta: refused, never overwritten
    no_room,       // her reserved tokens or receipts cannot hold this append: her owner grows them first
    invalid,       // no identity, no sequence, or a null delta with a length
};

inline const char * llama_persona_append_status_name(llama_persona_append_status s) {
    switch (s) {
        case llama_persona_append_status::appended:      return "appended";
        case llama_persona_append_status::replayed:      return "replayed";
        case llama_persona_append_status::wrong_persona: return "wrong_persona";
        case llama_persona_append_status::wrong_epoch:   return "wrong_epoch";
        case llama_persona_append_status::out_of_order:  return "out_of_order";
        case llama_persona_append_status::moved:         return "moved";
        case llama_persona_append_status::changed:       return "changed";
        case llama_persona_append_status::no_room:       return "no_room";
        case llama_persona_append_status::invalid:       return "invalid";
    }
    return "unknown";
}

struct llama_persona_receipt {
    llama_persona_append_status status = llama_persona_append_status::invalid;
    uint64_t seq  = 0;
    uint64_t from = 0; // where the delta starts in her history
    uint64_t to   = 0; // her frontier after it
};

class llama_persona_tokens {
public:
    // Her history is owned from the start and sized by nothing: it begins empty and her owner grows it ahead of
    // need, off the hot path (grow). An accepted append only writes into what has been grown; it never allocates.
    llama_persona_tokens(uint64_t persona, uint64_t epoch) : persona_(persona), epoch_(epoch) {}

    // Add room for `more_tokens` ids and `more_receipts` appends, keeping everything she holds. Called by her owner
    // between acts, never inside one. False on allocation failure, with her history unchanged.
    bool grow(size_t more_tokens, size_t more_receipts) {
        if (more_tokens > SIZE_MAX - capacity_ || more_receipts > SIZE_MAX - receipt_capacity_) {
            return false;
        }
        std::unique_ptr<llama_token[]> ids(new (std::nothrow) llama_token[capacity_ + more_tokens]);
        std::unique_ptr<commit_record[]> committed(new (std::nothrow) commit_record[receipt_capacity_ + more_receipts]);
        if (!ids || !committed) {
            return false;
        }
        if (n_ > 0) {
            std::memcpy(ids.get(), ids_.get(), n_ * sizeof(llama_token));
        }
        for (uint64_t i = 0; i < n_committed_; ++i) {
            committed[i] = committed_[i];
        }
        ids_.swap(ids);
        committed_.swap(committed);
        capacity_ += more_tokens;
        receipt_capacity_ += more_receipts;
        return true;
    }

    uint64_t persona() const { return persona_; }
    uint64_t epoch() const { return epoch_; }
    uint64_t frontier() const { return n_; }
    uint64_t committed() const { return n_committed_; }
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

    // One validated transaction: check her identity, epoch, sequence and frontier, then write the delta
    // into the reserved tail. `delta` is read in place (the caller's buffer, valid for the call).
    llama_persona_receipt append(const llama_persona_append_request & h, const llama_token * delta, size_t n_delta) {
        llama_persona_receipt r;
        r.seq = h.seq;
        if (h.request.empty() || h.seq == 0 || (delta == nullptr && n_delta != 0)) {
            return r; // invalid
        }
        if (h.persona != persona_) {
            r.status = llama_persona_append_status::wrong_persona;
            return r;
        }
        if (h.epoch != epoch_) {
            r.status = llama_persona_append_status::wrong_epoch;
            return r;
        }
        // A committed sequence: its own receipt for the same request and delta, refused for anything else.
        if (h.seq <= n_committed_) {
            const commit_record & c = committed_[h.seq - 1];
            const bool same = c.request == h.request && c.expected == h.frontier && n_delta == c.to - c.from &&
                              (n_delta == 0 || std::memcmp(ids_.get() + c.from, delta, n_delta * sizeof(llama_token)) == 0);
            r.status = same ? llama_persona_append_status::replayed : llama_persona_append_status::changed;
            r.from   = c.from;
            r.to     = c.to;
            return r;
        }
        if (h.seq != n_committed_ + 1) {
            r.status = llama_persona_append_status::out_of_order;
            return r;
        }
        if (h.frontier != n_) {
            r.status = llama_persona_append_status::moved;
            r.to     = n_;
            return r;
        }
        // Both reservations are checked before anything is written: a refusal leaves her exactly as she was.
        if (n_delta > capacity_ - n_ || n_committed_ == receipt_capacity_) { // n_ <= capacity_ always: no overflow
            r.status = llama_persona_append_status::no_room;
            r.to     = n_;
            return r;
        }
        if (n_delta != 0) {
            std::memcpy(ids_.get() + n_, delta, n_delta * sizeof(llama_token));
        }
        r.status = llama_persona_append_status::appended;
        r.from   = n_;
        r.to     = n_ + n_delta;
        committed_[n_committed_++] = { h.request, h.frontier, r.from, r.to };
        n_ = r.to;
        return r;
    }

private:
    // One committed append, at index seq - 1: who appended it, what it expected, and where it landed.
    struct commit_record {
        llama_persona_request_id request;
        uint64_t expected;
        uint64_t from;
        uint64_t to;
    };

    uint64_t                       persona_;
    uint64_t                       epoch_;
    size_t                         capacity_ = 0;
    std::unique_ptr<llama_token[]> ids_;
    uint64_t                       n_ = 0;
    size_t                           receipt_capacity_ = 0;
    std::unique_ptr<commit_record[]> committed_;
    uint64_t                         n_committed_ = 0;
};
