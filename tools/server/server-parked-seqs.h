#pragma once

// Continuum fork: checkpoints as PARKED SEQUENCES, by id, never serialized (CONTINUUM-ENGINE 4.7.11).
//
// A context provisioned with n_seq_max = slots + parked holds `parked` spare sequence ids. A checkpoint
// takes one: park copies the live sequence's partial state (its recurrent cell) to the parked id by
// reference (`seq_cp(live, parked, 0, 0)`: an empty attention range moves no KV cells), restore copies
// it back (`seq_cp(parked, live, 0, 0)`) before the caller trims attention by position, and releasing
// it frees the cell (`seq_rm(parked, -1, -1)`) and returns the id. No bytes cross to the host.
//
// The ids are handles: a parked_seq is reference counted; the last handle frees the cell and returns
// the id, so every erase, clear and cache eviction at every call site releases by construction, and a
// step frame (server-step-frame.h) that retains a handle keeps the cell until the step retires.
//
// The memory operations are injected (`parked_seq_ops`), so tests/test-server-parked-seqs.cpp proves
// the ownership without a model; the server binds them to llama_memory_seq_cp / llama_memory_seq_rm.

#include <cstdint>
#include <expected>
#include <functional>
#include <memory>
#include <mutex>
#include <vector>

using parked_seq_id = int32_t;

// The memory operations a parked sequence needs (llama_memory_seq_cp / seq_rm in the server).
struct parked_seq_ops {
    std::function<void(int32_t src, int32_t dst)> copy_partial; // seq_cp(src, dst, 0, 0)
    std::function<void(int32_t seq)>              free_all;     // seq_rm(seq, -1, -1)
};

enum class parked_seq_refusal : uint8_t {
    exhausted, // every provisioned parked id is in use: the caller evicts one (by value) or skips
};

class parked_seq_pool;

// One parked checkpoint. Immutable once parked; the last handle frees it.
class parked_seq {
public:
    parked_seq(const parked_seq &)             = delete;
    parked_seq & operator=(const parked_seq &) = delete;
    ~parked_seq();

    parked_seq_id id() const { return id_; }

    // Copy the parked partial state into `live` (the caller then trims attention past the checkpoint).
    void restore_into(int32_t live) const;

private:
    friend class parked_seq_pool;
    parked_seq(std::weak_ptr<parked_seq_pool> pool, parked_seq_id id) : pool_(std::move(pool)), id_(id) {}

    std::weak_ptr<parked_seq_pool> pool_;
    parked_seq_id                  id_;
};

using parked_seq_handle = std::shared_ptr<const parked_seq>;

class parked_seq_pool : public std::enable_shared_from_this<parked_seq_pool> {
public:
    // Ids [first, first + count) are the parked range (first = the number of live slots).
    static std::shared_ptr<parked_seq_pool> make(parked_seq_ops ops, int32_t first, int32_t count) {
        return std::shared_ptr<parked_seq_pool>(new parked_seq_pool(std::move(ops), first, count));
    }

    // Park the live sequence's partial state under a free id.
    std::expected<parked_seq_handle, parked_seq_refusal> park(int32_t live) {
        parked_seq_id id;
        {
            std::scoped_lock lock(mutex_);
            if (free_.empty()) {
                refused_++;
                return std::unexpected(parked_seq_refusal::exhausted);
            }
            id = free_.back();
            free_.pop_back();
            parked_++;
        }
        ops_.copy_partial(live, id);
        return parked_seq_handle(new parked_seq(weak_from_this(), id));
    }

    int32_t  capacity() const { return count_; }
    int32_t  in_use()   const { std::scoped_lock lock(mutex_); return count_ - (int32_t) free_.size(); }
    uint64_t parked()   const { std::scoped_lock lock(mutex_); return parked_; }
    uint64_t refused()  const { std::scoped_lock lock(mutex_); return refused_; }

private:
    friend class parked_seq;
    parked_seq_pool(parked_seq_ops ops, int32_t first, int32_t count) : ops_(std::move(ops)), count_(count) {
        for (int32_t i = count - 1; i >= 0; --i) {
            free_.push_back(first + i); // lowest id handed out first
        }
    }

    void release(parked_seq_id id) {
        ops_.free_all(id); // the cell is freed before the id can be handed out again
        std::scoped_lock lock(mutex_);
        free_.push_back(id);
    }

    parked_seq_ops             ops_;
    int32_t                    count_;
    mutable std::mutex         mutex_;
    std::vector<parked_seq_id> free_;
    uint64_t                   parked_  = 0;
    uint64_t                   refused_ = 0;
};

inline parked_seq::~parked_seq() {
    if (auto pool = pool_.lock()) {
        pool->release(id_);
    }
}

inline void parked_seq::restore_into(int32_t live) const {
    if (auto pool = pool_.lock()) {
        pool->ops_.copy_partial(id_, live);
    }
}
