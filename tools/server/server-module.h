#pragma once

// Continuum fork: the engine-side module, the C++ counterpart of continuum's Rust ServiceModule
// (CBAR-SUBSTRATE-ARCHITECTURE.md), shaped like a game engine's job system, not a promise library.
//
// One owner per concern: it owns its thread and a FIXED ring of jobs sized by its owner, and publishes
// its state as an immutable snapshot readers copy without waiting on its work. A job writes its result
// into storage its OWNER already owns (a frame slot, the arena, a field of the owner); nothing is
// allocated to carry a result back. Completion is a monotonic counter: `submit` returns a ticket (the
// job's sequence number), `completed()` is how many have finished, and a caller that must block uses
// `wait(ticket)`, an atomic wait on that counter (a futex on the platforms we run), never a future.
//
// A job is called exactly once: with `job_state::run` to do its work, or with `job_state::cancelled`
// when a stop dropped it, so it can mark its owner's slot. Stop cancels: the job in flight finishes,
// every queued job is called cancelled and destroyed (its captures released), and the counter still
// advances past every ticket, so no waiter waits forever. The worker sleeps until there is work or a
// stop: never a poll, never a timer. Jobs capture a pointer to their owner's storage, not payloads:
// memory-holding data is a pool-charged handle (host_bytes, parked_seq), never a copy.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <expected>
#include <functional>
#include <memory>
#include <mutex>
#include <stop_token>
#include <string>
#include <thread>
#include <utility>
#include <vector>

// An immutable value published by one writer and read by many: a reader takes a pointer, never a lock
// held across the writer's work. Swapping the pointer is the only critical section.
template <typename T>
class snapshot_cell {
public:
    explicit snapshot_cell(T initial) : current_(std::make_shared<const T>(std::move(initial))) {}

    std::shared_ptr<const T> read() const {
        std::scoped_lock lock(mutex_);
        return current_;
    }

    void publish(T next) {
        auto fresh = std::make_shared<const T>(std::move(next));
        std::scoped_lock lock(mutex_);
        current_.swap(fresh);
    }

private:
    mutable std::mutex       mutex_;
    std::shared_ptr<const T> current_;
};

// Why a module did not take a job.
enum class module_refusal : uint8_t {
    stopping, // the module is shutting down: the job was not queued
    full,     // the owner's capacity is reached: the job was not queued
};

// How a job is called: once, either to run or because a stop dropped it.
enum class job_state : uint8_t { run, cancelled };

// A job's sequence number; it is complete once completed() >= its ticket.
using job_ticket = uint64_t;

// What every module measures about itself.
struct module_stats {
    uint64_t queued     = 0; // jobs waiting now
    uint64_t done       = 0; // jobs run since start
    uint64_t cancelled  = 0; // queued jobs a stop dropped
    uint64_t refused    = 0; // submissions refused (stopping or full)
    uint64_t busy_us    = 0; // time spent in jobs since start
    uint64_t max_job_us = 0; // the longest single job since start
};

template <typename Snapshot>
class engine_module {
public:
    using job = std::function<void(job_state)>;

    // `capacity`: how many jobs may wait, from what the owner owns (never zero). The ring is allocated
    // once, here.
    engine_module(std::string name, Snapshot initial, size_t capacity)
        : name_(std::move(name)), snapshot_(std::move(initial)), ring_(std::max<size_t>(capacity, 1)),
          worker_([this](std::stop_token stop) { run(stop); }) {}

    engine_module(const engine_module &)             = delete;
    engine_module & operator=(const engine_module &) = delete;

    // A module's own destructor calls stop_and_join() first: its members die before this base's
    // destructor runs, so a job still running would otherwise touch them. Enforced here, loudly.
    virtual ~engine_module() {
        if (worker_.joinable()) {
            std::fprintf(stderr, "engine_module '%s': the derived destructor must call stop_and_join() first\n", name_.c_str());
            std::abort();
        }
    }

    const std::string & name() const { return name_; }

    // Queue `j`; its ticket is complete once completed() reaches it.
    std::expected<job_ticket, module_refusal> submit(job j) {
        job_ticket ticket;
        {
            std::scoped_lock lock(mutex_);
            if (worker_.get_stop_token().stop_requested()) {
                stats_.refused++;
                return std::unexpected(module_refusal::stopping);
            }
            if (count_ == ring_.size()) {
                stats_.refused++;
                return std::unexpected(module_refusal::full);
            }
            ring_[(head_ + count_) % ring_.size()] = std::move(j);
            count_++;
            ticket = ++submitted_;
            stats_.queued = count_;
        }
        wake_.notify_one();
        return ticket;
    }

    // How many jobs have finished (run or cancelled), in submission order.
    uint64_t completed() const { return completed_.load(std::memory_order_acquire); }

    // Block until `ticket` is complete: an atomic wait on the completion counter.
    void wait(job_ticket ticket) const {
        for (uint64_t seen = completed(); seen < ticket; seen = completed()) {
            completed_.wait(seen, std::memory_order_acquire);
        }
    }

    std::shared_ptr<const Snapshot> snapshot() const { return snapshot_.read(); }

    module_stats stats() const {
        std::scoped_lock lock(mutex_);
        return stats_;
    }

protected:
    void publish(Snapshot next) { snapshot_.publish(std::move(next)); }

    // Stop taking work, let the job in flight finish, join the worker; every job still queued is called
    // cancelled and destroyed, and the counter advances past it. Idempotent. Every module's destructor
    // calls it before anything else.
    void stop_and_join() {
        {
            std::scoped_lock lock(mutex_);
            worker_.request_stop();
        }
        wake_.notify_all();
        if (worker_.joinable()) {
            worker_.join();
        }
        cancel_queued();
    }

private:
    void finish_one() {
        completed_.fetch_add(1, std::memory_order_release);
        completed_.notify_all();
    }

    void run(std::stop_token stop) {
        while (true) {
            job j;
            {
                std::unique_lock lock(mutex_);
                // wakes on work or on stop. A stop wins over queued work: stop_and_join cancels it.
                wake_.wait(lock, stop, [this] { return count_ > 0; });
                if (stop.stop_requested()) {
                    return;
                }
                j = std::move(ring_[head_]);
                head_ = (head_ + 1) % ring_.size();
                count_--;
                stats_.queued = count_;
            }
            const auto started = std::chrono::steady_clock::now();
            j(job_state::run);
            j = nullptr; // its captures are released before completion is announced
            const auto us = (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - started).count();
            {
                std::scoped_lock lock(mutex_);
                stats_.done++;
                stats_.busy_us += us;
                stats_.max_job_us = std::max(stats_.max_job_us, us);
            }
            finish_one();
        }
    }

    // After the worker joined: every job still queued is called cancelled, destroyed, and counted.
    void cancel_queued() {
        std::vector<job> dropped;
        {
            std::scoped_lock lock(mutex_);
            dropped.reserve(count_);
            for (; count_ > 0; count_--) {
                dropped.push_back(std::move(ring_[head_]));
                head_ = (head_ + 1) % ring_.size();
            }
            stats_.cancelled += dropped.size();
            stats_.queued = 0;
        }
        for (auto & j : dropped) {
            j(job_state::cancelled);
            j = nullptr;
            finish_one();
        }
    }

    std::string                  name_;
    snapshot_cell<Snapshot>      snapshot_;
    mutable std::mutex           mutex_;
    std::condition_variable_any  wake_;
    std::vector<job>             ring_;
    size_t                       head_  = 0;
    size_t                       count_ = 0;
    job_ticket                   submitted_ = 0;
    std::atomic<uint64_t>        completed_{0};
    module_stats                 stats_;
    std::jthread                 worker_; // last: started after every member it uses exists
};
