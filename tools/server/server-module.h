#pragma once

// Continuum fork: the engine-side module, the C++ counterpart of continuum's Rust ServiceModule
// (CBAR-SUBSTRATE-ARCHITECTURE.md), shaped like a game engine's job system, not a promise library.
//
// One owner per concern: it owns its thread and a FIXED ring of jobs sized by its owner, and publishes
// its state as a snapshot readers copy with no lock. A job writes its result into storage its OWNER
// already owns (a frame slot, the arena, a field of the owner); nothing is allocated to carry a result
// back, and a job itself is stored inline in the ring (a fixed capture size, move-only, never a heap
// allocation). Completion is a monotonic counter: `submit` returns a ticket (the job's sequence number),
// `completed()` is how many have finished, and a caller that must block uses `wait(ticket)`, an atomic
// wait on that counter (a futex on the platforms we run), never a future.
//
// A job is called exactly once: with `job_state::run` to do its work, or with `job_state::cancelled`
// when a stop dropped it, so it can mark its owner's slot. A call that throws is contained: the fault is
// written as the job's typed terminal outcome into its owner's storage, the ticket still completes, and the
// worker lives on. Stop cancels:
// the job in flight finishes, every queued job is called cancelled and destroyed (its captures released),
// and the counter still advances past every ticket, so no waiter waits forever. The worker sleeps until
// there is work or a stop: never a poll, never a timer. Jobs capture a pointer to their owner's storage,
// not payloads: memory-holding data is a pool-charged handle (host_bytes, parked_seq), never a copy.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <mutex>
#include <new>
#include <stop_token>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

// A value published by one writer and read by many, with no lock (Codex on #135: a mutex let a reader wait on
// a publication). A seqlock over atomic words: the writer never waits on readers; a reader copies the words
// and retries when a publish overlapped its copy, so a reader is NOT wait-free: a writer descheduled mid-
// publish stalls readers until it resumes (they yield, never spin hot). A snapshot is counters and ids, so it
// is trivially copyable, enforced.
template <typename T>
class snapshot_cell {
    static_assert(std::is_trivially_copyable_v<T>, "a snapshot is counters and ids, copied by value");
    static constexpr size_t WORDS = (sizeof(T) + sizeof(uint64_t) - 1) / sizeof(uint64_t);

public:
    explicit snapshot_cell(const T & initial) { publish(initial); }

    T read() const {
        uint64_t buf[WORDS];
        for (uint32_t attempt = 0;; ++attempt) {
            if (attempt >= 4) {
                std::this_thread::yield(); // a publish keeps overlapping: let the writer finish
            }
            const uint64_t before = seq_.load(std::memory_order_acquire);
            if (before & 1) {
                continue; // a publish is mid-copy
            }
            for (size_t i = 0; i < WORDS; ++i) {
                buf[i] = words_[i].load(std::memory_order_relaxed);
            }
            std::atomic_thread_fence(std::memory_order_acquire);
            if (seq_.load(std::memory_order_relaxed) == before) {
                T out;
                std::memcpy(&out, buf, sizeof(T));
                return out;
            }
        }
    }

    // One writer at a time (the module's worker, or its owner before the worker runs).
    void publish(const T & next) {
        uint64_t buf[WORDS] = {};
        std::memcpy(buf, &next, sizeof(T));
        const uint64_t s = seq_.load(std::memory_order_relaxed);
        seq_.store(s + 1, std::memory_order_relaxed);
        std::atomic_thread_fence(std::memory_order_release);
        for (size_t i = 0; i < WORDS; ++i) {
            words_[i].store(buf[i], std::memory_order_relaxed);
        }
        seq_.store(s + 2, std::memory_order_release);
    }

private:
    std::atomic<uint64_t> seq_{0};
    std::atomic<uint64_t> words_[WORDS] = {};
};

// A move-only callable stored INLINE (Codex on #135: std::function requires a copyable target and may
// allocate). The capture must fit `Bytes`, checked at compile time; nothing is ever allocated to hold it.
template <size_t Bytes, typename... Args>
class inplace_job {
public:
    inplace_job() = default;

    template <typename F, typename D = std::decay_t<F>>
        requires(!std::is_same_v<D, inplace_job> && std::is_invocable_v<D &, Args...>)
    inplace_job(F && f) { // implicit, like the lambda handed to submit()
        static_assert(sizeof(D) <= Bytes, "a job captures pointers to its owner's storage, not payloads: it must fit");
        static_assert(alignof(D) <= alignof(std::max_align_t), "a job's capture must not be over-aligned");
        static_assert(std::is_nothrow_move_constructible_v<D>, "a job moves between ring slots without throwing");
        ::new (static_cast<void *>(store_)) D(std::forward<F>(f));
        ops_ = &ops_for<D>;
    }

    inplace_job(inplace_job && o) noexcept { take(o); }
    inplace_job & operator=(inplace_job && o) noexcept {
        if (this != &o) {
            reset();
            take(o);
        }
        return *this;
    }
    inplace_job(const inplace_job &)             = delete;
    inplace_job & operator=(const inplace_job &) = delete;
    ~inplace_job() { reset(); }

    explicit operator bool() const { return ops_ != nullptr; }
    void operator()(Args... args) { ops_->invoke(store_, args...); }

    void reset() {
        if (ops_) {
            ops_->destroy(store_);
            ops_ = nullptr;
        }
    }

private:
    struct ops {
        void (*invoke)(std::byte *, Args...);
        void (*move)(std::byte * to, std::byte * from) noexcept;
        void (*destroy)(std::byte *) noexcept;
    };
    template <typename D>
    static constexpr ops ops_for = {
        [](std::byte * s, Args... a) { (*std::launder(reinterpret_cast<D *>(s)))(a...); },
        [](std::byte * to, std::byte * from) noexcept {
            D * src = std::launder(reinterpret_cast<D *>(from));
            ::new (static_cast<void *>(to)) D(std::move(*src));
            src->~D();
        },
        [](std::byte * s) noexcept { std::launder(reinterpret_cast<D *>(s))->~D(); },
    };

    void take(inplace_job & o) noexcept {
        if (o.ops_) {
            o.ops_->move(store_, o.store_);
            ops_   = o.ops_;
            o.ops_ = nullptr;
        }
    }

    alignas(std::max_align_t) std::byte store_[Bytes];
    const ops * ops_ = nullptr;
};

// Why a module did not take a job.
enum class module_refusal : uint8_t {
    stopping, // the module is shutting down: the job was not queued
    full,     // the owner's capacity is reached: the job was not queued
    empty,    // the job holds no callable (default-constructed or moved-from): never queued, never called
};

// How a job is called: once, either to run or because a stop dropped it.
enum class job_state : uint8_t { run, cancelled };

// A job's sequence number, from 1; it is complete once completed() >= its ticket.
using job_ticket = uint64_t;

// What submit() did: queued under `ticket` (never 0), or refused for `refusal` (ticket 0). A plain typed result:
// the C++23 library result type is missing from grid toolchains (Ubuntu 22.04's libstdc++).
struct module_submit {
    job_ticket     ticket  = 0;
    module_refusal refusal = module_refusal::stopping; // meaningful only when !queued()
    bool queued() const { return ticket != 0; }
};

// What every module measures about itself.
struct module_stats {
    uint64_t queued     = 0; // jobs waiting now
    uint64_t done       = 0; // jobs run since start
    uint64_t cancelled  = 0; // queued jobs a stop dropped
    uint64_t refused    = 0; // submissions refused (stopping or full)
    uint64_t faulted    = 0; // calls (run or cancel) that threw: contained, their tickets still completed
    uint64_t busy_us    = 0; // time spent in jobs since start
    uint64_t max_job_us = 0; // the longest single job since start
};

// How a job ended, written by the module into storage its OWNER passed with the job (Codex on #135: one
// "last fault" slot lost earlier failures, and recording what() allocated a string inside the catch). Typed,
// one per job, no text: a name is rendered only at an export edge (job_outcome_name).
enum class job_outcome : uint8_t {
    pending        = 0, // not ended yet
    ran            = 1,
    run_faulted    = 2, // its run call threw: contained
    cancelled      = 3,
    cancel_faulted = 4, // its cancel call threw: contained
};

inline const char * job_outcome_name(job_outcome o) {
    switch (o) {
        case job_outcome::pending:        return "pending";
        case job_outcome::ran:            return "ran";
        case job_outcome::run_faulted:    return "run_faulted";
        case job_outcome::cancelled:      return "cancelled";
        case job_outcome::cancel_faulted: return "cancel_faulted";
    }
    return "unknown";
}

// `JobBytes`: the inline capture size of every job of this module, fixed at compile time.
template <typename Snapshot, size_t JobBytes = 64>
class engine_module {
public:
    using job = inplace_job<JobBytes, job_state>;

    // `capacity`: how many jobs may wait, from what the owner owns (never zero). The ring is allocated
    // once, here.
    engine_module(std::string name, const Snapshot & initial, size_t capacity)
        : name_(std::move(name)), snapshot_(initial), ring_(std::max<size_t>(capacity, 1)),
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

    // Queue `j`; its ticket is complete once completed() reaches it. `outcome`, when given, is the owner's
    // storage for how the job ended: the module writes it before the ticket completes, so it must stay valid
    // (not moved, not freed) until the owner has observed that completion (completed() >= ticket, or wait()).
    module_submit submit(job j, std::atomic<job_outcome> * outcome = nullptr) {
        job_ticket ticket;
        if (!j) {
            std::scoped_lock lock(mutex_);
            stats_.refused++;
            return module_submit{ 0, module_refusal::empty };
        }
        {
            std::scoped_lock lock(mutex_);
            if (worker_.get_stop_token().stop_requested()) {
                stats_.refused++;
                return module_submit{ 0, module_refusal::stopping };
            }
            if (count_ == ring_.size()) {
                stats_.refused++;
                return module_submit{ 0, module_refusal::full };
            }
            ticket = ++submitted_;
            ring_[(head_ + count_) % ring_.size()] = slot{ std::move(j), ticket, outcome };
            count_++;
            stats_.queued = count_;
        }
        wake_.notify_one();
        return module_submit{ ticket, module_refusal::stopping };
    }

    // A stop was requested: no job is taken from here on.
    bool stopping() const { return worker_.get_stop_token().stop_requested(); }

    // How many jobs have finished (run or cancelled), in submission order.
    uint64_t completed() const { return completed_.load(std::memory_order_acquire); }

    // Block until `ticket` is complete: an atomic wait on the completion counter.
    void wait(job_ticket ticket) const {
        for (uint64_t seen = completed(); seen < ticket; seen = completed()) {
            completed_.wait(seen, std::memory_order_acquire);
        }
    }

    // The last published state, copied with no lock.
    Snapshot snapshot() const { return snapshot_.read(); }

    module_stats stats() const {
        std::scoped_lock lock(mutex_);
        module_stats out = stats_;
        out.faulted      = faulted_.load(std::memory_order_relaxed);
        return out;
    }

protected:
    void publish(const Snapshot & next) { snapshot_.publish(next); }

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
    struct slot {
        job                       fn;
        job_ticket                ticket  = 0;
        std::atomic<job_outcome> * outcome = nullptr; // the owner's storage, or none
    };

    void finish_one() {
        completed_.fetch_add(1, std::memory_order_release);
        completed_.notify_all();
    }

    // A job's call, contained: whatever it throws ends it as a typed fault in its owner's storage (nothing is
    // allocated, nothing can throw here); the worker lives on.
    void call(slot & s, job_state state) {
        bool faulted = false;
        try {
            s.fn(state);
        } catch (...) {
            faulted = true;
        }
        if (faulted) {
            faulted_.fetch_add(1, std::memory_order_relaxed);
        }
        if (s.outcome != nullptr) {
            const job_outcome o = state == job_state::run ? (faulted ? job_outcome::run_faulted : job_outcome::ran)
                                                          : (faulted ? job_outcome::cancel_faulted : job_outcome::cancelled);
            s.outcome->store(o, std::memory_order_release);
        }
    }

    void run(std::stop_token stop) {
        while (true) {
            slot s;
            {
                std::unique_lock lock(mutex_);
                // wakes on work or on stop. A stop wins over queued work: stop_and_join cancels it.
                wake_.wait(lock, stop, [this] { return count_ > 0; });
                if (stop.stop_requested()) {
                    return;
                }
                s     = std::move(ring_[head_]);
                head_ = (head_ + 1) % ring_.size();
                count_--;
                stats_.queued = count_;
            }
            const auto started = std::chrono::steady_clock::now();
            call(s, job_state::run);
            s.fn.reset(); // its captures are released before completion is announced
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

    // After the worker joined, this thread is the ring's only user: each queued job is taken, called
    // cancelled (contained), destroyed and counted; no allocation, no lock held across a call.
    void cancel_queued() {
        while (true) {
            slot s;
            {
                std::scoped_lock lock(mutex_);
                if (count_ == 0) {
                    stats_.queued = 0;
                    return;
                }
                s     = std::move(ring_[head_]);
                head_ = (head_ + 1) % ring_.size();
                count_--;
                stats_.cancelled++;
            }
            call(s, job_state::cancelled);
            s.fn.reset();
            finish_one();
        }
    }

    std::string                  name_;
    snapshot_cell<Snapshot>      snapshot_;
    mutable std::mutex           mutex_;
    std::condition_variable_any  wake_;
    std::vector<slot>            ring_;
    size_t                       head_  = 0;
    size_t                       count_ = 0;
    job_ticket                   submitted_ = 0;
    std::atomic<uint64_t>        completed_{0};
    module_stats                 stats_;
    std::atomic<uint64_t>        faulted_{0};
    std::jthread                 worker_; // last: started after every member it uses exists
};
