#pragma once

// Continuum fork: the engine-side module, the C++ counterpart of continuum's Rust ServiceModule
// (CBAR-SUBSTRATE-ARCHITECTURE.md). One owner per concern: it owns its thread, takes work as
// move-only jobs on its queue, and publishes its state as an immutable snapshot that readers copy
// without ever waiting on its work. It stops by token, wakes only on work or stop (never a poll or a
// timer), refuses typed, and measures itself. The engine's off-thread concerns (the checkpoint copier,
// per-sequence sampling, the host pool's report) are modules of this one shape, never hand-built
// threads.
//
// Stop CANCELS: the job in flight finishes; every job still queued completes at once with
// module_refusal::cancelled (a typed result in its future, never a broken promise) and is destroyed
// there, so whatever it captured (pool handles: host_bytes) is released by the same RAII that charges
// it. An owner that needs a drain waits on its futures before stopping. The queue is bounded by a
// capacity its OWNER supplies from what it owns (e.g. the engine's slots), never a constant here; a
// full queue refuses typed. Job payloads that hold memory carry pool-charged handles, never raw
// copies, so pending work is never an uncharged second pool.

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <deque>
#include <expected>
#include <future>
#include <memory>
#include <mutex>
#include <stop_token>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>

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
    stopping,  // the module is shutting down: the job was not queued
    full,      // the owner's capacity is reached: the job was not queued
    cancelled, // queued, then the module stopped before it ran
};

// What every module measures about itself, published with its own snapshot.
struct module_stats {
    uint64_t queued      = 0; // jobs waiting now
    uint64_t done        = 0; // jobs finished since start
    uint64_t busy_us     = 0; // time spent in jobs since start
    uint64_t max_job_us  = 0; // the longest single job since start
    uint64_t cancelled   = 0; // queued jobs completed as cancelled by a stop
    uint64_t refused     = 0; // submissions refused (stopping or full)
};

// A module that owns one worker thread. `Snapshot` is what it tells the rest of the engine; the
// module's jobs read and publish it.
template <typename Snapshot>
class engine_module {
public:
    // `capacity`: how many jobs may wait, from what the owner owns (never zero).
    engine_module(std::string name, Snapshot initial, size_t capacity)
        : name_(std::move(name)), snapshot_(std::move(initial)), capacity_(std::max<size_t>(capacity, 1)),
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

    // What a submitted job's future yields: its result, or why it never ran.
    template <typename R>
    using outcome = std::expected<R, module_refusal>;

    // Queue `job`; its future carries the result, or module_refusal::cancelled if a stop dropped it.
    // Refused (typed) once the module is stopping or its queue is at capacity.
    template <typename Job>
    std::expected<std::future<outcome<std::invoke_result_t<Job>>>, module_refusal> submit(Job && job) {
        using R = std::invoke_result_t<Job>;
        auto entry  = std::make_unique<queued<R, std::decay_t<Job>>>(std::forward<Job>(job));
        auto future = entry->promise.get_future();
        {
            std::scoped_lock lock(mutex_);
            if (worker_.get_stop_token().stop_requested() || queue_.size() >= capacity_) {
                stats_.refused++;
                return std::unexpected(worker_.get_stop_token().stop_requested() ? module_refusal::stopping
                                                                                 : module_refusal::full);
            }
            queue_.push_back(std::move(entry));
            stats_.queued = queue_.size();
        }
        wake_.notify_one();
        return future;
    }

    std::shared_ptr<const Snapshot> snapshot() const { return snapshot_.read(); }

    module_stats stats() const {
        std::scoped_lock lock(mutex_);
        return stats_;
    }

protected:
    void publish(Snapshot next) { snapshot_.publish(std::move(next)); }

    // Stop taking work, let the job in flight finish, join the worker; every job still queued completes
    // as module_refusal::cancelled and is destroyed (its captures released). Idempotent. Every
    // module's destructor calls it before anything else.
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
    // One queued job, type-erased: it either runs (fulfilling its promise) or is cancelled (fulfilling
    // it with module_refusal::cancelled). Destroying it releases whatever it captured.
    struct job_base {
        virtual ~job_base()   = default;
        virtual void run()    = 0;
        virtual void cancel() = 0;
    };
    template <typename R, typename F>
    struct queued final : job_base {
        explicit queued(F f) : fn(std::move(f)) {}
        void run() override {
            if constexpr (std::is_void_v<R>) {
                fn();
                promise.set_value(outcome<R>{});
            } else {
                promise.set_value(outcome<R>(fn()));
            }
        }
        void cancel() override { promise.set_value(outcome<R>(std::unexpected(module_refusal::cancelled))); }
        F                           fn;
        std::promise<outcome<R>>    promise;
    };

    void run(std::stop_token stop) {
        while (true) {
            std::unique_ptr<job_base> job;
            {
                std::unique_lock lock(mutex_);
                // wakes on work or on stop: never a poll, never a timer. A stop wins over queued work:
                // queued jobs are cancelled by stop_and_join, never drained here.
                wake_.wait(lock, stop, [this] { return !queue_.empty(); });
                if (stop.stop_requested()) {
                    return;
                }
                job = std::move(queue_.front());
                queue_.pop_front();
                stats_.queued = queue_.size();
            }
            const auto started = std::chrono::steady_clock::now();
            job->run();
            job.reset(); // its captures are released before the next job is taken
            const auto us = (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - started).count();
            std::scoped_lock lock(mutex_);
            stats_.done++;
            stats_.busy_us += us;
            stats_.max_job_us = std::max(stats_.max_job_us, us);
        }
    }

    // After the worker joined: every job still queued completes as cancelled and is destroyed.
    void cancel_queued() {
        std::deque<std::unique_ptr<job_base>> dropped;
        {
            std::scoped_lock lock(mutex_);
            dropped.swap(queue_);
            stats_.cancelled += dropped.size();
            stats_.queued = 0;
        }
        for (auto & job : dropped) {
            job->cancel();
        }
    }

    std::string                             name_;
    snapshot_cell<Snapshot>                 snapshot_;
    mutable std::mutex                      mutex_;
    std::condition_variable_any             wake_;
    std::deque<std::unique_ptr<job_base>>   queue_;
    module_stats                            stats_;
    size_t                                  capacity_;
    std::jthread                            worker_; // last: started after every member it uses exists
};
