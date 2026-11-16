#pragma once

// Continuum fork: the engine-side module, the C++ counterpart of continuum's Rust ServiceModule
// (CBAR-SUBSTRATE-ARCHITECTURE.md). One owner per concern: it owns its thread, takes work as
// move-only jobs on its queue, and publishes its state as an immutable snapshot that readers copy
// without ever waiting on its work. It stops by token, wakes only on work or stop (never a poll or a
// timer), refuses typed, and measures itself. The engine's off-thread concerns (the checkpoint copier,
// per-sequence sampling, the host pool's report) are modules of this one shape, never hand-built
// threads.

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
    stopping, // the module is shutting down: the job was not queued
};

// What every module measures about itself, published with its own snapshot.
struct module_stats {
    uint64_t queued      = 0; // jobs waiting now
    uint64_t done        = 0; // jobs finished since start
    uint64_t busy_us     = 0; // time spent in jobs since start
    uint64_t max_job_us  = 0; // the longest single job since start
};

// A module that owns one worker thread. `Snapshot` is what it tells the rest of the engine; the
// module's jobs read and publish it.
template <typename Snapshot>
class engine_module {
public:
    engine_module(std::string name, Snapshot initial)
        : name_(std::move(name)), snapshot_(std::move(initial)),
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

    // Queue `job`; its future carries the result. Refused (typed) once the module is stopping.
    template <typename Job>
    std::expected<std::future<std::invoke_result_t<Job>>, module_refusal> submit(Job && job) {
        using R = std::invoke_result_t<Job>;
        std::packaged_task<R()> task(std::forward<Job>(job));
        auto future = task.get_future();
        {
            std::scoped_lock lock(mutex_);
            if (worker_.get_stop_token().stop_requested()) {
                return std::unexpected(module_refusal::stopping);
            }
            queue_.emplace_back([t = std::move(task)]() mutable { t(); });
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

    // Stop taking work, let the job in flight finish, join the worker. Queued jobs are dropped and
    // their futures broken. Idempotent. Every module's destructor calls it before anything else.
    void stop_and_join() {
        {
            std::scoped_lock lock(mutex_);
            worker_.request_stop();
        }
        wake_.notify_all();
        if (worker_.joinable()) {
            worker_.join();
        }
    }

private:
    void run(std::stop_token stop) {
        while (true) {
            std::packaged_task<void()> job;
            {
                std::unique_lock lock(mutex_);
                // wakes on work or on stop: never a poll, never a timer
                if (!wake_.wait(lock, stop, [this] { return !queue_.empty(); })) {
                    return; // stop requested with nothing taken
                }
                job = std::move(queue_.front());
                queue_.pop_front();
                stats_.queued = queue_.size();
            }
            const auto started = std::chrono::steady_clock::now();
            job();
            const auto us = (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - started).count();
            std::scoped_lock lock(mutex_);
            stats_.done++;
            stats_.busy_us += us;
            stats_.max_job_us = std::max(stats_.max_job_us, us);
        }
    }

    std::string                             name_;
    snapshot_cell<Snapshot>                 snapshot_;
    mutable std::mutex                      mutex_;
    std::condition_variable_any             wake_;
    std::deque<std::packaged_task<void()>>  queue_;
    module_stats                            stats_;
    std::jthread                            worker_; // last: started after every member it uses exists
};
