#pragma once

// Continuum fork: a fixed set of lanes that run one step's independent jobs at once (CONTINUUM-ENGINE
// 4.7.14). Built for jobs of tens to hundreds of microseconds, every few milliseconds:
// - a job is claimed lock-free from one 64-bit word (run id << 32 | next index), so a lane finishing an
//   earlier run can never claim in a new one;
// - each lane has its own wake word: a run wakes only as many lanes as it has jobs beyond the caller's,
//   with an atomic notify (a futex: __ulock on macOS), never a mutex and condition variable;
// - the caller is a lane and never sleeps: it runs jobs, then spins on the done count for the last few
//   microseconds of the slowest lane (no syscall, no core clocking down before the next GPU submit);
// - an idle lane sleeps on its wake word and burns nothing.
// A job's exception is carried back to the caller.

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#if defined(__APPLE__)
#include <pthread/qos.h>
#endif

// The one OS-specific line: lanes run where the step's latency is decided (macOS: performance cores).
inline void server_lane_priority() {
#if defined(__APPLE__)
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif
}

class server_lanes {
public:
    // `spawn` starts one lane's thread (std::thread by default; a test injects a failure). If starting any
    // lane throws, the lanes already started are stopped and joined before the exception leaves, so no
    // joinable std::thread is ever destroyed (Codex on #129: that is std::terminate).
    using spawn_fn = std::function<std::thread(std::function<void()>)>;

    explicit server_lanes(size_t n_workers, const spawn_fn & spawn = [](std::function<void()> f) { return std::thread(std::move(f)); })
        : lanes(n_workers) {
        try {
            for (size_t i = 0; i < n_workers; ++i) {
                lanes[i].thread = spawn([this, i] { work(lanes[i]); });
            }
        } catch (...) {
            shutdown();
            throw;
        }
    }

    server_lanes(const server_lanes &)             = delete;
    server_lanes & operator=(const server_lanes &) = delete;

    ~server_lanes() { shutdown(); }

    size_t size() const { return lanes.size() + 1; }

    // Run fn(i) for every i in [0, n) across the caller and up to n - 1 lanes; return when all have run.
    // The first job exception is rethrown here.
    void run(size_t n, const std::function<void(size_t)> & fn) {
        if (n == 0) {
            return;
        }
        const size_t helpers = std::min(n - 1, lanes.size());
        if (helpers == 0) {
            for (size_t i = 0; i < n; ++i) {
                fn(i);
            }
            return;
        }
        done.store(0, std::memory_order_relaxed);
        failure = nullptr;
        // one immutable record per run, published BEFORE the run's claim word: a lane claims only when the
        // word's run id matches the record's, so a word from an earlier run can never pair with this
        // run's jobs or bounds (Codex on #125)
        run_record & rec = runs[++run_id % 2];
        rec.fn.store(&fn, std::memory_order_relaxed);
        rec.n.store(n, std::memory_order_relaxed);
        rec.id.store(run_id, std::memory_order_release);
        record.store(&rec, std::memory_order_release);
        next.store(run_id << 32, std::memory_order_release);
        for (size_t i = 0; i < helpers; ++i) {
            lanes[i].wake.fetch_add(1, std::memory_order_release);
            lanes[i].wake.notify_one();
        }
        take();
        while (done.load(std::memory_order_acquire) != n) {
            std::this_thread::yield(); // the slowest lane's last microseconds; the caller never sleeps
        }
        if (failure) {
            std::rethrow_exception(failure);
        }
    }

private:
    // stop every started lane, wake it, and join it; lanes never started have no thread
    void shutdown() {
        stop.store(true, std::memory_order_release);
        for (auto & lane : lanes) {
            lane.wake.fetch_add(1, std::memory_order_release);
            lane.wake.notify_one();
        }
        for (auto & lane : lanes) {
            if (lane.thread.joinable()) {
                lane.thread.join();
            }
        }
    }

    struct lane {
        std::thread           thread;
        std::atomic<uint64_t> wake{0};
    };

    // claim and run the current run's jobs until none are left
    void take() {
        uint64_t claim = next.load(std::memory_order_acquire);
        while (true) {
            // read after the claim word: a word of run r implies run r's record is published (it was
            // stored first, with release), so a mismatch can only mean the word is stale
            const run_record * run = record.load(std::memory_order_acquire);
            const size_t i = (size_t) (claim & 0xffffffffu);
            if (run == nullptr || (claim >> 32) != run->id.load(std::memory_order_acquire) ||
                i >= run->n.load(std::memory_order_relaxed)) {
                return; // no run, a stale word, or this run's jobs are all claimed
            }
            if (!next.compare_exchange_weak(claim, claim + 1, std::memory_order_acq_rel, std::memory_order_acquire)) {
                continue; // another lane claimed it, or a new run began: `claim` holds the current word
            }
            try {
                (*run->fn.load(std::memory_order_relaxed))(i);
            } catch (...) {
                std::lock_guard<std::mutex> lock(failure_mutex);
                if (!failure) {
                    failure = std::current_exception();
                }
            }
            done.fetch_add(1, std::memory_order_acq_rel);
            claim = next.load(std::memory_order_acquire);
        }
    }

    void work(lane & self) {
        server_lane_priority();
        // from 0, never from a load at start: a wake (or the destructor's) that came before this thread
        // ran is still seen; and stop is checked before every sleep (Codex on #125: a stop signalled before
        // the first wait was missed, and the join hung)
        uint64_t seen = 0;
        while (true) {
            if (stop.load(std::memory_order_acquire)) {
                return;
            }
            self.wake.wait(seen, std::memory_order_acquire);
            seen = self.wake.load(std::memory_order_acquire);
            if (stop.load(std::memory_order_acquire)) {
                return;
            }
            take();
        }
    }

    std::vector<lane>                                lanes;
    std::atomic<bool>                                stop{false};
    std::atomic<uint64_t>                            next{0};
    uint64_t                                         run_id = 0; // the caller's only
    std::atomic<size_t>                              done{0};
    struct run_record {
        std::atomic<uint64_t>                            id{0};
        std::atomic<const std::function<void(size_t)> *> fn{nullptr};
        std::atomic<size_t>                              n{0};
    };
    // two records alternate. A record is rewritten only two runs later, after its run's every job was
    // done; a lane that was descheduled across that can read mixed fields, but its claim word is then two
    // runs stale and its compare-exchange fails, so it never runs a job. The fields are atomic so that
    // read is never a data race.
    run_record                                       runs[2];
    std::atomic<const run_record *>                  record{nullptr};
    std::mutex                                       failure_mutex; // only when a job throws
    std::exception_ptr                               failure;
};
