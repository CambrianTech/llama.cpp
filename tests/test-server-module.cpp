// what this catches: the engine module's contract (server-module.h) without a model: jobs run off the
// caller's thread, in order, with their results through futures; a snapshot is read while the module
// works without waiting on it; the module wakes on work or stop (never a poll); a stop CANCELS: the
// job in flight finishes, every queued job completes typed as cancelled (never drained, never a
// broken promise) and what it captured is released at once (regression for Codex's review of fork
// 84aba863e: the wait predicate kept draining queued jobs after a stop); the owner's capacity bounds
// the queue with a typed refusal; and the module measures itself.
#include "../tools/server/server-module.h"

#include <atomic>
#include <cstdlib>
#include <iostream>
#include <vector>

static void require(bool condition, const char * message) {
    if (!condition) {
        std::cerr << "FAILED: " << message << '\n';
        std::exit(1);
    }
}

struct counter_state {
    int64_t total = 0;
};

// A minimal module: it adds numbers off the caller's thread and publishes the running total.
class adder final : public engine_module<counter_state> {
public:
    explicit adder(size_t capacity) : engine_module("adder", counter_state{}, capacity) {}
    ~adder() override { stop_and_join(); }

    auto add(int64_t n) {
        return submit([this, n] {
            counter_state next = *snapshot();
            next.total += n;
            publish(next);
            return next.total;
        });
    }

    auto worker_id() {
        return submit([] { return std::this_thread::get_id(); });
    }

    void stop() { stop_and_join(); }
};

static void jobs_run_in_order_off_the_caller() {
    adder module(128);
    auto id = module.worker_id();
    require(id.has_value(), "a running module takes a job");
    require(id->get().value() != std::this_thread::get_id(), "the job ran on the module's own thread");
    std::vector<std::future<adder::outcome<int64_t>>> totals;
    for (int64_t n = 1; n <= 100; ++n) {
        auto f = module.add(n);
        require(f.has_value(), "queued");
        totals.push_back(std::move(*f));
    }
    int64_t expected = 0;
    for (int64_t n = 1; n <= 100; ++n) {
        expected += n;
        require(totals[(size_t) n - 1].get().value() == expected, "jobs run in submission order, each result through its future");
    }
    require(module.snapshot()->total == 5050, "the published snapshot is the module's state");
    const module_stats s = module.stats();
    require(s.done == 101 && s.queued == 0, "the module counts what it ran");
}

static void a_snapshot_never_waits_on_work() {
    adder module(4);
    std::atomic<bool> release{false};
    auto blocker = module.submit([&] { while (!release.load()) { std::this_thread::yield(); } return 0; });
    require(blocker.has_value(), "queued");
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < 1000; ++i) {
        require(module.snapshot()->total == 0, "a reader sees the last published state");
    }
    const auto waited = std::chrono::steady_clock::now() - t0;
    require(waited < std::chrono::milliseconds(50), "1,000 snapshot reads during a long job never wait on it");
    release = true;
    blocker->get();
}

struct tracker {
    std::atomic<int> * alive;
    explicit tracker(std::atomic<int> * a) : alive(a) { ++*alive; }
    tracker(const tracker &) = delete;
    ~tracker() { --*alive; }
};

static void a_stop_cancels_queued_jobs_typed_and_releases_them() {
    adder module(64);
    std::atomic<bool> started{false}, release{false};
    std::atomic<int>  alive{0};
    auto running = module.submit([&] { started = true; while (!release.load()) { std::this_thread::yield(); } return 7; });
    while (!started.load()) { std::this_thread::yield(); }
    std::vector<std::future<adder::outcome<int>>> queued;
    for (int i = 0; i < 3; ++i) {
        auto held = std::make_shared<tracker>(&alive); // a captured buffer: released when its job goes
        auto f = module.submit([held] { return 1; });
        require(f.has_value(), "queued behind the running job");
        queued.push_back(std::move(*f));
    }
    require(alive.load() == 3, "the queued jobs hold their captures");
    std::thread stopper([&] { module.stop(); });
    // the stop is requested before the running job ends: probe with a no-op until it is refused
    int probes = 0;
    while (true) {
        auto probe = module.submit([] { return 0; });
        if (!probe) {
            require(probe.error() == module_refusal::stopping, "a stopping module refuses typed");
            break;
        }
        ++probes;
    }
    release = true;
    stopper.join();
    require(running->get().value() == 7, "the job in flight finishes");
    for (auto & f : queued) {
        auto r = f.get();
        require(!r && r.error() == module_refusal::cancelled, "every queued job completes typed as cancelled, never runs");
    }
    require(alive.load() == 0, "a cancelled job's captures are released by the stop");
    require(module.stats().cancelled == (uint64_t) (3 + probes), "every cancelled job is counted");
    require(module.stats().done == 1, "nothing queued ran after the stop");
}

static void the_owners_capacity_bounds_the_queue() {
    adder module(2);
    std::atomic<bool> started{false}, release{false};
    auto running = module.submit([&] { started = true; while (!release.load()) { std::this_thread::yield(); } return 0; });
    while (!started.load()) { std::this_thread::yield(); }
    require(module.add(1).has_value() && module.add(2).has_value(), "two may wait");
    auto third = module.add(3);
    require(!third && third.error() == module_refusal::full, "a full queue refuses typed, never grows");
    require(module.stats().refused == 1, "the refusal is counted");
    release = true;
    running->get();
}

int main() {
    jobs_run_in_order_off_the_caller();
    a_snapshot_never_waits_on_work();
    a_stop_cancels_queued_jobs_typed_and_releases_them();
    the_owners_capacity_bounds_the_queue();
    std::cout << "test-server-module: ok\n";
    return 0;
}
