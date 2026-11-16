// what this catches: the engine module's contract (server-module.h) without a model: jobs run off the
// caller's thread, in order, writing results into storage their owner owns (no promise, no future);
// completion is a monotonic ticket counter a caller can wait on; a snapshot is read while the module
// works without waiting on it; a stop CANCELS: the job in flight finishes, every queued job is called
// cancelled once and its captures released, and the counter advances past every ticket so no waiter
// hangs (regression for Codex's review of fork 84aba863e: queued jobs were drained after a stop); the
// owner's capacity bounds the fixed ring with a typed refusal; the module measures itself.
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

// A minimal module: it adds numbers off the caller's thread, writes each running total into the
// caller's slot, and publishes the total.
class adder final : public engine_module<counter_state> {
public:
    explicit adder(size_t capacity) : engine_module("adder", counter_state{}, capacity) {}
    ~adder() override { stop_and_join(); }

    // `out` is the owner's storage; the job writes there.
    auto add(int64_t n, int64_t * out) {
        return submit([this, n, out](job_state s) {
            if (s == job_state::cancelled) {
                *out = -1;
                return;
            }
            counter_state next = *snapshot();
            next.total += n;
            publish(next);
            *out = next.total;
        });
    }

    void stop() { stop_and_join(); }
};

static void jobs_write_into_owner_storage_in_order() {
    adder module(128);
    std::thread::id worker{};
    auto t = module.submit([&worker](job_state) { worker = std::this_thread::get_id(); });
    require(t.has_value(), "a running module takes a job");
    module.wait(*t);
    require(worker != std::this_thread::get_id(), "the job ran on the module's own thread");
    std::vector<int64_t> totals(100, 0); // the owner's storage, allocated once
    job_ticket last = 0;
    for (int64_t n = 1; n <= 100; ++n) {
        auto ticket = module.add(n, &totals[(size_t) n - 1]);
        require(ticket.has_value(), "queued");
        last = *ticket;
    }
    module.wait(last); // one wait covers every earlier ticket: completion is in order
    int64_t expected = 0;
    for (int64_t n = 1; n <= 100; ++n) {
        expected += n;
        require(totals[(size_t) n - 1] == expected, "jobs ran in submission order, each result in the owner's slot");
    }
    require(module.completed() == last, "the completion counter is the last ticket");
    require(module.snapshot()->total == 5050, "the published snapshot is the module's state");
    require(module.stats().done == 101 && module.stats().queued == 0, "the module counts what it ran");
}

static void a_snapshot_never_waits_on_work() {
    adder module(4);
    std::atomic<bool> release{false};
    auto blocker = module.submit([&](job_state) { while (!release.load()) { std::this_thread::yield(); } });
    require(blocker.has_value(), "queued");
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < 1000; ++i) {
        require(module.snapshot()->total == 0, "a reader sees the last published state");
    }
    require(std::chrono::steady_clock::now() - t0 < std::chrono::milliseconds(50), "1,000 snapshot reads during a long job never wait on it");
    release = true;
    module.wait(*blocker);
}

struct tracker {
    std::atomic<int> * alive;
    explicit tracker(std::atomic<int> * a) : alive(a) { ++*alive; }
    tracker(const tracker &) = delete;
    ~tracker() { --*alive; }
};

static void a_stop_cancels_queued_jobs_and_no_waiter_hangs() {
    adder module(64);
    std::atomic<bool> started{false}, release{false};
    std::atomic<int>  alive{0};
    int ran_value = 0;
    auto running = module.submit([&](job_state s) {
        if (s == job_state::cancelled) return;
        started = true;
        while (!release.load()) { std::this_thread::yield(); }
        ran_value = 7;
    });
    while (!started.load()) { std::this_thread::yield(); }
    std::vector<int> slots(3, 0); // the owner's storage for the queued jobs
    job_ticket last = 0;
    for (int i = 0; i < 3; ++i) {
        auto held = std::make_shared<tracker>(&alive); // a captured handle: released with its job
        auto t = module.submit([held, slot = &slots[(size_t) i]](job_state s) { *slot = s == job_state::run ? 1 : -1; });
        require(t.has_value(), "queued behind the running job");
        last = *t;
    }
    require(alive.load() == 3, "the queued jobs hold their captures");
    std::thread waiter([&] { module.wait(last); }); // must return even though these never run
    std::thread stopper([&] { module.stop(); });
    int probes = 0;
    while (true) {
        auto probe = module.submit([](job_state) {});
        if (!probe && probe.error() == module_refusal::full) {
            std::this_thread::yield(); // the ring filled before the stop landed: the stop is coming
            continue;
        }
        if (!probe) {
            require(probe.error() == module_refusal::stopping, "a stopping module refuses typed");
            break;
        }
        ++probes;
    }
    release = true;
    stopper.join();
    waiter.join();
    require(ran_value == 7, "the job in flight finishes");
    for (int v : slots) {
        require(v == -1, "every queued job is called cancelled once and marks its owner's slot");
    }
    require(alive.load() == 0, "a cancelled job's captures are released by the stop");
    require(module.stats().cancelled == (uint64_t) (3 + probes), "every cancelled job is counted");
    require(module.stats().refused >= 1, "refusals are counted");
    require(module.stats().done == 1, "nothing queued ran after the stop");
    require(module.completed() == *running + 3 + (uint64_t) probes, "the counter advanced past every ticket");
}

static void the_owners_capacity_bounds_the_ring() {
    adder module(2);
    std::atomic<bool> started{false}, release{false};
    auto running = module.submit([&](job_state) { started = true; while (!release.load()) { std::this_thread::yield(); } });
    while (!started.load()) { std::this_thread::yield(); }
    int64_t a = 0, b = 0, c = 0;
    require(module.add(1, &a).has_value() && module.add(2, &b).has_value(), "two may wait");
    auto third = module.add(3, &c);
    require(!third && third.error() == module_refusal::full, "a full ring refuses typed, never grows");
    require(module.stats().refused == 1, "the refusal is counted");
    release = true;
    module.wait(*running + 2);
    require(a == 1 && b == 3 && c == 0, "the queued jobs ran; the refused one never did");
}

int main() {
    jobs_write_into_owner_storage_in_order();
    a_snapshot_never_waits_on_work();
    a_stop_cancels_queued_jobs_and_no_waiter_hangs();
    the_owners_capacity_bounds_the_ring();
    std::cout << "test-server-module: ok\n";
    return 0;
}
