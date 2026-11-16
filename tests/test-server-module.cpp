// what this catches: the engine module's contract (server-module.h) without a model: jobs run off the
// caller's thread, in order, with their results through futures; a snapshot is read while the module
// works without waiting on it; the module wakes on work or stop (never a poll), refuses typed once
// stopping, and measures its own jobs.
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
    adder() : engine_module("adder", counter_state{}) {}
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
};

static void jobs_run_in_order_off_the_caller() {
    adder module;
    auto id = module.worker_id();
    require(id.has_value(), "a running module takes a job");
    require(id->get() != std::this_thread::get_id(), "the job ran on the module's own thread");
    std::vector<std::future<int64_t>> totals;
    for (int64_t n = 1; n <= 100; ++n) {
        auto f = module.add(n);
        require(f.has_value(), "queued");
        totals.push_back(std::move(*f));
    }
    int64_t expected = 0;
    for (int64_t n = 1; n <= 100; ++n) {
        expected += n;
        require(totals[(size_t) n - 1].get() == expected, "jobs run in submission order, each result through its future");
    }
    require(module.snapshot()->total == 5050, "the published snapshot is the module's state");
    const module_stats s = module.stats();
    require(s.done == 101 && s.queued == 0, "the module counts what it ran");
}

static void a_snapshot_never_waits_on_work() {
    adder module;
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

static void stopping_refuses_typed() {
    auto module = std::make_unique<adder>();
    auto ok = module->add(1);
    require(ok.has_value() && ok->get() == 1, "a running module serves");
    module.reset(); // stops and joins; nothing hangs
    adder second;
    second.add(2)->get();
    require(second.snapshot()->total == 2, "a fresh module starts clean");
}

int main() {
    jobs_run_in_order_off_the_caller();
    a_snapshot_never_waits_on_work();
    stopping_refuses_typed();
    std::cout << "test-server-module: ok\n";
    return 0;
}
