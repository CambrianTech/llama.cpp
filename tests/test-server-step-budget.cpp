// what this catches: the frame budget (server-step-budget.h) without a model: build, GPU and gap
// time are attributed from the three timestamps; the stall between steps counts only while work is
// pending (an idle engine is not stalling); unsynchronized steps are never counted as idle or busy;
// decode and prefill-carrying steps are kept apart; a reader on another thread sees monotonic counters
// it can difference into a window; the idle share matches a hand-computed loop.
#include "../tools/server/server-step-budget.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <thread>

static void require(bool condition, const char * message) {
    if (!condition) {
        std::cerr << "FAILED: " << message << '\n';
        std::exit(1);
    }
}

// One step: build `b`, GPU `g`, then `s` of sampling before end(); returns the end time.
static int64_t step(step_budget & sb, int64_t t, step_kind k, uint32_t n, int64_t b, int64_t g, int64_t s, bool pending) {
    sb.begin(t, k, n);
    sb.submitted(t + b);
    sb.synced(t + b + g);
    sb.end(t + b + g + s, pending);
    return t + b + g + s;
}

static void a_decode_loop_is_attributed_exactly() {
    step_budget sb;
    int64_t t = 1000;
    // the measured M5 shape: ~20% build, ~55% GPU, the rest between steps
    for (int i = 0; i < 10; ++i) {
        t = step(sb, t, step_kind::decode, 4, 200, 550, 150, true);
        t += 100; // bookkeeping before the next llama_decode
    }
    sb.begin(t, step_kind::decode, 4); // closes the 10th step's stall
    auto d = sb.snapshot().decode;
    require(d.steps == 10 && d.tokens == 40, "steps and tokens counted");
    require(d.build_us == 2000 && d.gpu_us == 5500, "build and GPU attributed from the timestamps");
    require(d.gap_us == 10 * (150 + 100), "sampling after the sync and the wait for the next submit are the gap");
    require(std::fabs(*d.gpu_idle_share() - 4500.0 / 10000.0) < 1e-9, "idle share = (build + gap) / (build + gpu + gap) = 45%");
}

static void an_idle_engine_is_not_a_stall() {
    step_budget sb;
    int64_t t = step(sb, 0, step_kind::decode, 1, 10, 50, 5, /*pending=*/false);
    sb.begin(t + 60'000'000, step_kind::decode, 1); // a minute without requests
    require(sb.snapshot().decode.gap_us == 5, "only the in-step sampling counts; the idle minute does not");
}

static void unsynced_steps_are_neither_idle_nor_busy() {
    step_budget sb;
    sb.begin(0, step_kind::with_prefill, 512);
    sb.submitted(300);
    sb.end(310, true); // a prefill chunk without outputs: no sync
    sb.begin(400, step_kind::with_prefill, 512);
    auto p = sb.snapshot().with_prefill;
    require(p.build_us == 300, "its build is still CPU time the GPU waited on");
    require(p.gpu_us == 0 && p.gap_us == 0 && p.unsynced_us == 100, "the rest is unsynced: not idle, not busy");
    require(sb.snapshot().decode.steps == 0, "prefill-carrying steps are kept apart from decode-only steps");
}

static void a_reader_differences_a_window_while_the_loop_runs() {
    step_budget sb;
    std::atomic<bool> done{false};
    std::thread loop([&] {
        int64_t t = 0;
        for (int i = 0; i < 200000; ++i) {
            t = step(sb, t, step_kind::decode, 1, 2, 5, 1, true);
        }
        done = true;
    });
    step_budget_counters last{};
    while (!done) {
        auto now = sb.snapshot().decode;
        require(now.steps >= last.steps && now.gpu_us >= last.gpu_us, "counters only rise");
        auto window = now - last;
        require(window.steps <= 200000, "a window is a difference of two snapshots");
        last = now;
    }
    loop.join();
    require(sb.snapshot().decode.steps == 200000, "every step counted");
}

int main() {
    a_decode_loop_is_attributed_exactly();
    an_idle_engine_is_not_a_stall();
    unsynced_steps_are_neither_idle_nor_busy();
    a_reader_differences_a_window_while_the_loop_runs();
    std::cout << "test-server-step-budget: ok\n";
    return 0;
}
