// what this catches: the decode loop's host timings (server-step-budget.h) without a model: the call,
// the sync wait and the time between steps are attributed from the three timestamps; time between
// steps counts only while work is pending (an idle engine is not stalling); an unsynchronized step is
// counted only as unsynced, never in the synchronized timings (regression for Codex's review of fork
// 6651dcb65: begin 0 / submitted 300 / end 310 once reported a share of 1.0); decode and
// prefill-carrying steps are kept apart; a reader on another thread sees monotonic counters it can
// difference into a window.
#include "../tools/server/server-step-budget.h"

#include <cstdlib>
#include <iostream>
#include <thread>

static void require(bool condition, const char * message) {
    if (!condition) {
        std::cerr << "FAILED: " << message << '\n';
        std::exit(1);
    }
}

// One synchronized step: call `c`, wait `w`, then `s` of sampling before end(); returns the end time.
static int64_t step(step_budget & sb, int64_t t, step_kind k, uint32_t n, int64_t c, int64_t w, int64_t s, bool pending) {
    sb.begin(t, k, n);
    sb.submitted(t + c);
    sb.synced(t + c + w);
    sb.end(t + c + w + s, pending);
    return t + c + w + s;
}

static void a_decode_loop_is_attributed_exactly() {
    step_budget sb;
    int64_t t = 1000;
    for (int i = 0; i < 10; ++i) {
        t = step(sb, t, step_kind::decode, 4, 200, 550, 150, true);
        t += 100; // bookkeeping before the next llama_decode
    }
    sb.begin(t, step_kind::decode, 4); // closes the 10th step's time between steps
    auto d = sb.snapshot().decode;
    require(d.steps == 10 && d.tokens == 40, "steps and tokens counted");
    require(d.decode_call_us == 2000 && d.sync_wait_us == 5500, "call and sync wait attributed from the timestamps");
    require(d.inter_step_us == 10 * (150 + 100), "sampling after the sync and the wait for the next call are between steps");
}

static void an_idle_engine_is_not_a_stall() {
    step_budget sb;
    int64_t t = step(sb, 0, step_kind::decode, 1, 10, 50, 5, /*pending=*/false);
    sb.begin(t + 60'000'000, step_kind::decode, 1); // a minute without requests
    require(sb.snapshot().decode.inter_step_us == 5, "only the in-step sampling counts; the idle minute does not");
}

static void an_unsynchronized_step_is_only_unsynced() {
    step_budget sb;
    sb.begin(0, step_kind::with_prefill, 512);
    sb.submitted(300);
    sb.end(310, true); // a prefill chunk without outputs: no sync
    sb.begin(400, step_kind::with_prefill, 512);
    auto p = sb.snapshot().with_prefill;
    require(p.steps == 0 && p.decode_call_us == 0 && p.sync_wait_us == 0 && p.inter_step_us == 0,
            "nothing of an unsynchronized step enters the synchronized timings");
    require(p.unsynced_steps == 1 && p.unsynced_us == 400, "its whole span, to the next call, is unsynced");
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
        require(now.steps >= last.steps && now.sync_wait_us >= last.sync_wait_us, "counters only rise");
        auto window = now - last;
        require(window.steps <= 200000, "a window is a difference of two snapshots");
        last = now;
    }
    loop.join();
    require(sb.snapshot().decode.steps == 200000, "every step counted");
}

// what this catches: llama_decode returning non-zero (the loop retries) or a throw left a begin open, and
// either the next begin overwrote it (the cost vanished) or closed it late (the idle time after it counted)
static void a_failed_attempt_is_closed_where_it_fails() {
    step_budget sb;
    sb.begin(0, step_kind::with_prefill, 512);
    sb.submitted(700);
    sb.abandon(700);                                   // returned non-zero: closed at the retry decision
    sb.abandon(800);                                   // exactly once
    auto p = sb.snapshot().with_prefill;
    require(p.failed_steps == 1 && p.failed_us == 700, "the failed attempt counts its begin to its exit, once");
    int64_t t = step(sb, 900, step_kind::with_prefill, 256, 200, 300, 50, true); // the retry
    p = sb.snapshot().with_prefill;
    require(p.steps == 1 && p.decode_call_us == 200 && p.sync_wait_us == 300 && p.failed_us == 700,
            "the retry alone is the synchronized step");
    sb.begin(t, step_kind::decode, 1);
    sb.abandon(t + 10);                                // a throw, then a minute with no requests
    require(sb.snapshot().decode.failed_us == 10, "visible before the next request, without the idle minute");
    sb.begin(t + 60'000'000, step_kind::decode, 1);
    auto d = sb.snapshot().decode;
    require(d.failed_us == 10 && d.inter_step_us == 0 && d.unsynced_us == 0, "the idle minute is nobody's cost");
}

int main() {
    a_failed_attempt_is_closed_where_it_fails();
    a_decode_loop_is_attributed_exactly();
    an_idle_engine_is_not_a_stall();
    an_unsynchronized_step_is_only_unsynced();
    a_reader_differences_a_window_while_the_loop_runs();
    std::cout << "test-server-step-budget: ok\n";
    return 0;
}
