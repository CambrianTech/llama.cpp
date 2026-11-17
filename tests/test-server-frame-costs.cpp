// what this catches: the engine's frame recorder (tools/server/server-frame-costs.h, CONTINUUM-ENGINE 4.13 step 1)
// claiming what it did not measure: host uptime before activation counted as idle; a mixed frame filed as prefill or
// decode; a failed or retried batch counted as plain work; a reader that fell behind not told what it lost; a window
// the ring overwrote presented as complete. Clocks start far from zero, as ggml_time_us does. Model-free.
#include "../tools/server/server-frame-costs.h"

#include <cmath>
#include <cstdlib>
#include <iostream>

static void require(bool condition, const char * message) {
    if (!condition) {
        std::cerr << "FAILED: " << message << '\n';
        std::exit(1);
    }
}

static frame_cost frame(int64_t end_us, int64_t loop, int64_t batch, int64_t learn, int32_t gen, int32_t prompt,
                        frame_outcome outcome = frame_outcome::ok) {
    frame_cost f;
    f.end_us        = end_us;
    f.host_loop_us  = loop;
    f.host_batch_us = batch;
    f.host_learn_us = learn;
    f.gen_accepted  = gen;
    f.prompt_done   = prompt;
    f.sequences     = 4;
    f.max_context   = 60'000;
    f.outcome       = outcome;
    return f;
}

static void a_window_never_reaches_before_activation() {
    const int64_t t0 = 3'600'000'000; // an hour of host uptime before recording began
    frame_costs<8> costs(t0);
    costs.record(frame(t0 + 100'000, 60'000, 50'000, 0, 4, 0));
    const frame_window w = costs.window(0, t0 + 1'000'000);
    require(w.wall_us == 1'000'000, "wall time starts at activation, not at the clock's origin");
    require(w.outside_loop_us() == 940'000, "outside-loop time is the measured interval minus loop time");
}

static void work_is_filed_by_what_completed() {
    const int64_t t0 = 5'000'000;
    frame_costs<8> costs(t0);
    costs.record(frame(t0 + 100'000, 60'000, 50'000, 0, 4, 0));          // decode
    costs.record(frame(t0 + 200'000, 230'000, 200'000, 0, 0, 2048));     // prefill
    costs.record(frame(t0 + 400'000, 130'000, 100'000, 20'000, 3, 512)); // mixed, with learning after
    costs.record(frame(t0 + 500'000, 40'000, 30'000, 0, 0, 0, frame_outcome::failed));
    costs.record(frame(t0 + 600'000, 70'000, 60'000, 0, 4, 0, frame_outcome::retried));
    const frame_window w = costs.window(t0, t0 + 1'000'000);
    require(w.frames == 5, "every frame in the window");
    require(w.batch_us[(size_t) frame_kind::decode] == 110'000, "decode-only frames, including the retried one");
    require(w.batch_us[(size_t) frame_kind::prefill] == 200'000, "prefill-only frames");
    require(w.batch_us[(size_t) frame_kind::mixed] == 100'000, "a mixed frame stays mixed");
    require(w.batch_us[(size_t) frame_kind::none] == 30'000, "a failed frame completed nothing");
    require(w.failed == 1 && w.retried == 1, "outcomes counted");
    require(w.gen_accepted == 11 && w.prompt_done == 2560, "completed work only");
    require(w.host_learn_us == 20'000, "learning time kept apart");
    require(w.unclassified_us == 10'000 + 30'000 + 10'000 + 10'000 + 10'000, "loop time no named segment covers");
    require(std::fabs(w.gen_per_wall_s() - 11.0) < 1e-9, "accepted output over wall time");
    require(w.complete, "nothing overwritten");
}

static void a_reader_that_falls_behind_is_told_what_it_lost() {
    const int64_t t0 = 9'000'000;
    frame_costs<4> costs(t0);
    for (int i = 1; i <= 6; ++i) {
        costs.record(frame(t0 + i * 10, 5, 4, 0, 1, 0));
    }
    std::vector<frame_cost> out;
    const frame_drain d = costs.drain(0, out);
    require(d.dropped == 2 && out.size() == 4, "two overwritten, the newest four returned");
    require(out.front().seq == 2 && out.back().seq == 5 && d.next_seq == 6, "oldest first, sequence numbers intact");
    out.clear();
    costs.record(frame(t0 + 70, 5, 4, 0, 1, 0));
    const frame_drain again = costs.drain(d.next_seq, out);
    require(again.dropped == 0 && out.size() == 1 && out.front().seq == 6, "a caught-up reader gets only what is new");

    const frame_window w = costs.window(t0, t0 + 1000);
    require(!w.complete && w.frames == 4, "a window the ring overwrote is marked incomplete");
    require(costs.window(t0 + 45, t0 + 1000).complete, "a window entirely within the ring is complete");
}

int main() {
    a_window_never_reaches_before_activation();
    work_is_filed_by_what_completed();
    a_reader_that_falls_behind_is_told_what_it_lost();
    std::cout << "test-server-frame-costs: ok\n";
    return 0;
}
