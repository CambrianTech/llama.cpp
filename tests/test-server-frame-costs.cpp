// what this catches: the engine's per-frame cost aggregation (tools/server/server-frame-costs.h, CONTINUUM-ENGINE
// 4.13 step 1) misattributing the device's time: a prefill frame counted as decode, learning time lost, idle time
// invented, or the ring dropping frames inside the window. Model-free.
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

int main() {
    frame_costs<8> costs;
    // a 1 s window: two decode-only frames of 4 sequences, one prefill frame, learning after one frame
    costs.record({ 100'000, 50'000, 0, 4, 0, 4, 60'000 });
    costs.record({ 200'000, 50'000, 30'000, 4, 0, 4, 70'000 });
    costs.record({ 500'000, 200'000, 0, 4, 2048, 4, 90'000 });
    costs.record({ 2'000'000, 50'000, 0, 4, 0, 4, 10 }); // outside the window
    const frame_window w = costs.window(0, 1'000'000);
    require(w.frames == 3, "only frames inside the window");
    require(w.decode_us == 100'000 && w.prefill_us == 200'000 && w.learn_us == 30'000, "time attributed by kind");
    require(w.idle_us() == 670'000, "idle is the wall time no frame used");
    require(w.gen_tokens == 12 && w.prompt_tokens == 2048, "tokens summed");
    require(w.max_sequences == 4 && w.max_context == 90'000, "the widest batch and longest context");
    require(std::fabs(w.gen_tps() - 12.0) < 1e-9, "aggregate generation over wall time");
    require(std::fabs(w.decode_tps() - 40.0) < 1e-9, "generation over the steps that decoded");

    frame_costs<4> small;
    for (int i = 1; i <= 6; ++i) {
        small.record({ i * 10, 1, 0, 1, 0, 1, 1 });
    }
    require(small.window(0, 1000).frames == 4, "the ring keeps the newest N, never more");
    require(small.recorded() == 6, "every frame was recorded");
    std::cout << "test-server-frame-costs: ok\n";
    return 0;
}
