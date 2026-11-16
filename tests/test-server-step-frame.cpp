// what this catches: a decode step as a frame (server-step-frame.h) without a model: a resident whose
// last outside handle drops while a frame still uses it stays alive until that frame retires; recording
// N+1 and executing N are distinct frames that swap at the boundary; boundary commands run in order at
// the swap, queued from any thread; the per-step arena is a bump allocator reset on retire, its spans
// valid for the whole step even when it grows.
#include "../tools/server/server-step-frame.h"

#include <cstdlib>
#include <iostream>
#include <thread>

static void require(bool condition, const char * message) {
    if (!condition) {
        std::cerr << "FAILED: " << message << '\n';
        std::exit(1);
    }
}

struct resident {
    explicit resident(bool * f) : freed(f) {}
    resident(const resident &) = delete;
    ~resident() { *freed = true; }
    bool * freed;
};

static void a_resident_lives_until_the_frame_that_uses_it_retires() {
    step_pipeline pipe;
    bool freed = false;
    auto r = std::make_shared<resident>(&freed);
    pipe.recording().retain(r);    // step N+1 records a command that reads it
    r.reset();                     // its owner lets go (an eviction, an erase)
    require(!freed, "the recording frame still holds it");
    pipe.swap();                   // N+1 is now executing on the GPU
    require(!freed, "the executing frame still holds it");
    pipe.swap();                   // the GPU finished it: it retires at the next boundary
    require(freed, "released when the frame that used it retired, never before");
}

static void two_frames_swap_and_number_in_order() {
    step_pipeline pipe;
    const uint64_t first = pipe.recording().number();
    pipe.swap();
    require(pipe.executing().number() == first, "the recorded frame becomes the executing one");
    require(pipe.recording().number() == first + 1, "a retired frame is reused for the next step");
    require(&pipe.recording() != &pipe.executing(), "recording and executing are distinct frames");
}

static void boundary_commands_run_in_order_from_any_thread() {
    step_pipeline pipe;
    std::vector<int> applied;
    std::thread core([&] {
        for (int i = 0; i < 100; ++i) {
            pipe.queue([&applied, i] { applied.push_back(i); });
        }
    });
    core.join();
    require(applied.empty(), "nothing applies before the boundary");
    require(pipe.swap() == 100, "every queued command applies at the swap");
    for (int i = 0; i < 100; ++i) {
        require(applied[(size_t) i] == i, "in the order queued");
    }
    require(pipe.swap() == 0, "each applies once");
}

static void the_arena_bumps_and_resets_per_step() {
    step_pipeline pipe;
    step_arena & a = pipe.recording().arena();
    auto ids = a.allocate_array<int32_t>(16);
    for (int32_t i = 0; i < 16; ++i) {
        ids[(size_t) i] = i;
    }
    auto big = a.allocate_array<float>(100000); // outgrows the first block mid-step
    big[99999] = 1.0f;
    require(ids[15] == 15, "spans from earlier in the step stay valid after the arena grows");
    require(a.used() >= 100000 * sizeof(float), "the step's scratch is accounted");
    const size_t cap = a.capacity();
    pipe.swap();
    pipe.swap(); // that frame retires
    require(pipe.recording().arena().used() == 0, "the retired frame's arena is reset");
    require(pipe.recording().arena().capacity() == cap, "and keeps its grown block: no allocation next step");
}

int main() {
    a_resident_lives_until_the_frame_that_uses_it_retires();
    two_frames_swap_and_number_in_order();
    boundary_commands_run_in_order_from_any_thread();
    the_arena_bumps_and_resets_per_step();
    std::cout << "test-server-step-frame: ok\n";
    return 0;
}
