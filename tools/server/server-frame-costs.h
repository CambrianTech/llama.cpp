#pragma once

// Continuum fork (CONTINUUM-ENGINE 4.13, step 1: measure before changing anything): what each pass of the serving
// loop spent on the device, recorded into a fixed ring (no allocation per frame), and aggregated into where the GPU
// timeline went: decode, prefill, learning, idle. The same records become the frame loop's cost curves and the
// core's ServingObserver numbers. Pure, so tests/test-server-frame-costs.cpp proves the aggregation without a model.

#include <array>
#include <cstddef>
#include <cstdint>

struct frame_cost {
    int64_t at_us        = 0; // when the frame ended (ggml_time_us)
    int64_t step_us      = 0; // the batch's decode + post-decode (sampling syncs the device)
    int64_t learn_us     = 0; // the learning work that ran after it, this frame
    int32_t gen_tokens   = 0; // generation tokens in the batch (decoding sequences, drafts included)
    int32_t prompt_tokens = 0; // prefill tokens in the batch
    int32_t sequences    = 0; // sequences that decoded a token this frame
    int32_t max_context  = 0; // the longest context among them
};

// Where the timeline went over one window of frames.
struct frame_window {
    int64_t wall_us       = 0;
    int64_t frames        = 0;
    int64_t decode_us     = 0; // frames with no prefill
    int64_t prefill_us    = 0; // frames carrying prefill (their decode tokens ride along)
    int64_t learn_us      = 0;
    int64_t gen_tokens    = 0;
    int64_t prompt_tokens = 0;
    int64_t max_sequences = 0;
    int64_t max_context   = 0;

    int64_t busy_us() const { return decode_us + prefill_us + learn_us; }
    int64_t idle_us() const { return wall_us > busy_us() ? wall_us - busy_us() : 0; }
    // aggregate generation throughput over the window's wall time: the number the 80 tok/s goal is measured by
    double gen_tps() const { return wall_us > 0 ? 1e6 * (double) gen_tokens / (double) wall_us : 0.0; }
    // generation throughput while decoding only: the engine's rate when nothing else shares the step
    double decode_tps() const { return decode_us > 0 ? 1e6 * (double) gen_tokens / (double) (decode_us + prefill_us) : 0.0; }
};

template <size_t N>
class frame_costs {
    static_assert(N > 0 && (N & (N - 1)) == 0, "a power of two");

public:
    void record(const frame_cost & f) {
        ring_[head_ & (N - 1)] = f;
        ++head_;
    }

    // Aggregate the frames that ended in (from_us, to_us]; the window's wall time is to_us - from_us.
    frame_window window(int64_t from_us, int64_t to_us) const {
        frame_window w;
        w.wall_us = to_us > from_us ? to_us - from_us : 0;
        const uint64_t n = head_ < N ? head_ : N;
        for (uint64_t i = 0; i < n; ++i) {
            const frame_cost & f = ring_[(head_ - 1 - i) & (N - 1)];
            if (f.at_us <= from_us || f.at_us > to_us) {
                continue;
            }
            ++w.frames;
            (f.prompt_tokens > 0 ? w.prefill_us : w.decode_us) += f.step_us;
            w.learn_us      += f.learn_us;
            w.gen_tokens    += f.gen_tokens;
            w.prompt_tokens += f.prompt_tokens;
            if (f.sequences > w.max_sequences) { w.max_sequences = f.sequences; }
            if (f.max_context > w.max_context) { w.max_context = f.max_context; }
        }
        return w;
    }

    uint64_t recorded() const { return head_; }

private:
    std::array<frame_cost, N> ring_{};
    uint64_t                  head_ = 0;
};
