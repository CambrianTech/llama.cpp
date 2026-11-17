#pragma once

// Continuum fork (CONTINUUM-ENGINE 4.13, step 1: measure before changing anything). One record per serving-loop
// iteration that submitted a batch, kept in a fixed ring (no allocation per frame) and read through a typed drain:
// every record carries a sequence number, and a reader that falls behind is told exactly how many it lost.
//
// What a record claims, and what it does not (Codex review of llama.cpp#142):
// - Work is what COMPLETED, taken from the owners that complete it: generated tokens are the accepted tokens
//   (`n_tokens_generated`, incremented only for accepted output, after speculative verification); prompt tokens are
//   those `metrics_post_decode` counted for a decode that succeeded. Draft proposals and accepted drafts are kept
//   apart from both. A batch that was retried or failed says so in `outcome`.
// - The batch's shape (sequences, longest context) is snapshotted BEFORE decode, so a slot released by its own
//   completion inside the frame is still counted.
// - Times are HOST elapsed, never device time: `host_batch_us` is decode + post-decode as the host saw them (sampling
//   synchronizes the device), `host_learn_us` the learning work after the batch, `host_loop_us` the whole iteration.
//   The rest of the iteration is `unclassified`, not idle; the time between iterations is outside the loop, not GPU idle.
// - A frame with both prompt and generated work is MIXED; it is never filed as a pure prefill or decode cost.

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

enum class frame_outcome : uint8_t {
    ok      = 1,
    retried = 2, // the engine shrank the batch and decoded again (no KV room at first)
    failed  = 3, // decode or post-decode threw; the frame's slots were aborted
};

enum class frame_kind : uint8_t {
    decode  = 1, // generated tokens only
    prefill = 2, // prompt tokens only
    mixed   = 3, // both: its time is the pair's, never attributed to either alone
    none    = 4, // nothing completed (a failed or empty frame)
};

struct frame_cost {
    uint64_t      seq             = 0; // assigned by the recorder
    int64_t       end_us          = 0; // when the iteration ended (ggml_time_us)
    int64_t       host_loop_us    = 0;
    int64_t       host_batch_us   = 0;
    int64_t       host_learn_us   = 0;
    int32_t       gen_accepted    = 0; // accepted output tokens
    int32_t       prompt_done     = 0; // prompt tokens of decodes that succeeded
    int32_t       draft_proposed  = 0;
    int32_t       draft_accepted  = 0;
    int32_t       sequences       = 0; // distinct sequences in the batch, before decode
    int32_t       max_context     = 0; // longest of them, before decode
    frame_outcome outcome         = frame_outcome::ok;

    frame_kind kind() const {
        if (gen_accepted > 0 && prompt_done > 0) { return frame_kind::mixed; }
        if (gen_accepted > 0) { return frame_kind::decode; }
        if (prompt_done > 0) { return frame_kind::prefill; }
        return frame_kind::none;
    }
    int64_t unclassified_us() const {
        const int64_t named = host_batch_us + host_learn_us;
        return host_loop_us > named ? host_loop_us - named : 0;
    }
};

struct frame_drain {
    uint64_t next_seq = 0; // pass back as `after` next time
    uint64_t dropped  = 0; // records overwritten before this reader saw them
};

// Totals over the frames that ENDED in (from_us, to_us]; a frame crossing `from_us` belongs to the window it ends in.
struct frame_window {
    int64_t wall_us          = 0; // to_us - max(from_us, activation)
    int64_t frames           = 0;
    int64_t host_loop_us     = 0;
    int64_t host_learn_us    = 0;
    int64_t unclassified_us  = 0;
    int64_t batch_us[5]      = {}; // indexed by frame_kind
    int64_t gen_accepted     = 0;
    int64_t prompt_done      = 0;
    int64_t draft_proposed   = 0;
    int64_t draft_accepted   = 0;
    int64_t retried          = 0;
    int64_t failed           = 0;
    int64_t max_sequences    = 0;
    int64_t max_context      = 0;
    bool    complete         = true; // false: the ring overwrote frames of this window; its totals undercount

    // wall time with no loop iteration running: outside the loop, NOT device idle
    int64_t outside_loop_us() const { return wall_us > host_loop_us ? wall_us - host_loop_us : 0; }
    // accepted output per second of wall time: the number the 80 tok/s goal is read against
    double gen_per_wall_s() const { return wall_us > 0 ? 1e6 * (double) gen_accepted / (double) wall_us : 0.0; }
};

template <size_t N>
class frame_costs {
    static_assert(N > 0 && (N & (N - 1)) == 0, "a power of two");

public:
    // `activation_us` is when recording began: no window reaches before it, so host uptime is never counted.
    explicit frame_costs(int64_t activation_us) : activation_us_(activation_us) {}

    int64_t activation_us() const { return activation_us_; }

    uint64_t record(frame_cost f) {
        f.seq = next_seq_++;
        ring_[f.seq & (N - 1)] = f;
        return f.seq;
    }

    // Copy every record with seq >= after into `out` (oldest first).
    frame_drain drain(uint64_t after, std::vector<frame_cost> & out) const {
        frame_drain d;
        const uint64_t oldest = next_seq_ > N ? next_seq_ - N : 0;
        uint64_t from = after;
        if (from < oldest) {
            d.dropped = oldest - from;
            from      = oldest;
        }
        for (uint64_t s = from; s < next_seq_; ++s) {
            out.push_back(ring_[s & (N - 1)]);
        }
        d.next_seq = next_seq_;
        return d;
    }

    frame_window window(int64_t from_us, int64_t to_us) const {
        frame_window w;
        const int64_t from = from_us > activation_us_ ? from_us : activation_us_;
        w.wall_us = to_us > from ? to_us - from : 0;
        const uint64_t oldest = next_seq_ > N ? next_seq_ - N : 0;
        for (uint64_t s = next_seq_; s > oldest; --s) {
            const frame_cost & f = ring_[(s - 1) & (N - 1)];
            if (f.end_us > to_us) {
                continue;
            }
            if (f.end_us <= from) {
                break; // records are in end order; everything older is before the window
            }
            ++w.frames;
            w.host_loop_us    += f.host_loop_us;
            w.host_learn_us   += f.host_learn_us;
            w.unclassified_us += f.unclassified_us();
            w.batch_us[(size_t) f.kind()] += f.host_batch_us;
            w.gen_accepted    += f.gen_accepted;
            w.prompt_done     += f.prompt_done;
            w.draft_proposed  += f.draft_proposed;
            w.draft_accepted  += f.draft_accepted;
            w.retried         += f.outcome == frame_outcome::retried;
            w.failed          += f.outcome == frame_outcome::failed;
            w.max_sequences = std::max<int64_t>(w.max_sequences, f.sequences);
            w.max_context   = std::max<int64_t>(w.max_context, f.max_context);
            if (s - 1 == oldest && oldest > 0) {
                // the oldest record the ring still holds is inside the window: older ones in it were overwritten
                w.complete = false;
            }
        }
        return w;
    }

    uint64_t recorded() const { return next_seq_; }

private:
    std::array<frame_cost, N> ring_{};
    uint64_t                  next_seq_ = 0;
    int64_t                   activation_us_;
};
