#pragma once

// A training step cut into segments short enough that serving never waits behind it.
//
// THE PROBLEM (continuum S4, measured 2026-10-07 on an M5 serving Qwen3.8-27B beside 3 lanes):
// the trainer yields to serving only between steps, and one step (512 tokens, 8 adapted
// layers, Metal backward) ran 11-40 s, so a turn arriving mid-step waited the whole step out:
// her decode fell from 9.75 to 2.16 tok/s inside a window and the 10% slowdown bound was
// overrun (11.7%).
//
// THE SEAM: ggml_backend_sched computes a graph in views [j0, j1] up to every node its eval
// callback asks for (ask = true), synchronizes, then calls back (ask = false); returning true
// continues with the graph's allocation intact. So the trainer asks at the node where the
// estimated cost since the last cut reaches a budget, and yields at the call that follows.
// The budget adapts to the measured cost per millisecond, aiming each segment at one of her
// decode steps.

#include "ggml.h"

#include <algorithm>
#include <cstdint>

// A node's cost in estimated FLOPs: the matrix products dominate a training step, so they are
// counted exactly (2 * K per output element); every other op is counted by its output size,
// which the measured rate calibrates along with everything else.
inline double train_node_cost(const ggml_tensor * t) {
    switch (t->op) {
        case GGML_OP_MUL_MAT:
        case GGML_OP_MUL_MAT_ID:
            // dst[n, m] = src0[k, n]^T x src1[k, m]: K = src0->ne[0]
            return 2.0 * (double) t->src[0]->ne[0] * (double) ggml_nelements(t);
        case GGML_OP_OUT_PROD:
            // dst[a, b] = sum over k of src0[a, k] * src1[b, k]: K = src0->ne[1]
            return 2.0 * (double) t->src[0]->ne[1] * (double) ggml_nelements(t);
        default:
            return (double) ggml_nelements(t);
    }
}

// How long a segment may take, from her measured decode: one of her decode steps, the S4 gate's
// "enter and leave cost at most one decode step". tokens_per_ms counts every slot's tokens, so
// one step of a batch of busy_slots streams takes busy_slots / tokens_per_ms. Unmeasured, or
// out of range, takes TRAIN_SEGMENT_DEFAULT_MS, clamped so a fast lane does not cut every node
// and a slow one does not hold her for seconds.
static constexpr double TRAIN_SEGMENT_DEFAULT_MS = 100.0;
static constexpr double TRAIN_SEGMENT_MIN_MS     = 20.0;
static constexpr double TRAIN_SEGMENT_MAX_MS     = 250.0;

inline double train_segment_target_ms(double tokens_per_ms, int busy_slots) {
    if (!(tokens_per_ms > 0) || busy_slots <= 0) {
        return TRAIN_SEGMENT_DEFAULT_MS;
    }
    return std::clamp((double) busy_slots / tokens_per_ms, TRAIN_SEGMENT_MIN_MS, TRAIN_SEGMENT_MAX_MS);
}

struct train_segmenter {
    // the cost a first segment plans before any segment has been measured: small, so the
    // first cut comes early and the rate is learned within the first step
    static constexpr double FIRST_BUDGET = 1e9;
    // how much a newly measured rate moves the estimate (the rest is the history)
    static constexpr double RATE_GAIN = 0.3;

    double since_cut   = 0; // cost planned since the last cut
    double planned     = 0; // cost of the segment that the last cut closed
    double cost_per_ms = 0; // measured; 0 until the first segment is timed

    // ask = true: count this node, and cut after it when the segment has reached its budget
    bool cut_after(double node_cost, double target_ms) {
        since_cut += node_cost;
        const double budget = cost_per_ms > 0 ? target_ms * cost_per_ms : FIRST_BUDGET;
        if (since_cut < budget) {
            return false;
        }
        planned   = since_cut;
        since_cut = 0;
        return true;
    }

    // ask = false: the segment just closed took ms; learn the rate from it
    void measured(double ms) {
        if (!(ms > 0) || !(planned > 0)) {
            return;
        }
        const double r = planned / ms;
        cost_per_ms = cost_per_ms > 0 ? (1 - RATE_GAIN) * cost_per_ms + RATE_GAIN * r : r;
    }

    // a new graph starts: whatever was planned before belongs to no segment
    void reset_plan() { since_cut = 0; planned = 0; }
};
