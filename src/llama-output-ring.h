#pragma once

// Continuum: a decode's outputs kept readable while the next decode is queued behind it.
//
// A context has one output buffer and reads it only after synchronize(), which waits for everything queued, so a
// caller serving several minds could not queue mind B's step until it had read mind A's: on the M5 (Qwen3.6-35B-A3B,
// 2026-11-28) each token was ~5.5 ms of submit with the GPU idle, then ~15.5 ms of GPU, strictly in turn.
//
// With the ring on, the context keeps two output buffers. Each decode takes the spare by pointer swap before it
// reserves its outputs, so the previous decode's outputs stay where they are, and records an event once its own
// output copies are queued. A ticket names one decode: its logits are read after waiting on that decode's event
// alone, so A is sampled while B computes. A ticket is readable until the decode after next begins.

#include "ggml-backend.h"
#include "ggml-cpp.h"

#include <array>
#include <cstdint>

struct llama_output_ticket_slot {
    int64_t                ticket    = -1;      // the decode it holds, or -1
    int32_t                n_outputs = 0;
    float *                logits    = nullptr; // its rows, in the order the decode produced them
    ggml_backend_event_ptr event;               // recorded once its output copies were queued
};

struct llama_output_ring {
    bool                    on      = false;
    ggml_backend_t          backend = nullptr; // where the output copies run, so where the events are recorded
    ggml_backend_buffer_ptr spare;             // the previous decode's buffer while the context writes the other
    std::array<llama_output_ticket_slot, 2> slots; // [the latest decode, the one before]
    int64_t                 decodes = 0;
};
