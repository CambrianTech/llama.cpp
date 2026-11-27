#pragma once

// A mind's routing as she lived it (Continuum; IN-PROCESS-LEARNING.md, "Routing is part of what was
// lived"). On a mixture-of-experts base a near-tie in a row's top-k picks a different expert when the
// arithmetic differs slightly (a tile of 64 rows against one of 512, CPU against CUDA), and that discrete
// choice turns a 1e-4 difference into a different function. So serving records each row's chosen
// experts, per sequence, position and layer, and a lesson replays them: it trains the function she
// served.
//
// One owner per context:
// - the table: per sequence, [position][layer][n_used] expert ids as u8 (a model with more than 256
//   experts is refused) beside a flag per position and layer that serving routed it (the last
//   layer runs only a ubatch's output rows). A later serving decode at a position overwrites it; a
//   rewind keeps it, since a lesson rewinds to the served prefix and must still find what she lived.
// - capture: serving graphs (never training ones) hand their selected ids to the table, copied
//   asynchronously after compute and written at the context's synchronize, so the GPU never waits.
// - the pin: rows of one sequence inside a span replay the table, per row and per layer; every other
//   row in the same ubatch routes as it serves. While a pin is set every MoE layer carries it, so a
//   graph's shape never depends on which rows a ubatch covers, and setting or clearing a pin drops the
//   cached graphs.

#include "llama.h"
#include "llama-graph.h"

#include "ggml-cpp.h"

#include <cstdint>
#include <vector>

struct llama_lived_routing {
    static constexpr int32_t MAX_EXPERTS = 256; // ids are u8

    int32_t n_layer = 0;
    int32_t n_used  = 0;
    bool    capture = false;

    // the pin: rows of `pin_seq` in [pin0, pin0 + pin_n) replay the table (pin_n 0: none)
    llama_seq_id pin_seq = -1;
    llama_pos    pin0    = 0;
    int32_t      pin_n   = 0;

    // the capture buffer: I32 [n_used, n_layer * n_ubatch] on the output device. A serving graph copies
    // layer il's choice into rows [il * n_tokens, il * n_tokens + rows), so one read per ubatch carries
    // every layer (a read per layer cost a command buffer each on Metal: 405 us per decode, 2026-11-27)
    ggml_context_ptr        capture_ctx;
    ggml_backend_buffer_ptr capture_buf;
    ggml_tensor *           capture_dev = nullptr;

    // host time spent capturing (queue and settle), for the probe that prices it
    int64_t host_us = 0;

    // per sequence id: ids [position][layer][n_used], and whether serving routed that layer [position][layer]
    struct seq_rows {
        std::vector<uint8_t> ids;
        std::vector<uint8_t> lived;
    };
    std::vector<seq_rows> seqs;

    // the layer's lived ids at (seq, pos), or nullptr where serving never routed it
    const uint8_t * at(llama_seq_id seq, llama_pos pos, int32_t il) const;
    void write(llama_seq_id seq, llama_pos pos, int32_t il, const int32_t * ids);
    // every position of [pos0, pos0 + n) of `seq` was routed by serving (its first layer recorded)
    bool lived(llama_seq_id seq, llama_pos pos0, int32_t n) const;
    // row i of `ubatch` lies in the pin
    bool pinned(const llama_ubatch & ubatch, uint32_t i) const;
    size_t bytes() const;
};

// The graph input that carries, for the rows a layer sees, each pinned and lived row's experts and a
// mask (1 replayed, 0 routed). Rows are the ubatch's, or after the last layer's cut only its outputs,
// in ubatch order (as out_ids keeps them).
class llm_graph_input_pinned_experts : public llm_graph_input_i {
public:
    llm_graph_input_pinned_experts(const llama_lived_routing & lived, int32_t il) : lived(lived), il(il) {}
    virtual ~llm_graph_input_pinned_experts() = default;

    void set_input(const llama_ubatch * ubatch) override;
    bool can_reuse(const llm_graph_params & params) override;

    ggml_tensor * ids  = nullptr; // I32 [n_used, n_rows], 0 on a routed row
    ggml_tensor * mask = nullptr; // F32 [1, n_rows]

    const llama_lived_routing & lived;
    const int32_t il;
};

// The one hook in build_moe_ffn, after the router: `routed` with pinned rows replaced by what she lived,
// and in a serving graph the selection registered for capture.
ggml_tensor * llama_route_experts(const llm_graph_context & graph, int32_t il, ggml_tensor * routed);
