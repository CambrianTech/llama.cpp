#pragma once

// A lesson's routing as it was lived (Continuum): each position's chosen experts, per layer, so a
// mixture-of-experts graph that trains those rows replays them instead of routing again. A near-tie in
// a row's top-k picks a different expert when the arithmetic differs slightly (a tile of 64 rows
// against one of 512, CPU against CUDA), and that discrete choice turns a 1e-4 difference into a
// different function; replaying the lived choice trains the function that was served.

#include "llama.h"
#include "llama-graph.h"

#include <cstdint>
#include <vector>

struct llama_pinned_experts {
    int32_t   n_layer = 0;
    int32_t   n_used  = 0;
    llama_pos pos0    = 0;
    int32_t   n_pos   = 0;
    // [layer][position - pos0][n_used]
    std::vector<int32_t> ids;

    // every row of `ubatch` has a pinned choice: only then does a graph replay it
    bool covers(const llama_ubatch & ubatch) const;
    const int32_t * at(int32_t il, llama_pos pos) const;
};

// The graph input that carries a layer's pinned experts for the rows of a ubatch.
class llm_graph_input_pinned_experts : public llm_graph_input_i {
public:
    llm_graph_input_pinned_experts(const llama_pinned_experts & pinned, int32_t il) : pinned(pinned), il(il) {}
    virtual ~llm_graph_input_pinned_experts() = default;

    void set_input(const llama_ubatch * ubatch) override;

    ggml_tensor * ids = nullptr; // I32 [n_used, n_rows]: every row of the ubatch, or after the last layer's cut only its outputs, in ubatch order (as out_ids keeps them)

    const llama_pinned_experts & pinned;
    const int32_t il;
};

// The pinned experts of layer `il` for this graph's rows, or nullptr when nothing is pinned or a row is
// not covered (the router then chooses, as it serves).
ggml_tensor * llama_build_pinned_experts(const llm_graph_context & graph, int32_t il, int64_t n_used, int64_t n_rows);
