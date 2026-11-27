#pragma once

// A lesson's routing as it was lived (Continuum): each position's chosen experts, per layer, for ONE
// sequence, so a mixture-of-experts graph that trains those rows replays them instead of routing
// again. A near-tie in a row's top-k picks a different expert when the arithmetic differs slightly (a
// tile of 64 rows against one of 512, CPU against CUDA), and that discrete choice turns a 1e-4
// difference into a different function; replaying the lived choice trains the function that was
// served. The pin is per row: a row of the pinned sequence inside its range replays, every other row
// (a neighbour's, or the lesson's own outside the range) routes as it serves, in the same ubatch.

#include "llama.h"
#include "llama-graph.h"

#include <cstdint>
#include <vector>

struct llama_pinned_experts {
    llama_seq_id seq     = -1;
    int32_t      n_layer = 0;
    int32_t      n_used  = 0;
    llama_pos    pos0    = 0;
    int32_t      n_pos   = 0;
    // [layer][position - pos0][n_used]
    std::vector<int32_t> ids;

    // row i of `ubatch` replays: it belongs to the pinned sequence alone and its position is covered
    bool pinned(const llama_ubatch & ubatch, uint32_t i) const;
    const int32_t * at(int32_t il, llama_pos pos) const;
};

// The graph input that carries, for the rows a layer sees, each pinned row's experts and a mask (1
// pinned, 0 routed). Rows are the ubatch's, or after the last layer's cut only its outputs, in ubatch
// order (as out_ids keeps them).
class llm_graph_input_pinned_experts : public llm_graph_input_i {
public:
    llm_graph_input_pinned_experts(const llama_pinned_experts & pinned, int32_t il) : pinned(pinned), il(il) {}
    virtual ~llm_graph_input_pinned_experts() = default;

    void set_input(const llama_ubatch * ubatch) override;

    ggml_tensor * ids  = nullptr; // I32 [n_used, n_rows], 0 on a routed row
    ggml_tensor * mask = nullptr; // F32 [1, n_rows]

    const llama_pinned_experts & pinned;
    const int32_t il;
};

// `routed` ([n_used, n_rows], the router's choice) with every pinned row of this graph replaced by its
// lived choice; `routed` itself when no row of the ubatch is pinned.
ggml_tensor * llama_pin_experts(const llm_graph_context & graph, int32_t il, ggml_tensor * routed);
