#pragma once

// Continuum: each mind's decode graph stays built and allocated across a switch to another mind.
//
// A backend scheduler holds the split and the allocation of ONE graph (ggml_backend_sched: splits, graph copy, galloc),
// so a context serving several minds one step at a time rebuilt, re-split and re-allocated its graph at every switch:
// on the M5 (Qwen3.6-35B-A3B, four residents) 0 of 74 decode steps reused a graph, with 78 switches in that turn.
//
// A slot is a second scheduler with its own graph results. Entering a slot SWAPS the context's scheduler, graph results,
// active graph and adapter set with the slot's, so every use of the context's scheduler is unchanged; leaving swaps them
// back. A swap is a pointer exchange, never a copy. A slot is active only inside decode(), for ubatches of one token per
// sequence; prefill and training run on the context's own scheduler, reserved for the worst case.
//
// The slot keeps the adapter set its graphs were built with, so the graph's own reuse check (llm_graph_params::allow_reuse,
// which compares the set by pointer) holds when the mind returns. An adapter freed on the model may come back at the same
// address, so a slot built before any free is dropped rather than compared (llama_model::lora_frees).

#include "llama-adapter.h"
#include "llama-graph.h"

#include "ggml-cpp.h"

#include <array>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

struct llama_graph_slot {
    std::vector<uint64_t>             key;                          // whom its ubatch's sequences serve
    llama_adapter_loras_ptr           loras;                        // the set its graphs were built with
    ggml_backend_sched_ptr            sched;
    std::array<llm_graph_result_ptr, 2> gf_res_prev;                 // per output shape, as the context's own
    llm_graph_result *                gf_res_prev_active = nullptr;
    uint64_t                          lora_frees = 0;               // the model's adapter frees when the slot was made
    uint64_t                          used       = 0;               // the step it last served, for eviction
};

struct llama_graph_slots {
    size_t                                         capacity = 0; // set by the owner from the minds it keeps resident
    std::vector<std::unique_ptr<llama_graph_slot>> slots;
    llama_graph_slot *                             active   = nullptr;
    bool                                           in_decode = false; // slots serve decode() only (llama_context::graph_slot_scope)
    uint64_t                                       clock    = 0;
    std::unordered_map<llama_seq_id, uint64_t>     named;             // whom each sequence serves, as its owner named her
    std::vector<uint64_t>                          key;               // this step's, reused across steps
};
