#pragma once

// THE WALK's plan and its host state, apart from the context that runs it
// (llama_context::opt_epoch_iter): which positions train in which chunk, how far back each
// chunk's gradient reaches, and what the exact walk keeps in host memory. The plan is plain
// data from the labels and the model's shape, so it is tested without a model
// (test-walk-plan).

#include "llama.h"

#include "ggml-cpp.h"

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

struct llama_cparams;
struct llama_hparams;
class llama_memory_recurrent;

// at most n_batch positions of one run of the window, all labelled or all context
struct llama_walk_chunk {
    uint32_t c0;
    uint32_t n;
    uint32_t n_labels;
    // the exact walk's marks (llama_walk_plan::mark_exact)
    uint32_t grad_from  = 0;     // its gradient taps cover cached positions [grad_from, c0)
    bool     surrogate  = false; // a later chunk's gradient sits on its own K/V
    bool     needed     = false;
    bool     period_end = false; // the window's optimizer step
    float    loss_scale = 0.0f;  // its labelled positions over the window's
};

// chunks [j0, j1) of the plan: one labelled run, which trains given everything before it as
// served (the shadow adapter applies to labelled rows only)
struct llama_walk_run {
    size_t j0;
    size_t j1;
};

struct llama_walk_plan {
    // the window's runs, each cut every n_batch positions from its start; both walks take
    // them in this order. Nothing after the last label is in the plan.
    std::vector<llama_walk_chunk> chunks;
    std::vector<llama_walk_run>   runs; // the labelled ones, in order
    int64_t  last_label = -1;    // -1: the window has no label, nothing to learn from it
    uint32_t n_labels   = 0;
    int64_t  first      = -1;    // the exact walk's period end: the chunk its reverse pass trains last

    static llama_walk_plan of(const std::vector<llama_token> & labels, uint32_t n_ctx, uint32_t n_batch);

    // positions [begin, end) of a run
    uint32_t begin(const llama_walk_run & r) const { return chunks[r.j0].c0; }
    uint32_t end  (const llama_walk_run & r) const { return chunks[r.j1 - 1].c0 + chunks[r.j1 - 1].n; }

    // which chunks the exact walk trains (the labelled ones: the context before a run is her
    // stable set's and carries no gradient), how far back each gradient reaches (never before
    // its run's start), and the period end (the first chunk of the last run, trained last)
    void mark_exact(uint32_t horizon);

    // the first cached position a chunk at c0 carries gradient to (horizon 0: the whole prefix)
    static uint32_t grad_from(uint32_t c0, uint32_t horizon);

    // THE HOST BUDGET (Fable): the K/V accumulators, plus a recurrent state checkpoint every
    // `stride` chunks of the run being walked (the largest run's)
    size_t host_bytes(size_t stride, size_t kv_bytes, size_t state_bytes) const;
    // the smallest stride under the budget (0: no cap); the reverse pass rebuilds a chunk's
    // entry state by decoding forward from its checkpoint, so a longer stride trades host
    // memory for decode, never for exactness. When none fits, the stride of one checkpoint.
    size_t stride_for(size_t budget, size_t kv_bytes, size_t state_bytes) const;
};

// THE EXACT WALK's gradients, in host memory, each added into in place by the backward: per
// attention layer dL/d(cached K) and dL/d(cached V) of the window, [n_embd_*_gqa, position]
// F32 (null for a recurrent layer), which the graph reads through cparams while bound (never
// past the accumulator's life); per recurrent state tensor (a layer's conv state r_l and its
// recurrent state s_l) dL/d(the state entering the chunk trained last), which is what the
// chunk before it puts on the state it leaves (state_ready once one is in)
struct llama_walk_acc {
    std::vector<ggml_tensor *> k, v;
    std::unordered_map<const ggml_tensor *, ggml_tensor *> state;
    bool state_ready = false;

    llama_walk_acc() = default;
    llama_walk_acc(const llama_walk_acc &) = delete;
    llama_walk_acc & operator=(const llama_walk_acc &) = delete;
    ~llama_walk_acc();

    // zeroed accumulators for n_pos positions, bound into cparams; false: the host memory
    // could not be allocated
    bool init(const llama_hparams & hparams, const llama_memory_recurrent * recr, uint32_t n_pos, llama_cparams & cparams);

    // what init allocates for the K/V of n_pos positions, and one sequence's recurrent state
    static size_t kv_bytes(const llama_hparams & hparams, uint32_t n_pos);
    static size_t state_bytes(const llama_memory_recurrent * recr);

private:
    llama_cparams *         bound = nullptr;
    ggml_context_ptr        ctx;
    ggml_backend_buffer_ptr buf;
};

// positions [c0, c0 + n) of the window into the batch, one sequence, no output row: the
// caller marks the rows it wants out
void llama_walk_fill(llama_batch & batch, const std::vector<llama_token> & tokens, uint32_t c0, uint32_t n);
