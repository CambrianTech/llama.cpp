#pragma once

// Continuum: the walk's object, shared by the epoch (llama-context-opt.cpp) and a live reply learning in
// tiles (llama-context-learn-tiles.cpp). Private to the fork's sources.

#include "llama-context.h"
#include "llama-walk.h"

#include "ggml-opt.h"

#include <cstdint>
#include <vector>

class llama_kv_cache;
class llama_memory_recurrent;

// THE WALK (Fable's design, continuum 2026-10-06; objective 2026-10-14): one pass over a window
// in one context. A run with no label is context: a plain decode into the cache, as served, no
// backward. A labelled run is her reply, trained GIVEN EVERYTHING BEFORE IT AS SERVED: only its
// rows see the shadow adapter (the label mask), so the context before it carries no gradient and
// is never trained. The run trains in chunks of n_batch, each attending to everything before it
// in the cache; its chunks write their own cells under the shadow, so a later chunk of the run
// reads an earlier one as the shadow computes it. When the run is done it is RESTORED: its cells
// and the recurrent state are decoded again under her stable set, so the cache is always her
// stable self's and the next run reads this one as served. Memory is chunk x window, never
// window x window, so her whole lived window trains. Nothing after the last label is computed.
// One per window; a member of the context, so it runs on the context's own state.
struct llama_context::opt_walk {
    llama_context                  & lctx;
    ggml_opt_dataset_t               dataset;
    ggml_opt_result_t                result;
    const std::vector<llama_token> & tokens;
    const std::vector<llama_token> & labels;
    llama_batch                    & batch;
    ggml_opt_epoch_callback          callback;
    bool                             train;
    int64_t                          idata_in_loop;
    int64_t                          ndata_in_loop;
    int64_t                          t_loop_start;
    uint32_t                         n_ctx;
    uint32_t                         n_batch;
    // the sequence it walks: 0 for an epoch over a dataset window; a live reply's own sequence when it
    // learns in tiles beside serving (UNIFIED-PAGING.md, "The join")
    llama_seq_id                     seq;
    llama_walk_plan                  plan;

    // a recurrent part (pure recurrent or hybrid) carries its state across chunks; a hybrid's
    // attention part removes a chunk's cells like a plain cache
    llama_memory_recurrent * recr = nullptr;
    llama_kv_cache         * attn = nullptr;

    bool first_unit = true;
    // THE EXACT WALK's chunk being trained (null in the plain walk) and its host gradients
    const llama_walk_chunk * xc = nullptr;
    llama_walk_acc           acc;

    opt_walk(llama_context & lctx, ggml_opt_dataset_t dataset, ggml_opt_result_t result,
            const std::vector<llama_token> & tokens, const std::vector<llama_token> & labels, llama_batch & batch,
            ggml_opt_epoch_callback callback, bool train, int64_t idata_in_loop, int64_t ndata_in_loop, int64_t t_loop_start,
            uint32_t n_ctx, uint32_t n_batch, llama_seq_id seq)
        : lctx(lctx), dataset(dataset), result(result), tokens(tokens), labels(labels), batch(batch), callback(callback),
          train(train), idata_in_loop(idata_in_loop), ndata_in_loop(ndata_in_loop), t_loop_start(t_loop_start),
          n_ctx(n_ctx), n_batch(n_batch), seq(seq), plan(llama_walk_plan::of(labels, n_ctx, n_batch)) {}

    bool yield_point();
    bool decode_span(uint32_t p0, uint32_t p1, bool shadow);
    std::vector<uint8_t> checkpoint();
    bool rewind(uint32_t p_pop, const std::vector<uint8_t> & snap, uint32_t p_to);
    bool train_chunk(uint32_t pos_ctx, uint32_t n_chunk);
    bool restore(const llama_walk_run & run, const std::vector<uint8_t> & snap);
    void run_exact();
    void run_plain();

    // the exact walk's graph flags in cparams, which build_attn and build_rs read
    void set_flags(bool exact, uint32_t grad_from, bool surrogate, bool state_surrogate) {
        lctx.cparams.walk_exact           = exact;
        lctx.cparams.walk_grad_from       = grad_from;
        lctx.cparams.walk_surrogate       = surrogate;
        lctx.cparams.walk_state_surrogate = state_surrogate;
    }
};
