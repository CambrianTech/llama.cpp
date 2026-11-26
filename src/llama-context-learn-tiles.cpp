// Continuum: a live reply learning in tiles through the exact walk, one tile per step (UNIFIED-PAGING.md,
// "A long reply learns in tiles" and "The join"). The whole-reply graph holds every reply row against every
// lesson row at once, so a long reply cannot learn on any card (the 5090, 2026-11-26: lessons measured
// 64 GB, 91 GB and 1.9 TB on a 34 GB device). The exact walk keeps the gradient the whole graph carries
// (a later token's loss through the earlier tokens' K/V and recurrent state: Codex on 45e05f176) while its
// device memory is one tile's graph. Here it is made preemptible: each step is one unit of the walk, so a
// turn waiting is never more than one tile behind, the way an OS pages a long job at page boundaries.
//
// The units, per labelled run of the reply: decode the context before it under her stable set; forward
// each tile under the shadow, checkpointing the recurrent state every `stride` tiles; then reverse, last
// tile first, each trained with its gradient accumulated into the tiles before it. The first tile of the
// last run ends the period: one optimizer update with the whole reply's denominator. Then the reply's own
// restoration (llama-context-learn.cpp) puts her back exactly as served.

#include "llama-opt-walk.h"

#include "llama-impl.h"
#include "llama-kv-cache.h"
#include "llama-memory-hybrid.h"
#include "llama-memory-recurrent.h"
#include "llama-model.h"

#include <algorithm>
#include <string>
#include <vector>

struct llama_context::opt_tiles {
    // the whole sequence by position: [0, prefix) is the served prefix, already in the cache and never read
    std::vector<llama_token> tokens, labels;
    llama_batch batch;
    opt_walk walk;
    uint32_t tile;
    size_t run = 0;        // the run being walked
    size_t chunk = 0;      // the next chunk of its phase
    bool before = true;    // the context before the run is not yet decoded
    bool reverse = false;  // the run's forward is done
    uint32_t served = 0;   // the cache holds [0, served) as served
    size_t stride = 1;
    std::vector<std::vector<uint8_t>> snaps;

    opt_tiles(llama_context & lctx, const opt_live_reply & s, uint32_t tile)
        : tokens(make(s, /*labels =*/ false)), labels(make(s, /*labels =*/ true)),
          batch(llama_batch_init((int32_t) tile, 0, 1)),
          walk(lctx, /*dataset =*/ nullptr, s.result, tokens, labels, batch, /*callback =*/ nullptr, /*train =*/ true,
               0, 0, ggml_time_us(), (uint32_t) tokens.size(), tile, s.seq),
          tile(tile), served((uint32_t) s.prefix) {}

    ~opt_tiles() { llama_batch_free(batch); }

    static std::vector<llama_token> make(const opt_live_reply & s, bool labels) {
        std::vector<llama_token> by_pos((size_t) s.prefix + (size_t) s.n_train, labels ? -1 : 0);
        const auto & from = labels ? s.labels : s.tokens;
        std::copy(from.begin(), from.begin() + s.n_train, by_pos.begin() + s.prefix);
        return by_pos;
    }
};

void llama_context::opt_tiles_free::operator()(opt_tiles * tiles) const { delete tiles; }

bool llama_context::opt_learn_tile(uint32_t tile) {
    if (!opt_reply || opt_reply->restoring || opt_reply->cursor != 0 || opt_reply->graph_nodes != 0 || opt_reply->tiles) {
        opt_fail("llama_opt_learn_tile: call it once, after begin and before the first step");
        return false;
    }
    if (tile == 0) {
        opt_fail("llama_opt_learn_tile: a tile of zero rows");
        return false;
    }
    if (!opt_walk_exact) {
        // independent tiles lose the gradient a later token carries into the earlier ones (Codex): exact or none
        opt_fail("llama_opt_learn_tile: tiles learn only through the exact walk (walk_exact)");
        return false;
    }
    auto & s = *opt_reply;
    if (s.n_train <= 0) {
        opt_fail("llama_opt_learn_tile: the reply has no labelled row");
        return false;
    }
    auto tiles = std::unique_ptr<opt_tiles, opt_tiles_free>(new opt_tiles(*this, s, tile));
    opt_walk & walk = tiles->walk;
    if (walk.plan.last_label < 0) {
        opt_fail("llama_opt_learn_tile: the reply has no labelled row");
        return false;
    }
    if (auto * hybrid = dynamic_cast<llama_memory_hybrid *>(memory.get())) {
        walk.recr = hybrid->get_mem_recr();
        walk.attn = hybrid->get_mem_attn();
    } else {
        walk.recr = dynamic_cast<llama_memory_recurrent *>(memory.get());
    }
    if (walk.recr != nullptr && walk.recr->rollback_pending(s.seq)) {
        opt_fail("llama_opt_learn_tile: the sequence has a pending recurrent rollback: finish or drop the draft first");
        return false;
    }
    // the walk's prologue (run_exact): the horizon as requested, the plan marked, the host budget
    opt_walk_horizon = opt_walk_horizon_req;
    walk.plan.mark_exact(opt_walk_horizon);
    const uint32_t n_pos = walk.plan.longest_run();
    const size_t kv_bytes = llama_walk_acc::kv_bytes(model.hparams, n_pos);
    const size_t state_bytes = llama_walk_acc::state_bytes(walk.recr);
    tiles->stride = walk.plan.stride_for(opt_walk_host_budget, kv_bytes, state_bytes);
    const size_t bytes = walk.plan.host_bytes(tiles->stride, kv_bytes, state_bytes);
    opt_walk_host_bytes = std::max(opt_walk_host_bytes, bytes);
    if (opt_walk_host_budget > 0 && bytes > opt_walk_host_budget) {
        opt_fail("llama_opt_learn_tile: the exact walk's host memory for this reply (" + std::to_string(bytes)
            + " bytes, its K/V gradient) is past its budget of " + std::to_string(opt_walk_host_budget));
        return false;
    }
    if (!walk.acc.init(model.hparams, walk.recr, n_pos, cparams)) {
        opt_fail("llama_opt_learn_tile: the exact walk's gradient accumulator (host memory) could not be allocated");
        return false;
    }
    opt_stop_requested.store(false);
    opt_alloc_failed.store(false);
    tiles->walk.acc.unbind(); // bound only while a step runs
    s.tiles = std::move(tiles);
    return true;
}

// One unit of the walk. The accumulator is bound into the context only while the unit runs, so a neighbour's
// decode between steps never reads it (Codex on the join). A failure hands the reply to its restoration, its
// accumulated gradient cleared, the way a failed graph does: she is always restored as served.
llama_opt_learn_status llama_context::opt_tiles_step() {
    auto & s = *opt_reply;
    bool finished = false;
    {
        auto & t = *s.tiles;
        // a tile is one batch and one ubatch: its backward needs all of its own K/V in one graph
        struct scope {
            llama_cparams & params;
            llama_walk_acc & acc;
            uint32_t batch, ubatch;
            ~scope() { params.n_batch = batch; params.n_ubatch = ubatch; acc.unbind(); }
        } guard{cparams, t.walk.acc, cparams.n_batch, cparams.n_ubatch};
        cparams.n_batch = cparams.n_ubatch = t.tile;
        t.walk.acc.bind(cparams);
        t.walk.first_unit = true; // each step is one unit: nothing inside it waits for a yield
        finished = !opt_tiles_unit(t);
    }
    if (finished) {
        s.tiles.reset();
    }
    return LLAMA_OPT_LEARN_MORE; // the next step trains on, or restores her
}

// One unit; false when the walk is over (its update made, or failed and its gradient cleared), and the
// reply's cursor at its end so the restoration follows.
bool llama_context::opt_tiles_unit(opt_tiles & t) {
    auto & s = *opt_reply;
    auto & walk = t.walk;
    auto & plan = walk.plan;
    const auto fail = [&](const char * what) {
        s.failure = opt_failure.empty() ? what : opt_failure;
        ggml_opt_reset(opt_ctx, false); // gradients accumulated for an update that will not happen
        s.cursor = s.n_train;
        return false;
    };
    const llama_walk_run & run = plan.runs[t.run];
    const uint32_t r0 = plan.begin(run);
    if (t.before) {
        if (!walk.decode_span(t.served, r0, /*shadow =*/ false)) {
            return fail("the context before a run of the reply did not decode");
        }
        t.before = false;
        t.chunk = run.j0;
        t.snaps.assign(run.j1 - run.j0, {});
        if (t.served < r0) {
            return true; // the context was a unit of its own; the run's first tile is the next
        }
        // nothing came before the run: its first tile is this unit, never an empty step
    }
    if (!t.reverse) {
        const llama_walk_chunk & c = plan.chunks[t.chunk];
        if ((t.chunk - run.j0) % t.stride == 0) {
            if (!walk.checkpoint(t.snaps[t.chunk - run.j0])) {
                return fail("a tile's recurrent checkpoint could not be read");
            }
        }
        if (!walk.decode_span(c.c0, c.c0 + c.n, /*shadow =*/ true)) {
            return fail("a tile's forward under the shadow did not decode");
        }
        if (++t.chunk == run.j1) {
            t.reverse = true;
            walk.acc.start_run(r0);
        }
        return true;
    }
    // the reverse pass, last tile first: back to its checkpoint, forward to its start, train it at the full
    // horizon. A tile that does not fit is refused, never retried at a shorter horizon: that would drop
    // attention gradient and still call itself exact (Codex). The seat learns at a smaller tile instead.
    const size_t j = --t.chunk;
    const llama_walk_chunk & c = plan.chunks[j];
    const size_t cp = walk.recr != nullptr ? (j - run.j0) - (j - run.j0) % t.stride : j - run.j0;
    if (!walk.rewind(plan.chunks[run.j0 + cp].c0, t.snaps[cp], c.c0)) {
        return fail("a tile's rewind to its checkpoint failed");
    }
    const bool state_surrogate = walk.recr != nullptr && j + 1 < run.j1 && walk.acc.state_ready;
    llama_walk_chunk cc = c;
    cc.grad_from = std::max(r0, llama_walk_plan::grad_from(cc.c0, opt_walk_horizon));
    walk.set_flags(true, cc.grad_from, c.surrogate, state_surrogate);
    walk.xc = &cc;
    const bool ok = walk.train_chunk(cc.c0, cc.n);
    walk.xc = nullptr;
    walk.set_flags(false, 0, false, false);
    opt_walk_horizon_used = opt_walk_horizon;
    if (!ok) {
        return fail("a tile did not fit at its full gradient horizon: learn this reply at a smaller tile");
    }
    if (j != run.j0) {
        return true;
    }
    walk.acc.state_ready = false;
    if (t.run + 1 < plan.runs.size()) {
        // the run restored to her stable self, so the next run reads it as served
        if (!walk.restore(run, t.snaps[0])) {
            return fail("a run of the reply could not be restored under her stable set");
        }
        t.served = plan.end(run);
        ++t.run;
        t.before = true;
        t.reverse = false;
        return true;
    }
    // the last run's first tile ended the period: the update is made; the reply's restoration follows
    s.labels_done = s.n_labels;
    s.cursor = s.n_train;
    return false;
}

bool llama_opt_learn_tile(llama_context * ctx, uint32_t tile) { return ctx && ctx->opt_learn_tile(tile); }
