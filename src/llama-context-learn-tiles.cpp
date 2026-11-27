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
    // EVERY UNIT IS ONE TILE OF WORK (Codex on c6d40d7917; Joel: "everyone comply"). A span the walk must decode
    // (the context before a run, a reverse replay from its stride checkpoint, a run restored under her stable set)
    // is pending here and decoded one tile per unit, so no unit holds the Core longer than the tile it was sized by.
    uint32_t span_at = 0, span_to = 0;
    bool span_shadow = false;
    bool rewound = false;  // the reverse tile's checkpoint is set back; its replay (if any) is the pending span

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
    if (cparams.kv_unified && cparams.n_seq_max > 1) {
        // the walk's contract, stated where it starts (llama-graph-train.cpp reads her window as the cells
        // [0, n_win) of one stream): her positions are her stream's slots. A cache shared by several sequences
        // puts her cells after others', and the walk would read theirs. The Core keeps each mind's KV in her own
        // stream; a context that does not is never here for tiles, said before anything runs.
        opt_fail("llama_opt_learn_tile: tiles walk her own stream; this context shares one cache among sequences");
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
    if (getenv("WALK_TILES_DEBUG")) { for (const auto & r : walk.plan.runs) { for (size_t j = r.j0; j < r.j1; ++j) { const auto & c = walk.plan.chunks[j]; fprintf(stderr, "TILES chunk %zu c0=%u n=%u labels=%u scale=%f period_end=%d grad_from=%u surrogate=%d\n", j, c.c0, c.n, c.n_labels, c.loss_scale, (int) c.period_end, c.grad_from, (int) c.surrogate); } } fprintf(stderr, "TILES prefix=%d n_train=%d plan.n_labels=%u last_label=%lld tokens=%zu stride=%zu host_bytes=%zu\n", (int) s.prefix, s.n_train, walk.plan.n_labels, (long long) walk.plan.last_label, tiles->tokens.size(), tiles->stride, bytes); }
    s.tiles = std::move(tiles);
    return true;
}

// At begin: a reply whose labelled rows reach past one ubatch can only learn as an exact tile walk, so the context
// must be able to run one: the exact walk configured, and her own stream (a cache shared among sequences puts her
// cells after others'). A context that cannot is the caller's contract broken, refused before anything moves.
bool llama_context::opt_tiles_admit(const llama_token * labels, int32_t n) {
    int32_t n_train = n;
    while (n_train > 0 && labels[n_train - 1] < 0) { --n_train; }
    if (n_train <= (int32_t) n_ubatch()) {
        return true;
    }
    if (cparams.kv_unified && cparams.n_seq_max > 1) {
        opt_fail("llama_opt_learn: a reply past one ubatch walks in tiles over her own stream; this context shares one cache among sequences");
        return false;
    }
    if (!opt_walk_exact) {
        opt_fail("llama_opt_learn: a reply past one ubatch learns only through the exact walk (walk_exact)");
        return false;
    }
    return true;
}

// A reply's first gradient step, when no tile was asked for. decode() splits a chunk past one ubatch into
// ubatches that each run their own backward, so the gradient a later row carries into the earlier ones would be
// lost (measured on Qwen3.5-0.8B at ubatch 64, untiled: cosine 0.949, norm 0.889 against the exact step, exact walk
// on or off). Such a reply walks exactly, in tiles of one ubatch; a reply within one ubatch is one exact graph.
// The one-graph quantum (graph_nodes) widens its ubatch to the reply and never splits.
// Whether this context can walk the reply exactly is the caller's contract, said at begin before anything is
// touched (opt_tiles_admit); only what the walk measures as it starts (its host budget, its accumulator) can
// still stop it here.
// false: the walk could not start; the reply goes to its restoration with that reason, the way a refused chunk
// does, and the context's stop flags are cleared so the restoring decodes and a neighbour's run as served.
bool llama_context::opt_tiles_past_ubatch() {
    auto & s = *opt_reply;
    if (s.tiles || s.restoring || s.cursor != 0 || s.graph_nodes != 0 || s.n_train <= (int32_t) n_ubatch()) {
        return true;
    }
    if (opt_learn_tile(n_ubatch())) {
        return true;
    }
    s.failure = opt_failure;
    s.cursor = s.n_train;
    opt_stop_requested.store(false);
    opt_alloc_failed.store(false);
    return false;
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
        opt_last_unit = {0, 0};
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
    // a pending span goes first, one tile of it per unit: the unit's work is never more than the tile
    const auto span = [&](uint32_t from, uint32_t to, bool shadow) {
        t.span_at = from;
        t.span_to = to;
        t.span_shadow = shadow;
    };
    if (t.span_at < t.span_to) {
        const uint32_t to = std::min(t.span_to, t.span_at + t.tile);
        if (!walk.decode_span(t.span_at, to, t.span_shadow)) {
            return fail("a tile of a span the walk decodes (context, replay or restore) did not decode");
        }
        opt_last_unit.decoded += to - t.span_at;
        t.span_at = to;
        return true;
    }
    const llama_walk_run & run = plan.runs[t.run];
    const uint32_t r0 = plan.begin(run);
    if (t.before) {
        t.before = false;
        t.chunk = run.j0;
        t.snaps.assign(run.j1 - run.j0, {});
        if (t.served < r0) {
            span(t.served, r0, /*shadow =*/ false); // the context before the run, under her stable set
            return true;
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
        opt_last_unit.decoded += c.n;
        if (!walk.decode_span(c.c0, c.c0 + c.n, /*shadow =*/ true)) {
            return fail("a tile's forward under the shadow did not decode");
        }
        if (++t.chunk == run.j1) {
            t.reverse = true;
            walk.acc.start_run(r0);
        }
        return true;
    }
    // the reverse pass, last tile first: back to its checkpoint, forward to its start (one tile per unit), then
    // train it at the full horizon
    const size_t j = t.chunk - 1;
    const llama_walk_chunk & c = plan.chunks[j];
    if (!t.rewound) {
        const size_t cp = walk.recr != nullptr ? (j - run.j0) - (j - run.j0) % t.stride : j - run.j0;
        const uint32_t at = plan.chunks[run.j0 + cp].c0;
        if (!walk.rewind(at, t.snaps[cp], at)) {
            return fail("a tile's rewind to its checkpoint failed");
        }
        t.rewound = true;
        if (at < c.c0) {
            span(at, c.c0, /*shadow =*/ true); // the replay from the stride checkpoint to the tile
            return true;
        }
    }
    t.chunk = j;
    t.rewound = false;
    if (getenv("WALK_TILES_DEBUG")) { fprintf(stderr, "TILES reverse j=%zu c0=%u pos_max(seq %d)=%d pos_min=%d\n", j, c.c0, (int) s.seq, (int) memory->seq_pos_max(s.seq), (int) memory->seq_pos_min(s.seq)); }
    const bool state_surrogate = walk.recr != nullptr && j + 1 < run.j1 && walk.acc.state_ready;
    llama_walk_chunk cc = c;
    cc.grad_from = std::max(r0, llama_walk_plan::grad_from(cc.c0, opt_walk_horizon));
    walk.set_flags(true, cc.grad_from, c.surrogate, state_surrogate);
    walk.xc = &cc;
    opt_last_unit.decoded += cc.n;
    opt_last_unit.trained += cc.n;
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
        // the run restored to her stable self, one tile per unit, so the next run reads it as served
        if (!walk.rewind(r0, t.snaps[0], r0)) {
            return fail("a run of the reply could not be set back to its checkpoint");
        }
        span(r0, plan.end(run), /*shadow =*/ false);
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

llama_opt_learn_unit llama_opt_learn_last_unit(const llama_context * ctx) { return ctx ? ctx->opt_last_unit : llama_opt_learn_unit{0, 0}; }
