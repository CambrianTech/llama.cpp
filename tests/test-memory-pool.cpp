// what this catches: the engine's memory pool (tools/server/engine/memory_pool_llama, CONTINUUM-ENGINE 4.12.1)
// claiming a state moved when it did not: a load reported loaded with only the target taken, a refused load leaving
// part of a sequence behind in either context, a save writing past a caller's buffer, or a load overwriting a
// sequence that already held state. Model-backed like test-state-restore-fragmented (a second context of the same
// model stands in for the draft); the validators are checked model-free first.

#include "arg.h"
#include "common.h"
#include "llama.h"

#include "../tools/server/engine/memory_pool_llama.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

static void require(bool condition, const char * message) {
    if (!condition) {
        fprintf(stderr, "FAILED: %s\n", message);
        std::exit(1);
    }
}

static bool holds(llama_context * ctx, llama_seq_id seq) {
    return llama_memory_seq_pos_max(llama_get_memory(ctx), seq) >= 0;
}

static void validators() {
    seq_state_size want;
    want.tgt = 100;
    want.dft = 40;
    require(save_pair_valid(want, 100, 40), "exact buffers are valid");
    require(!save_pair_valid(want, 99, 40) && !save_pair_valid(want, 100, 39), "a short buffer is invalid");
    require(!save_pair_valid(want, 101, 40), "a larger buffer is invalid too: a save fills its buffers exactly");
    require(load_pair_valid(true, 10, 5) && load_pair_valid(false, 10, 0), "target always, draft exactly when present");
    require(!load_pair_valid(true, 10, 0), "a draft pool given no draft state");
    require(!load_pair_valid(false, 10, 5), "a target-only pool given a draft state");
    require(!load_pair_valid(false, 0, 0), "no target state");
}

int main(int argc, char ** argv) {
    validators();

    common_params params;
    params.kv_unified = true;
    params.n_parallel = 3;
    params.n_ctx      = 512;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }
    ggml_backend_load_all();
    common_init_result_ptr init = common_init_from_params(params);
    llama_model *   model = init->model();
    llama_context * tgt   = init->context();
    require(model != nullptr && tgt != nullptr, "model loaded");
    llama_context * dft = llama_init_from_model(model, common_context_params_to_llama(params));
    require(dft != nullptr, "a second context stands in for the draft");

    memory_pool_llama pool(tgt, dft);

    // sequence 0 holds 40 tokens in both contexts
    llama_batch batch = llama_batch_init(40, 0, 1);
    for (int i = 0; i < 40; ++i) {
        common_batch_add(batch, 1, i, { 0 }, i == 39);
    }
    require(pool.place(batch) == placement::placed, "the target places the prompt");
    require(llama_decode(dft, batch) == 0, "the draft holds it too");

    const seq_state_size n = pool.state_size(0, LLAMA_STATE_SEQ_FLAGS_NONE);
    require(n.tgt > 0 && n.dft > 0, "both contexts report state");
    std::vector<uint8_t> t(n.tgt), d(n.dft), short_t(n.tgt - 1);

    require(pool.save(0, LLAMA_STATE_SEQ_FLAGS_NONE, { short_t.data(), short_t.size() }, { d.data(), d.size() }) ==
                save_outcome::invalid, "a short buffer is refused before any byte is written");
    require(pool.save(0, LLAMA_STATE_SEQ_FLAGS_NONE, { t.data(), t.size() }, { d.data(), d.size() }) == save_outcome::saved,
            "an exact save");

    // round trip into sequence 1
    require(pool.load(1, LLAMA_STATE_SEQ_FLAGS_NONE, { t.data(), t.size() }, { d.data(), d.size() }) == load_outcome::loaded,
            "loaded into an empty sequence");
    require(holds(tgt, 1) && holds(dft, 1), "both contexts hold it");
    require(pool.range(1).max == pool.range(0).max, "the same positions");

    // a destination that already holds state is refused, untouched
    require(pool.load(1, LLAMA_STATE_SEQ_FLAGS_NONE, { t.data(), t.size() }, { d.data(), d.size() }) == load_outcome::invalid,
            "never loaded over a sequence that holds state");
    require(holds(tgt, 1) && holds(dft, 1), "and that sequence is untouched");

    // malformed pairs, before any change
    require(pool.load(2, LLAMA_STATE_SEQ_FLAGS_NONE, { t.data(), t.size() }, {}) == load_outcome::invalid,
            "a draft pool given no draft state");
    require(!holds(tgt, 2) && !holds(dft, 2), "nothing loaded");

    // a truncated target state is refused and leaves nothing in either context
    require(pool.load(2, LLAMA_STATE_SEQ_FLAGS_NONE, { t.data(), t.size() / 2 }, { d.data(), d.size() }) ==
                load_outcome::refused, "a truncated target state");
    require(!holds(tgt, 2) && !holds(dft, 2), "refused means empty in both contexts");

    // a corrupted draft state, after the target took its half: the target's half is removed
    std::vector<uint8_t> bad_d(d.size(), 0xFF);
    require(pool.load(2, LLAMA_STATE_SEQ_FLAGS_NONE, { t.data(), t.size() }, { bad_d.data(), bad_d.size() }) ==
                load_outcome::refused, "a corrupted draft state");
    require(!holds(tgt, 2) && !holds(dft, 2), "refused means empty in both contexts, the target's half included");

    // move: a whole sequence to another id in both contexts
    require(pool.move(1, 2) == edit_outcome::done, "a whole-sequence move");
    require(!holds(tgt, 1) && !holds(dft, 1) && holds(tgt, 2) && holds(dft, 2), "moved in both contexts");

    llama_batch_free(batch);
    llama_free(dft);
    fprintf(stderr, "test-memory-pool: ok\n");
    return 0;
}
