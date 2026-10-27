// A sequence's saved state restores into a context with a different stream count.
//
// A non-unified KV cache has one stream per parallel sequence, so a server relaunched with
// another slot count (3 -> 2 lanes) used to refuse every saved slot with "n_stream mismatch",
// and every sequence re-prefilled its whole context cold. One sequence's cells live in a single
// stream whatever the count, so they restore into the stream the new context gives that sequence.
//
// Checked here: a sequence saved at n_parallel=3 (non-unified) restores at n_parallel=2
// (non-unified) and into a unified cache, and the next token's logits after the restore equal
// the original context's, so the restored KV is the same KV, not merely accepted.

#include "arg.h"
#include "common.h"
#include "llama.h"

#include <cmath>
#include <cstdio>
#include <vector>

static llama_context * make_ctx(llama_model * model, const common_params & base, int n_parallel, bool unified) {
    common_params params = base;
    params.n_parallel = n_parallel;
    params.kv_unified = unified;
    params.n_ctx = 128 * n_parallel;
    llama_context_params cparams = common_context_params_to_llama(params);
    return llama_init_from_model(model, cparams);
}

// decode `tok` at `pos` on `seq` and copy the logits out
static bool next_logits(llama_context * ctx, llama_token tok, int pos, llama_seq_id seq, int n_vocab, std::vector<float> & out) {
    llama_batch batch = llama_batch_init(1, 0, 1);
    common_batch_add(batch, tok, pos, {seq}, true);
    const bool ok = llama_decode(ctx, batch) == 0;
    llama_batch_free(batch);
    if (!ok) {
        return false;
    }
    const float * logits = llama_get_logits_ith(ctx, -1);
    out.assign(logits, logits + n_vocab);
    return true;
}

static float max_abs_diff(const std::vector<float> & a, const std::vector<float> & b) {
    float worst = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        worst = std::fmax(worst, std::fabs(a[i] - b[i]));
    }
    return worst;
}

int main(int argc, char ** argv) {
    common_params params;
    params.sampling.seed = 1234;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    ggml_backend_load_all();

    llama_model_params mparams = common_model_params_to_llama(params);
    llama_model * model = llama_model_load_from_file(params.model.path.c_str(), mparams);
    if (model == nullptr) {
        fprintf(stderr, "%s : failed to load model\n", __func__);
        return 1;
    }
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    const llama_seq_id seq = 1;
    const int n_prompt = 40;

    // the writer: three lanes, one stream each
    llama_context * writer = make_ctx(model, params, 3, false);
    if (writer == nullptr) {
        fprintf(stderr, "%s : failed to create the writer context\n", __func__);
        return 1;
    }
    {
        llama_batch batch = llama_batch_init(n_prompt, 0, 1);
        for (int i = 0; i < n_prompt; ++i) {
            common_batch_add(batch, (llama_token) (1 + i % 7), i, {seq}, false);
        }
        if (llama_decode(writer, batch)) {
            fprintf(stderr, "%s : failed to decode the prompt\n", __func__);
            return 1;
        }
        llama_batch_free(batch);
    }

    std::vector<uint8_t> state(llama_state_seq_get_size(writer, seq));
    if (llama_state_seq_get_data(writer, state.data(), state.size(), seq) != state.size()) {
        fprintf(stderr, "%s : failed to save seq %d\n", __func__, seq);
        return 1;
    }

    const llama_token next = 3;
    std::vector<float> expected;
    if (!next_logits(writer, next, n_prompt, seq, n_vocab, expected)) {
        fprintf(stderr, "%s : failed to decode the next token on the writer\n", __func__);
        return 1;
    }
    llama_free(writer);

    struct reader_layout { int n_parallel; bool unified; const char * name; };
    const reader_layout readers[] = {
        { 2, false, "2 lanes, non-unified" },
        { 2, true,  "2 lanes, unified" },
    };

    int failures = 0;
    for (const auto & layout : readers) {
        llama_context * reader = make_ctx(model, params, layout.n_parallel, layout.unified);
        if (reader == nullptr) {
            fprintf(stderr, "%s : [%s] failed to create the context\n", __func__, layout.name);
            ++failures;
            continue;
        }
        const size_t nset = llama_state_seq_set_data(reader, state.data(), state.size(), seq);
        if (nset != state.size()) {
            fprintf(stderr, "%s : [%s] FAILED to restore seq %d saved at 3 lanes (got %zu of %zu bytes)\n",
                    __func__, layout.name, seq, nset, state.size());
            ++failures;
            llama_free(reader);
            continue;
        }
        std::vector<float> got;
        if (!next_logits(reader, next, n_prompt, seq, n_vocab, got)) {
            fprintf(stderr, "%s : [%s] FAILED to decode after the restore\n", __func__, layout.name);
            ++failures;
            llama_free(reader);
            continue;
        }
        const float diff = max_abs_diff(expected, got);
        // the same KV and the same token give the same logits; the bound only absorbs
        // floating-point reassociation between batch layouts
        if (diff > 1e-3f) {
            fprintf(stderr, "%s : [%s] FAILED: logits after the restore differ (max |d| = %g)\n", __func__, layout.name, diff);
            ++failures;
        } else {
            fprintf(stderr, "%s : [%s] restored %zu bytes; next-token logits match (max |d| = %g)\n",
                    __func__, layout.name, nset, diff);
        }
        llama_free(reader);
    }

    llama_model_free(model);

    if (failures > 0) {
        fprintf(stderr, "%s : %d layout(s) failed\n", __func__, failures);
        return 1;
    }
    fprintf(stderr, "%s : SUCCESS - a sequence restores across stream counts\n", __func__);
    return 0;
}
