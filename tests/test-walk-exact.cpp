// what this catches: the exact walk (llama_opt_params::walk_exact) training on anything other
// than the gradient of the window's whole loss. The plain walk stops the gradient at every
// chunk boundary; the exact walk carries each chunk's gradient back through the cached K/V of
// every chunk before it, so how the window is CHUNKED must not change the step it takes.
//
// One SGD step from the same fresh adapter, read off the adapter itself: SGD moves every A and B
// by exactly -lr * gradient, so the steps compare gradients directly (Fable: compare gradients,
// not losses; a loss-change proxy was measured to be nonlinear at practical step sizes):
//   1. every position labelled: a chunk of the whole window is ONE graph, the true gradient; the
//      exact walk in 4 chunks must match it, and the plain walk in 4 chunks must not
//   2. context, then a reply (a masked window, the walk's real case): the exact walk in 4 chunks
//      must take the step it takes with one chunk per run, and the plain walk must not
//
// Run on a pure-attention model and on a hybrid (attention + gated delta-net):
//   test-walk-exact -m Qwen2.5-Coder-1.5B-Instruct-Q4_K_M.gguf
//   test-walk-exact -m Qwen3.5-0.8B-Q8_0.gguf

#include "arg.h"
#include "common.h"
#include "llama.h"
#include "../src/llama-adapter.h"

#include <cmath>
#include <map>
#include <cstdlib>
#include <cstdio>
#include <string>
#include <vector>

static const float    LR     = 1e-3f; // an SGD step is linear in the gradient at any lr
static const uint32_t WINDOW = 512;

static ggml_opt_optimizer_params sgd_pars(void *) {
    ggml_opt_optimizer_params p = ggml_opt_get_default_optimizer_params(nullptr);
    p.sgd.alpha = LR;
    p.sgd.wd    = 0.0f;
    return p;
}

// the cache the walk reads its prefix from: F32 for the exactness checks; the served types (f16,
// q8_0, in a flash-attention context that stores V untransposed) measure the straight-through
// approximation the walk makes there (Fable on #47: the gradient is taken at the cache's values)
static ggml_type g_cache_type = GGML_TYPE_F32;
// recurrent rollback slots in the context (serving keeps some for speculative decoding)
static uint32_t g_n_rs_seq = 0;

static llama_context * make_ctx(const common_params & params, llama_model * model, uint32_t chunk) {
    auto cparams = common_context_params_to_llama(params);
    cparams.n_ctx           = WINDOW;
    cparams.n_batch         = chunk;
    cparams.n_ubatch        = chunk;
    cparams.n_seq_max       = 2;    // the plain walk snapshots a recurrent state into a scratch sequence
    cparams.kv_unified      = true; // ...and the window keeps every cell
    // a quantized V cache needs flash attention (it stores V untransposed); training graphs take
    // the explicit path either way
    cparams.flash_attn_type = g_cache_type == GGML_TYPE_F32 ? LLAMA_FLASH_ATTN_TYPE_DISABLED : LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cparams.type_k          = g_cache_type;
    cparams.type_v          = g_cache_type;
    cparams.n_rs_seq        = g_n_rs_seq;
    return llama_init_from_model(model, cparams);
}

// every A and B tensor of the adapter, in name order: SGD moves them by exactly -lr * gradient
static std::vector<float> adapter_params(const llama_adapter_lora * adapter) {
    std::map<std::string, const llama_adapter_lora_weight *> sorted;
    for (const auto & [name, w] : adapter->ab_map) {
        sorted[name] = &w;
    }
    std::vector<float> out;
    for (const auto & [name, w] : sorted) {
        for (ggml_tensor * t : { w->a, w->b }) {
            GGML_ASSERT(t->type == GGML_TYPE_F32);
            const size_t n = out.size();
            out.resize(n + ggml_nelements(t));
            ggml_backend_tensor_get(t, out.data() + n, 0, ggml_nbytes(t));
        }
    }
    return out;
}

// one SGD step from the fresh adapter: the step itself, -lr * the gradient it took
static std::vector<float> step_delta(const common_params & params, llama_model * model, const std::string & init,
                                     const std::vector<llama_token> & tokens, const std::vector<uint8_t> & labelled,
                                     uint32_t chunk, bool exact, size_t host_budget = 0, size_t * host_bytes = nullptr) {
    llama_adapter_lora * adapter = llama_adapter_lora_init(model, init.c_str());
    GGML_ASSERT(adapter != nullptr);
    const std::vector<float> before = adapter_params(adapter);

    llama_context * ctx = make_ctx(params, model, chunk);
    float scale = 1.0f;
    GGML_ASSERT(llama_set_adapters_lora(ctx, &adapter, 1, &scale) == 0);
    llama_opt_params lopt{};
    lopt.n_ctx_train     = 0;
    lopt.param_filter    = llama_opt_param_filter_all;
    lopt.get_opt_pars    = sgd_pars;
    lopt.optimizer_type  = GGML_OPT_OPTIMIZER_TYPE_SGD;
    lopt.adapter         = adapter;
    lopt.walk_exact      = exact;
    lopt.walk_horizon    = 0;
    llama_opt_init(ctx, model, lopt);
    llama_opt_set_walk_host_budget(ctx, host_budget);
    std::vector<std::vector<llama_token>> seqs = { std::vector<llama_token>(tokens.begin(), tokens.begin() + WINDOW + 1) };
    std::vector<std::vector<uint8_t>>     loss = { labelled };
    ggml_opt_dataset_t dataset = common_opt_dataset_init_masked(WINDOW, seqs, loss, tokens[0]);
    ggml_opt_result_t  result  = ggml_opt_result_init();
    llama_opt_epoch(ctx, dataset, result, nullptr, /*idata_split =*/ 1, nullptr, nullptr);
    GGML_ASSERT(!llama_opt_failed(ctx));
    {
        double l = 0.0, unc = 0.0;
        ggml_opt_result_loss(result, &l, &unc);
        printf("  chunk %3u %-5s: training loss %.6f (the forward the step saw)\n", chunk, exact ? "exact" : "plain", l);
    }
    if (host_bytes) {
        *host_bytes = llama_opt_walk_host_bytes(ctx);
    }
    ggml_opt_result_free(result);
    ggml_opt_dataset_free(dataset);
    llama_free(ctx);

    const std::vector<float> after = adapter_params(adapter);
    llama_adapter_lora_free(adapter);
    std::vector<float> d(before.size());
    for (size_t i = 0; i < d.size(); ++i) {
        d[i] = after[i] - before[i];
    }
    return d;
}

struct agreement {
    double cosine;
    double norm_ratio;
};

// the angle and the size of a step against the reference's: a wrong direction (a misrouted
// gradient) shows in the cosine, a wrong scale (a misweighted loss) in the norm ratio
static agreement compare(const char * what, const std::vector<float> & a, const std::vector<float> & ref) {
    double dot = 0.0, na = 0.0, nr = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += (double) a[i] * ref[i];
        na  += (double) a[i] * a[i];
        nr  += (double) ref[i] * ref[i];
    }
    const agreement r = { dot / std::sqrt(na * nr), std::sqrt(na / nr) };
    printf("  %s: cosine %.4f, norm ratio %.4f\n", what, r.cosine, r.norm_ratio);
    return r;
}

// Measured on the 1.5B Q4_K_M (5090): the exact walk agrees with one graph at cosine 0.998 and
// norm 0.995 (what is left is the quantized matmuls' batch-size variance: a 128-row chunk and a
// 512-row graph take different kernels); the plain walk sits at cosine 0.81, norm 2.1.
// the served cache types' bar (see 1b). Measured on the 5090: f16 0.9986 (1.5B) / 0.9981 (hybrid),
// q8_0 0.9978 / 0.9979, against F32's 0.998: the straight-through approximation costs nothing
// measurable, so the bar sits just under F32's own agreement.
static const double STRAIGHT_THROUGH_COSINE = 0.99;

static bool same_step(const agreement & r) {
    return r.cosine > 0.995 && std::fabs(r.norm_ratio - 1.0) < 0.02;
}

int main(int argc, char ** argv) {
    common_params params;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }
    params.no_extra_bufts = true; // OUT_PROD dequantizes the standard layout only
    llama_backend_init();
    auto mparams = common_model_params_to_llama(params);
    llama_model * model = llama_model_load_from_file(params.model.path.c_str(), mparams);
    GGML_ASSERT(model != nullptr);

    const std::string init = "test-walk-exact.init.gguf";
    GGML_ASSERT(common_lora_write_fresh(model, init, 8, 16, { "attn_q", "attn_v" }, 42));

    // words in a fixed pseudo-random order: a window the model cannot already predict, so its
    // loss, and the gradient under test, is well above the kernels' batch-size noise
    static const char * words[] = { "room", "card", "claim", "review", "patch", "lease", "board", "turn",
        "engine", "window", "chunk", "cache", "gradient", "adapter", "verdict", "submission" };
    std::string text;
    uint32_t lcg = 12345;
    while (text.size() < 8 * WINDOW) {
        lcg = lcg * 1664525u + 1013904223u;
        text += words[(lcg >> 16) % 16];
        text += (lcg >> 8) % 7 == 0 ? ".\n" : " ";
    }
    llama_context * tok_ctx = make_ctx(params, model, WINDOW);
    std::vector<llama_token> tokens = common_tokenize(tok_ctx, text, true);
    llama_free(tok_ctx);
    GGML_ASSERT(tokens.size() > WINDOW + 1);

    int failures = 0;
    {
        // 1. every position labelled: chunk = window is one graph, the true gradient
        std::vector<uint8_t> all(WINDOW + 1, 1);
        all[0] = 0;
        const auto ref   = step_delta(params, model, init, tokens, all, WINDOW, false);
        size_t full_bytes = 0;
        const auto exact = step_delta(params, model, init, tokens, all, WINDOW / 4, true, 0, &full_bytes);
        const auto plain = step_delta(params, model, init, tokens, all, WINDOW / 4, false);
        if (!same_step(compare("all labelled: exact walk in 4 chunks vs one graph", exact, ref))) {
            fprintf(stderr, "FAILED: the exact walk's step is not the window's gradient\n");
            ++failures;
        }
        if (same_step(compare("all labelled: plain walk in 4 chunks vs one graph", plain, ref))) {
            fprintf(stderr, "FAILED: the plain walk matches one graph too: this window cannot tell the two apart\n");
            ++failures;
        }
        if (llama_model_is_hybrid(model)) {
            // a host budget below every chunk's state snapshot: the walk checkpoints every few
            // chunks and decodes forward from the checkpoint, and the step must not change
            size_t strided_bytes = 0;
            const auto strided = step_delta(params, model, init, tokens, all, WINDOW / 4, true, full_bytes * 3 / 4, &strided_bytes);
            printf("  host memory: %.1f MiB with a snapshot per chunk, %.1f MiB checkpointed\n", full_bytes / 1048576.0, strided_bytes / 1048576.0);
            if (!(strided_bytes < full_bytes) || !same_step(compare("all labelled: checkpointed state vs a snapshot per chunk", strided, exact))) {
                fprintf(stderr, "FAILED: checkpointing the recurrent state changed the step (or saved nothing)\n");
                ++failures;
            }
        }
    }
    {
        // 1b. the served cache types: the walk reads the prefix at the cache's precision and its
        // gradient is taken there, then applied to the chunk's own full-precision K/V (straight-
        // through). Measured against the same one-graph reference, which never reads the cache.
        std::vector<uint8_t> all(WINDOW + 1, 1);
        all[0] = 0;
        const auto ref = step_delta(params, model, init, tokens, all, WINDOW, false);
        for (ggml_type t : { GGML_TYPE_F16, GGML_TYPE_Q8_0 }) {
            g_cache_type = t;
            const auto exact = step_delta(params, model, init, tokens, all, WINDOW / 4, true);
            g_cache_type = GGML_TYPE_F32;
            const agreement r = compare((std::string("all labelled, ") + ggml_type_name(t) + " cache: exact walk in 4 chunks vs one graph").c_str(), exact, ref);
            if (!(r.cosine > STRAIGHT_THROUGH_COSINE)) {
                fprintf(stderr, "FAILED: at a %s cache the exact walk's step strays from the window's gradient (cosine %.4f)\n", ggml_type_name(t), r.cosine);
                ++failures;
            }
        }
    }
    {
        // 2. context, then a reply (the walk's real case): the exact walk is invariant to how the
        // window is chunked, context chunks included (they train through the surrogate alone)
        std::vector<uint8_t> reply(WINDOW + 1, 0);
        for (uint32_t i = WINDOW / 2 + 37; i <= WINDOW; ++i) {
            reply[i] = 1;
        }
        const auto runs   = step_delta(params, model, init, tokens, reply, WINDOW, true); // one chunk per run
        const auto exact4 = step_delta(params, model, init, tokens, reply, WINDOW / 4, true);
        const auto plain4 = step_delta(params, model, init, tokens, reply, WINDOW / 4, false);
        if (!same_step(compare("context + reply: exact walk in 4 chunks vs one chunk per run", exact4, runs))) {
            fprintf(stderr, "FAILED: the exact walk's step depends on the chunking\n");
            ++failures;
        }
        if (same_step(compare("context + reply: plain walk in 4 chunks vs the exact walk", plain4, runs))) {
            fprintf(stderr, "FAILED: the plain walk matches the exact one: this window cannot tell the two apart\n");
            ++failures;
        }
    }

    {
        // 3. the host budget: a window whose accumulators and snapshots exceed it refuses by
        // name before anything runs, and the adapter is untouched
        llama_adapter_lora * adapter = llama_adapter_lora_init(model, init.c_str());
        const std::vector<float> before = adapter_params(adapter);
        llama_context * ctx = make_ctx(params, model, WINDOW / 4);
        float scale = 1.0f;
        GGML_ASSERT(llama_set_adapters_lora(ctx, &adapter, 1, &scale) == 0);
        llama_opt_params lopt{};
        lopt.param_filter   = llama_opt_param_filter_all;
        lopt.get_opt_pars   = sgd_pars;
        lopt.optimizer_type = GGML_OPT_OPTIMIZER_TYPE_SGD;
        lopt.adapter        = adapter;
        lopt.walk_exact     = true;
        llama_opt_init(ctx, model, lopt);
        llama_opt_set_walk_host_budget(ctx, 1);
        std::vector<std::vector<llama_token>> seqs = { std::vector<llama_token>(tokens.begin(), tokens.begin() + WINDOW + 1) };
        std::vector<std::vector<uint8_t>>     loss = { std::vector<uint8_t>(WINDOW + 1, 1) };
        ggml_opt_dataset_t dataset = common_opt_dataset_init_masked(WINDOW, seqs, loss, tokens[0]);
        llama_opt_epoch(ctx, dataset, nullptr, nullptr, /*idata_split =*/ 1, nullptr, nullptr);
        const std::string why = llama_opt_failure(ctx);
        printf("  host budget of 1 byte: failed=%d, \"%s\", needs %zu MiB\n", (int) llama_opt_failed(ctx), why.c_str(), llama_opt_walk_host_bytes(ctx) >> 20);
        if (!llama_opt_failed(ctx) || why.find("host memory") == std::string::npos || adapter_params(adapter) != before) {
            fprintf(stderr, "FAILED: a window over the host budget was not refused by name before it ran\n");
            ++failures;
        }
        ggml_opt_dataset_free(dataset);
        llama_free(ctx);
        llama_adapter_lora_free(adapter);
    }

    if (llama_model_is_hybrid(model)) {
        // 4. regression for the 5090 crash (2026-10-07 04:09Z): a recurrent context with rollback
        // slots (serving's, inherited by the training context) made the exact walk ASSERT inside
        // the serving process. It must refuse by name, and the adapter stays untouched.
        llama_adapter_lora * adapter = llama_adapter_lora_init(model, init.c_str());
        const std::vector<float> before = adapter_params(adapter);
        g_n_rs_seq = 4;
        llama_context * ctx = make_ctx(params, model, WINDOW / 4);
        g_n_rs_seq = 0;
        float scale = 1.0f;
        GGML_ASSERT(llama_set_adapters_lora(ctx, &adapter, 1, &scale) == 0);
        llama_opt_params lopt{};
        lopt.param_filter   = llama_opt_param_filter_all;
        lopt.get_opt_pars   = sgd_pars;
        lopt.optimizer_type = GGML_OPT_OPTIMIZER_TYPE_SGD;
        lopt.adapter        = adapter;
        lopt.walk_exact     = true;
        llama_opt_init(ctx, model, lopt);
        std::vector<std::vector<llama_token>> seqs = { std::vector<llama_token>(tokens.begin(), tokens.begin() + WINDOW + 1) };
        std::vector<std::vector<uint8_t>>     loss = { std::vector<uint8_t>(WINDOW + 1, 1) };
        ggml_opt_dataset_t dataset = common_opt_dataset_init_masked(WINDOW, seqs, loss, tokens[0]);
        llama_opt_epoch(ctx, dataset, nullptr, nullptr, /*idata_split =*/ 1, nullptr, nullptr);
        const std::string why = llama_opt_failure(ctx);
        printf("  rollback slots: failed=%d, \"%s\"\n", (int) llama_opt_failed(ctx), why.c_str());
        if (!llama_opt_failed(ctx) || why.find("rollback") == std::string::npos || adapter_params(adapter) != before) {
            fprintf(stderr, "FAILED: a recurrent context with rollback slots was not refused by name\n");
            ++failures;
        }
        ggml_opt_dataset_free(dataset);
        llama_free(ctx);
        llama_adapter_lora_free(adapter);
    }

    std::remove(init.c_str());
    llama_model_free(model);
    llama_backend_free();
    if (failures) {
        return 1;
    }
    printf("test-walk-exact: OK\n");
    return 0;
}
