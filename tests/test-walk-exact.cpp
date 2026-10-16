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
//   7. per-layer recompute under the exact walk: the same masked window takes the same step with
//      the layer outputs as checkpoints as without them, in a smaller chunk graph
//
// Run on a pure-attention model and on a hybrid (attention + gated delta-net):
//   test-walk-exact -m Qwen2.5-Coder-1.5B-Instruct-Q4_K_M.gguf
//   test-walk-exact -m Qwen3.5-0.8B-Q8_0.gguf

#include "arg.h"
#include "common.h"
#include "llama.h"
#include "../src/llama-adapter.h"
#include "../src/llama-memory-hybrid.h"
#include "../src/llama-memory-recurrent.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <cstdlib>
#include <cstdio>
#include <string>
#include <chrono>
#include <vector>

static const float    LR     = 1e-3f; // an SGD step is linear in the gradient at any lr
static const uint32_t WINDOW = 512;
// the window a step trains: WINDOW for every case but the device refusal (6), whose horizon must
// change the chunk graph's size, which a window this short does not
static const uint32_t REFUSAL_WINDOW = 2048;
static uint32_t g_window = WINDOW;

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
// the gradient horizon a step asks for (0 = the whole window) and the device budget its graphs
// may add (0 = the device's own free figure), and what the last step measured: its largest
// graph and the horizon it ended at
static uint32_t g_walk_horizon     = 0;
static size_t   g_device_budget    = 0;
static size_t   g_last_graph_bytes = 0;
static uint32_t g_last_horizon     = 0;
// per-layer recompute (llama_opt_params::recompute): the layer outputs are the backward pass's
// checkpoints, so a chunk holds one layer's activations instead of every layer's
static bool     g_recompute        = false;

// counts the recurrent memory's "non-consecutive token position" warnings: a state restore that
// leaves the cell's position at the window's end makes every next chunk read as non-consecutive
// (the 5090 log, 2026-10-07: "position 65640 after 66345"), passing everything else through
static int g_nonconsecutive = 0;
// and, for the device refusal (6): the horizon retries, the ubatches that failed to prepare, and
// the chunk graphs the device budget refused
static int g_retries         = 0;
static int g_ubatch_failures = 0;
static int g_refusals        = 0;
static void count_log(ggml_log_level level, const char * text, void * ud) {
    if (text && strstr(text, "non-consecutive token position")) {
        ++g_nonconsecutive;
    }
    if (text && strstr(text, "did not fit with a gradient horizon")) {
        ++g_retries;
    }
    if (text && strstr(text, "failed to prepare attention ubatches")) {
        ++g_ubatch_failures;
    }
    if (text && strstr(text, "it may add: not allocating it")) {
        ++g_refusals;
    }
    GGML_UNUSED(level);
    GGML_UNUSED(ud);
    fputs(text, stderr);
}

static llama_context * make_ctx(const common_params & params, llama_model * model, uint32_t chunk) {
    auto cparams = common_context_params_to_llama(params);
    cparams.n_ctx           = g_window;
    cparams.n_batch         = chunk;
    cparams.n_ubatch        = chunk;
    cparams.kv_unified      = true; // the window keeps every cell
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

// the name and length of each tensor adapter_params lays out, in the same order
static std::vector<std::pair<std::string, size_t>> adapter_layout(const llama_adapter_lora * adapter) {
    std::map<std::string, const llama_adapter_lora_weight *> sorted;
    for (const auto & [name, w] : adapter->ab_map) {
        sorted[name] = &w;
    }
    std::vector<std::pair<std::string, size_t>> out;
    for (const auto & [name, w] : sorted) {
        out.emplace_back(name + ".a", (size_t) ggml_nelements(w->a));
        out.emplace_back(name + ".b", (size_t) ggml_nelements(w->b));
    }
    return out;
}
static std::vector<std::pair<std::string, size_t>> g_layout;

// one SGD step from the fresh adapter: the step itself, -lr * the gradient it took
static std::vector<float> step_delta(const common_params & params, llama_model * model, const std::string & init,
                                     const std::vector<llama_token> & tokens, const std::vector<uint8_t> & labelled,
                                     uint32_t chunk, bool exact, size_t host_budget = 0, size_t * host_bytes = nullptr,
                                     std::vector<uint8_t> * state_after = nullptr) {
    llama_adapter_lora * adapter = llama_adapter_lora_init(model, init.c_str());
    GGML_ASSERT(adapter != nullptr);
    const std::vector<float> before = adapter_params(adapter);
    g_layout = adapter_layout(adapter);

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
    lopt.walk_horizon    = g_walk_horizon;
    lopt.recompute       = g_recompute;
    llama_opt_set_memory_budget(ctx, g_device_budget);
    llama_opt_init(ctx, model, lopt);
    llama_opt_set_walk_host_budget(ctx, host_budget);
    std::vector<std::vector<llama_token>> seqs = { std::vector<llama_token>(tokens.begin(), tokens.begin() + g_window + 1) };
    std::vector<std::vector<uint8_t>>     loss = { labelled };
    ggml_opt_dataset_t dataset = common_opt_dataset_init_masked(g_window, seqs, loss, tokens[0]);
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
    g_last_graph_bytes = llama_opt_graph_bytes(ctx);
    g_last_horizon     = llama_opt_walk_horizon(ctx);
    if (state_after) {
        state_after->resize(llama_state_seq_get_size(ctx, 0));
        GGML_ASSERT(llama_state_seq_get_data(ctx, state_after->data(), state_after->size(), 0) == state_after->size());
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

// her stable self's cache after [0, r0) and then [r0, r1): no adapter, decoded in chunks from
// each start as the walk decodes them (a matmul's result depends on its batch size), so a walk
// that restores its run leaves exactly these bytes
static std::vector<uint8_t> stable_state(const common_params & params, llama_model * model,
                                         const std::vector<llama_token> & tokens, uint32_t r0, uint32_t r1, uint32_t chunk) {
    llama_context * ctx = make_ctx(params, model, chunk);
    llama_batch batch = llama_batch_init(chunk, 0, 1);
    for (const auto & [p0, p1] : { std::make_pair(0u, r0), std::make_pair(r0, r1) }) {
        for (uint32_t c0 = p0; c0 < p1; c0 += chunk) {
            const uint32_t n = std::min(c0 + chunk, p1) - c0;
            common_batch_clear(batch);
            for (uint32_t i = 0; i < n; ++i) {
                common_batch_add(batch, tokens[c0 + i], c0 + i, { 0 }, i + 1 == n);
            }
            GGML_ASSERT(llama_decode(ctx, batch) == 0);
        }
    }
    std::vector<uint8_t> state(llama_state_seq_get_size(ctx, 0));
    GGML_ASSERT(llama_state_seq_get_data(ctx, state.data(), state.size(), 0) == state.size());
    llama_batch_free(batch);
    llama_free(ctx);
    return state;
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

// a step that strays names where: the tensors whose own cosine is lowest, so a wrong op
// shows as the layers (recurrent or attention, which depth) that carry it
static void worst_tensors(const std::vector<float> & a, const std::vector<float> & ref, int n) {
    std::vector<std::pair<double, std::string>> rows;
    size_t off = 0;
    for (const auto & [name, len] : g_layout) {
        double dot = 0.0, na = 0.0, nr = 0.0;
        for (size_t i = off; i < off + len; ++i) {
            dot += (double) a[i] * ref[i];
            na  += (double) a[i] * a[i];
            nr  += (double) ref[i] * ref[i];
        }
        off += len;
        if (na > 0.0 && nr > 0.0) {
            rows.emplace_back(dot / std::sqrt(na * nr), name + " norm ratio " + std::to_string(std::sqrt(na / nr)));
        }
    }
    std::sort(rows.begin(), rows.end());
    for (int i = 0; i < n && i < (int) rows.size(); ++i) {
        printf("    worst: cosine %.4f %s\n", rows[i].first, rows[i].second.c_str());
    }
}

// Measured on the 1.5B Q4_K_M (5090): the exact walk agrees with one graph at cosine 0.998 and
// norm 0.995 (what is left is the quantized matmuls' batch-size variance: a 128-row chunk and a
// 512-row graph take different kernels); the plain walk sits at cosine 0.81, norm 2.1.

static bool same_step(const agreement & r) {
    return r.cosine > 0.995 && std::fabs(r.norm_ratio - 1.0) < 0.02;
}

static std::vector<uint8_t> sequence_state(llama_context * ctx, llama_seq_id seq) {
    std::vector<uint8_t> bytes(llama_state_seq_get_size(ctx, seq));
    GGML_ASSERT(llama_state_seq_get_data(ctx, bytes.data(), bytes.size(), seq) == bytes.size());
    return bytes;
}

static void mixed_decode(const common_params & params, llama_model * model, const std::string & init,
                         const std::vector<llama_token> & tokens, bool flash, ggml_type cache_type, uint32_t undo = 0) {
    auto cp = common_context_params_to_llama(params);
    cp.n_ctx = 128;
    cp.n_batch = cp.n_ubatch = undo ? 8 : 2;
    cp.n_seq_max = 2;
    cp.kv_unified = true;
    cp.n_rs_seq = undo;
    cp.flash_attn_type = flash ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.type_k = cp.type_v = cache_type;
    llama_context * ctx = llama_init_from_model(model, cp);
    GGML_ASSERT(ctx);
    llama_adapter_lora * adapter = llama_adapter_lora_init(model, init.c_str());
    float scale = 1.0f;
    GGML_ASSERT(llama_set_adapters_lora(ctx, &adapter, 1, &scale) == 0);
    llama_opt_params op{};
    op.param_filter = llama_opt_param_filter_all;
    op.get_opt_pars = sgd_pars;
    op.optimizer_type = GGML_OPT_OPTIMIZER_TYPE_SGD;
    op.adapter = adapter;
    llama_batch batch = llama_batch_init(2, 0, 1);
    common_batch_add(batch, tokens[0], 0, {0}, true);
    common_batch_add(batch, tokens[1], 0, {1}, true);
    GGML_ASSERT(llama_decode(ctx, batch) == 0);
    const auto reference_neighbor = sequence_state(ctx, 1);
    const auto reference_target = sequence_state(ctx, 0);
    batch.label[0] = tokens[2];
    batch.pos[0] = batch.pos[1] = 1;
    GGML_ASSERT(llama_decode(ctx, batch) != 0);
    GGML_ASSERT(sequence_state(ctx, 0) == reference_target && sequence_state(ctx, 1) == reference_neighbor);
    llama_memory_clear(llama_get_memory(ctx), true);
    batch.pos[0] = batch.pos[1] = 0;
    llama_opt_init(ctx, model, op);
    const auto initial_weights = adapter_params(adapter);
    batch.label[0] = tokens[2];
    GGML_ASSERT(llama_decode(ctx, batch) == 0);
    GGML_ASSERT(llama_get_logits_ith(ctx, 0) && llama_get_logits_ith(ctx, 1));
    const auto neighbor = sequence_state(ctx, 1);
    GGML_ASSERT(neighbor == reference_neighbor);
    const auto trained = sequence_state(ctx, 0);
    GGML_ASSERT(neighbor != trained);
    const auto weights = adapter_params(adapter);
    GGML_ASSERT(weights != initial_weights);
    common_batch_clear(batch);
    common_batch_add(batch, tokens[2], 1, {0}, true);
    batch.label[0] = llama_vocab_n_tokens(llama_model_get_vocab(model));
    GGML_ASSERT(llama_decode(ctx, batch) != 0);
    GGML_ASSERT(sequence_state(ctx, 0) == trained && sequence_state(ctx, 1) == neighbor);
    GGML_ASSERT(adapter_params(adapter) == weights);
    batch.label[0] = tokens[3];
    GGML_ASSERT(llama_decode(ctx, batch) == 0);
    GGML_ASSERT(sequence_state(ctx, 1) == neighbor);
    GGML_ASSERT(adapter_params(adapter) != weights);
    if (undo) {
        auto * memory = llama_get_memory(ctx);
        auto * recurrent = dynamic_cast<llama_memory_recurrent *>(memory);
        if (auto * hybrid = dynamic_cast<llama_memory_hybrid *>(memory)) {
            recurrent = hybrid->get_mem_recr();
        }
        GGML_ASSERT(recurrent);
        recurrent->set_rs_idx(0, 1);
        const auto pending = sequence_state(ctx, 0);
        const auto pending_weights = adapter_params(adapter);
        batch.pos[0] = 2;
        GGML_ASSERT(llama_decode(ctx, batch) != 0);
        GGML_ASSERT(sequence_state(ctx, 0) == pending && sequence_state(ctx, 1) == neighbor);
        GGML_ASSERT(adapter_params(adapter) == pending_weights);
    }
    printf("mixed decode flash=%d cache=%s undo=%u: same-position sequences, neighbour preservation, admission refusal OK\n", flash, ggml_type_name(cache_type), undo);
    llama_batch_free(batch);
    llama_free(ctx);
    llama_adapter_lora_free(adapter);
}

static void multi_label_decode(const common_params & params, llama_model * model, const std::string & init,
                               const std::vector<llama_token> & tokens, bool flash, ggml_type cache_type) {
    // weights empty: an unweighted batch (label_weight NULL)
    auto step = [&](const std::vector<int> & sequences, const std::vector<float> & weights = {}) {
        auto cp = common_context_params_to_llama(params);
        cp.n_ctx = 128;
        cp.n_batch = cp.n_ubatch = 2;
        cp.n_seq_max = 2;
        cp.kv_unified = true;
        cp.flash_attn_type = flash ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
        cp.type_k = cp.type_v = cache_type;
        llama_context * ctx = llama_init_from_model(model, cp);
        GGML_ASSERT(ctx);
        llama_adapter_lora * adapter = llama_adapter_lora_init(model, init.c_str());
        float scale = 1.0f;
        GGML_ASSERT(llama_set_adapters_lora(ctx, &adapter, 1, &scale) == 0);
        llama_opt_params op{};
        op.param_filter = llama_opt_param_filter_all;
        op.get_opt_pars = sgd_pars;
        op.optimizer_type = GGML_OPT_OPTIMIZER_TYPE_SGD;
        op.adapter = adapter;
        llama_opt_init(ctx, model, op);
        const auto before = adapter_params(adapter);
        llama_batch batch = llama_batch_init(2, 0, 1);
        for (size_t k = 0; k < sequences.size(); ++k) {
            const int seq = sequences[k];
            common_batch_add(batch, tokens[seq], 0, {seq}, true);
            batch.label[batch.n_tokens - 1] = tokens[seq + 2];
            if (!weights.empty()) {
                batch.label_weight[batch.n_tokens - 1] = weights[k];
            }
        }
        float * kept_weights = batch.label_weight;
        if (weights.empty()) {
            batch.label_weight = nullptr; // unweighted: the loss every backend has
        }
        GGML_ASSERT(llama_decode(ctx, batch) == 0);
        batch.label_weight = kept_weights;
        auto delta = adapter_params(adapter);
        for (size_t i = 0; i < delta.size(); ++i) {
            delta[i] -= before[i];
        }
        llama_batch_free(batch);
        llama_free(ctx);
        llama_adapter_lora_free(adapter);
        return delta;
    };
    auto mean = step({0});
    const auto second = step({1});
    for (size_t i = 0; i < mean.size(); ++i) {
        mean[i] = (mean[i] + second[i]) / 2;
    }
    const auto joint = step({0, 1});
    GGML_ASSERT(same_step(compare("two labelled sequences vs mean independent steps", joint, mean)));
    GGML_ASSERT(same_step(compare("two labelled sequences with reversed row order", step({1, 0}), joint)));
    // what this catches: the per-row weight not reaching the loss, or reaching it wrong (each row
    // counts w_i * its loss / rows, its gradient scaled by w_i; a mind's relevance per token)
    auto scaled = [](std::vector<float> v, float s) { for (float & x : v) { x *= s; } return v; };
    GGML_ASSERT(same_step(compare("weights 1,1 vs an unweighted batch", step({0, 1}, {1.0f, 1.0f}), joint)));
    GGML_ASSERT(same_step(compare("weights 2,2 vs twice the unweighted step", step({0, 1}, {2.0f, 2.0f}), scaled(joint, 2.0f))));
    GGML_ASSERT(same_step(compare("weights 1,0 vs half of row 0's own step (a mean over 2 rows)", step({0, 1}, {1.0f, 0.0f}), scaled(step({0}), 0.5f))));
}
static void sparse_head_decode(const common_params & params, llama_model * model, const std::string & init,
                               const std::vector<llama_token> & tokens) {
    struct result {
        std::vector<float> delta, logits;
        std::vector<uint8_t> neighbour;
    };
    auto step = [&](int public_mode, bool reversed, bool train = true) {
        auto cp = common_context_params_to_llama(params);
        cp.n_ctx = 128;
        cp.n_batch = cp.n_ubatch = 4;
        cp.n_seq_max = 2;
        cp.kv_unified = true;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        auto * ctx = llama_init_from_model(model, cp);
        auto * adapter = llama_adapter_lora_init(model, init.c_str());
        GGML_ASSERT(ctx && adapter);
        float scale = 1.0f;
        GGML_ASSERT(llama_set_adapters_lora(ctx, &adapter, 1, &scale) == 0);
        llama_opt_params op{};
        op.param_filter = llama_opt_param_filter_all;
        op.get_opt_pars = sgd_pars;
        op.optimizer_type = GGML_OPT_OPTIMIZER_TYPE_SGD;
        op.adapter = adapter;
        if (train) {
            llama_opt_init(ctx, model, op);
        }
        const auto before = adapter_params(adapter);
        llama_batch batch = llama_batch_init(3, 0, 1);
        int neighbour_row = -1;
        const std::vector<int> order = reversed ? std::vector<int>{2, 0, 1} : std::vector<int>{0, 1, 2};
        for (int row : order) {
            common_batch_add(batch, tokens[row], row == 1 ? 1 : 0, {row == 2 ? 1 : 0},
                    public_mode == 2 || (public_mode == 1 && row == 2));
            if (train && row == 1) {
                batch.label[batch.n_tokens - 1] = tokens[3];
            }
            if (row == 2) {
                neighbour_row = batch.n_tokens - 1;
            }
        }
        GGML_ASSERT(llama_decode(ctx, batch) == 0);
        printf("sparse head allocation: train=%d public=%d reversed=%d cache=%s graph_bytes=%zu\n",
                train, public_mode, reversed, ggml_type_name(cp.type_k), llama_opt_graph_bytes(ctx));
        result out;
        out.delta = adapter_params(adapter);
        for (size_t i = 0; i < before.size(); ++i) {
            out.delta[i] -= before[i];
        }
        if (public_mode > 0) {
            const float * logits = llama_get_logits_ith(ctx, neighbour_row);
            GGML_ASSERT(logits);
            out.logits.assign(logits, logits + llama_vocab_n_tokens(llama_model_get_vocab(model)));
        }
        out.neighbour = sequence_state(ctx, 1);
        llama_batch_free(batch);
        llama_free(ctx);
        llama_adapter_lora_free(adapter);
        return out;
    };
    const auto dense = step(2, false);
    const auto serving = step(1, false, false);
    GGML_ASSERT(serving.logits == dense.logits);
    GGML_ASSERT(serving.neighbour == dense.neighbour);
    for (bool reversed : {false, true}) {
        for (int public_mode : {0, 1}) {
            const auto sparse = step(public_mode, reversed);
            GGML_ASSERT(same_step(compare("sparse public rows vs dense objective", sparse.delta, dense.delta)));
            GGML_ASSERT(public_mode == 0 || sparse.logits == dense.logits);
            GGML_ASSERT(sparse.neighbour == dense.neighbour);
        }
    }
    printf("sparse head: label-only rows, public logits, neighbour bytes and row reordering OK\n");
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

    // one file per run: two runs in one directory (a measurement beside an exactness check)
    // overwrote each other's adapter and failed to load it (the M5, 2026-10-11)
    const std::string init = "test-walk-exact." + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".init.gguf";
    GGML_ASSERT(common_lora_write_fresh(model, init, 8, 16, { "attn_q", "attn_v" }, 42));

    // words in a fixed pseudo-random order: a window the model cannot already predict, so its
    // loss, and the gradient under test, is well above the kernels' batch-size noise
    static const char * words[] = { "room", "card", "claim", "review", "patch", "lease", "board", "turn",
        "engine", "window", "chunk", "cache", "gradient", "adapter", "verdict", "submission" };
    std::string text;
    uint32_t lcg = 12345;
    while (text.size() < 8 * REFUSAL_WINDOW) {
        lcg = lcg * 1664525u + 1013904223u;
        text += words[(lcg >> 16) % 16];
        text += (lcg >> 8) % 7 == 0 ? ".\n" : " ";
    }
    llama_context * tok_ctx = make_ctx(params, model, WINDOW);
    std::vector<llama_token> tokens = common_tokenize(tok_ctx, text, true);
    llama_free(tok_ctx);
    GGML_ASSERT(tokens.size() > REFUSAL_WINDOW + 1);

    if (getenv("WALK_HEAD_ONLY")) {
        sparse_head_decode(params, model, init, tokens);
        if (llama_model_get_tensor(model, "output.weight")) {
            const auto head_init = init + ".head.gguf";
            GGML_ASSERT(common_lora_write_fresh(model, head_init, 8, 16, { "output.weight" }, 42));
            sparse_head_decode(params, model, head_init, tokens);
            std::remove(head_init.c_str());
        } else {
            printf("output adapter: unavailable (model has no independent output.weight tensor)\n");
        }
        std::remove(init.c_str());
        llama_model_free(model);
        llama_backend_free();
        return 0;
    }

    if (getenv("WALK_MIXED_ONLY")) {
        multi_label_decode(params, model, init, tokens, false, GGML_TYPE_F32);
        multi_label_decode(params, model, init, tokens, true, GGML_TYPE_Q8_0);
        mixed_decode(params, model, init, tokens, false, GGML_TYPE_F32);
        for (ggml_type cache : { GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_Q8_0 }) {
            mixed_decode(params, model, init, tokens, true, cache);
        }
        if (llama_model_is_hybrid(model) || llama_model_is_recurrent(model)) {
            mixed_decode(params, model, init, tokens, true, GGML_TYPE_Q8_0, 2);
        }
        std::remove(init.c_str());
        llama_model_free(model);
        llama_backend_free();
        return 0;
    }

    // MEASUREMENT MODE (WALK_MEASURE="2048,8192,..."): the device graph per chunk of a context +
    // reply window (the last 512 positions labelled), on the explicit path (F32 cache, no flash
    // attention) and through FLASH_ATTN_EXT (f16 cache, flash attention), at each window. Prints a
    // table and exits; the bytes are the run's own allocation preflight (llama_opt_graph_bytes).
    if (const char * env = getenv("WALK_MEASURE")) {
        std::string more = text;
        while (more.size() < 6 * 65536) {
            lcg = lcg * 1664525u + 1013904223u;
            more += words[(lcg >> 16) % 16];
            more += (lcg >> 8) % 7 == 0 ? ".\n" : " ";
        }
        llama_context * tc = make_ctx(params, model, WINDOW);
        const std::vector<llama_token> long_tokens = common_tokenize(tc, more, true);
        llama_free(tc);
        const uint32_t chunk = getenv("WALK_MEASURE_CHUNK") ? (uint32_t) atoi(getenv("WALK_MEASURE_CHUNK")) : 256;
        g_recompute = getenv("WALK_MEASURE_RECOMPUTE") != nullptr; // per-layer recompute, as the served runs train
        g_walk_horizon = getenv("WALK_MEASURE_HORIZON") ? (uint32_t) atoi(getenv("WALK_MEASURE_HORIZON")) : 0; // 0 = the whole window
        printf("  window  chunk  explicit MiB  flash MiB  flash q8_0 MiB  ratio\n");
        for (const char * p = env; *p; ) {
            char * end = nullptr;
            const uint32_t w = (uint32_t) strtoul(p, &end, 10);
            p = *end == ',' ? end + 1 : end;
            if (w == 0 || long_tokens.size() <= w + 1) { break; }
            g_window = w;
            std::vector<uint8_t> reply(w + 1, 0);
            for (uint32_t i = w - 511; i <= w; ++i) reply[i] = 1;
            double mib[3] = { 0, 0, 0 };
            // explicit attention at an F32 cache; flash at an f16 cache; flash at q8_0 (her lane's cache)
            const ggml_type types[3] = { GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_Q8_0 };
            for (int a = 0; a < 3; ++a) {
                g_cache_type = types[a];
                step_delta(params, model, init, long_tokens, reply, chunk, true);
                mib[a] = g_last_graph_bytes / 1048576.0;
            }
            g_cache_type = GGML_TYPE_F32;
            printf("  %6u  %5u  %12.1f  %9.1f  %14.1f  %5.2fx\n", w, chunk, mib[0], mib[1], mib[2], mib[1] > 0 ? mib[0] / mib[1] : 0.0);
            fflush(stdout);
        }
        std::remove(init.c_str());
        return 0;
    }

    const bool recompute_only = getenv("WALK_RECOMPUTE_ONLY") != nullptr;
    const bool live_only = getenv("WALK_LIVE_ONLY") != nullptr;
    GGML_ASSERT(!(live_only && recompute_only));
    if (live_only) {
        printf("WALK_LIVE_ONLY: live-sequence and generation-restore cases only; full cache/walk suite not selected\n");
    }
    int failures = 0;
    if (!recompute_only) {
    if (!live_only) {
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
            worst_tensors(exact, ref, 24);
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
        // 1b. the served cache types. Training attends to the cache in place, as serving does:
        // every K/V it reads, the chunk's own included, is at the cache's precision, and the
        // gradient passes straight through the cast. Two claims, measured apart: the walk is
        // EXACT at that precision (against one graph reading the same cache), and what the
        // precision itself costs against the F32 gradient (printed, the served model's own).
        std::vector<uint8_t> all(WINDOW + 1, 1);
        all[0] = 0;
        const auto truth = step_delta(params, model, init, tokens, all, WINDOW, false);
        for (ggml_type t : { GGML_TYPE_F16, GGML_TYPE_Q8_0 }) {
            g_cache_type = t;
            const auto one   = step_delta(params, model, init, tokens, all, WINDOW, false);
            const auto exact = step_delta(params, model, init, tokens, all, WINDOW / 4, true);
            g_cache_type = GGML_TYPE_F32;
            const std::string name = std::string("all labelled, ") + ggml_type_name(t) + " cache: ";
            if (!same_step(compare((name + "exact walk in 4 chunks vs one graph at the same cache").c_str(), exact, one))) {
                fprintf(stderr, "FAILED: at a %s cache the exact walk is not the window's gradient\n", ggml_type_name(t));
                ++failures;
            }
            compare((name + "one graph vs the F32 gradient (the cache's own precision)").c_str(), one, truth);
        }
    }
    {
        // 2. context, then a reply (the walk's real case): the reply trains given the context as
        // served, so only its own chunks train, and the exact walk is invariant to how the reply
        // is chunked
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

    } // !live_only
    {
        // 9. LEARNING ON A LIVE SEQUENCE (llama_opt_learn, P2). What this catches: training that
        // needs its own context, re-decodes her context, disturbs another slot, or leaves the shadow
        // in her cache. A serving-shaped context holds two sequences; sequence 0 is her context as
        // served, sequence 1 a neighbour. Her reply trains on sequence 0 from the served prefix:
        // the step must be the walk's exact step for the same window (one chunk per run), her
        // sequence must end as a stable decode of the reply would leave it, and the neighbour's
        // state must not change by a byte.
        std::vector<uint8_t> reply(WINDOW + 1, 0);
        for (uint32_t i = WINDOW / 2 + 37; i <= WINDOW; ++i) {
            reply[i] = 1;
        }
        const llama_pos r0 = WINDOW / 2 + 36; // the first labelled position
        const auto want_f32 = step_delta(params, model, init, tokens, reply, WINDOW, true);
        g_cache_type = GGML_TYPE_F16;
        const auto want_f16 = step_delta(params, model, init, tokens, reply, WINDOW, true);
        g_cache_type = GGML_TYPE_F32;

      for (const bool streams : { false, true }) {
        // unified F32 cache with explicit attention, then a stream per slot with an f16 cache and
        // flash attention: how a serving lane runs (each slot attends over its own stream)
        printf("  live sequence, %s:\n", streams ? "a stream per slot, f16 cache, flash attention" : "one unified F32 cache, explicit attention");
        auto two_seq_ctx = [&]() {
            auto cp = common_context_params_to_llama(params);
            cp.n_ctx = 2*WINDOW; cp.n_batch = WINDOW; cp.n_ubatch = WINDOW; cp.n_seq_max = 2; cp.kv_unified = !streams;
            cp.flash_attn_type = streams ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
            cp.type_k = streams ? GGML_TYPE_F16 : GGML_TYPE_F32; cp.type_v = cp.type_k;
            return llama_init_from_model(model, cp);
        };
        auto decode_span = [&](llama_context * c, llama_seq_id sq, const llama_token * t, llama_pos p0, int32_t n) {
            llama_batch b = llama_batch_init(WINDOW, 0, 1);
            common_batch_clear(b);
            for (int32_t i = 0; i < n; ++i) {
                common_batch_add(b, t[i], p0 + i, { sq }, i + 1 == n);
            }
            GGML_ASSERT(llama_decode(c, b) == 0);
            llama_batch_free(b);
        };
        auto seq_state = [&](llama_context * c, llama_seq_id sq) {
            std::vector<uint8_t> st(llama_state_seq_get_size(c, sq));
            GGML_ASSERT(llama_state_seq_get_data(c, st.data(), st.size(), sq) == st.size());
            return st;
        };
        const llama_token * neighbour = tokens.data() + WINDOW + 1; // other text, beyond her window

        // the control: her context, then her reply, decoded plainly beside the same neighbour
        llama_context * ctl = two_seq_ctx();
        decode_span(ctl, 1, neighbour, 0, 200);
        decode_span(ctl, 0, tokens.data(), 0, r0);
        decode_span(ctl, 0, tokens.data() + r0, r0, WINDOW - r0);
        const std::vector<uint8_t> want_her = seq_state(ctl, 0);
        llama_free(ctl);

        llama_adapter_lora * adapter = llama_adapter_lora_init(model, init.c_str());
        const std::vector<float> before = adapter_params(adapter);
        llama_context * ctx = two_seq_ctx();
        float scale = 1.0f;
        GGML_ASSERT(llama_set_adapters_lora(ctx, &adapter, 1, &scale) == 0);
        decode_span(ctx, 1, neighbour, 0, 200);
        decode_span(ctx, 0, tokens.data(), 0, r0); // her context, as served
        const std::vector<uint8_t> neighbour_before = seq_state(ctx, 1);
        llama_opt_params lopt{};
        lopt.param_filter   = llama_opt_param_filter_all;
        lopt.get_opt_pars   = sgd_pars;
        lopt.optimizer_type = GGML_OPT_OPTIMIZER_TYPE_SGD;
        lopt.adapter        = adapter;
        llama_opt_init(ctx, model, lopt);
        std::vector<llama_token> labels(WINDOW - r0);
        for (int32_t i = 0; i < (int32_t) WINDOW - r0; ++i) {
            labels[i] = tokens[r0 + i + 1];
        }
        // refusals come before any change: her sequence byte-identical after each
        {
            const std::vector<uint8_t> her_before = seq_state(ctx, 0);
            const bool r_null = llama_opt_learn(ctx, 0, r0, tokens.data() + r0, nullptr, WINDOW - r0, nullptr);
            const bool r_long = llama_opt_learn(ctx, 0, r0, tokens.data() + r0, labels.data(), 2*WINDOW, nullptr);
            const bool r_more = llama_opt_learn(ctx, 0, r0 + 5, tokens.data() + r0, labels.data(), 10, nullptr);
            const bool same   = seq_state(ctx, 0) == her_before;
            printf("  live sequence refusals: null labels %s, past the context %s, prefix not held %s; her sequence %s\n",
                    r_null ? "ACCEPTED" : "refused", r_long ? "ACCEPTED" : "refused", r_more ? "ACCEPTED" : "refused",
                    same ? "byte-identical" : "CHANGED");
            if (r_null || r_long || r_more || !same) {
                fprintf(stderr, "FAILED: a refused live learn was accepted or changed her sequence\n");
                ++failures;
            }
        }
        const auto generation_prefix = seq_state(ctx, 0);
        const bool learned = llama_opt_learn(ctx, 0, r0, tokens.data() + r0, labels.data(), WINDOW - r0, nullptr);
        const std::vector<uint8_t> her_after = seq_state(ctx, 0);
        const std::vector<uint8_t> neighbour_after = seq_state(ctx, 1);
        if (learned && !streams) {
            // Generation restore must prepare both components before touching weights, preserve
            // their addresses, and keep uncertain shadow state out of serving and learning.
            GGML_ASSERT(llama_set_adapters_lora(ctx, nullptr, 0, nullptr) == 0);
            const auto saved = adapter_params(adapter);
            const auto old_a = adapter->ab_map.begin()->second.a;
            const std::string checkpoint = init + ".generation";
            // Exercise aLoRA metadata custody without changing this fixture's learning template.
            adapter->alora_invocation_tokens = {tokens[0], tokens[1]};
            GGML_ASSERT(llama_adapter_lora_save(adapter, checkpoint.c_str()) == 0);
            FILE * weights = fopen(checkpoint.c_str(), "rb");
            FILE * optimizer = tmpfile();
            GGML_ASSERT(weights && optimizer && llama_opt_state_write(ctx, optimizer, "fixture-generation"));
            adapter->alora_invocation_tokens.clear();
            GGML_ASSERT(llama_state_seq_set_data(ctx, generation_prefix.data(), generation_prefix.size(), 0) == generation_prefix.size());
            GGML_ASSERT(llama_opt_learn(ctx, 0, r0, tokens.data() + r0, labels.data(), WINDOW - r0, nullptr));
            const auto uninterrupted_next = adapter_params(adapter);
            adapter->alora_invocation_tokens = {tokens[0], tokens[1]};
            // Change every A/B value deterministically, not just metadata or one selected tensor.
            for (const auto & entry : adapter->ab_map) {
                for (ggml_tensor * tensor : {entry.second.a, entry.second.b}) {
                    std::vector<float> changed(ggml_nelements(tensor));
                    ggml_backend_tensor_get(tensor, changed.data(), 0, ggml_nbytes(tensor));
                    for (float & value : changed) { value = value * 0.5f + 0.125f; }
                    ggml_backend_tensor_set(tensor, changed.data(), 0, ggml_nbytes(tensor));
                }
            }
            const auto changed = adapter_params(adapter);
            GGML_ASSERT(changed != saved);
            rewind(optimizer);
            GGML_ASSERT(!llama_opt_state_restore(ctx, adapter, weights, optimizer, "wrong-generation"));
            GGML_ASSERT(adapter_params(adapter) == changed && !adapter->restore_quarantined);
            rewind(weights);
            auto * prepared = llama_adapter_lora_state_prepare(adapter, weights);
            GGML_ASSERT(prepared && adapter_params(adapter) == changed);
            GGML_ASSERT(llama_adapter_lora_state_apply(adapter, prepared));
            llama_adapter_lora_state_free(prepared);
            GGML_ASSERT(adapter->restore_quarantined);
            GGML_ASSERT(llama_adapter_lora_save(adapter, checkpoint.c_str()) != 0);
            GGML_ASSERT(llama_set_adapters_lora(ctx, &adapter, 1, &scale) != 0);
            llama_batch quarantined_batch = llama_batch_get_one(tokens.data(), 1);
            quarantined_batch.label = labels.data();
            GGML_ASSERT(llama_decode(ctx, quarantined_batch) == -3);
            GGML_ASSERT(std::string(llama_opt_failure(ctx)).find("restore is uncertain") != std::string::npos);
            GGML_ASSERT(llama_encode(ctx, quarantined_batch) == -3);
            GGML_ASSERT(std::string(llama_opt_failure(ctx)).find("restore is uncertain") != std::string::npos);
            float zero_scale = 0.0f;
            GGML_ASSERT(llama_set_adapters_lora(ctx, &adapter, 1, &zero_scale) == 0);
            rewind(weights); rewind(optimizer);
            GGML_ASSERT(!llama_opt_state_restore(ctx, adapter, weights, optimizer, "wrong-generation"));
            GGML_ASSERT(adapter->restore_quarantined && adapter_params(adapter) == saved);
            GGML_ASSERT(!llama_opt_learn(ctx, 0, r0, tokens.data() + r0, labels.data(), WINDOW - r0, nullptr));
            rewind(weights); rewind(optimizer);
            GGML_ASSERT(llama_opt_state_restore(ctx, adapter, weights, optimizer, "fixture-generation"));
            GGML_ASSERT(!adapter->restore_quarantined && adapter_params(adapter) == saved);
            GGML_ASSERT(adapter->ab_map.begin()->second.a == old_a);
            adapter->alora_invocation_tokens.clear();
            GGML_ASSERT(llama_state_seq_set_data(ctx, generation_prefix.data(), generation_prefix.size(), 0) == generation_prefix.size());
            GGML_ASSERT(llama_opt_learn(ctx, 0, r0, tokens.data() + r0, labels.data(), WINDOW - r0, nullptr));
            const auto resumed_next = adapter_params(adapter);
            GGML_ASSERT(resumed_next.size() == uninterrupted_next.size());
            std::vector<float> resumed_step(saved.size()), uninterrupted_step(saved.size());
            for (size_t i = 0; i < saved.size(); ++i) {
                resumed_step[i] = resumed_next[i] - saved[i];
                uninterrupted_step[i] = uninterrupted_next[i] - saved[i];
            }
            GGML_ASSERT(same_step(compare("restored generation next step", resumed_step, uninterrupted_step)));
            // Restore the original generation for the enclosing live-sequence step comparison.
            adapter->alora_invocation_tokens = {tokens[0], tokens[1]};
            rewind(weights); rewind(optimizer);
            GGML_ASSERT(llama_opt_state_restore(ctx, adapter, weights, optimizer, "fixture-generation"));
            GGML_ASSERT(adapter_params(adapter) == saved);
            adapter->alora_invocation_tokens.clear();
            rewind(weights); rewind(optimizer);
            GGML_ASSERT(!llama_opt_state_restore(ctx, adapter, weights, optimizer, "fixture-generation"));
            GGML_ASSERT(adapter_params(adapter) == saved && !adapter->restore_quarantined);
            fclose(weights); fclose(optimizer);
            std::remove(checkpoint.c_str());
            printf("  generation restore: prepare refusal unchanged, aLoRA exact, addresses stable, changed weights restored, next step equivalent, quarantine recovered\n");
        }
        llama_free(ctx);
        const std::vector<float> after = adapter_params(adapter);
        llama_adapter_lora_free(adapter);
        std::vector<float> got(before.size());
        for (size_t i = 0; i < got.size(); ++i) {
            got[i] = after[i] - before[i];
        }
        printf("  live sequence: learned=%d \"%s\"\n", (int) learned, learned ? "" : "see the log");
        if (!learned || !same_step(compare("live sequence: her reply learned in place vs the walk's exact step", got, streams ? want_f16 : want_f32))) {
            fprintf(stderr, "FAILED: learning on a live sequence is not the walk's step\n");
            ++failures;
        }
        const bool her_ok = her_after == want_her;
        const bool nb_ok  = neighbour_after == neighbour_before;
        printf("  live sequence: her cache %s a stable decode of the reply; the neighbour %s\n",
                her_ok ? "byte-identical to" : "DIFFERS from", nb_ok ? "byte-identical" : "CHANGED");
        if (!her_ok || !nb_ok) {
            fprintf(stderr, "FAILED: learning on a live sequence disturbed the cache\n");
            ++failures;
        }
      }
    }

    if (live_only) {
        std::remove(init.c_str());
        llama_model_free(model);
        llama_backend_free();
        printf("test-walk-exact live/generation selection: %s\n", failures ? "FAILED" : "OK");
        return failures ? 1 : 0;
    }
    {
        // 8. what this catches: a walk that leaves the shadow in her cache. The reply's cells
        // are written under the shadow while it trains, then restored, so after either walk the
        // sequence (every attention cell, the recurrent state) is byte-identical to her stable
        // self's decode of the same window: serving never reads what training wrote.
        std::vector<uint8_t> reply(WINDOW + 1, 0);
        for (uint32_t i = WINDOW / 2 + 37; i <= WINDOW; ++i) {
            reply[i] = 1;
        }
        const uint32_t r0 = WINDOW / 2 + 36; // the first labelled position (it predicts reply[r0 + 1])
        const std::vector<uint8_t> want = stable_state(params, model, tokens, r0, WINDOW, WINDOW / 4);
        for (bool exact : { true, false }) {
            std::vector<uint8_t> got;
            step_delta(params, model, init, tokens, reply, WINDOW / 4, exact, 0, nullptr, &got);
            const bool same = got == want;
            printf("  cache after the %s walk: %zu bytes, %s her stable decode\n", exact ? "exact" : "plain", got.size(),
                    same ? "byte-identical to" : "DIFFERS from");
            if (!same) {
                fprintf(stderr, "FAILED: the %s walk left the shadow in the cache\n", exact ? "exact" : "plain");
                ++failures;
            }
        }
    }

    }

    {
        // 7. per-layer recompute under the exact walk. The chunk's activations, every layer's
        // kept for the backward pass, are what a large model's chunk graph is made of (Kimi's
        // first exact run on the 5090, 2026-10-10: 5.6 GiB more at chunk 256 under every
        // gradient horizon from 1155 down to 256). Recompute keeps the layer outputs and rebuilds
        // the rest in the backward pass, so the step must not move and the graph must shrink.
        std::vector<uint8_t> reply(WINDOW + 1, 0);
        for (uint32_t i = WINDOW / 2 + 37; i <= WINDOW; ++i) {
            reply[i] = 1;
        }
        const auto kept  = step_delta(params, model, init, tokens, reply, WINDOW / 4, true);
        const size_t kept_bytes = g_last_graph_bytes;
        g_recompute = true;
        const auto rebuilt = step_delta(params, model, init, tokens, reply, WINDOW / 4, true);
        const size_t rebuilt_bytes = g_last_graph_bytes;
        g_recompute = false;
        printf("  chunk graph: %.1f MiB keeping every layer's activations, %.1f MiB with recompute\n",
                kept_bytes / 1048576.0, rebuilt_bytes / 1048576.0);
        if (!same_step(compare("context + reply: exact walk with recompute vs without", rebuilt, kept))) {
            fprintf(stderr, "FAILED: recompute changed the exact walk's step\n");
            ++failures;
        }
        if (kept_bytes == 0 && rebuilt_bytes == 0) {
            printf("  chunk graph reduction: unavailable (device-only allocation counter)\n");
        } else if (!(rebuilt_bytes < kept_bytes)) {
            fprintf(stderr, "FAILED: recompute did not shrink the exact walk's chunk graph\n");
            ++failures;
        }
    }

    if (!recompute_only) {
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
        // 5. what this catches: a state restore that does not restore the recurrent cell's
        // position. The reverse pass restores each chunk's entry state; with the cell left at the
        // window's end, every chunk after the first restore reads as non-consecutive.
        std::vector<uint8_t> all(WINDOW + 1, 1);
        all[0] = 0;
        g_nonconsecutive = 0;
        llama_log_set(count_log, nullptr);
        step_delta(params, model, init, tokens, all, WINDOW / 4, true);
        llama_log_set(nullptr, nullptr); // the default logger again
        printf("  state restore: %d non-consecutive position warnings in the reverse pass\n", g_nonconsecutive);
        if (g_nonconsecutive != 0) {
            fprintf(stderr, "FAILED: a restored recurrent state left its cell at another position\n");
            ++failures;
        }
    }
    if (llama_model_is_hybrid(model)) {
        // 4. TRAINING IN THE SERVING CONTEXT (Joel, 2026-10-13: "training is basically llama inference with
        // backprop"). Serving keeps recurrent rollback slots for drafts; a training graph keeps one state, the
        // live one (llm_graph_context::rs_keep), so a context WITH rollback slots takes the same step as one
        // without. Was a refusal (the 5090 crash, 2026-10-07 04:09Z), which forced a second training
        // context beside serving.
        std::vector<uint8_t> all(WINDOW + 1, 1);
        const auto without = step_delta(params, model, init, tokens, all, WINDOW / 4, true);
        g_n_rs_seq = 4;
        const auto with    = step_delta(params, model, init, tokens, all, WINDOW / 4, true);
        g_n_rs_seq = 0;
        if (!same_step(compare("rollback slots: exact walk in a context with n_rs_seq 4 vs 0", with, without))) {
            fprintf(stderr, "FAILED: rollback slots changed the exact walk's step\n");
            worst_tensors(with, without, 24);
            ++failures;
        }
    }

    {
        // 6. regression for the 5090 (2026-10-10 04:34Z): after the device budget refuses a chunk
        // graph, the walk shrinks the gradient horizon and tries the chunk again. The refusal comes
        // AFTER the chunk's ubatch was applied, so the retry used to run on top of the chunk's own
        // cells and fail to prepare its ubatch at all ("init_batch: failed to prepare attention
        // ubatches"), stopping the epoch instead of reaching a horizon that fits. A budget just
        // under the measured graph refuses EVERY horizon, so this pins the retry itself: each retry
        // must reach the device preflight again (and be refused there), never fail in init_batch,
        // and the epoch then refuses by name with the adapter untouched. Needs a device: a CPU run
        // measures no graph, and says so.
        g_window = REFUSAL_WINDOW;
        std::vector<uint8_t> all(g_window + 1, 1);
        all[0] = 0;
        step_delta(params, model, init, tokens, all, WINDOW / 4, true);
        const size_t graph_bytes = g_last_graph_bytes;
        printf("  device graph: %.1f MiB per chunk\n", graph_bytes / 1048576.0);
        // the budget caps GPU memory only (ggml_opt_alloc), so a CPU-only run cannot refuse; it
        // still measures a graph (in host buffers), which is why the graph size alone was the
        // wrong test for "a CPU-only run"
        const bool cpu_only = !params.devices.empty() && params.devices[0] == nullptr;
        if (graph_bytes == 0 || cpu_only) {
            printf("  device refusal: skipped (a CPU-only run: the device budget caps GPU memory only)\n");
        } else {
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
            // what a graph may add is the budget less the allocator's 512 MiB margin (ggml-opt), and
            // "add" is growth past the buffers the walk's own forward decodes already sized: one
            // byte past the margin lets a chunk graph grow by nothing, so every horizon is refused
            llama_opt_set_memory_budget(ctx, 512u*1024*1024 + 1);
            llama_opt_init(ctx, model, lopt);
            std::vector<std::vector<llama_token>> seqs = { std::vector<llama_token>(tokens.begin(), tokens.begin() + g_window + 1) };
            std::vector<std::vector<uint8_t>>     loss = { all };
            ggml_opt_dataset_t dataset = common_opt_dataset_init_masked(g_window, seqs, loss, tokens[0]);
            g_retries = 0;
            g_ubatch_failures = 0;
            g_refusals = 0;
            llama_log_set(count_log, nullptr);
            llama_opt_epoch(ctx, dataset, nullptr, nullptr, /*idata_split =*/ 1, nullptr, nullptr);
            llama_log_set(nullptr, nullptr); // the default logger again
            const std::string why = llama_opt_failure(ctx);
            printf("  device refusal: %d retries, %d graph refusals, %d ubatch failures, failed=%d \"%s\"\n",
                    g_retries, g_refusals, g_ubatch_failures, (int) llama_opt_failed(ctx), why.c_str());
            if (g_retries < 1 || g_refusals < 2 || g_ubatch_failures != 0 ||
                !llama_opt_failed(ctx) || adapter_params(adapter) != before) {
                fprintf(stderr, "FAILED: a refused chunk graph was not retried cleanly at a smaller horizon\n");
                ++failures;
            }
            ggml_opt_dataset_free(dataset);
            llama_free(ctx);
            llama_adapter_lora_free(adapter);
        }
        g_window = WINDOW;
    }

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
