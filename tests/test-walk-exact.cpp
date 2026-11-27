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
#include "../src/llama-context.h"
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

// The test drives the labelled public llama_batch (llama_opt_batch_init, decoded by walk_decode below); upstream removed
// common_batch_clear/common_batch_add with the move to llama_batch_ext, so the test keeps its own two.
static void walk_batch_clear(llama_batch & batch) {
    batch.n_tokens = 0;
}

static void walk_batch_add(llama_batch & batch, llama_token id, llama_pos pos,
                           const std::vector<llama_seq_id> & seq_ids, bool logits) {
    GGML_ASSERT(batch.seq_id[batch.n_tokens] && "llama_batch size exceeded");
    batch.token   [batch.n_tokens] = id;
    batch.pos     [batch.n_tokens] = pos;
    batch.n_seq_id[batch.n_tokens] = (int32_t) seq_ids.size();
    for (size_t i = 0; i < seq_ids.size(); ++i) {
        batch.seq_id[batch.n_tokens][i] = seq_ids[i];
    }
    batch.logits  [batch.n_tokens] = logits;
    batch.n_tokens++;
}

// one decode of the labelled batch: the context's own decode, as the deprecated llama_decode wrapper calls it
static int32_t walk_decode(llama_context * ctx, const llama_batch & batch) {
    return ctx->decode(batch);
}

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


// A lesson's routing as it was lived: every layer's chosen experts for each position of the window,
// read from the graph's own ffn_moe_topk tensors during one served forward of the window.
struct lived_routing {
    int32_t n_layer = 0;
    int32_t n_used  = 0;
    int32_t n_pos   = 0;
    std::vector<int32_t> ids; // [layer][position][n_used]
    int32_t row0 = 0;         // the row the next decode's first row records at (several decodes, one record)
};
static const lived_routing * g_pinned = nullptr;

static bool routing_eval(ggml_tensor * t, bool ask, void * user_data) {
    const char * name = ggml_get_name(t);
    const bool topk = std::strncmp(name, "ffn_moe_topk-", 13) == 0;
    if (ask) {
        return topk;
    }
    if (!topk) {
        return true;
    }
    auto * r = (lived_routing *) user_data;
    const int32_t il = std::atoi(name + 13);
    const int64_t n_used = t->ne[0], n_tokens = t->ne[1];
    if (r->n_used == 0) {
        r->n_used = (int32_t) n_used;
        r->ids.assign((size_t) r->n_layer * r->n_pos * n_used, 0);
    }
    GGML_ASSERT(n_used == r->n_used && r->row0 + n_tokens <= r->n_pos && il < r->n_layer);
    for (int64_t i = 0; i < n_tokens; ++i) { // a view of the argsort: read row by row
        ggml_backend_tensor_get(t, &r->ids[((size_t) il * r->n_pos + r->row0 + i) * n_used], i * t->nb[1], n_used * sizeof(int32_t));
    }
    return true;
}

static lived_routing record_routing(common_params params, llama_model * model, const std::string & init,
                                    const std::vector<llama_token> & tokens) {
    lived_routing r;
    r.n_layer = llama_model_n_layer(model);
    r.n_pos   = (int32_t) g_window;
    params.cb_eval           = routing_eval;
    params.cb_eval_user_data = &r;
    llama_adapter_lora * adapter = llama_adapter_lora_init(model, init.c_str());
    GGML_ASSERT(adapter != nullptr);
    llama_context * ctx = make_ctx(params, model, g_window);
    float scale = 1.0f;
    GGML_ASSERT(llama_set_adapters_lora(ctx, &adapter, 1, &scale) == 0);
    llama_batch batch = llama_batch_init((int32_t) g_window, 0, 1);
    for (uint32_t i = 0; i < g_window; ++i) {
        batch.token[i]     = tokens[i];
        batch.pos[i]       = (llama_pos) i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]    = true; // every row an output, so the last layer routes them all too
    }
    batch.n_tokens = (int32_t) g_window;
    GGML_ASSERT(llama_decode(ctx, batch) == 0);
    llama_batch_free(batch);
    llama_free(ctx);
    llama_adapter_lora_free(adapter);
    GGML_ASSERT(r.n_used > 0 && "the model has no MoE layer to pin");
    return r;
}

// Two residents in one ubatch at the same positions, the lesson's range covering only part of hers:
// routing per row, every row an output, rows in batch order (seq 0's, then seq 1's).
static lived_routing route_two(const common_params & params_in, llama_model * model, const std::vector<llama_token> & tokens,
                               uint32_t n, const lived_routing * pin, llama_pos pin0, int32_t pin_n) {
    common_params params = params_in;
    lived_routing r;
    r.n_layer = llama_model_n_layer(model);
    r.n_pos   = (int32_t) (2 * n);
    params.cb_eval           = routing_eval;
    params.cb_eval_user_data = &r;
    auto cparams = common_context_params_to_llama(params);
    cparams.n_ctx      = 4 * n;
    cparams.n_batch    = 2 * n;
    cparams.n_ubatch   = 2 * n;
    cparams.n_seq_max  = 2;
    cparams.kv_unified = true;
    llama_context * ctx = llama_init_from_model(model, cparams);
    GGML_ASSERT(ctx != nullptr);
    if (pin) {
        // the lesson's lived ids for positions [pin0, pin0 + pin_n) of seq 0, from its rows of `pin`
        std::vector<int32_t> ids((size_t) pin->n_layer * pin_n * pin->n_used);
        for (int32_t il = 0; il < pin->n_layer; ++il) {
            for (int32_t p = 0; p < pin_n; ++p) {
                std::memcpy(&ids[((size_t) il * pin_n + p) * pin->n_used],
                            &pin->ids[((size_t) il * pin->n_pos + pin0 + p) * pin->n_used], sizeof(int32_t) * pin->n_used);
            }
        }
        GGML_ASSERT(llama_set_pinned_experts(ctx, 0, pin->n_layer, pin->n_used, pin0, pin_n, ids.data()));
    }
    llama_batch batch = llama_batch_init((int32_t) (2 * n), 0, 2);
    for (uint32_t s = 0; s < 2; ++s) {
        for (uint32_t i = 0; i < n; ++i) {
            const uint32_t k = s * n + i;
            batch.token[k]     = tokens[s * n + i]; // a neighbour with its own text at the same positions
            batch.pos[k]       = (llama_pos) i;
            batch.n_seq_id[k]  = 1;
            batch.seq_id[k][0] = (llama_seq_id) s;
            batch.logits[k]    = true;
        }
    }
    batch.n_tokens = (int32_t) (2 * n);
    GGML_ASSERT(llama_decode(ctx, batch) == 0);
    llama_batch_free(batch);
    llama_free(ctx);
    GGML_ASSERT(r.n_used > 0 && "the model has no MoE layer to pin");
    return r;
}

// the model's expert count, from its own metadata (<arch>.expert_count)
static int32_t model_n_expert(const llama_model * model) {
    char arch[64], val[32];
    GGML_ASSERT(llama_model_meta_val_str(model, "general.architecture", arch, sizeof(arch)) > 0);
    GGML_ASSERT(llama_model_meta_val_str(model, (std::string(arch) + ".expert_count").c_str(), val, sizeof(val)) > 0);
    return (int32_t) std::atoi(val);
}
// the model's experts per token, from its own metadata (<arch>.expert_used_count)
static int32_t model_n_expert_used(const llama_model * model) {
    char arch[64], val[32];
    GGML_ASSERT(llama_model_meta_val_str(model, "general.architecture", arch, sizeof(arch)) > 0);
    GGML_ASSERT(llama_model_meta_val_str(model, (std::string(arch) + ".expert_used_count").c_str(), val, sizeof(val)) > 0);
    return (int32_t) std::atoi(val);
}
// what this catches (Codex on fork #167): a context-wide pin that hands a neighbour the lesson's experts
// at the same positions, or drops the whole ubatch to the router when one row is uncovered.
static int pinned_rows_are_per_sequence(const common_params & params, llama_model * model, const std::vector<llama_token> & tokens) {
    const uint32_t n = 64;
    const llama_pos pin0 = 32;
    const int32_t   pin_n = 32;
    const lived_routing served = route_two(params, model, tokens, n, nullptr, 0, 0);
    const int32_t n_expert = model_n_expert(model);
    lived_routing fake = served; // deliberately not what routing picks, so a replay is unmistakable
    for (auto & e : fake.ids) {
        e = (e + 1) % n_expert;
    }
    const lived_routing pinned = route_two(params, model, tokens, n, &fake, pin0, pin_n);
    int wrong = 0;
    for (int32_t il = 0; il < served.n_layer; ++il) {
        for (uint32_t row = 0; row < 2 * n; ++row) {
            const size_t o = ((size_t) il * served.n_pos + row) * served.n_used;
            const bool covered = row < n && (llama_pos) row >= pin0 && (llama_pos) row < pin0 + pin_n;
            const auto & want = covered ? fake.ids : served.ids;
            if (std::memcmp(&pinned.ids[o], &want[o], sizeof(int32_t) * served.n_used) != 0) {
                ++wrong;
            }
        }
    }
    printf("  pinned per sequence: %d of %u rows x %d layers differ from (seq 0 [%d,%d) replayed, the rest routed)\n",
           wrong, 2 * n, served.n_layer, pin0, pin0 + pin_n);
    return wrong == 0 ? 0 : 1;
}

// One sequence decoded as two same-shape ubatches, [0, n) then [n, 2n), on one context (the second may
// reuse the first's graph). `when` says when the pin (of positions [n, 2n), from `pin`) is set: never,
// before the first decode, or between the two.
enum class pin_when { never, before, between };
static lived_routing route_halves(const common_params & params_in, llama_model * model, const std::vector<llama_token> & tokens,
                                  uint32_t n, const lived_routing * pin, pin_when when) {
    common_params params = params_in;
    lived_routing r;
    r.n_layer = llama_model_n_layer(model);
    r.n_pos   = (int32_t) (2 * n);
    params.cb_eval           = routing_eval;
    params.cb_eval_user_data = &r;
    auto cparams = common_context_params_to_llama(params);
    cparams.n_ctx      = 4 * n;
    cparams.n_batch    = n;
    cparams.n_ubatch   = n;
    cparams.kv_unified = true;
    llama_context * ctx = llama_init_from_model(model, cparams);
    GGML_ASSERT(ctx != nullptr);
    auto set_pin = [&]() {
        std::vector<int32_t> ids((size_t) pin->n_layer * n * pin->n_used);
        for (int32_t il = 0; il < pin->n_layer; ++il) {
            std::memcpy(&ids[(size_t) il * n * pin->n_used], &pin->ids[((size_t) il * pin->n_pos + n) * pin->n_used],
                        sizeof(int32_t) * n * pin->n_used);
        }
        GGML_ASSERT(llama_set_pinned_experts(ctx, 0, pin->n_layer, pin->n_used, (llama_pos) n, (int32_t) n, ids.data()));
    };
    if (when == pin_when::before) {
        set_pin();
    }
    llama_batch batch = llama_batch_init((int32_t) n, 0, 1);
    for (uint32_t half = 0; half < 2; ++half) {
        if (half == 1 && when == pin_when::between) {
            set_pin();
        }
        for (uint32_t i = 0; i < n; ++i) {
            batch.token[i]     = tokens[half * n + i];
            batch.pos[i]       = (llama_pos) (half * n + i);
            batch.n_seq_id[i]  = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i]    = true;
        }
        batch.n_tokens = (int32_t) n;
        r.row0 = (int32_t) (half * n);
        GGML_ASSERT(llama_decode(ctx, batch) == 0);
    }
    llama_batch_free(batch);
    llama_free(ctx);
    return r;
}

// what this catches (Codex on fork #167 at 83d4edd52): a graph built with no covered row, reused for a
// covered ubatch of the same shape, skipping the pin; and one built before the pin was set, reused after.
static int pinned_rows_survive_graph_reuse(const common_params & params, llama_model * model, const std::vector<llama_token> & tokens) {
    const uint32_t n = 32;
    const lived_routing served = route_halves(params, model, tokens, n, nullptr, pin_when::never);
    const int32_t n_expert = model_n_expert(model);
    lived_routing fake = served;
    for (auto & e : fake.ids) {
        e = (e + 1) % n_expert;
    }
    int failed = 0;
    for (const pin_when when : { pin_when::before, pin_when::between }) {
        const lived_routing got = route_halves(params, model, tokens, n, &fake, when);
        int wrong = 0;
        for (int32_t il = 0; il < served.n_layer; ++il) {
            for (uint32_t row = 0; row < 2 * n; ++row) {
                const size_t o = ((size_t) il * served.n_pos + row) * served.n_used;
                const auto & want = row >= n ? fake.ids : served.ids;
                wrong += std::memcmp(&got.ids[o], &want[o], sizeof(int32_t) * served.n_used) != 0;
            }
        }
        printf("  pinned across graph reuse (pin set %s): %d of %u rows x %d layers differ from ([0,%u) routed, [%u,%u) replayed)\n",
               when == pin_when::before ? "before the first decode" : "between the decodes", wrong, 2 * n, served.n_layer, n, n, 2 * n);
        failed += wrong != 0;
    }
    return failed;
}

// what this catches: serving's capture recording something other than what the graph selected, or a
// last-layer row serving never routed (the cut keeps only outputs) recorded as lived. One serving
// decode of n rows, only the last an output (a prompt's prefill), with capture on and the graph's own
// ffn_moe_topk recorded beside it: every layer but the last matches row by row, and the last holds
// only the output row.
static int capture_records_what_serving_routed(const common_params & params_in, llama_model * model, const std::vector<llama_token> & tokens) {
    const uint32_t n = 32;
    common_params params = params_in;
    lived_routing r;
    r.n_layer = llama_model_n_layer(model);
    r.n_pos   = (int32_t) n;
    params.cb_eval           = routing_eval;
    params.cb_eval_user_data = &r;
    auto cparams = common_context_params_to_llama(params);
    cparams.n_ctx      = 2 * n;
    cparams.n_batch    = n;
    cparams.n_ubatch   = n;
    cparams.kv_unified = true;
    llama_context * ctx = llama_init_from_model(model, cparams);
    GGML_ASSERT(ctx != nullptr);
    GGML_ASSERT(llama_set_routing_capture(ctx, true) == LLAMA_ROUTING_CAPTURE_RECORDING);
    llama_batch batch = llama_batch_init((int32_t) n, 0, 1);
    for (uint32_t i = 0; i < n; ++i) {
        batch.token[i]     = tokens[i];
        batch.pos[i]       = (llama_pos) i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]    = i + 1 == n;
    }
    batch.n_tokens = (int32_t) n;
    GGML_ASSERT(llama_decode(ctx, batch) == 0);
    llama_batch_free(batch);
    std::vector<int32_t> table((size_t) r.n_layer * n * r.n_used);
    GGML_ASSERT(llama_get_lived_routing(ctx, 0, 0, (int32_t) n, table.data()));
    const size_t bytes = llama_lived_routing_bytes(ctx);
    llama_free(ctx);
    int wrong = 0;
    const int32_t last = r.n_layer - 1;
    for (int32_t il = 0; il < r.n_layer; ++il) {
        for (uint32_t p = 0; p < n; ++p) {
            const int32_t * got = &table[((size_t) il * n + p) * r.n_used];
            if (il == last && p + 1 < n) {
                wrong += got[0] != -1; // never routed by serving
                continue;
            }
            // the recorder keeps the last layer's single output row at its row 0
            const size_t row = il == last ? 0 : p;
            wrong += std::memcmp(got, &r.ids[((size_t) il * r.n_pos + row) * r.n_used], sizeof(int32_t) * r.n_used) != 0;
        }
    }
    printf("  capture: %d of %u rows x %d layers differ from what the graph selected (the last layer: its output only); table %zu bytes\n",
           wrong, n, r.n_layer, bytes);
    return wrong == 0 ? 0 : 1;
}

// WALK_ROUTING_BENCH: what capture costs serving. One context, a prefill, then single-token decodes
// (the generation shape) with capture off and on, alternated so the node's other load falls on both
// arms alike; each arm reports tokens per second over its decodes.
static void routing_capture_bench(const common_params & params, llama_model * model, const std::vector<llama_token> & tokens) {
    const uint32_t n_prompt = 64, n_gen = 64, rounds = 8;
    auto cparams = common_context_params_to_llama(params);
    cparams.n_ctx      = n_prompt + rounds * 2 * n_gen + 8;
    cparams.n_batch    = n_prompt;
    cparams.n_ubatch   = n_prompt;
    cparams.kv_unified = true;
    llama_context * ctx = llama_init_from_model(model, cparams);
    GGML_ASSERT(ctx != nullptr);
    llama_batch batch = llama_batch_init((int32_t) n_prompt, 0, 1);
    auto put = [&](uint32_t k, llama_token t, llama_pos p, bool out) {
        batch.token[k] = t; batch.pos[k] = p; batch.n_seq_id[k] = 1; batch.seq_id[k][0] = 0; batch.logits[k] = out;
    };
    for (uint32_t i = 0; i < n_prompt; ++i) {
        put(i, tokens[i], (llama_pos) i, i + 1 == n_prompt);
    }
    batch.n_tokens = (int32_t) n_prompt;
    GGML_ASSERT(llama_decode(ctx, batch) == 0);
    llama_synchronize(ctx);
    llama_pos pos = (llama_pos) n_prompt;
    double tps[2][8] = {};
    for (uint32_t r = 0; r < rounds; ++r) {
        for (int on = 0; on < 2; ++on) {
            GGML_ASSERT(llama_set_routing_capture(ctx, on != 0) == (on ? LLAMA_ROUTING_CAPTURE_RECORDING : LLAMA_ROUTING_CAPTURE_OFF));
            const int64_t t0 = ggml_time_us();
            for (uint32_t g = 0; g < n_gen; ++g) {
                put(0, tokens[(size_t) pos % tokens.size()], pos, true);
                batch.n_tokens = 1;
                GGML_ASSERT(llama_decode(ctx, batch) == 0);
                (void) llama_get_logits_ith(ctx, -1); // the sampler's read: the step's synchronize
                ++pos;
            }
            tps[on][r] = n_gen * 1e6 / (double) (ggml_time_us() - t0);
        }
    }
    llama_batch_free(batch);
    auto median = [&](int on) {
        std::vector<double> v(tps[on], tps[on] + rounds);
        std::sort(v.begin(), v.end());
        return (v[rounds / 2 - 1] + v[rounds / 2]) / 2.0;
    };
    printf("  capture bench (%u rounds of %u decodes, alternated): median off %.2f tok/s, on %.2f tok/s (on/off %.3f); table %zu bytes; capture host time %.1f us per captured decode\n",
           rounds, n_gen, median(0), median(1), median(1) / median(0), llama_lived_routing_bytes(ctx), (double) llama_lived_routing_host_us(ctx) / (rounds * n_gen));
    for (uint32_t r = 0; r < rounds; ++r) {
        printf("    round %u: off %.2f  on %.2f\n", r, tps[0][r], tps[1][r]);
    }
    llama_free(ctx);
}
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
    if (g_pinned) { // the lesson's routing as it was lived, replayed by every graph that trains its rows
        GGML_ASSERT(llama_set_pinned_experts(ctx, 0, g_pinned->n_layer, g_pinned->n_used, 0, g_pinned->n_pos, g_pinned->ids.data()));
    }
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
    llama_batch batch = llama_opt_batch_init(chunk);
    for (const auto & [p0, p1] : { std::make_pair(0u, r0), std::make_pair(r0, r1) }) {
        for (uint32_t c0 = p0; c0 < p1; c0 += chunk) {
            const uint32_t n = std::min(c0 + chunk, p1) - c0;
            walk_batch_clear(batch);
            for (uint32_t i = 0; i < n; ++i) {
                walk_batch_add(batch, tokens[c0 + i], c0 + i, { 0 }, i + 1 == n);
            }
            GGML_ASSERT(walk_decode(ctx, batch) == 0);
        }
    }
    std::vector<uint8_t> state(llama_state_seq_get_size(ctx, 0));
    GGML_ASSERT(llama_state_seq_get_data(ctx, state.data(), state.size(), 0) == state.size());
    llama_opt_batch_free(batch);
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
    llama_batch batch = llama_opt_batch_init(2);
    walk_batch_add(batch, tokens[0], 0, {0}, true);
    walk_batch_add(batch, tokens[1], 0, {1}, true);
    GGML_ASSERT(walk_decode(ctx, batch) == 0);
    const auto reference_neighbor = sequence_state(ctx, 1);
    const auto reference_target = sequence_state(ctx, 0);
    batch.label[0] = tokens[2];
    batch.pos[0] = batch.pos[1] = 1;
    GGML_ASSERT(walk_decode(ctx, batch) != 0);
    GGML_ASSERT(sequence_state(ctx, 0) == reference_target && sequence_state(ctx, 1) == reference_neighbor);
    llama_memory_clear(llama_get_memory(ctx), true);
    batch.pos[0] = batch.pos[1] = 0;
    llama_opt_init(ctx, model, op);
    const auto initial_weights = adapter_params(adapter);
    batch.label[0] = tokens[2];
    GGML_ASSERT(walk_decode(ctx, batch) == 0);
    GGML_ASSERT(llama_get_logits_ith(ctx, 0) && llama_get_logits_ith(ctx, 1));
    const auto neighbor = sequence_state(ctx, 1);
    GGML_ASSERT(neighbor == reference_neighbor);
    const auto trained = sequence_state(ctx, 0);
    GGML_ASSERT(neighbor != trained);
    const auto weights = adapter_params(adapter);
    GGML_ASSERT(weights != initial_weights);
    walk_batch_clear(batch);
    walk_batch_add(batch, tokens[2], 1, {0}, true);
    batch.label[0] = llama_vocab_n_tokens(llama_model_get_vocab(model));
    GGML_ASSERT(walk_decode(ctx, batch) != 0);
    GGML_ASSERT(sequence_state(ctx, 0) == trained && sequence_state(ctx, 1) == neighbor);
    GGML_ASSERT(adapter_params(adapter) == weights);
    batch.label[0] = tokens[3];
    GGML_ASSERT(walk_decode(ctx, batch) == 0);
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
        GGML_ASSERT(walk_decode(ctx, batch) != 0);
        GGML_ASSERT(sequence_state(ctx, 0) == pending && sequence_state(ctx, 1) == neighbor);
        GGML_ASSERT(adapter_params(adapter) == pending_weights);
    }
    printf("mixed decode flash=%d cache=%s undo=%u: same-position sequences, neighbour preservation, admission refusal OK\n", flash, ggml_type_name(cache_type), undo);
    llama_opt_batch_free(batch);
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
        llama_batch batch = llama_opt_batch_init(2);
        for (size_t k = 0; k < sequences.size(); ++k) {
            const int seq = sequences[k];
            walk_batch_add(batch, tokens[seq], 0, {seq}, true);
            batch.label[batch.n_tokens - 1] = tokens[seq + 2];
            if (!weights.empty()) {
                batch.label_weight[batch.n_tokens - 1] = weights[k];
            }
        }
        float * kept_weights = batch.label_weight;
        if (weights.empty()) {
            batch.label_weight = nullptr; // unweighted: the loss every backend has
        }
        GGML_ASSERT(walk_decode(ctx, batch) == 0);
        batch.label_weight = kept_weights;
        auto delta = adapter_params(adapter);
        for (size_t i = 0; i < delta.size(); ++i) {
            delta[i] -= before[i];
        }
        llama_opt_batch_free(batch);
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
        llama_batch batch = llama_opt_batch_init(3);
        int neighbour_row = -1;
        const std::vector<int> order = reversed ? std::vector<int>{2, 0, 1} : std::vector<int>{0, 1, 2};
        for (int row : order) {
            walk_batch_add(batch, tokens[row], row == 1 ? 1 : 0, {row == 2 ? 1 : 0},
                    public_mode == 2 || (public_mode == 1 && row == 2));
            if (train && row == 1) {
                batch.label[batch.n_tokens - 1] = tokens[3];
            }
            if (row == 2) {
                neighbour_row = batch.n_tokens - 1;
            }
        }
        GGML_ASSERT(walk_decode(ctx, batch) == 0);
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
        llama_opt_batch_free(batch);
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

    // Fixture provisioning for the real server protocol tests, using the same adapter owner.
    if (const char * shadow = getenv("WALK_WRITE_SHADOW")) {
        GGML_ASSERT(std::rename(init.c_str(), shadow) == 0);
        llama_model_free(model);
        llama_backend_free();
        return 0;
    }

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

    // WALK_PINNED: on a mixture-of-experts base, is the tiled walk's disagreement with one graph the
    // walk, or routing flips between row shapes? Both are measured: each graph routing for itself, then
    // both replaying the routing the window was served with. If pinning restores agreement, routing
    // was the cause, and the walk replays the lived routing by design.
    if (getenv("WALK_ROUTING_BENCH")) {
        routing_capture_bench(params, model, tokens);
        std::remove(init.c_str());
        llama_model_free(model);
        llama_backend_free();
        return 0;
    }
    if (getenv("WALK_PINNED")) {
        if (capture_records_what_serving_routed(params, model, tokens) != 0 || pinned_rows_are_per_sequence(params, model, tokens) != 0 ||
            pinned_rows_survive_graph_reuse(params, model, tokens) != 0) {
            fprintf(stderr, "FAILED: a pinned row is not its own sequence's (Codex on fork #167)\n");
            return 1;
        }
        if (getenv("WALK_PINNED_CHECKS_ONLY")) { // the routing checks alone, without the learning comparison
            std::remove(init.c_str());
            llama_model_free(model);
            llama_backend_free();
            return 0;
        }
        std::vector<uint8_t> all(WINDOW + 1, 1);
        all[0] = 0;
        const uint32_t tile = getenv("WALK_PINNED_TILE") ? (uint32_t) atoi(getenv("WALK_PINNED_TILE")) : 64;
        const auto ref_routed  = step_delta(params, model, init, tokens, all, WINDOW, true);
        const auto tile_routed = step_delta(params, model, init, tokens, all, tile, true);
        compare("routed by each graph: tiles vs one graph", tile_routed, ref_routed);
        const lived_routing lived = record_routing(params, model, init, tokens);
        g_pinned = &lived;
        const auto ref_pinned  = step_delta(params, model, init, tokens, all, WINDOW, true);
        const auto tile_pinned = step_delta(params, model, init, tokens, all, tile, true);
        g_pinned = nullptr;
        const agreement pinned = compare("routing pinned as lived: tiles vs one graph", tile_pinned, ref_pinned);
        if (pinned.cosine < 0.997) {
            worst_tensors(tile_pinned, ref_pinned, 12);
        }
        std::remove(init.c_str());
        llama_model_free(model);
        llama_backend_free();
        return pinned.cosine < 0.997 ? 1 : 0;
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
        // WALK_LIVE_LATE_LABELS=K: her reply's first K rows unlabelled, so a run starts K rows past the served prefix
        // and the walk decodes that context before it (Codex: a long unlabelled prefix must still be one tile per unit)
        const uint32_t late = getenv("WALK_LIVE_LATE_LABELS") ? (uint32_t) atoi(getenv("WALK_LIVE_LATE_LABELS")) : 0;
        std::vector<uint8_t> reply(WINDOW + 1, 0);
        for (uint32_t i = WINDOW / 2 + 37 + late; i <= WINDOW; ++i) {
            reply[i] = 1;
        }
        const llama_pos r0 = WINDOW / 2 + 36; // the first labelled position
        const auto want_f32 = step_delta(params, model, init, tokens, reply, WINDOW, true);
        g_cache_type = GGML_TYPE_F16;
        const auto want_f16 = step_delta(params, model, init, tokens, reply, WINDOW, true);
        g_cache_type = GGML_TYPE_F32;

      for (const bool streams : { false, true }) {
        if (getenv("WALK_LIVE_STREAMS_ONLY") && !streams) { continue; } // the Core's layout alone: each mind's KV its own stream
        // unified F32 cache with explicit attention, then a stream per slot with an f16 cache and
        // flash attention: how a serving lane runs (each slot attends over its own stream)
        printf("  live sequence, %s:\n", streams ? "a stream per slot, f16 cache, flash attention" : "one unified F32 cache, explicit attention");
        size_t graph_segments = 0;
        auto two_seq_ctx = [&]() {
            auto cp = common_context_params_to_llama(params);
            cp.n_ctx = 2*WINDOW; cp.n_batch = WINDOW; cp.n_ubatch = WINDOW; cp.n_seq_max = 2; cp.kv_unified = !streams;
            if (getenv("WALK_LIVE_BRACKET_NEIGHBOURS")) { cp.n_ctx = 3*WINDOW; cp.n_seq_max = 3; }
            if (getenv("WALK_LIVE_GRAPH_SMALL_UBATCH")) { cp.n_ubatch = WINDOW / 8; }
            if (getenv("WALK_LIVE_SMALL_RESTORE_BATCH")) { cp.n_batch = cp.n_ubatch = WINDOW / 8; }
            cp.flash_attn_type = streams ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
            cp.type_k = streams ? GGML_TYPE_F16 : GGML_TYPE_F32; cp.type_v = cp.type_k;
            // Graph segmentation preserves the full token graph and its gradient.
            // This tests existing scheduler views only, NOT resumable serving yields.
            if (getenv("WALK_LIVE_GRAPH_SEGMENTS")) {
                cp.cb_eval = [](ggml_tensor *, bool ask, void * data) {
                    if (!ask) { ++*static_cast<size_t *>(data); }
                    return true;
                };
                cp.cb_eval_user_data = &graph_segments;
            }
            llama_context * c = llama_init_from_model(model, cp);
            if (getenv("WALK_LIVE_PINNED")) { // serving records her routing; her lesson replays it
                GGML_ASSERT(llama_set_routing_capture(c, true) == LLAMA_ROUTING_CAPTURE_RECORDING);
            }
            return c;
        };
        auto decode_span = [&](llama_context * c, llama_seq_id sq, const llama_token * t, llama_pos p0, int32_t n) {
            llama_batch b = llama_opt_batch_init(WINDOW);
            for (int32_t first = 0; first < n;) {
                const int32_t last = std::min(n, first + (int32_t) llama_n_batch(c));
                walk_batch_clear(b);
                for (int32_t i = first; i < last; ++i) {
                    walk_batch_add(b, t[i], p0 + i, { sq }, i + 1 == n);
                }
                GGML_ASSERT(walk_decode(c, b) == 0);
                first = last;
            }
            llama_opt_batch_free(b);
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
        if (getenv("WALK_LIVE_BRACKET_NEIGHBOURS")) { decode_span(ctl, 2, neighbour, 0, 200); }
        decode_span(ctl, 0, tokens.data() + r0, r0, WINDOW - r0);
        const std::vector<uint8_t> want_her = seq_state(ctl, 0);
        if (getenv("WALK_LIVE_BRACKET_NEIGHBOURS")) {
            auto paired = llama_opt_batch_init(2);
            paired.n_tokens = 2;
            for (int i = 0; i < 2; ++i) {
                paired.token[i] = neighbour[200]; paired.pos[i] = 200;
                paired.n_seq_id[i] = 1; paired.seq_id[i][0] = i + 1; paired.logits[i] = true;
            }
            if (llama_model_is_hybrid(model) || llama_model_is_recurrent(model)) {
                decode_span(ctl, 1, neighbour + 200, 200, 1);
                decode_span(ctl, 2, neighbour + 200, 200, 1);
            } else {
                GGML_ASSERT(walk_decode(ctl, paired) == 0);
            }
            llama_opt_batch_free(paired);
        } else {
            decode_span(ctl, 1, neighbour + 200, 200, 1);
        }
        const auto want_neighbour_turn = seq_state(ctl, 1);
        std::vector<uint8_t> want_second_neighbour;
        if (getenv("WALK_LIVE_BRACKET_NEIGHBOURS")) {
            want_second_neighbour = seq_state(ctl, 2);
        }
        // WALK_LIVE_PINNED: her routing as served, and the reference step that replays it
        lived_routing served_routing;
        std::vector<float> want_pinned;
        if (getenv("WALK_LIVE_PINNED")) {
            served_routing.n_layer = llama_model_n_layer(model);
            served_routing.n_used  = model_n_expert_used(model);
            served_routing.n_pos   = (int32_t) WINDOW;
            served_routing.ids.resize((size_t) served_routing.n_layer * WINDOW * served_routing.n_used);
            GGML_ASSERT(llama_get_lived_routing(ctl, 0, 0, (int32_t) WINDOW, served_routing.ids.data()));
            const ggml_type kept = g_cache_type;
            g_cache_type = streams ? GGML_TYPE_F16 : GGML_TYPE_F32; // the cache the live context keeps
            g_pinned     = &served_routing;
            want_pinned  = step_delta(params, model, init, tokens, reply, WINDOW, true);
            g_pinned     = nullptr;
            g_cache_type = kept;
        }
        llama_free(ctl);

        llama_adapter_lora * adapter = llama_adapter_lora_init(model, init.c_str());
        const std::vector<float> before = adapter_params(adapter);
        llama_context * ctx = two_seq_ctx();
        float scale = 1.0f;
        GGML_ASSERT(llama_set_adapters_lora(ctx, &adapter, 1, &scale) == 0);
        decode_span(ctx, 1, neighbour, 0, 200);
        decode_span(ctx, 0, tokens.data(), 0, r0); // her context, as served
        if (getenv("WALK_LIVE_PINNED")) {
            // her reply was served in the control: carry its routing here, as a sequence paged in carries
            // it, then lift the pin so only her lesson replays it (the table keeps it)
            const int32_t span = (int32_t) WINDOW - r0;
            std::vector<int32_t> reply_routing((size_t) served_routing.n_layer * span * served_routing.n_used);
            for (int32_t il = 0; il < served_routing.n_layer; ++il) {
                std::memcpy(&reply_routing[(size_t) il * span * served_routing.n_used],
                            &served_routing.ids[((size_t) il * WINDOW + r0) * served_routing.n_used],
                            sizeof(int32_t) * span * served_routing.n_used);
            }
            GGML_ASSERT(llama_set_pinned_experts(ctx, 0, served_routing.n_layer, served_routing.n_used, r0, span, reply_routing.data()));
            GGML_ASSERT(llama_set_pinned_experts(ctx, 0, served_routing.n_layer, served_routing.n_used, 0, 0, nullptr));
        }
        if (getenv("WALK_LIVE_BRACKET_NEIGHBOURS")) { decode_span(ctx, 2, neighbour, 0, 200); }
        const std::vector<uint8_t> neighbour_before = seq_state(ctx, 1);
        struct admission_counts { size_t statics = 0, workspace = 0; bool allow_workspace = true; std::map<std::string, size_t> limits; } stale_admission, current_admission;
        llama_opt_params lopt{};
        lopt.param_filter   = llama_opt_param_filter_all;
        lopt.get_opt_pars   = sgd_pars;
        lopt.optimizer_type = GGML_OPT_OPTIMIZER_TYPE_SGD;
        lopt.adapter        = adapter;
        // a reply learning in tiles walks exactly (UNIFIED-PAGING.md, "A long reply learns in tiles")
        // a reply past one ubatch learns only through the exact walk (llama_opt_learn refuses it otherwise)
        lopt.walk_exact     = getenv("WALK_LIVE_TILES") != nullptr || getenv("WALK_LIVE_EXACT_ONLY") != nullptr
                           || getenv("WALK_LIVE_GRAPH_SMALL_UBATCH") != nullptr || getenv("WALK_LIVE_SMALL_RESTORE_BATCH") != nullptr;
        lopt.static_admission = [](ggml_backend_buffer_type_t, const char *, size_t, void * data) {
            ++static_cast<admission_counts *>(data)->statics;
            return true;
        };
        lopt.workspace_admission = [](ggml_backend_buffer_type_t buft, size_t required, size_t, void * data) {
            auto & admission = *static_cast<admission_counts *>(data);
            ++admission.workspace;
            if (!admission.allow_workspace) { return false; }
            if (admission.limits.empty()) { return true; }
            const auto limit = admission.limits.find(ggml_backend_buft_name(buft));
            return limit != admission.limits.end() && required <= limit->second;
        };
        lopt.static_admission_data = lopt.workspace_admission_data = &stale_admission;
        llama_opt_init(ctx, model, lopt);
        // What this catches: renewing authority for the already selected mind must replace
        // both callback bindings; retaining a prior request's user data is a dangling lease.
        lopt.static_admission_data = lopt.workspace_admission_data = &current_admission;
        GGML_ASSERT(llama_opt_init(ctx, model, lopt));
        // What this catches: the real served-window labelled graph can be priced
        // without allocating optimizer buffers or changing either resident's KV.
        {
            const auto her_preflight = seq_state(ctx, 0);
            const auto neighbour_preflight = seq_state(ctx, 1);
            size_t requirements = 0, allocated = 0;
            struct measured_plan { size_t * total; admission_counts * admission; } plan{&requirements, &current_admission};
            const bool measured = llama_opt_measure_memory(ctx,
                [](const llama_adapter_lora *, const char *, const char * buft, const char * kind,
                        size_t bytes, bool, void * data) {
                    auto & plan = *static_cast<measured_plan *>(data);
                    if (std::string(kind).find("required") != std::string::npos) { *plan.total += bytes; }
                    if (getenv("WALK_LIVE_GRAPH_NEIGHBOUR") && std::string(kind) == "shared_compute_required") {
                        plan.admission->limits[buft] = bytes;
                    }
                }, &plan);
            llama_opt_visit_memory(ctx, [](const llama_adapter_lora *, const char *, const char *,
                    const char *, size_t bytes, bool, void * data) {
                *static_cast<size_t *>(data) += bytes;
            }, &allocated);
            if (!measured || requirements == 0 || allocated != 0 || adapter_params(adapter) != before ||
                    seq_state(ctx, 0) != her_preflight || seq_state(ctx, 1) != neighbour_preflight) {
                fprintf(stderr, "FAILED: dry learning preflight changed residency/state or refused: %s\n", llama_opt_failure(ctx));
                return 1;
            }
            printf("  dry learning preflight: %zu required bytes; optimizer unallocated, both sequences unchanged\n", requirements);
        }
        std::vector<llama_token> labels(WINDOW - r0);
        for (int32_t i = 0; i < (int32_t) WINDOW - r0; ++i) {
            labels[i] = (uint32_t) i < late ? -1 : tokens[r0 + i + 1];
        }
        // refusals come before any change: her sequence byte-identical after each
        {
            const std::vector<uint8_t> her_before = seq_state(ctx, 0);
            const bool r_null = llama_opt_learn(ctx, 0, r0, tokens.data() + r0, nullptr, WINDOW - r0, nullptr);
            const bool r_long = llama_opt_learn(ctx, 0, r0, tokens.data() + r0, labels.data(), llama_n_ctx(ctx), nullptr);
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
        // What this catches: a reply past one ubatch learns only as an exact tile walk over her own stream; a
        // cache shared among sequences cannot run one, so begin refuses before anything moves (never a plain
        // split, which lost cosine 0.949 / norm 0.889 on Qwen3.5-0.8B at ubatch 64).
        if (!streams && (uint32_t) (WINDOW - r0) > llama_n_ubatch(ctx)) {
            const std::vector<uint8_t> her_before = seq_state(ctx, 0), nb_before = seq_state(ctx, 1);
            const bool began = llama_opt_learn_begin(ctx, 0, r0, tokens.data() + r0, labels.data(), WINDOW - r0,
                tokens.data() + r0, WINDOW - r0, nullptr);
            const bool same = seq_state(ctx, 0) == her_before && seq_state(ctx, 1) == nb_before;
            printf("  shared cache, reply past one ubatch: %s; both sequences %s\n", began ? "ACCEPTED" : "refused",
                    same ? "byte-identical" : "CHANGED");
            if (began || !same) {
                fprintf(stderr, "FAILED: a shared cache began a reply it cannot walk exactly\n");
                ++failures;
            }
            llama_free(ctx);
            llama_adapter_lora_free(adapter);
            continue;
        }
        const auto generation_prefix = seq_state(ctx, 0);
        // What this catches: revoked authority must stop NEW gradients without abandoning KV
        // restoration or freeing optimizer custody. A drain is not proof the reply was learned.
        GGML_ASSERT(llama_opt_learn_begin(ctx, 0, r0, tokens.data() + r0, labels.data(), WINDOW - r0,
            tokens.data() + r0, WINDOW - r0, nullptr));
        GGML_ASSERT(llama_opt_learn_needs_gradient(ctx));
        GGML_ASSERT(llama_opt_learn_drain(ctx));
        GGML_ASSERT(llama_opt_learn_drain(ctx));
        GGML_ASSERT(!llama_opt_learn_needs_gradient(ctx));
        llama_opt_learn_status drained;
        do { drained = llama_opt_learn_step(ctx); } while (drained == LLAMA_OPT_LEARN_MORE);
        GGML_ASSERT(drained == LLAMA_OPT_LEARN_DONE);
        GGML_ASSERT(!llama_opt_learn_drain(ctx));
        GGML_ASSERT(adapter_params(adapter) == before);
        GGML_ASSERT(seq_state(ctx, 0) == want_her);
        GGML_ASSERT(seq_state(ctx, 1) == neighbour_before);
        GGML_ASSERT(llama_state_seq_set_data(ctx, generation_prefix.data(), generation_prefix.size(), 0) == generation_prefix.size());

        if (getenv("WALK_LIVE_GRAPH_CONTINUATION")) {
            GGML_ASSERT(llama_opt_learn_begin(ctx, 0, r0, tokens.data() + r0, labels.data(), WINDOW - r0,
                tokens.data() + r0, WINDOW - r0, nullptr));
            ctx->opt_reply->graph_nodes = 1;
            GGML_ASSERT(llama_opt_learn_step(ctx) == LLAMA_OPT_LEARN_MORE);
            GGML_ASSERT(ctx->opt_reply->graph_preparing);
            GGML_ASSERT(llama_opt_learn_drain(ctx));
            do { drained = llama_opt_learn_step(ctx); } while (drained == LLAMA_OPT_LEARN_MORE);
            GGML_ASSERT(drained == LLAMA_OPT_LEARN_DONE);
            GGML_ASSERT(adapter_params(adapter) == before);
            GGML_ASSERT(seq_state(ctx, 0) == want_her && seq_state(ctx, 1) == neighbour_before);
            GGML_ASSERT(llama_state_seq_set_data(ctx, generation_prefix.data(), generation_prefix.size(), 0) == generation_prefix.size());

            GGML_ASSERT(llama_opt_learn_begin(ctx, 0, r0, tokens.data() + r0, labels.data(), WINDOW - r0,
                tokens.data() + r0, WINDOW - r0, nullptr));
            ctx->opt_reply->graph_nodes = 1;
            GGML_ASSERT(llama_opt_learn_step(ctx) == LLAMA_OPT_LEARN_MORE);
            GGML_ASSERT(ctx->opt_reply->graph_preparing);
            current_admission.allow_workspace = false;
            GGML_ASSERT(llama_opt_learn_step(ctx) == LLAMA_OPT_LEARN_MORE);
            current_admission.allow_workspace = true;
            do { drained = llama_opt_learn_step(ctx); } while (drained == LLAMA_OPT_LEARN_MORE);
            GGML_ASSERT(drained == LLAMA_OPT_LEARN_FAILED);
            GGML_ASSERT(adapter_params(adapter) == before);
            GGML_ASSERT(seq_state(ctx, 0) == want_her && seq_state(ctx, 1) == neighbour_before);
            GGML_ASSERT(llama_state_seq_set_data(ctx, generation_prefix.data(), generation_prefix.size(), 0) == generation_prefix.size());
        }

        // What this catches: yielding with training flags still installed, losing accumulated
        // labels between chunks, or restoring the neighbour instead of the learning sequence.
        // Neighbour inference runs at returned boundaries; learning covers the entire reply.
        if (getenv("WALK_LIVE_BRACKET_NEIGHBOURS")) {
            llama_memory_clear(llama_get_memory(ctx), true);
            decode_span(ctx, 1, neighbour, 0, 200);
            decode_span(ctx, 0, tokens.data(), 0, r0);
            decode_span(ctx, 2, neighbour, 0, 200);
        }
        ggml_opt_result_t tile_result = (getenv("WALK_LIVE_TILES") || getenv("WALK_LIVE_LOSS")) ? ggml_opt_result_init() : nullptr; // diagnostic: the learned loss
        GGML_ASSERT(llama_opt_learn_begin(ctx, 0, r0, tokens.data() + r0, labels.data(), WINDOW - r0, tokens.data() + r0, WINDOW - r0, tile_result));
        // What this catches (Codex on the join): tiles that lose the gradient a later token carries into the
        // earlier ones, a walk that disturbs her neighbour between its steps, or a restore that is not her served
        // self. The reply learns through the exact walk in tiles of WINDOW / 8, one per step, and every assertion
        // below holds: the step is the walk's exact step, her cache a stable decode, the neighbour untouched.
        // the walk's layout contract, both ways: refused up front on a shared cache (the reply still learns, by its
        // ordinary path), accepted on the Core's layout, a stream per mind
        // WALK_LIVE_HOST_BUDGET=bytes: a host budget that forces a checkpoint stride above 1, so the reverse replays
        // from a stride checkpoint to each tile (Codex: that replay must still be one tile per unit)
        if (getenv("WALK_LIVE_HOST_BUDGET")) { llama_opt_set_walk_host_budget(ctx, (size_t) atoll(getenv("WALK_LIVE_HOST_BUDGET"))); }
        if (getenv("WALK_LIVE_TILES")) { GGML_ASSERT(llama_opt_learn_tile(ctx, getenv("WALK_LIVE_TILE_ROWS") ? (uint32_t) atoi(getenv("WALK_LIVE_TILE_ROWS")) : WINDOW / 8) == streams); }
        // Diagnostic: change only scheduling boundaries, not the served graph/window
        // or the independent whole-reply gradient oracle. No production chunk policy.
        if (getenv("WALK_LIVE_SMALL_CHUNKS")) {
            ctx->opt_reply->chunk_tokens = WINDOW / 8;
        }
        if (getenv("WALK_LIVE_GRAPH_CONTINUATION")) {
            ctx->opt_reply->graph_nodes = 1;
        }
        // A checkpoint may not publish an optimizer/cursor pair from half a live reply.
        FILE * partial_state = tmpfile();
        GGML_ASSERT(partial_state != nullptr);
        GGML_ASSERT(!llama_opt_state_write(ctx, partial_state, "active-live-reply"));
        fclose(partial_state);
        llama_opt_learn_status progress;
        uint32_t tiled_units = 0, tiled_decoded = 0, tiled_most = 0; // every tiled step's work, for the receipt
        int yielded = 0;
        int graph_neighbour_turns = 0;
        bool shifted_neighbour_pending = false;
        bool injected_graph_failure = false;
        std::vector<float> uncertain_weights;
        do {
            if (getenv("WALK_LIVE_SMALL_CHUNKS") && !llama_opt_learn_needs_gradient(ctx)) {
                ctx->opt_reply->chunk_tokens = WINDOW; // retain the control's restoration segmentation
            }
            // what this catches (Codex on c6d40d7917): a tile step that decodes more than its tile in one unit
            // (the context before a run, a reverse replay from a stride checkpoint, a run restored between runs),
            // holding the Core past the slack the tile was sized to. Each tiled step reports its own work
            // (llama_opt_learn_last_unit), and no step decodes more than one tile. Counted at the decode, since a
            // reverse pops her cache back to its checkpoint before it replays: the position she ends at hides a replay.
            const bool tiled = ctx->opt_reply != nullptr && ctx->opt_reply->tiles != nullptr;
            progress = llama_opt_learn_step(ctx);
            if (tiled && getenv("WALK_LIVE_TILE_ROWS")) {
                const llama_opt_learn_unit unit = llama_opt_learn_last_unit(ctx);
                ++tiled_units;
                tiled_decoded += unit.decoded;
                tiled_most = std::max(tiled_most, unit.decoded);
                if (unit.decoded > (uint32_t) atoi(getenv("WALK_LIVE_TILE_ROWS"))) {
                    fprintf(stderr, "FAILED: one tiled unit decoded %u rows, past its tile of %s\n", unit.decoded, getenv("WALK_LIVE_TILE_ROWS"));
                    ++failures;
                }
            }
            if (progress == LLAMA_OPT_LEARN_MORE) {
                if (getenv("WALK_LIVE_BRACKET_NEIGHBOURS") && yielded == 0) {
                    auto paired = llama_opt_batch_init(2);
                    paired.n_tokens = 2;
                    for (int i = 0; i < 2; ++i) {
                        paired.token[i] = neighbour[200]; paired.pos[i] = 200;
                        paired.n_seq_id[i] = 1; paired.seq_id[i][0] = i + 1; paired.logits[i] = true;
                    }
                    const auto held0 = seq_state(ctx, 0);
                    GGML_ASSERT(walk_decode(ctx, paired) == 0);
                    llama_opt_batch_free(paired);
                    GGML_ASSERT(ctx->opt_reply->graph_meta);
                    GGML_ASSERT(seq_state(ctx, 0) == held0);
                }
                if (getenv("WALK_LIVE_GRAPH_FAILURE") && !injected_graph_failure && !ctx->opt_reply->graph_preparing) {
                    GGML_ASSERT(ctx->opt_reply->graph_meta);
                    // Model a parameter write before failure; recovery must preserve and quarantine it, not invent rollback.
                    auto * changed = adapter->ab_map.begin()->second.a;
                    float value;
                    ggml_backend_tensor_get(changed, &value, 0, sizeof(value));
                    value += 0.125f;
                    ggml_backend_tensor_set(changed, &value, 0, sizeof(value));
                    uncertain_weights = adapter_params(adapter);
                    ctx->opt_reply->graph_nodes = -1; // fail the next continuation with live graph custody
                    injected_graph_failure = true;
                }
                if (getenv("WALK_LIVE_GRAPH_DRAIN") && yielded == 1) {
                    GGML_ASSERT(ctx->opt_reply->graph_meta);
                    GGML_ASSERT(llama_opt_learn_drain(ctx));
                    GGML_ASSERT(ctx->opt_reply->drain_graph);
                }
                if (!getenv("WALK_LIVE_NO_NEIGHBOUR") && !shifted_neighbour_pending && (!getenv("WALK_LIVE_GRAPH_CONTINUATION") || (getenv("WALK_LIVE_GRAPH_NEIGHBOUR") && yielded % 128 == 0))) {
                    GGML_ASSERT(llama_state_seq_set_data(ctx, neighbour_before.data(), neighbour_before.size(), 1) == neighbour_before.size());
                    if (getenv("WALK_LIVE_GRAPH_SHIFT") && graph_neighbour_turns == 0 && llama_memory_can_shift(llama_get_memory(ctx))) {
                        // A queued K shift must not be consumed or silently skipped beside a retained graph.
                        llama_memory_seq_add(llama_get_memory(ctx), 1, 0, -1, 1);
                        auto shifted = llama_opt_batch_init(1);
                        shifted.n_tokens = 1;
                        shifted.token[0] = neighbour[200]; shifted.pos[0] = 201;
                        shifted.n_seq_id[0] = 1; shifted.seq_id[0][0] = 1; shifted.logits[0] = true;
                        GGML_ASSERT(walk_decode(ctx, shifted) != 0);
                        GGML_ASSERT(walk_decode(ctx, shifted) != 0);
                        llama_opt_batch_free(shifted);
                        shifted_neighbour_pending = true;
                        ++yielded;
                        continue; // keep the actual queued shift until the training graph completes
                    }
                    if (getenv("WALK_LIVE_GRAPH_NEIGHBOUR") && graph_neighbour_turns == 0) {
                        const auto held = llama_get_memory_breakdown(ctx);
                        auto denied = llama_opt_batch_init(1);
                        denied.n_tokens = 1;
                        denied.token[0] = neighbour[200]; denied.pos[0] = 200;
                        denied.n_seq_id[0] = 1; denied.seq_id[0][0] = 1; denied.logits[0] = true;
                        current_admission.allow_workspace = false;
                        GGML_ASSERT(walk_decode(ctx, denied) != 0);
                        current_admission.allow_workspace = true;
                        llama_opt_batch_free(denied);
                        const auto after_refusal = llama_get_memory_breakdown(ctx);
                        for (const auto & row : held) {
                            GGML_ASSERT(after_refusal.at(row.first).compute == row.second.compute);
                        }
                        GGML_ASSERT(llama_state_seq_set_data(ctx, neighbour_before.data(), neighbour_before.size(), 1) == neighbour_before.size());
                    }
                    decode_span(ctx, 1, neighbour + 200, 200, 1);
                    ++graph_neighbour_turns;
                    if (getenv("WALK_LIVE_TILES") && seq_state(ctx, 1) != want_neighbour_turn) { const auto got_n = seq_state(ctx, 1); size_t diff = 0, first = SIZE_MAX; for (size_t b = 0; b < std::min(got_n.size(), want_neighbour_turn.size()); ++b) { if (got_n[b] != want_neighbour_turn[b]) { ++diff; first = std::min(first, b); } } fprintf(stderr, "DEBUG neighbour differs at yield %d: size %zu vs %zu, %zu bytes differ, first at %zu\n", yielded, got_n.size(), want_neighbour_turn.size(), diff, first); }
                    GGML_ASSERT(seq_state(ctx, 1) == want_neighbour_turn);
                }
                ++yielded;
            }
        } while (progress == LLAMA_OPT_LEARN_MORE);
        if (tiled_units > 0) { printf("  tiled units: %u, %u rows decoded, at most %u in one\n", tiled_units, tiled_decoded, tiled_most); }
        if (tile_result) { double loss = 0, unc = 0; ggml_opt_result_loss(tile_result, &loss, &unc); printf("  DEBUG tiled walk: training loss %f\n", loss); ggml_opt_result_free(tile_result); }
        GGML_ASSERT(yielded >= 1);
        if (injected_graph_failure) {
            GGML_ASSERT(progress == LLAMA_OPT_LEARN_FAILED && !ctx->opt_reply);
            GGML_ASSERT(adapter->restore_quarantined && adapter_params(adapter) == uncertain_weights);
            GGML_ASSERT(std::string(llama_opt_failure(ctx)).find("effects uncertain") != std::string::npos);
            GGML_ASSERT(seq_state(ctx, 0) == want_her);
            FILE * refused_checkpoint = tmpfile();
            GGML_ASSERT(refused_checkpoint && !llama_opt_state_write(ctx, refused_checkpoint, "failed-graph"));
            fclose(refused_checkpoint);
            GGML_ASSERT(llama_set_adapters_lora(ctx, &adapter, 1, &scale) != 0);
            GGML_ASSERT(llama_set_adapters_lora(ctx, nullptr, 0, nullptr) == 0);
            GGML_ASSERT(llama_state_seq_set_data(ctx, neighbour_before.data(), neighbour_before.size(), 1) == neighbour_before.size());
            decode_span(ctx, 1, neighbour + 200, 200, 1);
            GGML_ASSERT(seq_state(ctx, 1) == want_neighbour_turn);
            GGML_ASSERT(ctx->opt_release(adapter));
            auto * replacement = llama_adapter_lora_init(model, init.c_str());
            GGML_ASSERT(replacement && llama_set_adapters_lora(ctx, &replacement, 1, &scale) == 0);
            lopt.adapter = replacement;
            llama_opt_init(ctx, model, lopt);
            GGML_ASSERT(llama_state_seq_set_data(ctx, generation_prefix.data(), generation_prefix.size(), 0) == generation_prefix.size());
            GGML_ASSERT(llama_opt_learn(ctx, 0, r0, tokens.data() + r0, labels.data(), WINDOW - r0, nullptr));
            llama_free(ctx);
            llama_adapter_lora_free(replacement);
            llama_adapter_lora_free(adapter);
            printf("  failed continuation: quarantined, stable sequence restored, neighbour serves, optimizer released\n");
            continue;
        }
        if (getenv("WALK_LIVE_GRAPH_CONTINUATION")) {
            GGML_ASSERT(yielded > 100);
            if (shifted_neighbour_pending) {
                auto * shift_control = two_seq_ctx();
                decode_span(shift_control, 1, neighbour, 0, 200);
                llama_memory_seq_add(llama_get_memory(shift_control), 1, 0, -1, 1);
                decode_span(shift_control, 1, neighbour + 200, 201, 1);
                decode_span(ctx, 1, neighbour + 200, 201, 1);
                GGML_ASSERT(seq_state(ctx, 1) == seq_state(shift_control, 1));
                llama_free(shift_control);
                GGML_ASSERT(llama_memory_seq_rm(llama_get_memory(ctx), 1, 0, -1));
                GGML_ASSERT(llama_state_seq_set_data(ctx, neighbour_before.data(), neighbour_before.size(), 1) == neighbour_before.size());
                decode_span(ctx, 1, neighbour + 200, 200, 1);
                printf("  queued neighbour shift: deferred twice, applied after graph, control byte-exact\n");
            } else if (getenv("WALK_LIVE_GRAPH_NEIGHBOUR")) { GGML_ASSERT(graph_neighbour_turns > 1); }
            printf("  native graph continuation returns: %d; neighbour turns: %d\n", yielded, graph_neighbour_turns);
        }
        // Restore the fixture's original neighbour for its unchanged-custody assertions below.
        GGML_ASSERT(llama_state_seq_set_data(ctx, neighbour_before.data(), neighbour_before.size(), 1) == neighbour_before.size());
        const bool learned = progress == LLAMA_OPT_LEARN_DONE;
        if (getenv("WALK_LIVE_GRAPH_SEGMENTS")) {
            GGML_ASSERT(graph_segments > 0);
            printf("  synchronous graph-view segments observed: %zu (not serving yields)\n", graph_segments);
        }
        GGML_ASSERT(current_admission.statics > 0 && current_admission.workspace > 0);
        GGML_ASSERT(stale_admission.statics == 0 && stale_admission.workspace == 0);
        const std::vector<uint8_t> her_after = seq_state(ctx, 0);
        const std::vector<uint8_t> neighbour_after = seq_state(ctx, 1);
        if (getenv("WALK_LIVE_BRACKET_NEIGHBOURS")) {
            GGML_ASSERT(seq_state(ctx, 2) == want_second_neighbour);
        }
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
            GGML_ASSERT(llama_set_adapters_lora(ctx, &adapter, 1, &scale) == 0);
            GGML_ASSERT(llama_state_seq_set_data(ctx, generation_prefix.data(), generation_prefix.size(), 0) == generation_prefix.size());
            GGML_ASSERT(llama_opt_learn(ctx, 0, r0, tokens.data() + r0, labels.data(), WINDOW - r0, nullptr));
            const auto uninterrupted_next = adapter_params(adapter);
            GGML_ASSERT(llama_set_adapters_lora(ctx, nullptr, 0, nullptr) == 0);
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
            llama_batch quarantined_batch {}; // what llama_batch_get_one returned: one token, positions and sequence inferred
            quarantined_batch.n_tokens = 1;
            quarantined_batch.token = tokens.data();
            quarantined_batch.label = labels.data() + late; // a labelled row: the guard is on training decodes
            GGML_ASSERT(walk_decode(ctx, quarantined_batch) == -3);
            GGML_ASSERT(std::string(llama_opt_failure(ctx)).find("restore is uncertain") != std::string::npos);
            GGML_ASSERT(ctx->encode(quarantined_batch) == -3); // the context's own encode, as the deprecated llama_encode calls it
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
            GGML_ASSERT(llama_set_adapters_lora(ctx, &adapter, 1, &scale) == 0);
            GGML_ASSERT(llama_state_seq_set_data(ctx, generation_prefix.data(), generation_prefix.size(), 0) == generation_prefix.size());
            llama_batch retained = llama_opt_batch_init(WINDOW - r0);
            walk_batch_clear(retained);
            for (uint32_t i = 0; i < WINDOW - r0; ++i) {
                walk_batch_add(retained, tokens[r0 + i], r0 + i, {0}, true);
                retained.label[i] = labels[i];
            }
            const auto used_seq = seq_state(ctx, 0);
            const auto used_neighbour = seq_state(ctx, 1);
            GGML_ASSERT(!llama_opt_prepare_restore(ctx, retained));
            GGML_ASSERT(adapter_params(adapter) == saved);
            GGML_ASSERT(seq_state(ctx, 0) == used_seq && seq_state(ctx, 1) == used_neighbour);
            GGML_ASSERT(llama_opt_learn(ctx, 0, r0, tokens.data() + r0, labels.data(), WINDOW - r0, nullptr));
            const auto resumed_next = adapter_params(adapter);
            GGML_ASSERT(llama_set_adapters_lora(ctx, nullptr, 0, nullptr) == 0);
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
            // A genuinely fresh context must restore without a warmup update or cache insertion.
            llama_context * fresh = two_seq_ctx();
            GGML_ASSERT(llama_set_adapters_lora(fresh, &adapter, 1, &scale) == 0);
            llama_opt_init(fresh, model, lopt);
            const auto fresh_seq = seq_state(fresh, 0);
            const auto fresh_neighbour = seq_state(fresh, 1);
            GGML_ASSERT(llama_opt_prepare_restore(fresh, retained));
            GGML_ASSERT(adapter_params(adapter) == saved);
            GGML_ASSERT(seq_state(fresh, 0) == fresh_seq && seq_state(fresh, 1) == fresh_neighbour);
            FILE * uninitialized = tmpfile();
            GGML_ASSERT(uninitialized && !llama_opt_state_write(fresh, uninitialized, "fixture-generation"));
            fclose(uninitialized);
            llama_opt_batch_free(retained);
            GGML_ASSERT(llama_set_adapters_lora(fresh, nullptr, 0, nullptr) == 0);
            adapter->alora_invocation_tokens = {tokens[0], tokens[1]};
            rewind(weights); rewind(optimizer);
            GGML_ASSERT(llama_opt_state_restore(fresh, adapter, weights, optimizer, "fixture-generation"));
            adapter->alora_invocation_tokens.clear();
            GGML_ASSERT(llama_set_adapters_lora(fresh, &adapter, 1, &scale) == 0);
            decode_span(fresh, 0, tokens.data(), 0, r0);
            GGML_ASSERT(llama_opt_learn(fresh, 0, r0, tokens.data() + r0, labels.data(), WINDOW - r0, nullptr));
            const auto cold_next = adapter_params(adapter);
            for (size_t i = 0; i < saved.size(); ++i) { resumed_step[i] = cold_next[i] - saved[i]; }
            GGML_ASSERT(same_step(compare("fresh context restored next step without warmup", resumed_step, uninterrupted_step)));
            GGML_ASSERT(llama_set_adapters_lora(fresh, nullptr, 0, nullptr) == 0);
            adapter->alora_invocation_tokens = {tokens[0], tokens[1]};
            rewind(weights); rewind(optimizer);
            GGML_ASSERT(llama_opt_state_restore(fresh, adapter, weights, optimizer, "fixture-generation"));
            adapter->alora_invocation_tokens.clear();
            llama_free(fresh);
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
        const auto & want_live = getenv("WALK_LIVE_PINNED") ? want_pinned : (streams ? want_f16 : want_f32);
        if (!learned || !same_step(compare("live sequence: her reply learned in place vs the walk's exact step", got, want_live))) {
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
