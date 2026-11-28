// Continuum: an adapter set per sequence (src/llama-adapter-seq.h).
//
// Four minds share one context, each in her own sequence at her own depth, each wearing her own adapters (two distinct
// nonzero adapters, both at once, and the base). The same teacher-forced tokens are decoded several ways: the
// REFERENCE steps one mind at a time under upstream's context-wide set (llama_set_adapters_lora), in both orders of a
// two-adapter set (upstream adds them in its hash map's order, and on a hybrid one ulp at a step moved the next step's
// logits by 0.19 on the CPU); the BASE does the same with no adapter, and the SHAPE batches it, measuring the kernels'
// own difference between one row and four; the BATCH steps all four in one decode, each naming her set on her
// sequence. What this catches:
//   - a delta added to another mind's rows, or a mind's rows given another set: each batched mind's logits must match
//     her nearer reference within the kernels' own difference (bit-exact on the CPU), and sit ten times closer to it
//     than the smallest adapter's effect (her reference against the base);
//   - rows reordered with their sequence ids kept (Codex's case), a set rebound under another scale on a warm graph,
//     and the base after an adapted owner: each phase below changes one of these and is held to the same bound;
//   - interleaved groups (one set on rows 0 and 2): gathered and written back by index, counted, and held to the bound;
//   - steps of two tokens with logits at the end: past inp_out_ids a tensor holds only the output rows, and each
//     group adds to its own among them (Codex, #172);
//   - an adapter nobody wears loaded and freed mid-phase: nothing named changes, every later step reuses its graph
//     (Codex, #172: a free anywhere on the model refused every named set);
//   - a token shared by sequences wearing different sets, and a set whose adapter was freed: refused.
// The adapters are fresh ones with a nonzero B (a fresh B is zero: its delta would be nothing to test).

#include "arg.h"
#include "common.h"
#include "llama.h"
#include "gguf.h"
#include "../src/llama-ext.h" // llama_get_memory_breakdown: the compute buffers each arm holds

#include <algorithm>
#include <chrono>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

static constexpr uint32_t N_MINDS = 4;
static constexpr uint32_t PHASE   = 6;  // steps per phase
static constexpr uint32_t PHASES  = 8;

using set_t = std::vector<std::pair<llama_adapter_lora *, float>>;

// the phases: what each mind wears (wears below), the order of her rows in the batch, and how many tokens each mind
// steps (logits only at her last: past inp_out_ids a tensor holds only those rows, a subset of the ubatch's)
struct phase_t {
    const char * what;
    uint32_t     order[N_MINDS];
    uint32_t     tokens;
};
static const phase_t phases[PHASES] = {
    { "each her own set, rows in order",                 { 0, 1, 2, 3 }, 1 },
    { "rows swapped, sequence ids kept",                 { 1, 0, 3, 2 }, 1 },
    { "mind 0 rebound at another scale (warm graph)",    { 0, 1, 2, 3 }, 1 },
    { "mind 1 on the base after her adapter",            { 0, 1, 2, 3 }, 1 },
    { "minds 0 and 2 share a set on rows 0 and 2",       { 0, 1, 2, 3 }, 1 },
    { "the same sets with their rows consecutive",       { 0, 2, 1, 3 }, 1 },
    { "two tokens each, logits at the end, interleaved", { 0, 1, 2, 3 }, 2 },
    { "two tokens each, logits at the end, consecutive", { 0, 2, 1, 3 }, 2 },
};
static constexpr uint32_t FREES_AN_UNNAMED_ADAPTER = 2; // the phase that loads and frees an adapter nobody wears, mid-phase

static set_t wears(uint32_t phase, uint32_t m, llama_adapter_lora * a, llama_adapter_lora * b) {
    const set_t sets[N_MINDS] = { { { a, 1.0f } }, { { b, 1.0f } }, { { a, 1.0f }, { b, 0.75f } }, {} };
    set_t set = sets[m];
    if (m == 0 && phase >= 2) { set = { { a, 0.5f } }; }
    if (m == 1 && phase >= 3) { set = {}; }
    if (m == 2 && phase >= 4) { set = { { a, 0.5f } }; }
    return set;
}

static bool ends_with(const std::string & s, const std::string & tail) {
    return s.size() >= tail.size() && s.compare(s.size() - tail.size(), tail.size(), tail) == 0;
}

static void fill(ggml_tensor * t, std::mt19937 & rng, float bound) {
    GGML_ASSERT(t->type == GGML_TYPE_F32);
    std::uniform_real_distribution<float> u(-bound, bound);
    float * v = (float *) t->data;
    for (int64_t j = 0; j < ggml_nelements(t); ++j) {
        v[j] = u(rng);
    }
}

// a fresh adapter with B drawn as A is, so its delta is nonzero; on a mixture of experts it also adapts every expert
// weight ([n_in, n_out, n_expert]: A [n_in, rank, n_expert], B [rank, n_out, n_expert]), so build_lora_mm_id is tested
static llama_adapter_lora * adapter(llama_model * model, const std::string & model_path, uint32_t seed) {
    constexpr int64_t rank = 8;
    const std::string path = "test-seq-adapters." + std::to_string(seed) + "." +
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".gguf";
    GGML_ASSERT(common_lora_write_fresh(model, path, (int32_t) rank, 16, { "attn_q", "attn_v", "ffn_up", "ffn_down" }, seed));
    ggml_context * data = nullptr;
    gguf_context * in = gguf_init_from_file(path.c_str(), { /*.no_alloc =*/ false, /*.ctx =*/ &data });
    GGML_ASSERT(in != nullptr && data != nullptr);
    gguf_context * out = gguf_init_empty();
    gguf_set_kv(out, in);
    std::mt19937 rng(seed);
    for (int64_t i = 0; i < gguf_get_n_tensors(in); ++i) {
        ggml_tensor * t = ggml_get_tensor(data, gguf_get_tensor_name(in, i));
        if (ends_with(ggml_get_name(t), ".lora_b")) {
            fill(t, rng, 0.02f);
        }
        gguf_add_tensor(out, t);
    }

    ggml_context * shapes = nullptr;
    gguf_context * base = gguf_init_from_file(model_path.c_str(), { /*.no_alloc =*/ true, /*.ctx =*/ &shapes });
    GGML_ASSERT(base != nullptr && shapes != nullptr);
    std::vector<std::pair<const ggml_tensor *, std::string>> experts;
    for (int64_t i = 0; i < gguf_get_n_tensors(base); ++i) {
        const std::string name = gguf_get_tensor_name(base, i);
        if (ends_with(name, "_exps.weight")) {
            experts.emplace_back(ggml_get_tensor(shapes, name.c_str()), name);
        }
    }
    ggml_context * extra = ggml_init({ /*.mem_size =*/ experts.size() * 2 * ggml_tensor_overhead(), /*.mem_buffer =*/ nullptr, /*.no_alloc =*/ true });
    std::vector<std::vector<float>> owned;
    owned.reserve(experts.size() * 2);
    for (const auto & [w, name] : experts) {
        ggml_tensor * a = ggml_new_tensor_3d(extra, GGML_TYPE_F32, w->ne[0], rank, w->ne[2]);
        ggml_tensor * b = ggml_new_tensor_3d(extra, GGML_TYPE_F32, rank, w->ne[1], w->ne[2]);
        ggml_set_name(a, (name + ".lora_a").c_str());
        ggml_set_name(b, (name + ".lora_b").c_str());
        for (ggml_tensor * t : { a, b }) {
            owned.emplace_back(ggml_nelements(t));
            t->data = owned.back().data();
            fill(t, rng, t == a ? 1.0f / std::sqrt((float) w->ne[0]) : 0.02f);
            gguf_add_tensor(out, t);
        }
    }
    GGML_ASSERT(gguf_write_to_file(out, path.c_str(), false));
    printf("seq-adapters: adapter %u adapts %lld expert weights\n", seed, (long long) experts.size());
    gguf_free(base);
    ggml_free(shapes);
    ggml_free(extra);
    gguf_free(out);
    gguf_free(in);
    ggml_free(data);
    llama_adapter_lora * loaded = llama_adapter_lora_init(model, path.c_str());
    GGML_ASSERT(loaded != nullptr);
    std::remove(path.c_str());
    return loaded;
}

// reference_reversed: upstream adds a set's adapters in its map's order, so a two-adapter set's reference is taken in
// both orders; base_batch: the batch with no adapter, the kernels' own difference
enum class arm { reference, reference_reversed, base, batch, base_batch };

struct arm_result {
    std::vector<std::vector<float>> logits; // [step * N_MINDS + mind][n_vocab]
    uint64_t gathered[PHASES] = {};
    bool refused_shared = false;
    bool refused_freed  = false;
    bool others_decode  = false; // a parked sequence's freed gene refuses only her
    bool renamed_decodes = false;
    size_t compute_bytes = 0; // the compute buffers the context holds after its steps
    int32_t reused[PHASES] = {}; // graphs reused in each phase after its first step
};

static arm_result run_arm(const common_params & params, llama_model * model, const std::vector<llama_token> & tokens,
                          llama_adapter_lora * a, llama_adapter_lora * b, arm kind, bool unified) {
    auto cparams = common_context_params_to_llama(params);
    cparams.n_seq_max  = N_MINDS;
    cparams.n_ctx      = N_MINDS * 1024;
    cparams.n_batch    = 1024;
    cparams.n_ubatch   = 1024;
    cparams.kv_unified = unified;
    cparams.no_perf    = false; // the graphs reused are read back
    llama_context * ctx = llama_init_from_model(model, cparams);
    GGML_ASSERT(ctx != nullptr);
    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    auto context_set = [&](const set_t & set) {
        std::vector<llama_adapter_lora *> adapters;
        std::vector<float> scales;
        for (const auto & [ad, sc] : set) { adapters.push_back(ad); scales.push_back(sc); }
        if (kind == arm::reference_reversed) {
            std::reverse(adapters.begin(), adapters.end());
            std::reverse(scales.begin(), scales.end());
        }
        GGML_ASSERT(llama_set_adapters_lora(ctx, adapters.data(), adapters.size(), scales.data()) == 0);
    };
    auto seq_set = [&](llama_seq_id seq, const set_t & set) {
        std::vector<llama_adapter_lora *> adapters;
        std::vector<float> scales;
        for (const auto & [ad, sc] : set) { adapters.push_back(ad); scales.push_back(sc); }
        GGML_ASSERT(llama_set_seq_adapters_lora(ctx, seq, adapters.data(), adapters.size(), scales.data()) == 0);
    };

    llama_batch_ext * batch = llama_batch_ext_init(ctx);
    auto put = [&](llama_token t, llama_pos p, llama_seq_id s, bool out) {
        const int32_t i = llama_batch_ext_add_token(batch, s, t);
        GGML_ASSERT(i >= 0 && llama_batch_ext_set_pos(batch, i, &p) && llama_batch_ext_set_output_logits(batch, i, out));
        return i;
    };
    auto decode = [&]() {
        const int32_t ret = llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch);
        llama_batch_ext_clear(batch);
        return ret;
    };

    // each mind prefilled at her own depth under her first set: the batch arm's named on her sequence first, as a bind
    // names it, so her prefill takes the same path as her steps (a whole ubatch in one group)
    llama_pos pos[N_MINDS];
    for (uint32_t m = 0; m < N_MINDS; ++m) {
        if (kind == arm::batch) {
            seq_set((llama_seq_id) m, wears(0, m, a, b));
        } else {
            context_set(kind == arm::base || kind == arm::base_batch ? set_t{} : wears(0, m, a, b));
        }
        const uint32_t n = 40 + 23 * m;
        for (uint32_t i = 0; i < n; ++i) {
            put(tokens[(i + 97 * m) % tokens.size()], (llama_pos) i, (llama_seq_id) m, i + 1 == n);
        }
        GGML_ASSERT(decode() == 0);
        pos[m] = (llama_pos) n;
    }
    // the batch arm's minds name their own sets, and the context wears b: a mind named to the base that fell back to the
    // context's set would wear b
    context_set(kind == arm::batch ? set_t{ { b, 1.0f } } : set_t{});

    arm_result r;
    r.logits.resize(PHASES * PHASE * N_MINDS);
    for (uint32_t ph = 0; ph < PHASES; ++ph) {
        if (kind == arm::batch) {
            for (uint32_t m = 0; m < N_MINDS; ++m) {
                seq_set((llama_seq_id) m, wears(ph, m, a, b)); // named once per phase, as a bind names it
            }
        }
        const uint64_t gathered0 = llama_seq_adapters_gathered(ctx);
        int32_t reused0 = 0;
        const uint32_t n = phases[ph].tokens;
        for (uint32_t g = 0; g < PHASE; ++g) {
            const uint32_t step = ph * PHASE + g;
            // mind m's tokens, logits at her last only
            auto steps = [&](uint32_t m) {
                for (uint32_t t = 0; t < n; ++t) {
                    put(tokens[(size_t) (pos[m] + t + 31 * m) % tokens.size()], pos[m] + (llama_pos) t, (llama_seq_id) m, t + 1 == n);
                }
            };
            if (kind == arm::batch || kind == arm::base_batch) {
                for (uint32_t row = 0; row < N_MINDS; ++row) {
                    steps(phases[ph].order[row]);
                }
                GGML_ASSERT(decode() == 0);
                for (uint32_t row = 0; row < N_MINDS; ++row) {
                    const float * l = llama_get_logits_ith(ctx, (int32_t) (row * n + n - 1));
                    r.logits[step * N_MINDS + phases[ph].order[row]].assign(l, l + n_vocab);
                }
            } else {
                for (uint32_t m = 0; m < N_MINDS; ++m) {
                    context_set(kind == arm::base ? set_t{} : wears(ph, m, a, b));
                    steps(m);
                    GGML_ASSERT(decode() == 0);
                    const float * l = llama_get_logits_ith(ctx, -1);
                    r.logits[step * N_MINDS + m].assign(l, l + n_vocab);
                }
            }
            for (uint32_t m = 0; m < N_MINDS; ++m) {
                pos[m] += (llama_pos) n;
            }
            if (g == 0) {
                reused0 = llama_perf_context(ctx).n_reused; // the phase's first step builds what its sets need
            }
            if (ph == FREES_AN_UNNAMED_ADAPTER && g == 2) {
                // an adapter nobody wears comes and goes (genes page in and out): nothing named changes, so every later
                // step decodes and reuses its graph as before
                llama_adapter_lora_free(adapter(model, params.model.path, 99));
            }
        }
        r.gathered[ph] = llama_seq_adapters_gathered(ctx) - gathered0;
        r.reused[ph]   = llama_perf_context(ctx).n_reused - reused0;
    }

    if (kind == arm::batch) {
        // minds 0 and 1 wear different sets (phase 5: a at 0.5, and the base): one token under both is refused
        const int32_t i = put(tokens[0], std::max(pos[0], pos[1]), 0, true);
        GGML_ASSERT(llama_batch_ext_add_seq(batch, i, 1));
        r.refused_shared = decode() != 0;
        // an adapter a sequence names is freed: her decode is refused until her set is named again
        llama_adapter_lora * gone = adapter(model, params.model.path, 77);
        seq_set(3, { { gone, 1.0f } });
        llama_adapter_lora_free(gone);
        // mind 3 is parked with it: mind 0 decodes alone (Codex, #172), mind 3 is refused until named again
        put(tokens[0], pos[0], 0, true);
        r.others_decode = decode() == 0;
        put(tokens[0], pos[3], 3, true);
        r.refused_freed = decode() != 0;
        seq_set(3, {});
        put(tokens[0], pos[3], 3, true);
        r.renamed_decodes = decode() == 0;
    }
    for (const auto & [type, held] : llama_get_memory_breakdown(ctx)) {
        r.compute_bytes += held.compute;
    }
    llama_batch_ext_free(batch);
    llama_free(ctx);
    return r;
}

static float max_diff(const std::vector<float> & x, const std::vector<float> & y) {
    float d = 0.0f;
    for (size_t i = 0; i < x.size(); ++i) {
        d = std::max(d, std::fabs(x[i] - y[i]));
    }
    return d;
}

static size_t argmax(const std::vector<float> & x) {
    return (size_t) (std::max_element(x.begin(), x.end()) - x.begin());
}

int main(int argc, char ** argv) {
    common_params params;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }
    llama_backend_init();
    llama_model * model = llama_model_load_from_file(params.model.path.c_str(), common_model_params_to_llama(params));
    GGML_ASSERT(model != nullptr);
    llama_adapter_lora * a = adapter(model, params.model.path, 42);
    llama_adapter_lora * b = adapter(model, params.model.path, 7);

    std::string text;
    for (int i = 0; i < 64; ++i) {
        text += "Each mind keeps her own notes about the board, the card, the claim and the review. ";
    }
    const std::vector<llama_token> tokens = common_tokenize(llama_model_get_vocab(model), text, true, false);

    // the Core's layout (a stream per mind, kv_unified off) and upstream's default (one unified cache). Off, a ubatch
    // takes its sequences in consecutive increasing ids (llama_batch_allocr::split_equal, sequential), so a batch whose
    // rows are out of that order is split into several ubatches; unified, the swapped and interleaved rows below share
    // one ubatch
    bool ok = true;
    for (const bool unified : { false, true }) {
        printf("seq-adapters: kv_unified %s\n", unified ? "on" : "off");
        const arm_result ref   = run_arm(params, model, tokens, a, b, arm::reference, unified);
        const arm_result rev   = run_arm(params, model, tokens, a, b, arm::reference_reversed, unified);
        const arm_result base  = run_arm(params, model, tokens, a, b, arm::base, unified);
        const arm_result batch = run_arm(params, model, tokens, a, b, arm::batch, unified);
        const arm_result shape = run_arm(params, model, tokens, a, b, arm::base_batch, unified);

        for (uint32_t ph = 0; ph < PHASES; ++ph) {
            float worst = 0.0f, effect = INFINITY, kernels = 0.0f, orders = 0.0f, scale = 0.0f;
            uint32_t flips = 0;
            for (uint32_t g = 0; g < PHASE; ++g) {
                const uint32_t step = ph * PHASE + g;
                for (uint32_t m = 0; m < N_MINDS; ++m) {
                    const size_t k = step * N_MINDS + m;
                    // her nearer reference: upstream's sum over two adapters takes either order
                    const float d_ref = max_diff(batch.logits[k], ref.logits[k]);
                    const float d_rev = max_diff(batch.logits[k], rev.logits[k]);
                    worst   = std::max(worst, std::min(d_ref, d_rev));
                    orders  = std::max(orders, max_diff(ref.logits[k], rev.logits[k]));
                    kernels = std::max(kernels, max_diff(shape.logits[k], base.logits[k]));
                    for (const float v : ref.logits[k]) { scale = std::max(scale, std::fabs(v)); }
                    flips += argmax(batch.logits[k]) != argmax((d_ref <= d_rev ? ref : rev).logits[k]);
                    if (!wears(ph, m, a, b).empty()) {
                        effect = std::min(effect, max_diff(ref.logits[k], base.logits[k]));
                    }
                }
            }
            // within the kernels' own difference and one ulp at the logits' scale: exact on the CPU; on Metal the
            // adapters' matmuls add their own shapes' difference (measured 1.0-1.6x the base's, 2026-11-28: bounded at 2x)
            const bool exact   = worst <= 2.0f * kernels + FLT_EPSILON * scale;
            const bool passed  = exact && worst * 10.0f < effect;
            const bool counted = batch.gathered[ph] == (ph == 4 || ph == 6 ? PHASE : 0); // phases 4 and 6 interleave, in one ubatch either way
            // groups never cost a reuse: after its first step, a phase reuses exactly the graphs upstream's batch does
            const bool reuses  = batch.reused[ph] == shape.reused[ph];
            printf("seq-adapters:   phase %u (%s): max |batch - nearer reference| %.3e (kernels' own %.3e, upstream's two orders %.3e), "
                   "smallest adapter effect %.3e, argmax flips %u of %u, gathered %llu, graphs reused %d (upstream %d)%s\n",
                   ph, phases[ph].what, worst, kernels, orders, effect, flips, PHASE * N_MINDS,
                   (unsigned long long) batch.gathered[ph], batch.reused[ph], shape.reused[ph],
                   passed && counted && reuses ? "" : "  <-- FAILED");
            ok = ok && passed && counted && reuses;
        }
        printf("seq-adapters:   a token under two sequences wearing different sets is %s\n", batch.refused_shared ? "refused" : "DECODED  <-- FAILED");
        ok = ok && batch.refused_shared;
        printf("seq-adapters:   a decode under a named set whose adapter was freed is %s\n", batch.refused_freed ? "refused" : "DECODED  <-- FAILED");
        ok = ok && batch.refused_freed;
        printf("seq-adapters:   with a parked mind's gene freed, another mind %s, and she decodes once named again: %s\n", batch.others_decode ? "decodes" : "is REFUSED  <-- FAILED", batch.renamed_decodes ? "yes" : "NO  <-- FAILED");
        ok = ok && batch.others_decode && batch.renamed_decodes;
        printf("seq-adapters:   compute buffers: batch %.2f MiB, upstream's batch with no adapter %.2f MiB, one mind at a time under the context's set %.2f MiB\n",
               batch.compute_bytes / 1048576.0, shape.compute_bytes / 1048576.0, ref.compute_bytes / 1048576.0);
    }

    // the genome: many genes loaded on the model, a few worn. What a context holds follows the worn ones, so the batch
    // under the same two worn genes holds the same compute buffers with sixteen more loaded and worn by no one
    {
        const arm_result few = run_arm(params, model, tokens, a, b, arm::batch, false);
        std::vector<llama_adapter_lora *> genome;
        for (uint32_t g = 0; g < 16; ++g) {
            genome.push_back(adapter(model, params.model.path, 1000 + g));
        }
        const arm_result many = run_arm(params, model, tokens, a, b, arm::batch, false);
        const bool same = many.compute_bytes == few.compute_bytes;
        printf("seq-adapters: the genome: the batch's compute buffers with 2 genes loaded %.2f MiB, with 18 loaded (2 worn) %.2f MiB%s\n",
               few.compute_bytes / 1048576.0, many.compute_bytes / 1048576.0, same ? "" : "  <-- FAILED");
        ok = ok && same;
        for (llama_adapter_lora * gene : genome) {
            llama_adapter_lora_free(gene);
        }
    }

    llama_adapter_lora_free(a);
    llama_adapter_lora_free(b);
    llama_model_free(model);
    llama_backend_free();
    printf("seq-adapters: %s\n", ok ? "OK" : "FAILED");
    return ok ? 0 : 1;
}
