// Continuum: each mind's decode graph kept across switches (src/llama-graph-slots.h).
//
// Four minds share one context, each at her own depth, three different adapter sets among them, and
// decode one token each in turn, as the lane serves residents. The same teacher-forced steps run twice, on two contexts:
// with no slots (every switch rebuilds, re-splits and re-allocates) and with one slot per mind. What this catches:
//   - a kept graph that serves a mind's step with another mind's K/V, adapter set or shape: the logits must be
//     bit-identical to the rebuilt graph's at every step;
//   - a slot that does not keep its graph: with slots, every step after a mind's first reuses its graph.
// It prints ms/step for both, the graphs reused, and the bytes the slots hold.

#include "arg.h"
#include "common.h"
#include "llama.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static constexpr uint32_t N_MINDS = 4;
static constexpr uint32_t N_GEN   = 48;

struct arm_result {
    std::vector<std::vector<float>> logits; // [step][n_vocab]
    double ms_per_step = 0.0;
    int32_t reused = 0;
    size_t slot_bytes = 0;
    bool attached = false;
};

static uint64_t persona_id(uint32_t m) { return 0xA000 + m; }

// One stream is the Core's shape: one mind is on the context at a time, bound into its one sequence by her blocks
// (llama_persona_bind) and saved out after her step, and each is named to the slots by her persona id. Many streams:
// each mind is her own sequence and the slots know her by it.
static arm_result run_arm(const common_params & params, llama_model * model, const std::vector<llama_token> & tokens,
                          llama_adapter_lora * a, llama_adapter_lora * b, uint32_t slots, bool one_stream) {
    auto cparams = common_context_params_to_llama(params);
    cparams.n_seq_max       = one_stream ? 1 : N_MINDS;
    cparams.n_ctx           = (one_stream ? 1 : N_MINDS) * 2048;
    cparams.n_batch         = 2048;
    cparams.n_ubatch        = 2048;
    cparams.kv_unified      = false;
    cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cparams.type_k          = GGML_TYPE_Q4_0;
    cparams.type_v          = GGML_TYPE_Q4_0;
    cparams.no_perf         = false;
    llama_context * ctx = llama_init_from_model(model, cparams);
    GGML_ASSERT(ctx != nullptr);
    llama_set_graph_slots(ctx, slots);
    const auto capture = llama_set_routing_capture(ctx, true); // as serving runs: each graph also records its routing
    GGML_ASSERT(capture == LLAMA_ROUTING_CAPTURE_RECORDING || capture == LLAMA_ROUTING_CAPTURE_NO_EXPERTS);

    // mind 0 and 2 wear a, mind 1 wears b, mind 3 serves the base
    float one = 1.0f;
    llama_adapter_lora * genes[N_MINDS] = { a, b, a, nullptr };
    auto wear = [&](uint32_t m) {
        GGML_ASSERT(llama_set_adapters_lora(ctx, genes[m] ? &genes[m] : nullptr, genes[m] ? 1 : 0, &one) == 0);
    };
    llama_persona * persona[N_MINDS] = {};
    // her storage is memory this test owns (the block layer's role: her mapped file), attached once, so binding her
    // points the stream at it and saving her points it away: the Core's move on a device that views host memory
    const size_t storage = one_stream ? llama_persona_storage_bytes(ctx) : 0;
    const bool   attaches = storage > 0 && llama_persona_storage_viewable(ctx);
    const size_t page = 1u << 16;
    std::vector<std::vector<uint8_t>> memory(attaches ? N_MINDS : 0, std::vector<uint8_t>(storage + 2 * page));
    auto aligned = [&](std::vector<uint8_t> & m) { return reinterpret_cast<void *>((reinterpret_cast<uintptr_t>(m.data()) + page - 1) & ~(uintptr_t) (page - 1)); };
    auto seq_of = [&](uint32_t m) { return one_stream ? (llama_seq_id) 0 : (llama_seq_id) m; };
    auto bind = [&](uint32_t m) {
        if (one_stream) {
            GGML_ASSERT(llama_persona_bind(ctx, 0, persona[m]) == 0);
            llama_set_graph_slot_key(ctx, 0, persona_id(m));
        }
    };
    auto park = [&](uint32_t m) {
        if (one_stream) {
            GGML_ASSERT(llama_persona_save(ctx, 0, persona[m]) == 0);
        }
    };

    llama_batch_ext * batch = llama_batch_ext_init(ctx);
    GGML_ASSERT(batch != nullptr);
    auto put = [&](llama_token t, llama_pos p, llama_seq_id s, bool out) {
        const int32_t i = llama_batch_ext_add_token(batch, s, t);
        GGML_ASSERT(i >= 0 && llama_batch_ext_set_pos(batch, i, &p) && llama_batch_ext_set_output_logits(batch, i, out));
    };
    auto decode = [&]() {
        GGML_ASSERT(llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch) == 0);
        llama_batch_ext_clear(batch);
    };
    llama_pos pos[N_MINDS];
    for (uint32_t m = 0; m < N_MINDS; ++m) { // each at its own depth, so each has its own n_kv
        const uint32_t n = 260 + 256 * m; // its N_GEN steps stay inside one 256-cell block: n_kv never changes
        if (one_stream) {
            persona[m] = llama_persona_init(ctx, persona_id(m), 1);
            GGML_ASSERT(persona[m] != nullptr);
            if (attaches) { // bound fresh before her first token, she decodes straight into her storage
                GGML_ASSERT(llama_persona_storage_attach(ctx, persona[m], aligned(memory[m]), storage + page) == 0);
                GGML_ASSERT(llama_persona_bind(ctx, 0, persona[m]) == 0);
            }
            llama_set_graph_slot_key(ctx, 0, persona_id(m));
        }
        wear(m);
        for (uint32_t i = 0; i < n; ++i) {
            put(tokens[(i + 97 * m) % tokens.size()], (llama_pos) i, seq_of(m), i + 1 == n);
        }
        decode();
        (void) llama_get_logits_ith(ctx, -1);
        park(m);
        pos[m] = (llama_pos) n;
    }
    llama_perf_context_reset(ctx);

    arm_result r;
    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const auto t0 = std::chrono::steady_clock::now();
    for (uint32_t g = 0; g < N_GEN; ++g) {
        for (uint32_t m = 0; m < N_MINDS; ++m) {
            bind(m);
            wear(m);
            put(tokens[(size_t) (pos[m] + 31 * m) % tokens.size()], pos[m], seq_of(m), true);
            decode();
            const float * l = llama_get_logits_ith(ctx, -1);
            r.logits.emplace_back(l, l + n_vocab);
            park(m);
            ++pos[m];
        }
    }
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    r.ms_per_step = ms / (N_GEN * N_MINDS);
    r.reused      = llama_perf_context(ctx).n_reused;
    r.slot_bytes  = llama_graph_slots_bytes(ctx);
    r.attached    = attaches;
    llama_batch_ext_free(batch);
    for (auto * p : persona) {
        llama_persona_free(p);
    }
    llama_free(ctx);
    return r;
}

int main(int argc, char ** argv) {
    common_params params;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }
    llama_backend_init();
    llama_model * model = llama_model_load_from_file(params.model.path.c_str(), common_model_params_to_llama(params));
    GGML_ASSERT(model != nullptr);

    const std::string init = "test-graph-slots." + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".gguf";
    GGML_ASSERT(common_lora_write_fresh(model, init, 8, 16, { "attn_q", "attn_v" }, 42));
    llama_adapter_lora * a = llama_adapter_lora_init(model, init.c_str());
    llama_adapter_lora * b = llama_adapter_lora_init(model, init.c_str());
    GGML_ASSERT(a != nullptr && b != nullptr);
    std::remove(init.c_str());

    std::string text;
    uint32_t lcg = 12345;
    static const char * words[] = { "room", "card", "claim", "review", "patch", "lease", "board", "turn" };
    while (text.size() < 16384) {
        lcg = lcg * 1664525u + 1013904223u;
        text += words[(lcg >> 16) % 8];
        text += (lcg >> 8) % 7 == 0 ? ".\n" : " ";
    }
    const std::vector<llama_token> tokens = common_tokenize(llama_model_get_vocab(model), text, true, false);
    GGML_ASSERT(tokens.size() > 2048);

    bool ok = true;
    for (const bool one_stream : { false, true }) {
        const arm_result rebuilt = run_arm(params, model, tokens, a, b, 0, one_stream);
        const arm_result kept    = run_arm(params, model, tokens, a, b, N_MINDS, one_stream);
        float max_diff = 0.0f;
        for (size_t s = 0; s < rebuilt.logits.size(); ++s) {
            for (size_t i = 0; i < rebuilt.logits[s].size(); ++i) {
                max_diff = std::max(max_diff, std::fabs(rebuilt.logits[s][i] - kept.logits[s][i]));
            }
        }
        const int32_t steps = (int32_t) (N_GEN * N_MINDS);
        printf("graph slots, %s: %u minds, %d steps: rebuilt %.2f ms/step (%d reused), kept %.2f ms/step (%d reused), "
               "slots hold %.1f MiB, max |logit diff| %g\n", one_stream ? (kept.attached ? "one stream, attached storage (the Core)" : "one stream, row copies") : "a stream each",
               N_MINDS, steps, rebuilt.ms_per_step, rebuilt.reused, kept.ms_per_step, kept.reused,
               kept.slot_bytes / 1024.0 / 1024.0, max_diff);
        if (max_diff != 0.0f) { printf("FAIL: a kept graph served different logits\n"); ok = false; }
        if (kept.reused < steps - (int32_t) N_MINDS) { printf("FAIL: a slot rebuilt a mind's graph it kept\n"); ok = false; }
    }

    llama_adapter_lora_free(a);
    llama_adapter_lora_free(b);
    llama_model_free(model);
    llama_backend_free();
    return ok ? 0 : 1;
}
