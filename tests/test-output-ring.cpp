// Continuum: a decode's outputs kept readable while the next decode is queued behind it (src/llama-output-ring.h).
//
// Two minds, each her own stream at her own depth, each with her graph kept (graph slots), decode one token each in
// turn, teacher-forced so both arms see the same inputs. Serial: decode A, read A, decode B, read B, as the Core
// serves today (each read waits for everything queued). Pipelined: B's step is queued before A's outputs are read, so
// A is read, and her next step encoded, while B computes. What this catches:
//   - a ticket that reads another decode's outputs (the ring's swap or its record misplaced): every step's logits
//     must be bit-identical to the serial arm's;
//   - a ticket read that waits for the decode queued after it (no overlap): it prints ms/step for both arms.

#include "arg.h"
#include "common.h"
#include "llama.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

static constexpr uint32_t N_MINDS = 2;
static constexpr uint32_t N_GEN   = 96;

// each mind at her own depth: OUTPUT_RING_DEPTH (default 1200) for the first, 700 more for each after
static uint32_t depth_of(uint32_t m) {
    const char * set = getenv("OUTPUT_RING_DEPTH");
    return (set ? (uint32_t) std::stoul(set) : 1200) + 700 * m;
}

struct arm_result {
    std::vector<std::vector<float>> logits; // [step][n_vocab], minds interleaved A, B, A, B, ...
    double ms_per_step = 0.0;
    double submit_ms_per_step = 0.0; // how long each decode call blocked the caller
};

static arm_result run_arm(const common_params & params, llama_model * model, const std::vector<llama_token> & tokens, bool pipelined) {
    auto cparams = common_context_params_to_llama(params);
    cparams.n_seq_max       = N_MINDS;
    cparams.n_ctx           = N_MINDS * (depth_of(N_MINDS - 1) + 4096);
    cparams.n_batch         = 2048;
    cparams.n_ubatch        = 2048;
    cparams.kv_unified      = false;
    cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cparams.type_k          = GGML_TYPE_Q4_0;
    cparams.type_v          = GGML_TYPE_Q4_0;
    llama_context * ctx = llama_init_from_model(model, cparams);
    GGML_ASSERT(ctx != nullptr);
    llama_set_graph_slots(ctx, N_MINDS);
    GGML_ASSERT(llama_set_output_ring(ctx, true));

    llama_batch_ext * batch = llama_batch_ext_init(ctx);
    GGML_ASSERT(batch != nullptr);
    auto put = [&](llama_token t, llama_pos p, llama_seq_id s, bool out) {
        const int32_t i = llama_batch_ext_add_token(batch, s, t);
        GGML_ASSERT(i >= 0 && llama_batch_ext_set_pos(batch, i, &p) && llama_batch_ext_set_output_logits(batch, i, out));
    };
    auto decode = [&]() {
        GGML_ASSERT(llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch) == 0);
        llama_batch_ext_clear(batch);
        return llama_output_ticket(ctx);
    };
    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    llama_pos pos[N_MINDS];
    for (uint32_t m = 0; m < N_MINDS; ++m) { // each at her own depth, prefilled a batch at a time
        const uint32_t n = depth_of(m);
        for (uint32_t from = 0; from < n; from += 2048) {
            const uint32_t to = std::min(n, from + 2048);
            for (uint32_t i = from; i < to; ++i) {
                put(tokens[(i + 97 * m) % tokens.size()], (llama_pos) i, (llama_seq_id) m, i + 1 == n);
            }
            const int64_t ticket = decode();
            if (to == n) {
                (void) llama_ticket_logits(ctx, ticket, 0);
            }
        }
        pos[m] = (llama_pos) n;
    }

    arm_result r;
    double submit_ms = 0.0;
    auto step = [&](uint32_t m) {
        put(tokens[(size_t) (pos[m] + 31 * m) % tokens.size()], pos[m], (llama_seq_id) m, true);
        ++pos[m];
        const auto t = std::chrono::steady_clock::now();
        const int64_t ticket = decode();
        submit_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
        return ticket;
    };
    auto keep = [&](int64_t ticket) {
        const float * l = llama_ticket_logits(ctx, ticket, 0);
        GGML_ASSERT(l != nullptr);
        r.logits.emplace_back(l, l + n_vocab);
    };
    const auto t0 = std::chrono::steady_clock::now();
    if (!pipelined) {
        for (uint32_t g = 0; g < N_GEN; ++g) {
            for (uint32_t m = 0; m < N_MINDS; ++m) {
                keep(step(m));
            }
        }
    } else {
        // one step always queued behind the one computing: B's goes in before A's is read, and A's next before B's
        int64_t held = step(0);
        for (uint32_t g = 0; g < N_GEN; ++g) {
            const int64_t b = step(1);
            keep(held);
            held = g + 1 < N_GEN ? step(0) : -1;
            keep(b);
        }
    }
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    r.ms_per_step = ms / (N_GEN * N_MINDS);
    r.submit_ms_per_step = submit_ms / (N_GEN * N_MINDS);
    llama_batch_ext_free(batch);
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

    std::string text;
    uint32_t lcg = 12345;
    static const char * words[] = { "room", "card", "claim", "review", "patch", "lease", "board", "turn" };
    while (text.size() < 32768) {
        lcg = lcg * 1664525u + 1013904223u;
        text += words[(lcg >> 16) % 8];
        text += (lcg >> 8) % 7 == 0 ? ".\n" : " ";
    }
    const std::vector<llama_token> tokens = common_tokenize(llama_model_get_vocab(model), text, true, false);
    GGML_ASSERT(tokens.size() > 4096); // the prompt wraps: depth is what is measured, not novelty

    const arm_result serial    = run_arm(params, model, tokens, false);
    const arm_result pipelined = run_arm(params, model, tokens, true);

    float max_diff = 0.0f;
    for (size_t s = 0; s < serial.logits.size(); ++s) {
        for (size_t i = 0; i < serial.logits[s].size(); ++i) {
            max_diff = std::max(max_diff, std::fabs(serial.logits[s][i] - pipelined.logits[s][i]));
        }
    }
    printf("output ring: %u minds at %u and %u, %u steps: serial %.2f ms/step (submit %.2f), pipelined %.2f ms/step (submit %.2f; %.0f%% of serial), max |logit diff| %g\n",
           N_MINDS, depth_of(0), depth_of(1), N_GEN * N_MINDS, serial.ms_per_step, serial.submit_ms_per_step, pipelined.ms_per_step, pipelined.submit_ms_per_step, 100.0 * pipelined.ms_per_step / serial.ms_per_step, max_diff);
    const bool ok = max_diff == 0.0f && serial.logits.size() == pipelined.logits.size();
    if (!ok) {
        printf("FAIL: a ticket read another decode's outputs\n");
    }
    llama_model_free(model);
    llama_backend_free();
    return ok ? 0 : 1;
}
