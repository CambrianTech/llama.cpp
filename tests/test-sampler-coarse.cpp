// what this catches: the ballpark-first sampler (common/sampling.cpp, coarse_cap; CONTINUUM-ENGINE 4.7.13)
// picks EXACTLY the token the full-vocabulary path picks, on a real model's logits, step after step:
// two samplers with the same parameters and seed, one eligible for the ballpark path and one forced onto
// the full path (n_probs = 1 makes it ineligible), sample every step of a long generation with penalties
// active and the window sliding; any difference fails. Prints each path's per-token time (alternating
// which goes first) and the ballpark path's counters (cap, reseeds, widens, mean candidates).
//
// usage: test-sampler-coarse -m <model.gguf>
#include "arg.h"
#include "common.h"
#include "sampling.h"
#include "llama.h"

#include <chrono>
#include <clocale>
#include <cstdio>
#include <vector>

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    common_params params;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }
    common_init();
    llama_backend_init();
    auto init = common_init_from_params(params);
    llama_model * model = init->model();
    llama_context * ctx = init->context();
    if (model == nullptr || ctx == nullptr) { std::fprintf(stderr, "FAILED: model did not load\n"); return 1; }

    common_params_sampling sp;
    sp.temp = 0.8f; sp.top_k = 40; sp.top_p = 0.95f; sp.min_p = 0.05f;
    sp.penalty_last_n = 256; sp.penalty_repeat = 1.1f; sp.penalty_freq = 0.2f; sp.penalty_present = 0.1f;
    sp.seed = 1234;
    common_params_sampling sp_full = sp;
    sp_full.n_probs = 1; // ineligible for the ballpark path: the full-vocabulary reference
    common_sampler * coarse = common_sampler_init(model, sp);
    common_sampler * full   = common_sampler_init(model, sp_full);

    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::vector<llama_token> prompt = common_tokenize(vocab, "Write a long story about a lighthouse keeper who repeats himself.", true, true);
    llama_batch batch = llama_batch_init(512, 0, 1);
    for (size_t i = 0; i < prompt.size(); ++i) common_batch_add(batch, prompt[i], (llama_pos) i, { 0 }, i + 1 == prompt.size());
    if (llama_decode(ctx, batch) != 0) { std::fprintf(stderr, "FAILED: prompt decode\n"); return 1; }
    llama_pos pos = (llama_pos) prompt.size();

    const int steps = 600;
    double us_coarse = 0, us_full = 0;
    int done = 0;
    for (int s = 0; s < steps; ++s) {
        llama_synchronize(ctx);
        llama_token a, b;
        auto time = [&](common_sampler * smpl, double & acc) {
            auto t0 = std::chrono::steady_clock::now();
            const llama_token id = common_sampler_sample(smpl, ctx, -1);
            acc += std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
            return id;
        };
        if (s % 2 == 0) { a = time(coarse, us_coarse); b = time(full, us_full); }
        else            { b = time(full, us_full); a = time(coarse, us_coarse); }
        if (a != b) {
            std::fprintf(stderr, "FAILED: step %d: ballpark picked %d, the full vocabulary picked %d\n", s, a, b);
            return 1;
        }
        common_sampler_accept(coarse, a, true);
        common_sampler_accept(full, b, true);
        done++;
        if (llama_vocab_is_eog(vocab, a)) break;
        common_batch_clear(batch);
        common_batch_add(batch, a, pos++, { 0 }, true);
        if (llama_decode(ctx, batch) != 0) { std::fprintf(stderr, "FAILED: decode\n"); return 1; }
    }
    size_t cap = 0; uint64_t calls = 0, reseeds = 0, widens = 0, cands = 0;
    common_sampler_coarse_stats(coarse, &cap, &calls, &reseeds, &widens, &cands);
    std::printf("identical tokens over %d steps; sampling per token: ballpark %.1f us, full vocabulary %.1f us\n",
                done, us_coarse / done, us_full / done);
    std::printf("ballpark path: cap %zu, calls %llu, reseeds %llu, widens %llu, mean candidates %.0f\n",
                cap, (unsigned long long) calls, (unsigned long long) reseeds, (unsigned long long) widens,
                calls ? (double) cands / (double) calls : 0.0);
    common_sampler_free(coarse);
    common_sampler_free(full);
    llama_batch_free(batch);
    std::printf("test-sampler-coarse: ok\n");
    return 0;
}
