// what this catches: the penalties sampler walks the recent history, not the vocabulary
// (CONTINUUM-ENGINE 4.7.13). Without a model: on id-ordered (the default chain), shuffled and truncated
// candidate arrays at a real vocabulary size, the result equals the reference per-candidate loop
// exactly; and the per-token time of the fast path is printed against the reference.
#include "llama.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <unordered_map>
#include <vector>

static void require(bool ok, const char * what) {
    if (!ok) { std::fprintf(stderr, "FAILED: %s\n", what); std::exit(1); }
}

// the reference: upstream's loop, one hash lookup per candidate
static void reference(std::vector<llama_token_data> & cur, const std::unordered_map<llama_token, int> & counts,
                      float repeat, float freq, float present) {
    for (auto & d : cur) {
        auto it = counts.find(d.id);
        if (it == counts.end()) continue;
        const int count = it->second;
        if (d.logit <= 0) d.logit *= repeat; else d.logit /= repeat;
        d.logit -= float(count) * freq + float(count > 0) * present;
    }
}

int main() {
    const int32_t V = 248320; // Qwen3.5/3.6 vocabulary
    const int32_t last_n = 256;
    const float repeat = 1.1f, freq = 0.2f, present = 0.1f;
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> logit(-20.0f, 20.0f);
    std::uniform_int_distribution<llama_token> tok(0, V - 1);

    llama_sampler * smpl = llama_sampler_init_penalties(V, last_n, repeat, freq, present);
    std::unordered_map<llama_token, int> counts;
    std::vector<llama_token> history;
    for (int i = 0; i < 400; ++i) { // more than last_n: the window slides
        const llama_token t = i % 5 == 0 ? 42 : tok(rng); // some repeats
        llama_sampler_accept(smpl, t);
        history.push_back(t);
    }
    for (size_t i = history.size() - (size_t) last_n; i < history.size(); ++i) counts[history[i]]++;

    std::vector<llama_token_data> base(V);
    for (int32_t i = 0; i < V; ++i) base[(size_t) i] = { i, logit(rng), 0.0f };

    auto run = [&](std::vector<llama_token_data> cur) {
        llama_token_data_array arr = { cur.data(), cur.size(), -1, false };
        llama_sampler_apply(smpl, &arr);
        cur.resize(arr.size);
        return cur;
    };
    auto same = [](const std::vector<llama_token_data> & a, const std::vector<llama_token_data> & b) {
        if (a.size() != b.size()) return false;
        for (size_t i = 0; i < a.size(); ++i) if (a[i].id != b[i].id || a[i].logit != b[i].logit) return false;
        return true;
    };

    // id-ordered (the default chain: penalties first)
    { auto ref = base; reference(ref, counts, repeat, freq, present); require(same(run(base), ref), "id-ordered equals the reference"); }
    // shuffled (an earlier sampler reordered)
    { auto sh = base; std::shuffle(sh.begin(), sh.end(), rng); auto ref = sh; reference(ref, counts, repeat, freq, present); require(same(run(sh), ref), "shuffled equals the reference"); }
    // truncated (top-k ran first)
    { std::vector<llama_token_data> tr(base.begin(), base.begin() + 40); auto ref = tr; reference(ref, counts, repeat, freq, present); require(same(run(tr), ref), "truncated equals the reference"); }
    // a sorted head that is NOT aligned but large
    { auto rv = base; std::reverse(rv.begin(), rv.end()); auto ref = rv; reference(ref, counts, repeat, freq, present); require(same(run(rv), ref), "reversed equals the reference"); }

    // the per-token receipt
    const int reps = 200;
    auto time_us = [&](auto && f) {
        auto work = base;
        const auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < reps; ++r) { work = base; f(work); }
        auto copy_t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < reps; ++r) { work = base; }
        const auto t1 = std::chrono::steady_clock::now();
        // subtract the per-rep copy of the candidate array (both variants pay it in this harness)
        return (std::chrono::duration<double, std::micro>(copy_t0 - t0).count() - std::chrono::duration<double, std::micro>(t1 - copy_t0).count()) / reps;
    };
    const double ref_us  = time_us([&](std::vector<llama_token_data> & w) { reference(w, counts, repeat, freq, present); });
    const double fast_us = time_us([&](std::vector<llama_token_data> & w) { llama_token_data_array a = { w.data(), w.size(), -1, false }; llama_sampler_apply(smpl, &a); });
    std::printf("penalties per token, V = %d, %zu distinct recent tokens: reference %.1f us, history walk %.1f us\n",
                V, counts.size(), ref_us, fast_us);

    llama_sampler_free(smpl);
    std::printf("test-sampler-penalties: ok\n");
    return 0;
}
