// what this catches: sampling one step's sequences on the lanes (server-lanes.h, CONTINUUM-ENGINE 4.7.14)
// choosing a different token than sampling them one after another, a lane run that loses or repeats a
// job, a job exception that does not reach the caller; and it reports the step's sampling time serial vs
// laned on real logits (needs -m <model>; the receipt is the per-step microseconds it prints).
#include "../tools/server/server-lanes.h"

#include "arg.h"
#include "common.h"
#include "llama.h"
#include "sampling.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

static void require(bool condition, const char * message) {
    if (!condition) {
        std::fprintf(stderr, "FAILED: %s\n", message);
        std::exit(1);
    }
}

static double now_us() {
    return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// the lanes alone: every job once, any count, exceptions carried back
static void lanes_run_every_job_once() {
    server_lanes lanes(5);
    for (size_t n : { (size_t) 1, (size_t) 2, (size_t) 6, (size_t) 64, (size_t) 1000 }) {
        for (int rep = 0; rep < 200; ++rep) {
            std::vector<std::atomic<int>> hits(n);
            lanes.run(n, [&](size_t i) { hits[i].fetch_add(1, std::memory_order_relaxed); });
            for (auto & h : hits) {
                require(h.load() == 1, "every job runs exactly once");
            }
        }
    }
    bool caught = false;
    try {
        lanes.run(6, [](size_t i) { if (i == 3) { throw std::runtime_error("job 3"); } });
    } catch (const std::runtime_error & e) {
        caught = std::string(e.what()) == "job 3";
    }
    require(caught, "a job's exception reaches the caller");
    std::atomic<int> after{0};
    lanes.run(6, [&](size_t) { after++; });
    require(after.load() == 6, "the lanes run normally after a failed run");
}

int main(int argc, char ** argv) {
    lanes_run_every_job_once();

    common_params params;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }
    const int n_seq = 6;
    params.n_parallel = n_seq;
    params.kv_unified = true;
    common_init();
    llama_backend_init();
    auto init = common_init_from_params(params);
    llama_model *   model = init->model();
    llama_context * ctx   = init->context();
    require(model && ctx, "model loads");
    const llama_vocab * vocab = llama_model_get_vocab(model);

    // one decode of six sequences, one output row each: the shape of a serving step
    const char * topics[n_seq] = { "the ledger", "a merge queue", "a kv cache", "the grid", "a resident", "a checkpoint" };
    llama_batch batch = llama_batch_init(512, 0, n_seq);
    std::vector<int> out_idx(n_seq);
    for (int s = 0; s < n_seq; ++s) {
        const auto toks = common_tokenize(vocab, std::string("Write about ") + topics[s] + ".", true, true);
        for (size_t i = 0; i < toks.size(); ++i) {
            const bool last = i + 1 == toks.size();
            if (last) {
                out_idx[s] = batch.n_tokens;
            }
            common_batch_add(batch, toks[i], (llama_pos) i, { s }, last);
        }
    }
    require(llama_decode(ctx, batch) == 0, "decode");

    common_params_sampling sp;
    sp.top_k          = 40;
    sp.penalty_last_n = 256;
    sp.penalty_repeat = 1.1f;
    std::vector<common_sampler *> base(n_seq);
    for (int s = 0; s < n_seq; ++s) {
        base[s] = common_sampler_init(model, sp);
    }
    std::vector<common_sampler_row> rows(n_seq);
    for (int s = 0; s < n_seq; ++s) {
        rows[s] = common_sampler_row_ith(ctx, out_idx[s]);
    }

    server_lanes lanes(n_seq - 1);
    const int reps = 400;
    double t_serial = 0, t_laned = 0;
    std::vector<llama_token> a(n_seq), b(n_seq);
    for (int rep = 0; rep < reps; ++rep) {
        std::vector<common_sampler *> s1(n_seq), s2(n_seq);
        for (int s = 0; s < n_seq; ++s) {
            s1[s] = common_sampler_clone(base[s]);
            s2[s] = common_sampler_clone(base[s]);
        }
        // alternate which goes first, so neither always meets a warm or cold cache
        auto serial = [&] {
            const double t0 = now_us();
            for (int s = 0; s < n_seq; ++s) {
                a[s] = common_sampler_sample_row(s1[s], rows[s]);
            }
            t_serial += now_us() - t0;
        };
        auto laned = [&] {
            const double t0 = now_us();
            lanes.run(n_seq, [&](size_t s) { b[s] = common_sampler_sample_row(s2[s], rows[s]); });
            t_laned += now_us() - t0;
        };
        if (rep % 2 == 0) { serial(); laned(); } else { laned(); serial(); }
        for (int s = 0; s < n_seq; ++s) {
            require(a[s] == b[s], "a laned sample picks exactly the serial token");
            common_sampler_free(s1[s]);
            common_sampler_free(s2[s]);
        }
    }
    std::printf("one step, %d sequences, %d reps: serial %.1f us/step, laned (%zu lanes) %.1f us/step, %.2fx\n",
            n_seq, reps, t_serial / reps, lanes.size(), t_laned / reps, t_serial / t_laned);

    for (auto * s : base) {
        common_sampler_free(s);
    }
    llama_batch_free(batch);
    std::printf("test-server-lanes: ok\n");
    return 0;
}
