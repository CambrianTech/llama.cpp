// what this catches: a training step cut into segments (the sched eval callback a trainer uses
// to yield to serving between segments, common/train-segment.h, continuum S4) must train EXACTLY
// as the uncut step: the same parameters, bit for bit, after several AdamW steps. A segment
// boundary that dropped, reordered or re-ran a node would move the weights. The cuts are counted,
// so a callback that never cut (a vacuous pass) fails too. The segmenter's own decisions (first
// budget, the budget learned from a measured rate, the target clamp) are pinned beside it.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-opt.h"
#include "train-segment.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CHECK(cond) do { if (!(cond)) { fprintf(stderr, "FAILED %s:%d: %s\n", __FILE__, __LINE__, #cond); exit(1); } } while (0)

static const int N = 16; // width
static const int B = 4;  // batch columns
static const int STEPS = 3;

enum class cut_mode { none, every_node, segmenter };

struct cut_state {
    cut_mode        mode;
    train_segmenter seg;
    int             cuts = 0;
};

static bool on_eval(ggml_tensor * t, bool ask, void * user_data) {
    auto & s = *static_cast<cut_state *>(user_data);
    if (ask) {
        switch (s.mode) {
            case cut_mode::none:       return false;
            case cut_mode::every_node: return true;
            // a budget far below one node's cost: the segmenter cuts at (almost) every node
            case cut_mode::segmenter:  return s.seg.cut_after(train_node_cost(t), 1e-9);
        }
    }
    s.cuts += 1;
    s.seg.measured(1.0);
    return true; // continue: only a cancel stops a graph
}

static ggml_opt_optimizer_params adamw_pars(void *) {
    ggml_opt_optimizer_params p = ggml_opt_get_default_optimizer_params(nullptr);
    p.adamw.alpha = 0.01f;
    return p;
}

// three AdamW steps of a two-layer net on fixed data; the final weights, W1 then W2
static std::vector<float> train(cut_mode mode, int * cuts_out) {
    // by type, not ggml_backend_cpu_init: in a dynamically loaded backend build (CI) the CPU
    // backend is its own library
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    GGML_ASSERT(cpu != nullptr);
    ggml_backend_t backends[] = { cpu };
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, nullptr, 1, GGML_DEFAULT_GRAPH_SIZE, false, true);
    cut_state state{ mode, {}, 0 };
    state.seg.cost_per_ms = 1.0; // as if measured: with the target below, the budget is under one node
    if (mode != cut_mode::none) {
        ggml_backend_sched_set_eval_callback(sched, on_eval, &state);
    }

    ggml_init_params sp = { 8 * ggml_tensor_overhead(), nullptr, true };
    ggml_context * ctx_static = ggml_init(sp);
    ggml_tensor * w1 = ggml_new_tensor_2d(ctx_static, GGML_TYPE_F32, N, N);
    ggml_tensor * w2 = ggml_new_tensor_2d(ctx_static, GGML_TYPE_F32, N, N);
    ggml_set_param(w1);
    ggml_set_param(w2);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx_static, cpu);
    std::vector<float> init(N * N);
    for (int i = 0; i < N * N; ++i) {
        init[i] = 0.05f * std::sin(0.7f * (float) i);
    }
    ggml_backend_tensor_set(w1, init.data(), 0, sizeof(float) * init.size());
    for (int i = 0; i < N * N; ++i) {
        init[i] = 0.05f * std::cos(0.3f * (float) i);
    }
    ggml_backend_tensor_set(w2, init.data(), 0, sizeof(float) * init.size());

    ggml_opt_params params = ggml_opt_default_params(sched, GGML_OPT_LOSS_TYPE_SUM);
    params.optimizer    = GGML_OPT_OPTIMIZER_TYPE_ADAMW;
    params.get_opt_pars = adamw_pars;
    params.opt_period   = 1;
    ggml_opt_context_t opt_ctx = ggml_opt_init(params); // no ctx_compute: graphs are built per step
    ggml_opt_result_t  result  = ggml_opt_result_init();

    std::vector<float> xs(N * B);
    for (int i = 0; i < N * B; ++i) {
        xs[i] = std::sin(0.1f * (float) (i + 1));
    }
    for (int step = 0; step < STEPS; ++step) {
        ggml_init_params cp = { GGML_DEFAULT_GRAPH_SIZE * ggml_tensor_overhead() + 4 * ggml_graph_overhead_custom(GGML_DEFAULT_GRAPH_SIZE, true), nullptr, true };
        ggml_context * ctx_compute = ggml_init(cp);
        ggml_tensor * x   = ggml_new_tensor_2d(ctx_compute, GGML_TYPE_F32, N, B);
        ggml_tensor * h   = ggml_mul_mat(ctx_compute, w1, x);
        ggml_tensor * y   = ggml_mul_mat(ctx_compute, w2, ggml_silu(ctx_compute, h));
        ggml_tensor * out = ggml_sum(ctx_compute, ggml_sqr(ctx_compute, y));
        ggml_cgraph * gf  = ggml_new_graph_custom(ctx_compute, GGML_DEFAULT_GRAPH_SIZE, true);
        ggml_build_forward_expand(gf, out);
        ggml_opt_prepare_alloc(opt_ctx, ctx_compute, gf, x, out);
        CHECK(ggml_opt_alloc(opt_ctx, true));
        ggml_backend_tensor_set(x, xs.data(), 0, sizeof(float) * xs.size());
        state.seg.reset_plan();
        ggml_opt_eval(opt_ctx, result);
        ggml_free(ctx_compute);
    }

    std::vector<float> weights(2 * N * N);
    ggml_backend_tensor_get(w1, weights.data(), 0, sizeof(float) * N * N);
    ggml_backend_tensor_get(w2, weights.data() + N * N, 0, sizeof(float) * N * N);
    *cuts_out = state.cuts;

    ggml_opt_result_free(result);
    ggml_opt_free(opt_ctx);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx_static);
    ggml_backend_sched_free(sched);
    ggml_backend_free(cpu);
    return weights;
}

static void segmenter_decisions() {
    // unmeasured: the first budget decides, and a cut resets the running cost
    train_segmenter s;
    CHECK(!s.cut_after(train_segmenter::FIRST_BUDGET / 2, 100.0));
    CHECK(s.cut_after(train_segmenter::FIRST_BUDGET / 2, 100.0));
    CHECK(s.planned == train_segmenter::FIRST_BUDGET && s.since_cut == 0);
    // measured: that segment took 10 ms, so 1e8 per ms, and a 100 ms target budgets 1e10
    s.measured(10.0);
    CHECK(std::fabs(s.cost_per_ms - 1e8) < 1.0);
    CHECK(!s.cut_after(9e9, 100.0));
    CHECK(s.cut_after(1e9, 100.0));
    // a new graph discards the uncut tail
    s.cut_after(5e8, 100.0);
    s.reset_plan();
    CHECK(s.since_cut == 0 && s.planned == 0);
    // a zero or negative duration teaches nothing
    const double r = s.cost_per_ms;
    s.measured(0.0);
    CHECK(s.cost_per_ms == r);

    // the target is one of her decode steps: 3 busy slots at 0.03 tokens/ms = 100 ms per step
    CHECK(std::fabs(train_segment_target_ms(0.03, 3) - 100.0) < 1e-9);
    CHECK(train_segment_target_ms(0.0, 3) == TRAIN_SEGMENT_DEFAULT_MS);  // unmeasured
    CHECK(train_segment_target_ms(10.0, 1) == TRAIN_SEGMENT_MIN_MS);     // a fast lane: no per-node cuts
    CHECK(train_segment_target_ms(0.001, 1) == TRAIN_SEGMENT_MAX_MS);    // a slow one: no multi-second holds
}

int main() {
    segmenter_decisions();

    int cuts_none = 0, cuts_every = 0, cuts_seg = 0;
    const std::vector<float> ref   = train(cut_mode::none, &cuts_none);
    const std::vector<float> every = train(cut_mode::every_node, &cuts_every);
    const std::vector<float> seg   = train(cut_mode::segmenter, &cuts_seg);
    printf("cuts: none %d, every node %d, segmenter %d\n", cuts_none, cuts_every, cuts_seg);
    CHECK(cuts_none == 0);
    CHECK(cuts_every > STEPS * 4); // forward, backward and the optimizer, cut throughout
    CHECK(cuts_seg > STEPS * 4);
    CHECK(std::memcmp(ref.data(), every.data(), sizeof(float) * ref.size()) == 0);
    CHECK(std::memcmp(ref.data(), seg.data(), sizeof(float) * ref.size()) == 0);
    // and the weights moved, so equality is not two untrained nets agreeing
    std::vector<float> init(N * N);
    for (int i = 0; i < N * N; ++i) {
        init[i] = 0.05f * std::sin(0.7f * (float) i);
    }
    CHECK(std::memcmp(ref.data(), init.data(), sizeof(float) * init.size()) != 0);

    printf("test-opt-segmented: OK\n");
    return 0;
}
