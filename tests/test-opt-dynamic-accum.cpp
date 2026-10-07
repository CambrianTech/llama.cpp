// what this catches: ggml-opt with graphs built per step (ggml_opt_prepare_alloc, the path
// llama_context's training takes) must start every optimizer period from ZERO gradient
// accumulators. Each step here is SGD on loss = sum(w * x), so its gradient is x and each
// step moves w by exactly -lr * x. If the accumulators carried the previous period's gradient,
// the second step would move w by -2 lr x, the third by -3 lr x.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-opt.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>

#define CHECK(cond) do { if (!(cond)) { fprintf(stderr, "FAILED %s:%d: %s\n", __FILE__, __LINE__, #cond); exit(1); } } while (0)

static const float LR = 0.125f;
static const float X  = 2.0f;

static ggml_opt_optimizer_params sgd_pars(void *) {
    ggml_opt_optimizer_params p = ggml_opt_get_default_optimizer_params(nullptr);
    p.sgd.alpha = LR;
    p.sgd.wd    = 0.0f;
    return p;
}

static void run(int32_t opt_period) {
    // by type, not ggml_backend_cpu_init: in a dynamically loaded backend build (CI) the CPU
    // backend is its own library
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    GGML_ASSERT(cpu != nullptr);
    ggml_backend_t backends[] = { cpu };
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, nullptr, 1, GGML_DEFAULT_GRAPH_SIZE, false, true);

    ggml_init_params sp = { 8 * ggml_tensor_overhead(), nullptr, true };
    ggml_context * ctx_static = ggml_init(sp);
    ggml_tensor * w = ggml_new_tensor_1d(ctx_static, GGML_TYPE_F32, 1);
    ggml_set_param(w);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx_static, cpu);
    float w_now = 1.0f;
    ggml_backend_tensor_set(w, &w_now, 0, sizeof(float));

    ggml_opt_params params = ggml_opt_default_params(sched, GGML_OPT_LOSS_TYPE_SUM);
    params.optimizer    = GGML_OPT_OPTIMIZER_TYPE_SGD;
    params.get_opt_pars = sgd_pars;
    params.opt_period   = opt_period;
    ggml_opt_context_t opt_ctx = ggml_opt_init(params); // no ctx_compute: graphs are built per step
    ggml_opt_result_t  result  = ggml_opt_result_init();

    const int n_evals = 3 * opt_period;
    for (int i = 0; i < n_evals; ++i) {
        ggml_init_params cp = { GGML_DEFAULT_GRAPH_SIZE * ggml_tensor_overhead() + 4 * ggml_graph_overhead_custom(GGML_DEFAULT_GRAPH_SIZE, true), nullptr, true };
        ggml_context * ctx_compute = ggml_init(cp);
        ggml_tensor * x   = ggml_new_tensor_1d(ctx_compute, GGML_TYPE_F32, 1);
        ggml_tensor * out = ggml_mul(ctx_compute, w, x);
        ggml_cgraph * gf  = ggml_new_graph_custom(ctx_compute, GGML_DEFAULT_GRAPH_SIZE, true);
        ggml_build_forward_expand(gf, out);
        ggml_opt_prepare_alloc(opt_ctx, ctx_compute, gf, x, out);
        CHECK(ggml_opt_alloc(opt_ctx, true));
        ggml_backend_tensor_set(x, &X, 0, sizeof(float));
        ggml_opt_eval(opt_ctx, result);
        ggml_free(ctx_compute);

        float w_after;
        ggml_backend_tensor_get(w, &w_after, 0, sizeof(float));
        if ((i + 1) % opt_period == 0) {
            // one period = opt_period evals of gradient X each, scaled by nothing (LOSS_TYPE_SUM)
            const float expected = w_now - LR * X * opt_period;
            if (std::fabs(w_after - expected) > 1e-6f) {
                fprintf(stderr, "opt_period %d, step %d: w moved to %f, expected %f (from %f)\n",
                        opt_period, (i + 1) / opt_period, w_after, expected, w_now);
                exit(1);
            }
            w_now = w_after;
        } else {
            CHECK(std::fabs(w_after - w_now) < 1e-7f && "no update inside a period");
        }
    }

    ggml_opt_result_free(result);
    ggml_opt_free(opt_ctx);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx_static);
    ggml_backend_sched_free(sched);
    ggml_backend_free(cpu);
}

// what this catches: ggml_opt_set_next_step, the training walk's one-step-per-window period.
// Three graphs on loss = sum(w * x): the first two only accumulate, at weights 0.5 and 0.25,
// the second with an extra term 3 * w * x; the third (weight 1) takes the step. So the step
// is SGD on (0.5 + 0.25 + 3 + 1) * x. Each graph's x is a GRAD leaf, whose gradient (the
// weight times w, plus 3 w in the second) must read back after its eval; a following period
// starts from zero.
static void run_manual() {
    // by type, not ggml_backend_cpu_init: in a dynamically loaded backend build (CI) the CPU
    // backend is its own library
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    GGML_ASSERT(cpu != nullptr);
    ggml_backend_t backends[] = { cpu };
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, nullptr, 1, GGML_DEFAULT_GRAPH_SIZE, false, true);

    ggml_init_params sp = { 8 * ggml_tensor_overhead(), nullptr, true };
    ggml_context * ctx_static = ggml_init(sp);
    ggml_tensor * w = ggml_new_tensor_1d(ctx_static, GGML_TYPE_F32, 1);
    ggml_set_param(w);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx_static, cpu);
    float w_now = 1.0f;
    ggml_backend_tensor_set(w, &w_now, 0, sizeof(float));

    ggml_opt_params params = ggml_opt_default_params(sched, GGML_OPT_LOSS_TYPE_CROSS_ENTROPY);
    params.optimizer    = GGML_OPT_OPTIMIZER_TYPE_SGD;
    params.get_opt_pars = sgd_pars;
    ggml_opt_context_t opt_ctx = ggml_opt_init(params);

    // cross-entropy over 2 classes of logits [w x, 0] with label class 0:
    // dCE/dlogit0 = softmax0 - 1, so d/dw = (s0 - 1) x and d/dx = (s0 - 1) w
    const float scales[3] = { 0.5f, 0.25f, 1.0f };
    for (int period = 0; period < 2; ++period) {
        float expected_grad_w = 0.0f;
        for (int k = 0; k < 3; ++k) {
            ggml_init_params cp = { GGML_DEFAULT_GRAPH_SIZE * ggml_tensor_overhead() + 4 * ggml_graph_overhead_custom(GGML_DEFAULT_GRAPH_SIZE, true), nullptr, true };
            ggml_context * ctx_compute = ggml_init(cp);
            ggml_tensor * x = ggml_new_tensor_1d(ctx_compute, GGML_TYPE_F32, 1);
            ggml_set_input(x);
            ggml_set_grad(x);
            ggml_tensor * z      = ggml_new_tensor_2d(ctx_compute, GGML_TYPE_F32, 1, 1);
            ggml_set_input(z);
            ggml_tensor * wx     = ggml_reshape_2d(ctx_compute, ggml_mul(ctx_compute, w, x), 1, 1);
            ggml_tensor * logits = ggml_concat(ctx_compute, wx, z, 0); // [2, 1]
            ggml_tensor * extra  = k == 1 ? ggml_scale(ctx_compute, ggml_sum(ctx_compute, ggml_mul(ctx_compute, w, x)), 3.0f) : nullptr;
            ggml_cgraph * gf = ggml_new_graph_custom(ctx_compute, GGML_DEFAULT_GRAPH_SIZE, true);
            ggml_build_forward_expand(gf, logits);
            if (extra) {
                ggml_build_forward_expand(gf, extra);
            }
            ggml_opt_prepare_alloc(opt_ctx, ctx_compute, gf, x, logits);
            ggml_opt_set_next_step(opt_ctx, /*period_end =*/ k == 2, scales[k], extra);
            CHECK(ggml_opt_alloc(opt_ctx, true));
            ggml_backend_tensor_set(x, &X, 0, sizeof(float));
            const float zero = 0.0f, one = 1.0f;
            ggml_backend_tensor_set(z, &zero, 0, sizeof(float));
            ggml_tensor * labels = ggml_opt_labels(opt_ctx);
            ggml_backend_tensor_set(labels, &one, 0, sizeof(float));
            ggml_backend_tensor_set(labels, &zero, sizeof(float), sizeof(float));
            ggml_opt_eval(opt_ctx, nullptr);

            const float s0 = 1.0f / (1.0f + std::exp(-w_now * X));
            const float dlogit = scales[k] * (s0 - 1.0f);
            const float dx = dlogit * w_now + (k == 1 ? 3.0f * w_now : 0.0f);
            expected_grad_w += dlogit * X + (k == 1 ? 3.0f * X : 0.0f);
            ggml_tensor * gx = ggml_opt_leaf_grad(opt_ctx, x);
            CHECK(gx != nullptr && "a GRAD leaf's gradient reads back after the eval");
            float gx_val;
            ggml_backend_tensor_get(gx, &gx_val, 0, sizeof(float));
            if (std::fabs(gx_val - dx) > 1e-5f) {
                fprintf(stderr, "period %d graph %d: dL/dx %f, expected %f\n", period, k, gx_val, dx);
                exit(1);
            }
            ggml_free(ctx_compute);

            float w_after;
            ggml_backend_tensor_get(w, &w_after, 0, sizeof(float));
            if (k < 2) {
                CHECK(std::fabs(w_after - w_now) < 1e-7f && "no step before the period ends");
            } else {
                const float expected = w_now - LR * expected_grad_w;
                if (std::fabs(w_after - expected) > 1e-5f) {
                    fprintf(stderr, "period %d: w moved to %f, expected %f (from %f)\n", period, w_after, expected, w_now);
                    exit(1);
                }
                w_now = w_after;
            }
        }
    }

    ggml_opt_free(opt_ctx);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx_static);
    ggml_backend_sched_free(sched);
    ggml_backend_free(cpu);
}

int main() {
    ggml_backend_load_all();
    run(1);
    run(2);
    run_manual();
    printf("test-opt-dynamic-accum: OK\n");
    return 0;
}
