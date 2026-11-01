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
#include <cstring>
#include <initializer_list>

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

static void run_out_prod_broadcast(ggml_type type, bool transpose) {
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    CHECK(cpu != nullptr);
    ggml_init_params cp = { 32 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(cp);
    ggml_tensor * input = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 32, 2, 2, 2);
    ggml_tensor * a = ggml_cast(ctx, input, type);
    ggml_tensor * decoded = ggml_cast(ctx, a, GGML_TYPE_F32);
    ggml_tensor * b_storage = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, transpose ? 2 : 1, transpose ? 1 : 2, 4, 4);
    ggml_tensor * b = transpose ? ggml_transpose(ctx, b_storage) : b_storage;
    ggml_tensor * out = ggml_out_prod(ctx, a, b);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);
    ggml_build_forward_expand(graph, decoded);
    CHECK(ggml_backend_supports_op(cpu, out));
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, cpu);
    float av[32 * 2 * 2 * 2], bv[2 * 4 * 4];
    for (int batch = 0; batch < 4; ++batch) {
        for (int row = 0; row < 2; ++row) {
            for (int x = 0; x < 32; ++x) {
                av[(batch * 2 + row) * 32 + x] = float((batch + 1) * (row + 1));
            }
        }
    }
    for (int batch = 0; batch < 16; ++batch) {
        bv[batch * 2] = float(batch + 1);
        bv[batch * 2 + 1] = -float(batch + 2);
    }
    ggml_backend_tensor_set(input, av, 0, sizeof(av));
    ggml_backend_tensor_set(b_storage, bv, 0, sizeof(bv));
    CHECK(ggml_backend_graph_compute(cpu, graph) == GGML_STATUS_SUCCESS);
    float actual[32 * 4 * 4];
    ggml_backend_tensor_get(out, actual, 0, sizeof(actual));
    ggml_backend_tensor_get(decoded, av, 0, sizeof(av));
    for (int z = 0; z < 4; ++z) {
        for (int y = 0; y < 4; ++y) {
            const int a_batch = (z / 2) * 2 + y / 2;
            const int b_batch = z * 4 + y;
            for (int x = 0; x < 32; ++x) {
                const float expected = av[a_batch * 64 + x] * bv[b_batch * 2]
                    + av[a_batch * 64 + 32 + x] * bv[b_batch * 2 + 1];
                if (std::fabs(actual[b_batch * 32 + x] - expected) >= 1e-4f) {
                    fprintf(stderr, "out_prod type=%s transpose=%d y=%d z=%d actual=%g expected=%g\n",
                            ggml_type_name(type), transpose, y, z, actual[b_batch * 32 + x], expected);
                }
                CHECK(std::fabs(actual[b_batch * 32 + x] - expected) < 1e-4f);
            }
        }
    }
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(cpu);
}

static void run_cache_view_gradient(ggml_type type) {
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    ggml_backend_t backends[] = { cpu };
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, nullptr, 1, GGML_DEFAULT_GRAPH_SIZE, false, true);
    ggml_init_params sp = { 8 * ggml_tensor_overhead(), nullptr, true };
    ggml_context * storage = ggml_init(sp);
    ggml_tensor * weights = ggml_new_tensor_2d(storage, GGML_TYPE_F32, 32, 1);
    ggml_tensor * cache = ggml_new_tensor_2d(storage, type, 32, 4);
    ggml_set_param(weights);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(storage, cpu);
    float values[32];
    for (float & value : values) { value = 1.0f; }
    ggml_backend_tensor_set(weights, values, 0, sizeof(values));
    ggml_backend_tensor_memset(cache, 0, 0, ggml_nbytes(cache));
    ggml_opt_params params = ggml_opt_default_params(sched, GGML_OPT_LOSS_TYPE_SUM);
    params.optimizer = GGML_OPT_OPTIMIZER_TYPE_SGD;
    params.get_opt_pars = sgd_pars;
    ggml_opt_context_t opt = ggml_opt_init(params);
    ggml_init_params cp = { GGML_DEFAULT_GRAPH_SIZE * ggml_tensor_overhead() + 4 * ggml_graph_overhead_custom(GGML_DEFAULT_GRAPH_SIZE, true), nullptr, true };
    ggml_context * ctx = ggml_init(cp);
    ggml_tensor * row = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    ggml_set_input(row);
    ggml_tensor * written = ggml_set_rows(ctx, cache, weights, row);
    ggml_tensor * view = ggml_view_2d(ctx, written, 32, 1, written->nb[1], 2 * written->nb[1]);
    view = ggml_cast(ctx, view, GGML_TYPE_F32);
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, GGML_DEFAULT_GRAPH_SIZE, true);
    ggml_build_forward_expand(graph, view);
    ggml_opt_prepare_alloc(opt, ctx, graph, row, view);
    CHECK(ggml_opt_alloc(opt, true));
    const int32_t index = 2;
    ggml_backend_tensor_set(row, &index, 0, sizeof(index));
    ggml_opt_eval(opt, nullptr);
    CHECK(ggml_opt_refusal(opt)[0] == '\0');
    ggml_backend_tensor_get(weights, values, 0, sizeof(values));
    for (float value : values) { CHECK(std::fabs(value - (1.0f - LR)) < 1e-6f); }
    ggml_opt_free(opt);
    ggml_free(ctx);
    ggml_backend_buffer_free(buf);
    ggml_free(storage);
    ggml_backend_sched_free(sched);
    ggml_backend_free(cpu);
}

static void run_label_projection(bool projected) {
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    CHECK(cpu != nullptr);
    ggml_backend_t backends[] = { cpu };
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, nullptr, 1, GGML_DEFAULT_GRAPH_SIZE, false, true);
    ggml_init_params sp = { 8 * ggml_tensor_overhead(), nullptr, true };
    ggml_context * storage = ggml_init(sp);
    ggml_tensor * weights = ggml_new_tensor_2d(storage, GGML_TYPE_F32, 2, 4);
    ggml_tensor * identity = ggml_new_tensor_2d(storage, GGML_TYPE_F32, 2, 2);
    ggml_set_param(weights);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(storage, cpu);
    const float before[] = { 8.0f, -8.0f, 0.5f, -0.5f, -8.0f, 8.0f, 0.25f, 0.75f };
    const float eye[] = { 1, 0, 0, 1 };
    ggml_backend_tensor_set(weights, before, 0, sizeof(before));
    ggml_backend_tensor_set(identity, eye, 0, sizeof(eye));
    ggml_opt_params params = ggml_opt_default_params(sched, GGML_OPT_LOSS_TYPE_CROSS_ENTROPY_TARGETS);
    params.optimizer = GGML_OPT_OPTIMIZER_TYPE_SGD;
    params.get_opt_pars = sgd_pars;
    ggml_opt_context_t opt = ggml_opt_init(params);
    ggml_init_params cp = { GGML_DEFAULT_GRAPH_SIZE * ggml_tensor_overhead() + 4 * ggml_graph_overhead_custom(GGML_DEFAULT_GRAPH_SIZE, true), nullptr, true };
    ggml_context * graph_ctx = ggml_init(cp);
    ggml_tensor * base = ggml_mul_mat(graph_ctx, identity, weights);
    ggml_mul_mat_set_prec(base, GGML_PREC_F32);
    ggml_tensor * rows = ggml_new_tensor_1d(graph_ctx, GGML_TYPE_I32, 2);
    ggml_tensor * public_rows = ggml_new_tensor_1d(graph_ctx, GGML_TYPE_I32, 3);
    ggml_set_input(rows);
    ggml_set_input(public_rows);
    ggml_tensor * selected = ggml_add(graph_ctx, ggml_get_rows(graph_ctx, base, rows),
            ggml_get_rows(graph_ctx, ggml_scale(graph_ctx, base, 0.25f), rows));
    ggml_tensor * logits = ggml_set_rows(graph_ctx, ggml_dup(graph_ctx, base), selected, rows);
    ggml_cgraph * gf = ggml_new_graph_custom(graph_ctx, GGML_DEFAULT_GRAPH_SIZE, true);
    ggml_build_forward_expand(gf, logits);
    ggml_tensor * public_logits = nullptr;
    ggml_tensor * public_shadow = nullptr;
    ggml_tensor * local_shadow[2] = {};
    if (projected) {
        public_shadow = ggml_new_tensor_1d(graph_ctx, GGML_TYPE_I32, 1);
        ggml_set_input(public_shadow);
        for (auto & local : local_shadow) {
            local = ggml_new_tensor_1d(graph_ctx, GGML_TYPE_I32, 1);
            ggml_set_input(local);
        }
        ggml_tensor * slices[2];
        CHECK(ggml_opt_project_head(graph_ctx, gf, weights, logits, rows, public_rows, rows,
                public_shadow, local_shadow, 1, &public_logits, slices, 2) == 2);
        ggml_opt_prepare_alloc_slices(opt, graph_ctx, gf, rows, slices, 2);
    } else {
        public_logits = ggml_get_rows(graph_ctx, logits, public_rows);
        ggml_build_forward_expand(gf, public_logits);
        ggml_opt_prepare_alloc(opt, graph_ctx, gf, rows, ggml_get_rows(graph_ctx, logits, rows));
    }
    ggml_set_output(public_logits);
    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
        auto * node = ggml_graph_node(gf, i);
        if (node->op == GGML_OP_MUL_MAT) {
            CHECK(node->op_params[0] == GGML_PREC_F32);
        }
    }
    CHECK(ggml_opt_alloc(opt, true));
    const int32_t labels[] = { 1, 3 }, public_ids[] = { 2, 1, 0 }, targets[] = { 0, 1 };
    ggml_backend_tensor_set(rows, labels, 0, sizeof(labels));
    ggml_backend_tensor_set(public_rows, public_ids, 0, sizeof(public_ids));
    if (projected) {
        const int32_t public_label = 1, local_label = 0;
        ggml_backend_tensor_set(public_shadow, &public_label, 0, sizeof(public_label));
        for (auto * local : local_shadow) {
            ggml_backend_tensor_set(local, &local_label, 0, sizeof(local_label));
        }
    }
    ggml_backend_tensor_set(ggml_opt_labels(opt), targets, 0, sizeof(targets));
    ggml_opt_eval(opt, nullptr);
    CHECK(ggml_opt_refusal(opt)[0] == '\0');
    float after[8], served[6];
    ggml_backend_tensor_get(weights, after, 0, sizeof(after));
    ggml_backend_tensor_get(public_logits, served, 0, sizeof(served));
    for (int row = 0; row < 4; ++row) {
        const bool labelled = row == 1 || row == 3;
        const float p0 = 1.0f / (1.0f + std::exp(1.25f * (before[2*row + 1] - before[2*row])));
        const float delta = labelled ? LR * 1.25f * ((row == 1 ? 1.0f : 0.0f) - p0) / 2 : 0;
        CHECK(std::fabs(after[2*row] - before[2*row] - delta) < 1e-6f);
        CHECK(std::fabs(after[2*row + 1] - before[2*row + 1] + delta) < 1e-6f);
    }
    for (int row = 0; row < 3; ++row) {
        const float scale = public_ids[row] == 1 ? 1.25f : 1;
        for (int col = 0; col < 2; ++col) {
            CHECK(served[2*row + col] == before[2*public_ids[row] + col] * scale);
        }
    }
    ggml_opt_free(opt);
    ggml_free(graph_ctx);
    ggml_backend_buffer_free(buf);
    ggml_free(storage);
    ggml_backend_sched_free(sched);
    ggml_backend_free(cpu);
}
// what this catches: a training graph the loss's gradient cannot pass through (here an in-place
// clamp, a view with no backward) must refuse ggml_opt_alloc by the op's name, not abort: the
// abort took the M5's serving engine down on the 35B-A3B's first /train (2026-11-01). The same
// context then trains a graph it can differentiate.
static void run_refuses_graph_without_backward() {
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    CHECK(cpu != nullptr);
    ggml_backend_t backends[] = { cpu };
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, nullptr, 1, GGML_DEFAULT_GRAPH_SIZE, false, true);

    ggml_init_params sp = { 8 * ggml_tensor_overhead(), nullptr, true };
    ggml_context * ctx_static = ggml_init(sp);
    ggml_tensor * w = ggml_new_tensor_1d(ctx_static, GGML_TYPE_F32, 1);
    ggml_set_param(w);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx_static, cpu);
    const float w0 = 1.0f;
    ggml_backend_tensor_set(w, &w0, 0, sizeof(float));

    ggml_opt_params params = ggml_opt_default_params(sched, GGML_OPT_LOSS_TYPE_SUM);
    params.optimizer    = GGML_OPT_OPTIMIZER_TYPE_SGD;
    params.get_opt_pars = sgd_pars;
    ggml_opt_context_t opt_ctx = ggml_opt_init(params);

    for (const bool differentiable : { false, true }) {
        ggml_init_params cp = { GGML_DEFAULT_GRAPH_SIZE * ggml_tensor_overhead() + 4 * ggml_graph_overhead_custom(GGML_DEFAULT_GRAPH_SIZE, true), nullptr, true };
        ggml_context * ctx_compute = ggml_init(cp);
        ggml_tensor * x   = ggml_new_tensor_1d(ctx_compute, GGML_TYPE_F32, 1);
        ggml_tensor * wx  = ggml_mul(ctx_compute, w, x);
        ggml_tensor * out = differentiable ? ggml_clamp(ctx_compute, wx, -8.0f, 8.0f) : ggml_clamp_inplace(ctx_compute, wx, -8.0f, 8.0f);
        ggml_cgraph * gf  = ggml_new_graph_custom(ctx_compute, GGML_DEFAULT_GRAPH_SIZE, true);
        ggml_build_forward_expand(gf, out);
        ggml_opt_prepare_alloc(opt_ctx, ctx_compute, gf, x, out);
        if (!differentiable) {
            CHECK(!ggml_opt_alloc(opt_ctx, true));
            CHECK(std::strstr(ggml_opt_refusal(opt_ctx), "no backward through CLAMP") != nullptr);
        } else {
            CHECK(ggml_opt_alloc(opt_ctx, true));
            ggml_backend_tensor_set(x, &X, 0, sizeof(float));
            ggml_opt_eval(opt_ctx, nullptr);
            CHECK(ggml_opt_refusal(opt_ctx)[0] == '\0');
            float w_after;
            ggml_backend_tensor_get(w, &w_after, 0, sizeof(float));
            // |w x| = 2 is inside the clamp, so its gradient is x and SGD moves w by -lr x
            CHECK(std::fabs(w_after - (w0 - LR * X)) < 1e-6f);
        }
        ggml_free(ctx_compute);
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
    run_refuses_graph_without_backward();
    run_label_projection(false);
    run_label_projection(true);
    for (ggml_type type : { GGML_TYPE_F16, GGML_TYPE_Q8_0 }) {
        run_cache_view_gradient(type);
        run_out_prod_broadcast(type, false);
        run_out_prod_broadcast(type, true);
    }
    printf("test-opt-dynamic-accum: OK\n");
    return 0;
}
