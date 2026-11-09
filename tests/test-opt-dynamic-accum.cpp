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

static void run(int32_t opt_period, bool test_residency_gate = false, bool refuse_host = false, bool test_measurement = false, int graph_nodes = 0, bool use_gpu = false) {
    // by type, not ggml_backend_cpu_init: in a dynamically loaded backend build (CI) the CPU
    // backend is its own library
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    GGML_ASSERT(cpu != nullptr);
    ggml_backend_t gpu = use_gpu ? ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr) : nullptr;
    CHECK(!use_gpu || gpu);
    ggml_backend_t backends[] = { gpu ? gpu : cpu, cpu };
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, nullptr, gpu ? 2 : 1, GGML_DEFAULT_GRAPH_SIZE, false, true);

    ggml_init_params sp = { 8 * ggml_tensor_overhead(), nullptr, true };
    ggml_context * ctx_static = ggml_init(sp);
    ggml_tensor * w = ggml_new_tensor_1d(ctx_static, GGML_TYPE_F32, 1);
    ggml_set_param(w);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx_static, gpu ? gpu : cpu);
    float w_now = 1.0f;
    ggml_backend_tensor_set(w, &w_now, 0, sizeof(float));

    ggml_opt_params params = ggml_opt_default_params(sched, GGML_OPT_LOSS_TYPE_SUM);
    params.optimizer    = GGML_OPT_OPTIMIZER_TYPE_SGD;
    params.get_opt_pars = sgd_pars;
    params.opt_period   = opt_period;
    ggml_opt_context_t opt_ctx = ggml_opt_init(params); // no ctx_compute: graphs are built per step
    ggml_opt_result_t  result  = ggml_opt_result_init();
    struct gate_state { bool allow = false; size_t measured = 0; bool refuse_host; bool host_seen = false; } gate{false, 0, refuse_host};
    if (test_residency_gate) {
        ggml_opt_set_static_allocation_callback(opt_ctx, [](ggml_backend_buffer_type_t,
                const char * kind, size_t bytes, void * data) {
            auto & gate = *static_cast<gate_state *>(data);
            if (strcmp(kind, "optimizer_static") == 0) { gate.measured = bytes; }
            if (strcmp(kind, "optimizer_host") == 0) { gate.host_seen = true; }
            return gate.allow || (gate.refuse_host && strcmp(kind, "optimizer_static") == 0);
        }, &gate);
    }


    bool measured_once = false;
    bool cancelled_preparation = false;
    size_t measured_static = 0, measured_host = 0, measured_graph = 0;
    const int n_evals = 3 * opt_period;
    for (int i = 0; i < n_evals; ++i) {
        ggml_init_params cp = { GGML_DEFAULT_GRAPH_SIZE * ggml_tensor_overhead() + 4 * ggml_graph_overhead_custom(GGML_DEFAULT_GRAPH_SIZE, true), nullptr, true };
        ggml_context * ctx_compute = ggml_init(cp);
        ggml_tensor * x   = ggml_new_tensor_1d(ctx_compute, GGML_TYPE_F32, 1);
        ggml_tensor * out = ggml_mul(ctx_compute, w, x);
        ggml_cgraph * gf  = ggml_new_graph_custom(ctx_compute, GGML_DEFAULT_GRAPH_SIZE, true);
        ggml_build_forward_expand(gf, out);
        if (test_measurement && !measured_once) {
            // The real optimizer graph must be measurable before either residency owner
            // allocates, then a rebuilt graph must allocate exactly the reported bytes.
            auto probe = ggml_opt_init(params);
            ggml_opt_prepare_alloc(probe, ctx_compute, gf, x, out);
            const size_t held = ggml_backend_sched_get_buffer_size(sched, cpu);
            CHECK(ggml_opt_measure_alloc(probe, true, &measured_static, &measured_host, &measured_graph));
            CHECK(measured_static > 0 && measured_host > 0 && measured_graph > 0);
            CHECK(ggml_backend_sched_get_buffer_size(sched, cpu) == held);
            size_t owned = 0;
            ggml_opt_visit_owned_buffers(probe, [](ggml_backend_buffer_t b, const char *, void * data) {
                *static_cast<size_t *>(data) += ggml_backend_buffer_get_size(b);
            }, &owned);
            CHECK(owned == 0);
            CHECK(!ggml_opt_alloc(probe, true)); // no reuse of a dry, rewritten graph
            float unchanged = 0;
            ggml_backend_tensor_get(w, &unchanged, 0, sizeof(unchanged));
            CHECK(unchanged == w_now);
            ggml_opt_free(probe);
            ggml_free(ctx_compute);
            measured_once = true;
            --i;
            continue;
        }
        ggml_opt_prepare_alloc(opt_ctx, ctx_compute, gf, x, out);
        if (test_residency_gate && !gate.allow) {
            // What this catches: denied residency is measured by the real allocator before
            // any optimizer buffer allocation, and re-preparing after admission is usable.
            CHECK(!ggml_opt_alloc(opt_ctx, true));
            CHECK(gate.measured > 0);
            CHECK(!refuse_host || gate.host_seen);
            size_t allocated = 0;
            ggml_opt_visit_owned_buffers(opt_ctx, [](ggml_backend_buffer_t b, const char *, void * data) {
                *static_cast<size_t *>(data) += ggml_backend_buffer_get_size(b);
            }, &allocated);
            CHECK(allocated == 0);
            gate.allow = true;
            ggml_free(ctx_compute);
            --i;
            continue;
        }
        if (graph_nodes > 0) {
            CHECK(ggml_opt_alloc_begin(opt_ctx, true));
            CHECK(!ggml_opt_alloc_begin(opt_ctx, true));
            CHECK(!ggml_opt_alloc(opt_ctx, false));
            CHECK(!ggml_opt_can_prepare_restore(opt_ctx));
            float held;
            ggml_backend_tensor_get(w, &held, 0, sizeof(held));
            CHECK(held == w_now);
            if (!cancelled_preparation) {
                CHECK(ggml_opt_alloc_cancel(opt_ctx));
                CHECK(!ggml_opt_alloc_cancel(opt_ctx));
                cancelled_preparation = true;
                ggml_free(ctx_compute);
                --i;
                continue;
            }
        }
        CHECK(ggml_opt_alloc(opt_ctx, true));
        if (test_measurement && i == 0) {
            struct sizes { size_t device = 0; size_t host = 0; } actual;
            ggml_opt_visit_owned_buffers(opt_ctx, [](ggml_backend_buffer_t b, const char * kind, void * data) {
                auto & actual = *static_cast<sizes *>(data);
                (strcmp(kind, "optimizer_static") == 0 ? actual.device : actual.host) += ggml_backend_buffer_get_size(b);
            }, &actual);
            CHECK(actual.device == measured_static && actual.host == measured_host);
            CHECK(ggml_backend_sched_get_buffer_size(sched, cpu) == measured_graph);
        }
        if (test_residency_gate) {
            size_t allocated = 0;
            ggml_opt_visit_owned_buffers(opt_ctx, [](ggml_backend_buffer_t b, const char * kind, void * data) {
                if (strcmp(kind, "optimizer_static") == 0) { *static_cast<size_t *>(data) = ggml_backend_buffer_get_size(b); }
            }, &allocated);
            CHECK(allocated == gate.measured);
        }
        ggml_backend_tensor_set(x, &X, 0, sizeof(float));
        if (graph_nodes > 0) {
            const auto before = ggml_opt_committed_steps(opt_ctx);
            bool complete = false;
            int calls = 0;
            do {
                CHECK(++calls < 1000);
                CHECK(ggml_opt_eval_step(opt_ctx, result, graph_nodes, &complete) == GGML_STATUS_SUCCESS);
                if (!complete) { CHECK(ggml_opt_committed_steps(opt_ctx) == before); }
            } while (!complete);
            CHECK(calls > 1);
            CHECK(ggml_opt_committed_steps(opt_ctx) == before + ((i + 1) % opt_period == 0));
        } else {
            ggml_opt_eval(opt_ctx, result);
        }
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
    if (gpu) { ggml_backend_free(gpu); }
    ggml_backend_free(cpu);
}

// what this catches: ggml_opt_set_next_step, the training walk's one-step-per-window period.
// Three graphs on loss = sum(w * x): the first two only accumulate, at weights 0.5 and 0.25,
// the second with an extra term 3 * w * x; the third (weight 1) takes the step. So the step
// is SGD on (0.5 + 0.25 + 3 + 1) * x. Each graph's x is a GRAD leaf, whose gradient (the
// weight times w, plus 3 w in the second) must read back after its eval; a following period
// starts from zero.
static void run_manual(bool abandon_first_chunk = false) {
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
            // What this catches: revoking a partial live reply clears only its accumulated
            // gradients, and the next complete period must match a clean optimizer.
            if (abandon_first_chunk && period == 0 && k == 0) {
                ggml_opt_reset(opt_ctx, false);
                expected_grad_w = 0.0f;
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

// what this catches: the MoE router's normalization, weights / clamp(sum_rows(weights), eps, inf)
// (llama-graph.cpp build_moe_ffn), with more than one expert used: the divisor is broadcast over
// the experts, and DIV's backward gave it a gradient of the dividend's shape. The build asserted
// on it and aborted the M5's serving engine during a /train on the 35B-A3B (8 experts used,
// 2026-11-07); the toy MoE in test-llama-archs uses 1, where the two shapes agree. Here w is the
// router's two weights for one token, the loss is w0 / (w0 + w1), and one SGD step must move w by
// -lr * (w1, -w0) / (w0 + w1)^2: the dividend's path and the broadcast divisor's, summed.
static void run_trains_through_broadcast_divisor() {
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    CHECK(cpu != nullptr);
    ggml_backend_t backends[] = { cpu };
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, nullptr, 1, GGML_DEFAULT_GRAPH_SIZE, false, true);

    ggml_init_params sp = { 8 * ggml_tensor_overhead(), nullptr, true };
    ggml_context * ctx_static = ggml_init(sp);
    ggml_tensor * w = ggml_new_tensor_2d(ctx_static, GGML_TYPE_F32, 2, 1); // [n_expert_used, n_tokens]
    ggml_set_param(w);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx_static, cpu);
    const float w0[2] = { 1.0f, 3.0f };
    ggml_backend_tensor_set(w, w0, 0, sizeof(w0));

    ggml_opt_params params = ggml_opt_default_params(sched, GGML_OPT_LOSS_TYPE_SUM);
    params.optimizer    = GGML_OPT_OPTIMIZER_TYPE_SGD;
    params.get_opt_pars = sgd_pars;
    ggml_opt_context_t opt_ctx = ggml_opt_init(params);

    ggml_init_params cp = { GGML_DEFAULT_GRAPH_SIZE * ggml_tensor_overhead() + 4 * ggml_graph_overhead_custom(GGML_DEFAULT_GRAPH_SIZE, true), nullptr, true };
    ggml_context * ctx_compute = ggml_init(cp);
    ggml_tensor * pick = ggml_new_tensor_2d(ctx_compute, GGML_TYPE_F32, 2, 1); // which normalized weight is the loss
    ggml_tensor * sum  = ggml_clamp(ctx_compute, ggml_sum_rows(ctx_compute, w), 6.103515625e-5f, INFINITY); // [1, n_tokens]
    ggml_tensor * out  = ggml_mul(ctx_compute, ggml_div(ctx_compute, w, sum), pick);
    ggml_cgraph * gf   = ggml_new_graph_custom(ctx_compute, GGML_DEFAULT_GRAPH_SIZE, true);
    ggml_build_forward_expand(gf, out);
    ggml_opt_prepare_alloc(opt_ctx, ctx_compute, gf, pick, out);
    CHECK(ggml_opt_alloc(opt_ctx, true));
    CHECK(ggml_opt_refusal(opt_ctx)[0] == '\0');
    const float pick0[2] = { 1.0f, 0.0f };
    ggml_backend_tensor_set(pick, pick0, 0, sizeof(pick0));
    ggml_opt_eval(opt_ctx, nullptr);
    CHECK(ggml_opt_refusal(opt_ctx)[0] == '\0');
    float w_after[2];
    ggml_backend_tensor_get(w, w_after, 0, sizeof(w_after));
    const float s2 = (w0[0] + w0[1]) * (w0[0] + w0[1]);
    CHECK(std::fabs(w_after[0] - (w0[0] - LR * ( w0[1] / s2))) < 1e-6f);
    CHECK(std::fabs(w_after[1] - (w0[1] - LR * (-w0[0] / s2))) < 1e-6f);

    ggml_free(ctx_compute);
    ggml_opt_free(opt_ctx);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx_static);
    ggml_backend_sched_free(sched);
    ggml_backend_free(cpu);
}

// what this catches: a backward that gives a source a gradient of another shape must be returned
// by ggml_build_backward_expand_or_refuse, naming the node, never asserted: the assert is what took
// the serving process down. The break is made here by a gradient accumulator of the wrong shape
// (w has 1 element, its accumulator 2), so the test needs no op with a broken backward.
static void run_refuses_misshaped_gradient() {
    ggml_init_params cp = { 64 * ggml_tensor_overhead() + 2 * ggml_graph_overhead_custom(GGML_DEFAULT_GRAPH_SIZE, true), nullptr, true };
    ggml_context * ctx = ggml_init(cp);
    ggml_tensor * w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ggml_set_param(w);
    ggml_tensor * x   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ggml_tensor * out = ggml_mul(ctx, w, x);
    ggml_set_name(out, "w_times_x");
    ggml_set_loss(out);
    ggml_cgraph * gb = ggml_new_graph_custom(ctx, GGML_DEFAULT_GRAPH_SIZE, true);
    ggml_build_forward_expand(gb, out);

    ggml_tensor * grad_accs[GGML_DEFAULT_GRAPH_SIZE] = { nullptr };
    for (int i = 0; i < ggml_graph_n_nodes(gb); ++i) {
        if (ggml_graph_node(gb, i) == w) {
            grad_accs[i] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2);
        }
    }
    const char * why = nullptr;
    const ggml_tensor * refused = ggml_build_backward_expand_or_refuse(ctx, gb, grad_accs, &why);
    CHECK(refused == out);
    CHECK(why != nullptr && std::strstr(why, "src[0]") != nullptr);
    ggml_free(ctx);
}

int main() {
    ggml_backend_load_all();
    for (int period : {1, 2, 3}) { run(period, false, false, false, 1); }
    if (ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU)) {
        for (int period : {1, 2, 3}) { run(period, false, false, false, 1, true); }
        printf("PASS CUDA optimizer continuation commits once across periods 1/2/3\n");
    }
    run(1);
    run(2);
    run(1, true);
    // Static admission alone must not allocate when the host lease is refused.
    run(1, true, true);
    run(1, false, false, true);
    run_manual();
    run_manual(true);
    run_refuses_graph_without_backward();
    run_trains_through_broadcast_divisor();
    run_refuses_misshaped_gradient();
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
