// what this catches: GGML_TENSOR_FLAG_GRAD, a leaf whose gradient the backward computes while
// no optimizer updates it (the training walk reads the gradient of the cached prefix K/V this
// way). The leaf must get its exact gradient beside a real parameter's, and a leaf WITHOUT the
// flag must stay a constant with no gradient.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>

#define CHECK(cond) do { if (!(cond)) { fprintf(stderr, "FAILED %s:%d: %s\n", __FILE__, __LINE__, #cond); exit(1); } } while (0)

int main() {
    // by type, not a CPU-backend symbol: in a dynamically loaded backend build (CI) the CPU
    // backend is its own library
    ggml_backend_load_all();
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    CHECK(cpu != nullptr);

    ggml_init_params params = { 64 * ggml_tensor_overhead() + ggml_graph_overhead_custom(GGML_DEFAULT_GRAPH_SIZE, true) * 2, nullptr, true };
    ggml_context * ctx = ggml_init(params);

    const int n = 4;
    ggml_tensor * x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n); // the GRAD leaf
    ggml_tensor * w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n); // a real parameter
    ggml_tensor * c = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n); // a plain constant
    ggml_set_grad(x);
    ggml_set_param(w);

    // loss = sum((w * x + c)^2): d/dx = 2 (w x + c) w, d/dw = 2 (w x + c) x
    ggml_tensor * y    = ggml_add(ctx, ggml_mul(ctx, w, x), c);
    ggml_tensor * loss = ggml_sum(ctx, ggml_sqr(ctx, y));
    ggml_set_loss(loss);

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, GGML_DEFAULT_GRAPH_SIZE, true);
    ggml_build_forward_expand(gf, loss);
    ggml_cgraph * gb = ggml_graph_dup(ctx, gf, true);
    ggml_build_backward_expand(ctx, gb, nullptr);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, cpu);
    CHECK(buf != nullptr);
    for (int i = 0; i < n; ++i) {
        const float xi = 0.5f + i, wi = 1.0f - 0.25f * i, ci = 2.0f;
        ggml_backend_tensor_set(x, &xi, i * sizeof(float), sizeof(float));
        ggml_backend_tensor_set(w, &wi, i * sizeof(float), sizeof(float));
        ggml_backend_tensor_set(c, &ci, i * sizeof(float), sizeof(float));
    }
    ggml_graph_reset(gb); // loss grad = 1, every other grad = 0
    CHECK(ggml_backend_graph_compute(cpu, gb) == GGML_STATUS_SUCCESS);

    ggml_tensor * gx = ggml_graph_get_grad(gb, x);
    ggml_tensor * gw = ggml_graph_get_grad(gb, w);
    CHECK(gx != nullptr && "a GRAD leaf gets a gradient");
    CHECK(gw != nullptr && "a PARAM still gets its gradient");
    CHECK(ggml_graph_get_grad(gb, c) == nullptr && "an unflagged constant gets none");
    for (int i = 0; i < n; ++i) {
        const float xi = 0.5f + i, wi = 1.0f - 0.25f * i, yi = wi * xi + 2.0f;
        float gxi, gwi;
        ggml_backend_tensor_get(gx, &gxi, i * sizeof(float), sizeof(float));
        ggml_backend_tensor_get(gw, &gwi, i * sizeof(float), sizeof(float));
        CHECK(std::fabs(gxi - 2.0f * yi * wi) < 1e-5f);
        CHECK(std::fabs(gwi - 2.0f * yi * xi) < 1e-5f);
    }
    CHECK(!(x->flags & GGML_TENSOR_FLAG_PARAM) && "the GRAD leaf is not a parameter");
    // regression: GRAD first shared COMPUTE's bit, so every computed node read as a GRAD leaf and
    // the backward asked for gradients of the quantized weights (GGML_ASSERT in the 1.5B walk)
    for (int i = 0; i < ggml_graph_n_nodes(gb); ++i) {
        ggml_tensor * node = ggml_graph_node(gb, i);
        CHECK((node == x || !(node->flags & GGML_TENSOR_FLAG_GRAD)) && "only the flagged leaf is a GRAD leaf");
    }

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(cpu);
    printf("test-grad-leaf: OK\n");
    return 0;
}
