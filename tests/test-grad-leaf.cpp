// what this catches: GGML_TENSOR_FLAG_GRAD, a leaf whose gradient the backward computes while
// no optimizer updates it (the training walk reads the gradient of the cached prefix K/V this
// way). The leaf must get its exact gradient beside a real parameter's, and a leaf WITHOUT the
// flag must stay a constant with no gradient.

#include "ggml.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>

#define CHECK(cond) do { if (!(cond)) { fprintf(stderr, "FAILED %s:%d: %s\n", __FILE__, __LINE__, #cond); exit(1); } } while (0)

int main() {
    ggml_init_params params = { 64u * 1024 * 1024, nullptr, false };
    ggml_context * ctx = ggml_init(params);

    const int n = 4;
    ggml_tensor * x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n); // the GRAD leaf
    ggml_tensor * w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n); // a real parameter
    ggml_tensor * c = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n); // a plain constant
    for (int i = 0; i < n; ++i) {
        ggml_set_f32_1d(x, i, 0.5f + i);
        ggml_set_f32_1d(w, i, 1.0f - 0.25f * i);
        ggml_set_f32_1d(c, i, 2.0f);
    }
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
    ggml_graph_reset(gb); // loss grad = 1, every other grad = 0
    ggml_graph_compute_with_ctx(ctx, gb, 1);

    ggml_tensor * gx = ggml_graph_get_grad(gb, x);
    ggml_tensor * gw = ggml_graph_get_grad(gb, w);
    CHECK(gx != nullptr && "a GRAD leaf gets a gradient");
    CHECK(gw != nullptr && "a PARAM still gets its gradient");
    CHECK(ggml_graph_get_grad(gb, c) == nullptr && "an unflagged constant gets none");
    for (int i = 0; i < n; ++i) {
        const float xi = 0.5f + i, wi = 1.0f - 0.25f * i, yi = wi * xi + 2.0f;
        CHECK(std::fabs(ggml_get_f32_1d(gx, i) - 2.0f * yi * wi) < 1e-5f);
        CHECK(std::fabs(ggml_get_f32_1d(gw, i) - 2.0f * yi * xi) < 1e-5f);
    }
    CHECK(!(x->flags & GGML_TENSOR_FLAG_PARAM) && "the GRAD leaf is not a parameter");
    // regression: GRAD first shared COMPUTE's bit, so every computed node read as a GRAD leaf and
    // the backward asked for gradients of the quantized weights (GGML_ASSERT in the 1.5B walk)
    for (int i = 0; i < ggml_graph_n_nodes(gb); ++i) {
        ggml_tensor * node = ggml_graph_node(gb, i);
        CHECK((node == x || !(node->flags & GGML_TENSOR_FLAG_GRAD)) && "only the flagged leaf is a GRAD leaf");
    }

    ggml_free(ctx);
    printf("test-grad-leaf: OK\n");
    return 0;
}
