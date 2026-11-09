// What this catches: a dry graph measurement must neither grow nor free serving buffers,
// and its exact sizes must match the guarded allocation path (including a refused lease).
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include <cstdio>
#include <cstdlib>
#define CHECK(x) do { if (!(x)) { fprintf(stderr, "FAILED line %d: %s\n", __LINE__, #x); exit(1); } } while (0)

struct graph_fixture {
    ggml_context * ctx;
    ggml_context * external_ctx = nullptr;
    ggml_cgraph * graph;
    ggml_tensor * x;
    ggml_tensor * out;
    explicit graph_fixture(int64_t n, bool separate_input = false) {
        ctx = ggml_init({1024*1024, nullptr, true});
        if (separate_input) { external_ctx = ggml_init({1024*1024, nullptr, true}); }
        x = ggml_new_tensor_1d(external_ctx ? external_ctx : ctx, GGML_TYPE_F32, n);
        ggml_set_input(x);
        out = separate_input ? ggml_sum(ctx, x) : ggml_scale(ctx, x, 2.0f);
        ggml_set_output(out);
        graph = ggml_new_graph_custom(ctx, 64, false);
        ggml_build_forward_expand(graph, out);
    }
    ~graph_fixture() { ggml_free(ctx); ggml_free(external_ctx); }
};
int main() {
    ggml_backend_load_all();
    auto cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    CHECK(cpu);
    ggml_backend_t backends[] = {cpu};
    auto sched = ggml_backend_sched_new(backends, nullptr, 1, 64, false, true);
    graph_fixture small(16);
    CHECK(ggml_backend_sched_alloc_graph(sched, small.graph));
    const size_t held = ggml_backend_sched_get_buffer_size(sched, cpu);
    CHECK(held > 0);
    ggml_backend_sched_reset(sched);
    size_t measured = 0;
    { graph_fixture big(65536);
      ggml_backend_sched_measure_graph(sched, big.graph, &measured);
      CHECK(measured > held);
      CHECK(ggml_backend_sched_get_buffer_size(sched, cpu) == held);
      CHECK(big.x->data == nullptr && big.out->data == nullptr);
      ggml_backend_sched_reset(sched);
    }
    size_t denied_size = 0, no_growth = 0;
    { graph_fixture big(65536);
      CHECK(!ggml_backend_sched_alloc_graph_within(sched, big.graph, &no_growth, &denied_size));
      CHECK(denied_size == measured);
      CHECK(ggml_backend_sched_get_buffer_size(sched, cpu) == held);
      ggml_backend_sched_reset(sched);
    }
    { graph_fixture big(65536);
      const size_t lease = measured - held;
      size_t actual = 0;
      CHECK(ggml_backend_sched_alloc_graph_within(sched, big.graph, &lease, &actual));
      CHECK(actual == measured);
      const float value = 3.0f;
      ggml_backend_tensor_set(big.x, &value, 0, sizeof(value));
      CHECK(ggml_backend_sched_graph_compute(sched, big.graph) == GGML_STATUS_SUCCESS);
      float result = 0;
      ggml_backend_tensor_get(big.out, &result, 0, sizeof(result));
      CHECK(result == 6.0f);
      ggml_backend_sched_reset(sched);
    }
    // Optimizer-owned leaves must not be charged again as workspace, and their
    // declared placement must match a later real allocation without fake pointers.
    size_t external_plan = 0;
    const size_t held_before_external = ggml_backend_sched_get_buffer_size(sched, cpu);
    { graph_fixture big(65536, true);
      ggml_tensor * external[] = {big.x};
      ggml_backend_sched_measure_graph_external(sched, big.graph, &external_plan, external, backends, 1);
      CHECK(big.x->data == nullptr && big.x->buffer == nullptr && big.out->data == nullptr);
      CHECK(ggml_backend_sched_get_buffer_size(sched, cpu) == held_before_external);
      CHECK(external_plan < measured);
      ggml_backend_sched_reset(sched);
    }
    { graph_fixture big(65536, true);
      auto input_buffer = ggml_backend_alloc_ctx_tensors(big.external_ctx, cpu);
      CHECK(input_buffer);
      size_t actual = 0, growth = external_plan > held_before_external ? external_plan - held_before_external : 0;
      CHECK(ggml_backend_sched_alloc_graph_within(sched, big.graph, &growth, &actual));
      CHECK(actual == external_plan);
      const float value = 5.0f;
      ggml_set_zero(big.x);
      ggml_backend_tensor_set(big.x, &value, 0, sizeof(value));
      CHECK(ggml_backend_sched_graph_compute(sched, big.graph) == GGML_STATUS_SUCCESS);
      float result = 0;
      ggml_backend_tensor_get(big.out, &result, 0, sizeof(result));
      CHECK(result == 5.0f);
      ggml_backend_sched_reset(sched);
      ggml_backend_buffer_free(input_buffer);
    }
    ggml_backend_sched_free(sched);
    ggml_backend_free(cpu);
    printf("PASS dry measurement preserves live buffers; exact lease allocates same plan\n");
}
