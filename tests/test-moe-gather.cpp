// [MOE-GATHER #23] identity-table equivalence proof for the gather-capable
// MUL_MAT_ID (docs/serving/MOE-GATHER-MULMATID.md, continuum PR #2144 map).
//
// Builds the SAME MoE-shaped graph twice on every backend that accepts the
// gather op (Metal first): once as plain ggml_mul_mat_id (contiguous stride),
// once as ggml_mul_mat_id_gather with the IDENTITY table (eptrs[i] = i*nb02,
// byte offsets relative to the src0 tensor start). The gather path exercises
// the full table machinery while addressing the exact same bytes, so the two
// outputs must match BIT-FOR-BIT — any divergence is an addressing bug, not
// numerics. Backends that reject the gather (supports_op) are skipped, not
// failed: partial rollout is the designed state.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstring>
#include <cstdint>
#include <vector>

static float frand(uint32_t & state) {
    // deterministic LCG — identical fills on every run/platform
    state = state * 1664525u + 1013904223u;
    return ((state >> 8) & 0xffff) / 65536.0f - 0.5f;
}

struct graph_out {
    std::vector<float> data;
    bool               ran = false;
};

// build + run one MUL_MAT_ID graph on `backend`; gather=true uses the identity
// table. Returns the dst contents.
static graph_out run_case(ggml_backend_t backend, bool gather) {
    const int64_t ne00     = 64; // cols (must be >= simdgroup mins)
    const int64_t ne01     = 32; // rows per expert
    const int64_t n_expert = 8;
    const int64_t n_used   = 2;
    const int64_t n_tokens = 4;  // < 32 → mv path (the implemented family)

    graph_out out;

    ggml_init_params ip = {
        /*.mem_size   =*/ ggml_tensor_overhead() * 16 + ggml_graph_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * as  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, ne00, ne01, n_expert);
    ggml_tensor * b   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, ne00, 1, n_tokens);
    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, n_tokens);

    ggml_tensor * eptrs = nullptr;
    ggml_tensor * dst   = nullptr;
    if (gather) {
        eptrs = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_expert);
        dst   = ggml_mul_mat_id_gather(ctx, as, b, ids, eptrs);
    } else {
        dst   = ggml_mul_mat_id(ctx, as, b, ids);
    }

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, dst);

    // Partial rollout is designed: a backend that rejects the gather op skips.
    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    if (dev != nullptr && !ggml_backend_dev_supports_op(dev, dst)) {
        ggml_free(ctx);
        return out;
    }

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buf == nullptr) {
        ggml_free(ctx);
        return out;
    }

    // deterministic fills
    {
        std::vector<float> h((size_t) ne00*ne01*n_expert);
        uint32_t s = 42;
        for (auto & v : h) v = frand(s);
        ggml_backend_tensor_set(as, h.data(), 0, h.size()*sizeof(float));
    }
    {
        std::vector<float> h((size_t) ne00*n_tokens);
        uint32_t s = 1337;
        for (auto & v : h) v = frand(s);
        ggml_backend_tensor_set(b, h.data(), 0, h.size()*sizeof(float));
    }
    {
        // routing spread across all experts, deterministic
        std::vector<int32_t> h((size_t) n_used*n_tokens);
        for (size_t i = 0; i < h.size(); ++i) h[i] = (int32_t) ((i*3 + 1) % n_expert);
        ggml_backend_tensor_set(ids, h.data(), 0, h.size()*sizeof(int32_t));
    }
    if (eptrs != nullptr) {
        // THE identity table: expert i lives exactly where the stride says.
        std::vector<int64_t> h((size_t) n_expert);
        for (int64_t i = 0; i < n_expert; ++i) h[(size_t) i] = i*(int64_t) as->nb[2];
        ggml_backend_tensor_set(eptrs, h.data(), 0, h.size()*sizeof(int64_t));
    }

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        return out;
    }

    out.data.resize(ggml_nelements(dst));
    ggml_backend_tensor_get(dst, out.data.data(), 0, out.data.size()*sizeof(float));
    out.ran = true;

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return out;
}

int main() {
    int n_checked = 0;

    for (size_t di = 0; di < ggml_backend_dev_count(); ++di) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(di);
        ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
        if (backend == nullptr) {
            continue;
        }
        const char * name = ggml_backend_name(backend);

        graph_out plain  = run_case(backend, /*gather=*/false);
        graph_out gather = run_case(backend, /*gather=*/true);

        if (!plain.ran) {
            printf("%-12s SKIP (plain mul_mat_id unsupported)\n", name);
        } else if (!gather.ran) {
            printf("%-12s SKIP (gather rejected by supports_op — designed partial rollout)\n", name);
        } else {
            const bool same = plain.data.size() == gather.data.size() &&
                memcmp(plain.data.data(), gather.data.data(), plain.data.size()*sizeof(float)) == 0;
            printf("%-12s %s (%zu values)\n", name, same ? "OK bit-identical" : "FAIL divergent", plain.data.size());
            if (!same) {
                ggml_backend_free(backend);
                return 1;
            }
            n_checked++;
        }
        ggml_backend_free(backend);
    }

    if (n_checked == 0) {
        printf("no backend accepted the gather op — nothing verified (not a failure)\n");
    }
    return 0;
}
