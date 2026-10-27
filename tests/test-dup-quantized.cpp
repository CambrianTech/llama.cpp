// The CPU backend copies a quantized tensor into F32, F16 and BF16.
//
// A training graph that walks a quantized cache reads it as F16 (llm_graph_context::build_attn_mha
// casts a q8_0 K/V view to F16). The CPU dup took only an F32 destination from a quantized source,
// so every CPU training run aborted the engine in its first step ("ggml-cpu/ops.cpp: fatal error";
// continuum card c11da2f6, enrico 2026-10-24..26).
//
// Checked here, on a contiguous source and on a permuted view (the cache's shape): the F16 and
// BF16 copies equal the F32 dequantization converted, element for element.

#include "ggml.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <vector>

static int check(const char * what, ggml_tensor * f32, ggml_tensor * other) {
    const int64_t n = ggml_nelements(f32);
    int bad = 0;
    for (int64_t i = 0; i < n; ++i) {
        const float want = ((const float *) f32->data)[i];
        float got;
        float expect;
        if (other->type == GGML_TYPE_F16) {
            got    = ggml_fp16_to_fp32(((const ggml_fp16_t *) other->data)[i]);
            expect = ggml_fp16_to_fp32(ggml_fp32_to_fp16(want));
        } else {
            got    = ggml_bf16_to_fp32(((const ggml_bf16_t *) other->data)[i]);
            expect = ggml_bf16_to_fp32(ggml_fp32_to_bf16(want));
        }
        if (got != expect) {
            if (bad < 3) {
                fprintf(stderr, "%s: element %lld is %g, expected %g\n", what, (long long) i, got, expect);
            }
            ++bad;
        }
    }
    fprintf(stderr, "%s: %lld elements, %d mismatched\n", what, (long long) n, bad);
    return bad;
}

int main() {
    ggml_init_params ip = { 64 * 1024 * 1024, nullptr, false };
    ggml_context * ctx = ggml_init(ip);

    // [64 per row, 4 rows, 2 planes]: rows of whole Q8_0 blocks, like a K cache [head_dim, tokens, heads]
    const int64_t ne0 = 64, ne1 = 4, ne2 = 2;
    std::vector<float> values(ne0 * ne1 * ne2);
    for (size_t i = 0; i < values.size(); ++i) {
        values[i] = 3.0f * std::sin(0.37f * (float) i) + 0.01f * (float) (i % 11);
    }
    ggml_tensor * q = ggml_new_tensor_3d(ctx, GGML_TYPE_Q8_0, ne0, ne1, ne2);
    ggml_quantize_chunk(GGML_TYPE_Q8_0, values.data(), q->data, 0, ne1 * ne2, ne0, nullptr);

    // a permuted view, as attention reads the cache: rows stay whole blocks, planes and rows swap
    ggml_tensor * view = ggml_permute(ctx, q, 0, 2, 1, 3);

    ggml_tensor * outs[6] = {
        ggml_cast(ctx, q, GGML_TYPE_F32),    ggml_cast(ctx, q, GGML_TYPE_F16),    ggml_cast(ctx, q, GGML_TYPE_BF16),
        ggml_cast(ctx, view, GGML_TYPE_F32), ggml_cast(ctx, view, GGML_TYPE_F16), ggml_cast(ctx, view, GGML_TYPE_BF16),
    };
    ggml_cgraph * gf = ggml_new_graph(ctx);
    for (ggml_tensor * t : outs) {
        ggml_build_forward_expand(gf, t);
    }
    if (ggml_graph_compute_with_ctx(ctx, gf, 2) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "graph compute failed\n");
        return 1;
    }

    int bad = 0;
    bad += check("contiguous q8_0 -> f16", outs[0], outs[1]);
    bad += check("contiguous q8_0 -> bf16", outs[0], outs[2]);
    bad += check("permuted q8_0 -> f16", outs[3], outs[4]);
    bad += check("permuted q8_0 -> bf16", outs[3], outs[5]);

    ggml_free(ctx);
    if (bad > 0) {
        return 1;
    }
    fprintf(stderr, "SUCCESS - the CPU copies q8_0 into f16 and bf16\n");
    return 0;
}
