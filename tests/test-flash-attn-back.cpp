// what this catches: the tiled FLASH_ATTN_EXT backward (ggml_flash_attn_ext_backward) giving a
// gradient other than attention's. Training reads attention through flash attention so its memory is
// queries x tile, never queries x window x heads (continuum, 2026-10-11), and every backend that runs
// it must give the gradients of plain attention: across several tiles and a ragged last one, grouped
// query heads, a causal mask that leaves whole tiles masked for some rows, and F32 or F16 K/V.
//
// The reference is the explicit path (mul_mat, soft_max_ext, mul_mat) on the CPU at F32, the path
// training took before. A finite difference is not used: through a kernel that rounds K/V or stages
// products at reduced precision its noise is larger than the defects this has to find (measured:
// Metal's half-staged mul_mm moved the tiled gradients 3.7e-4 relative, which this catches at 1e-4).
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

struct shape { int64_t d, h, hkv, m, n; int tile; };

// d(sum(out * w)) / d{q, k, v}, read in logical order
static bool grads(ggml_backend_t be, bool flash, const shape & s, ggml_type kv_type, std::vector<float> (&g)[3]) {
    ggml_init_params ip = { ggml_tensor_overhead()*16384 + ggml_graph_overhead_custom(1 << 16, true), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, s.d, s.n, s.h);
    ggml_tensor * k = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, s.d, s.m, s.hkv);
    ggml_tensor * v = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, s.d, s.m, s.hkv);
    ggml_tensor * w = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, s.d, s.h, s.n);
    ggml_tensor * m = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, s.m, GGML_PAD(s.n, 64), 1, 1);
    ggml_set_param(q);
    ggml_set_param(k);
    ggml_set_param(v);
    // F16 K/V: the flash path casts (as a cache feeds it); the reference reads the same F16-exact
    // values at F32 (its explicit backward cannot take a gradient through a transposed cast)
    ggml_tensor * kk = kv_type == GGML_TYPE_F32 || !flash ? k : ggml_cast(ctx, k, kv_type);
    ggml_tensor * vv = kv_type == GGML_TYPE_F32 || !flash ? v : ggml_cast(ctx, v, kv_type);
    const float scale = 1.0f / sqrtf((float) s.d);
    ggml_tensor * out;
    if (flash) {
        out = ggml_flash_attn_ext(ctx, q, kk, vv, m, scale, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);
        ggml_flash_attn_ext_set_back_tile(out, s.tile);
    } else {
        ggml_tensor * mv = ggml_cont(ctx, ggml_cast(ctx, ggml_view_2d(ctx, m, s.m, s.n, m->nb[1], 0), GGML_TYPE_F32));
        ggml_tensor * kq = ggml_mul_mat(ctx, kk, q);
        ggml_mul_mat_set_prec(kq, GGML_PREC_F32);
        kq = ggml_soft_max_ext(ctx, kq, mv, scale, 0.0f);
        ggml_tensor * kqv = ggml_mul_mat(ctx, ggml_cont(ctx, ggml_transpose(ctx, vv)), kq);
        out = ggml_cont(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3));
    }
    ggml_tensor * loss = ggml_sum(ctx, ggml_mul(ctx, out, w));
    ggml_set_loss(loss);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 1 << 16, true);
    ggml_build_forward_expand(gf, loss);
    ggml_build_backward_expand(ctx, gf, nullptr);
    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
        if (!ggml_backend_supports_op(be, ggml_graph_node(gf, i))) {
            ggml_free(ctx);
            return false;
        }
    }
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> uni(-1.0f, 1.0f);
    for (ggml_tensor * t : { q, k, v, w }) {
        std::vector<float> x(ggml_nelements(t));
        for (auto & e : x) {
            e = uni(rng);
            if (kv_type == GGML_TYPE_F16 && (t == k || t == v)) {
                e = ggml_fp16_to_fp32(ggml_fp32_to_fp16(e)); // F16-exact, so both paths see one value
            }
        }
        ggml_backend_tensor_set(t, x.data(), 0, x.size()*sizeof(float));
    }
    // causal over the LAST n positions, the way a chunk sits after its context
    std::vector<ggml_fp16_t> mm(ggml_nelements(m));
    for (int64_t i = 0; i < m->ne[1]; ++i) {
        for (int64_t j = 0; j < s.m; ++j) {
            mm[i*s.m + j] = ggml_fp32_to_fp16(i < s.n && j <= s.m - s.n + i ? 0.0f : -INFINITY);
        }
    }
    ggml_backend_tensor_set(m, mm.data(), 0, mm.size()*sizeof(ggml_fp16_t));
    ggml_graph_reset(gf);
    GGML_ASSERT(ggml_backend_graph_compute(be, gf) == GGML_STATUS_SUCCESS);
    ggml_tensor * ps[3] = { q, k, v };
    for (int i = 0; i < 3; ++i) {
        ggml_tensor * gr = ggml_graph_get_grad(gf, ps[i]);
        std::vector<char> raw(ggml_nbytes(gr));
        ggml_backend_tensor_get(gr, raw.data(), 0, raw.size());
        g[i].clear();
        for (int64_t i2 = 0; i2 < gr->ne[2]; ++i2) {
            for (int64_t i1 = 0; i1 < gr->ne[1]; ++i1) {
                for (int64_t i0 = 0; i0 < gr->ne[0]; ++i0) {
                    g[i].push_back(*(const float *) (raw.data() + i0*gr->nb[0] + i1*gr->nb[1] + i2*gr->nb[2]));
                }
            }
        }
    }
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return true;
}

int main() {
    ggml_backend_load_all();
    const shape shapes[] = {
        {  64,  6, 2,  300, 37,  64 }, // several tiles, a ragged last one, grouped heads
        { 256, 24, 4, 1000, 64, 256 }, // Qwen3.8-27B's attention layout
        {  64,  3, 1,   50, 50,  16 }, // early rows whose later tiles are wholly masked
        {  64,  4, 4,  200, 32, 512 }, // one tile, no grouping
    };
    const double max_rel = 1e-4;
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    int failures = 0;
    for (ggml_type kv : { GGML_TYPE_F32, GGML_TYPE_F16 }) {
        for (const shape & s : shapes) {
            std::vector<float> ref[3];
            GGML_ASSERT(grads(cpu, false, s, kv, ref));
            for (size_t di = 0; di < ggml_backend_dev_count(); ++di) {
                ggml_backend_dev_t dev = ggml_backend_dev_get(di);
                ggml_backend_t be = ggml_backend_dev_init(dev, nullptr);
                if (!be) {
                    continue;
                }
                std::vector<float> got[3];
                if (!grads(be, true, s, kv, got)) {
                    printf("  %-8s d=%lld h=%lld/%lld m=%lld n=%lld tile=%d %s: not supported\n", ggml_backend_name(be),
                           (long long) s.d, (long long) s.h, (long long) s.hkv, (long long) s.m, (long long) s.n, s.tile, ggml_type_name(kv));
                    ggml_backend_free(be);
                    continue;
                }
                double worst = 0.0;
                for (int i = 0; i < 3; ++i) {
                    double md = 0.0, mr = 0.0;
                    for (size_t j = 0; j < ref[i].size(); ++j) {
                        md = std::fmax(md, std::fabs(got[i][j] - ref[i][j]));
                        mr = std::fmax(mr, std::fabs(ref[i][j]));
                    }
                    worst = std::fmax(worst, md / mr);
                }
                const bool ok = worst < max_rel;
                printf("  %-8s d=%lld h=%lld/%lld m=%lld n=%lld tile=%d %s: max rel %.2e %s\n", ggml_backend_name(be),
                       (long long) s.d, (long long) s.h, (long long) s.hkv, (long long) s.m, (long long) s.n, s.tile,
                       ggml_type_name(kv), worst, ok ? "OK" : "FAILED");
                failures += ok ? 0 : 1;
                ggml_backend_free(be);
            }
        }
    }
    ggml_backend_free(cpu);
    printf("%s\n", failures == 0 ? "OK" : "FAILED");
    return failures == 0 ? 0 : 1;
}
