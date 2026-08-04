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
// table. n_tokens selects the kernel family on Metal: < 32 → mv_id (decode),
// >= 32 → mm_id (prefill). Returns the dst contents.
enum gather_mode { GM_OFF, GM_IDENTITY, GM_CROSS };

static ggml_backend_moe_gather_entry_t entry_proc(ggml_backend_t backend) {
    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
    return reg ? (ggml_backend_moe_gather_entry_t)
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_moe_gather_entry") : nullptr;
}

static graph_out run_case(ggml_backend_t backend, gather_mode mode, int64_t n_tokens, ggml_type wtype) {
    const bool gather = mode != GM_OFF;
    const int64_t ne00     = 256; // cols — a multiple of 256 so K-quant blocks (256) are legal too
    const int64_t ne01     = 32; // rows per expert
    const int64_t n_expert = 8;
    const int64_t n_used   = 2;

    graph_out out;

    ggml_init_params ip = {
        /*.mem_size   =*/ ggml_tensor_overhead() * 16 + ggml_graph_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * as  = ggml_new_tensor_3d(ctx, wtype, ne00, ne01, n_expert);
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
    // GM_CROSS puts the expert bytes in a SEPARATE backend buffer and points the table there — the
    // case a same-buffer identity table can never reach. The entry becomes the distance between two
    // independent allocations (multi-GB, often negative in a real serve), so any link in a backend's
    // chain narrower than 64 bits truncates and the kernel reads garbage.
    ggml_backend_buffer_t alt = nullptr;

    // deterministic fills. A QUANTIZED weight tensor is what routes the op to the quantized kernel
    // families (mmvq / MMQ on CUDA) — an F32 tensor only ever exercises mmf, so an F32-only test can
    // pass while the quantized path is broken. That is the whole point of sweeping wtype here.
    {
        std::vector<float> h((size_t) ne00*ne01*n_expert);
        uint32_t s = 42;
        for (auto & v : h) v = frand(s);
        if (wtype == GGML_TYPE_F32) {
            ggml_backend_tensor_set(as, h.data(), 0, h.size()*sizeof(float));
        } else {
            std::vector<uint8_t> q(ggml_nbytes(as));
            ggml_quantize_chunk(wtype, h.data(), q.data(), 0, ne01*n_expert, ne00, nullptr);
            ggml_backend_tensor_set(as, q.data(), 0, q.size());
        }
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
        std::vector<int64_t> h((size_t) n_expert);
        if (mode == GM_IDENTITY) {
            for (int64_t i = 0; i < n_expert; ++i) h[(size_t) i] = i*(int64_t) as->nb[2];
        } else {
            ggml_backend_moe_gather_entry_t proc = entry_proc(backend);
            if (proc == nullptr) { ggml_backend_buffer_free(buf); ggml_free(ctx); return out; }
            const size_t nb_as = ggml_nbytes(as);
            alt = ggml_backend_alloc_buffer(backend, nb_as);
            if (alt == nullptr) { ggml_backend_buffer_free(buf); ggml_free(ctx); return out; }
            std::vector<uint8_t> mirror(nb_as);
            ggml_backend_tensor_get(as, mirror.data(), 0, nb_as);
            ggml_tensor t = *as;
            t.buffer = alt;
            t.data   = ggml_backend_buffer_get_base(alt);
            ggml_backend_tensor_set(&t, mirror.data(), 0, nb_as);
            for (int64_t i = 0; i < n_expert; ++i) {
                if (!proc(as, alt, (size_t) i*as->nb[2], &h[(size_t) i])) {
                    ggml_backend_buffer_free(alt); ggml_backend_buffer_free(buf);
                    ggml_free(ctx); return out;
                }
            }
        }
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

    if (alt != nullptr) { ggml_backend_buffer_free(alt); }
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

        // Sweep BOTH kernel families (mv/decode vs mm/prefill) AND weight types. The type axis is the
        // one that matters most: quantized weights route to entirely different kernels (mmvq / MMQ)
        // than F32 (mmf), and a gather can be correct in one family while broken in the other.
        const struct { const char * label; int64_t n_tokens; ggml_type wt; } cases[] = {
            { "mv/f32",   4,  GGML_TYPE_F32  },
            { "mm/f32",   40, GGML_TYPE_F32  },
            { "mv/q8_0",  4,  GGML_TYPE_Q8_0 },
            { "mm/q8_0",  40, GGML_TYPE_Q8_0 },
            { "mv/q4_0",  4,  GGML_TYPE_Q4_0 },
            { "mm/q4_0",  40, GGML_TYPE_Q4_0 },
            { "mv/q4_K",  4,  GGML_TYPE_Q4_K },
            { "mm/q4_K",  40, GGML_TYPE_Q4_K },
        };

        bool failed = false;
        for (const auto & c : cases) {
            graph_out plain = run_case(backend, GM_OFF, c.n_tokens, c.wt);
            const struct { const char * tag; gather_mode m; } modes[] = {
                { "identity", GM_IDENTITY }, { "cross", GM_CROSS },
            };
            for (const auto & mo : modes) {
            graph_out gather = run_case(backend, mo.m, c.n_tokens, c.wt);

            if (!plain.ran) {
                printf("%-12s %-10s %-9s SKIP (plain mul_mat_id unsupported)\n", name, c.label, mo.tag);
            } else if (!gather.ran) {
                printf("%-12s %-10s %-9s SKIP (unsupported/entry-proc absent — designed partial rollout)\n", name, c.label, mo.tag);
            } else {
                const bool same = plain.data.size() == gather.data.size() &&
                    memcmp(plain.data.data(), gather.data.data(), plain.data.size()*sizeof(float)) == 0;
                printf("%-12s %-10s %-9s %s (%zu values)\n", name, c.label, mo.tag,
                       same ? "OK bit-identical" : "FAIL divergent", plain.data.size());
                if (!same) {
                    if (mo.m == GM_CROSS) {
                        printf("               ^ cross-allocation diverges while identity passes => the table\n"
                               "                 VALUE reaches the kernel wrong. Audit every link from the\n"
                               "                 gather-entry proc to the kernel address computation for a\n"
                               "                 type narrower than int64_t.\n");
                    }
                    failed = true;
                    break;
                }
                n_checked++;
            }
            if (failed) { break; }
            }
        }
        ggml_backend_free(backend);
        if (failed) {
            return 1;
        }
    }

    if (n_checked == 0) {
        printf("no backend accepted the gather op — nothing verified (not a failure)\n");
    }
    return 0;
}
