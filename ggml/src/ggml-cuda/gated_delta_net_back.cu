#include "gated_delta_net_back.cuh"
#include "ggml-cuda/common.cuh"

// The backward of the gated delta rule, one CUDA block per (sequence, head), in the math of
// the CPU reference (ggml-cpu/ops.cpp ggml_compute_forward_gated_delta_net_back_f32), with
// the state kept transposed as M[j][i] = S[i][j]:
//
//   S'  = a (.) S_prev          a = exp(g), a scalar or, for KDA, per row i
//   u   = S'^T k,  delta = beta (v - u),  S = S' + k delta^T,  o = scale S^T q
//
// run in reverse carrying dS. Thread j owns row j of M / M' / dM, so every row-local sum
// (u_j, dd_j, the M update) is a plain loop over i in that thread; the column sums the
// backward needs (dq_i, dk_i, the KDA dg_i) go through shared accumulators with each thread
// adding at a diagonal offset (i = j + s mod S_v at step s), so no two threads touch one
// address in the same step. The forward states are recomputed and STORED in a pool buffer
// ((T + 1) x S_v^2 per block): the backward needs every S and S', and inverting the decay
// would divide by exp(g). dq and dk are accumulated with atomics because q and k are shared
// across the heads that broadcast over them.
//
// First cut: S_v <= 128 (M' and dM live in dynamic shared memory, 2 x S_v^2 floats); the
// stored states cost (T + 1) x S_v^2 x 4 B per (sequence, head) — 67 MB at T = 1024,
// S_v = 128 — and recomputing in segments is the follow-on if that transient matters.

template <int S_v>
__global__ void __launch_bounds__(S_v, 1)
gated_delta_net_back_cuda(
        const float * __restrict__ q,     const float * __restrict__ k,     const float * __restrict__ v,
        const float * __restrict__ g,     const float * __restrict__ beta,  const float * __restrict__ s0,
        const float * __restrict__ grad,  float * __restrict__ states,
        float * __restrict__ dq,  float * __restrict__ dk,  float * __restrict__ dv,
        float * __restrict__ dg,  float * __restrict__ db,  float * __restrict__ ds,
        const int64_t H, const int64_t T, const int64_t n_seqs,
        const int64_t neq1, const int64_t neq3, const int64_t nek1, const int64_t nek3,
        const int64_t sq1, const int64_t sq2, const int64_t sq3,
        const int64_t sk1, const int64_t sk2, const int64_t sk3,
        const int64_t sv1, const int64_t sv2, const int64_t sv3,
        const int      kda, const int K, const float scale) {
    constexpr int SS = S_v * S_v;

    extern __shared__ float smem[];
    float * Mp = smem;        // S' for the current step, row j at Mp + j*S_v
    float * dM = smem + SS;   // dS carried backward

    __shared__ float a_s[S_v], q_s[S_v], k_s[S_v], v_s[S_v], gy_s[S_v];
    __shared__ float delta_s[S_v], u_s[S_v], dd_s[S_v], du_s[S_v];
    __shared__ float acc_dq[S_v], acc_dk[S_v], acc_dg[S_v];
    __shared__ float red[S_v];

    const int     j    = threadIdx.x;
    const int64_t iv1  = blockIdx.x;  // head
    const int64_t iv3  = blockIdx.y;  // sequence
    const int64_t rq3  = n_seqs / neq3;
    const int64_t rk3  = n_seqs / nek3;
    const int64_t neg0 = kda ? S_v : 1;

    // this block's slice of the stored states: (T + 1) states of SS floats
    states += (iv3 * H + iv1) * (T + 1) * SS;

    const float * s0_blk = s0 + iv3 * H * SS + iv1 * SS;
    const int64_t snap   = (int64_t) SS * H * n_seqs;
    const float * gy_all = grad;
    const float * gs_all = grad + (int64_t) S_v * H * T * n_seqs;

    auto q_at = [&](int64_t t) { return q + (iv3 / rq3) * sq3 + t * sq2 + (iv1 % neq1) * sq1; };
    auto k_at = [&](int64_t t) { return k + (iv3 / rk3) * sk3 + t * sk2 + (iv1 % nek1) * sk1; };
    auto v_at = [&](int64_t t) { return v + iv3 * sv3 + t * sv2 + iv1 * sv1; };
    auto g_at = [&](int64_t t) { return g + (iv3 * T + t) * H * neg0 + iv1 * neg0; };
    auto b_at = [&](int64_t t) { return beta[(iv3 * T + t) * H + iv1]; };

    // block-wide sum of one float per thread, result valid in every thread
    auto block_sum = [&](float x) {
        red[j] = x;
        __syncthreads();
        for (int stride = S_v / 2; stride > 0; stride >>= 1) {
            if (j < stride) {
                red[j] += red[j + stride];
            }
            __syncthreads();
        }
        const float r = red[0];
        __syncthreads();
        return r;
    };

    // ---- forward, storing every state ------------------------------------------------
    for (int i = 0; i < S_v; ++i) {
        states[j * S_v + i] = s0_blk[j * S_v + i];
    }
    for (int64_t t = 0; t < T; ++t) {
        const float * g_t = g_at(t);
        a_s[j] = expf(kda ? g_t[j] : g_t[0]);
        k_s[j] = k_at(t)[j];
        v_s[j] = v_at(t)[j];
        __syncthreads();
        const float   b     = b_at(t);
        const float * Mprev = states + t * SS + j * S_v;
        float       * Mnext = states + (t + 1) * SS + j * S_v;
        float u = 0.0f;
        for (int i = 0; i < S_v; ++i) {
            u += Mprev[i] * a_s[i] * k_s[i];
        }
        const float delta = (v_s[j] - u) * b;
        for (int i = 0; i < S_v; ++i) {
            Mnext[i] = Mprev[i] * a_s[i] + k_s[i] * delta;
        }
        __syncthreads();
    }

    // ---- backward, carrying dM ---------------------------------------------------------
    for (int i = 0; i < S_v; ++i) {
        dM[j * S_v + i] = 0.0f;
    }
    __syncthreads();

    for (int64_t t = T - 1; t >= 0; --t) {
        const float * g_t = g_at(t);
        a_s[j]  = expf(kda ? g_t[j] : g_t[0]);
        q_s[j]  = q_at(t)[j];
        k_s[j]  = k_at(t)[j];
        v_s[j]  = v_at(t)[j];
        gy_s[j] = gy_all[((iv3 * T + t) * H + iv1) * S_v + j];
        acc_dq[j] = 0.0f;
        acc_dk[j] = 0.0f;
        acc_dg[j] = 0.0f;
        __syncthreads();

        const float   b     = b_at(t);
        float       * dMj   = dM + j * S_v;
        float       * Mpj   = Mp + j * S_v;
        const float * M1j   = states + (t + 1) * SS + j * S_v;  // S after token t
        const float * M0j   = states + t * SS + j * S_v;        // S before token t

        // a snapshot slot that captured the state after token t
        const int64_t slot = T - 1 - t;
        if (slot < K) {
            const float * gs = gs_all + slot * snap + (iv3 * H + iv1) * SS + j * S_v;
            for (int i = 0; i < S_v; ++i) {
                dMj[i] += gs[i];
            }
        }

        // o = scale S^T q:  dq_i += scale sum_j M[j][i] gy_j ;  dM[j][i] += scale gy_j q_i
        {
            const float gyj = gy_s[j];
            for (int s = 0; s < S_v; ++s) {
                const int i = (j + s) & (S_v - 1);
                atomicAdd(&acc_dq[i], M1j[i] * gyj);
                dMj[i] += scale * gyj * q_s[i];
            }
        }

        // recompute S', u, delta for this step
        float u = 0.0f;
        for (int i = 0; i < S_v; ++i) {
            const float mp = M0j[i] * a_s[i];
            Mpj[i] = mp;
            u += mp * k_s[i];
        }
        u_s[j]     = u;
        delta_s[j] = (v_s[j] - u) * b;
        __syncthreads();

        // S = S' + k delta^T:  dd_j = sum_i dM[j][i] k_i ;  dk_i += sum_j dM[j][i] delta_j
        {
            float dd = 0.0f;
            const float dlj = delta_s[j];
            for (int s = 0; s < S_v; ++s) {
                const int i = (j + s) & (S_v - 1);
                dd += dMj[i] * k_s[i];
                atomicAdd(&acc_dk[i], dMj[i] * dlj);
            }
            dd_s[j] = dd;
            // delta = beta (v - u):  dv_j = beta dd_j ;  du_j = -beta dd_j
            dv[((iv3 * T + t) * H + iv1) * S_v + j] = b * dd;
            du_s[j] = -b * dd;
        }
        __syncthreads();

        // dbeta = sum_j dd_j (v_j - u_j)
        {
            const float dbeta = block_sum(dd_s[j] * (v_s[j] - u_s[j]));
            if (j == 0) {
                db[(iv3 * T + t) * H + iv1] = dbeta;
            }
        }

        // u = S'^T k:  dM[j][i] += k_i du_j ;  dk_i += sum_j S'[j][i] du_j
        {
            const float duj = du_s[j];
            for (int s = 0; s < S_v; ++s) {
                const int i = (j + s) & (S_v - 1);
                dMj[i] += k_s[i] * duj;
                atomicAdd(&acc_dk[i], Mpj[i] * duj);
            }
        }

        // S' = a (.) S_prev:  dg = sum dS' (.) S' (per row i for KDA, one scalar otherwise),
        // then dS_prev = a (.) dS'
        {
            float dg_local = 0.0f;
            for (int s = 0; s < S_v; ++s) {
                const int i = (j + s) & (S_v - 1);
                const float c = dMj[i] * Mpj[i];
                if (kda) {
                    atomicAdd(&acc_dg[i], c);
                } else {
                    dg_local += c;
                }
                dMj[i] *= a_s[i];
            }
            if (!kda) {
                const float dg0 = block_sum(dg_local);
                if (j == 0) {
                    dg[(iv3 * T + t) * H + iv1] = dg0;
                }
            }
        }
        __syncthreads();

        // the column sums of this step: dq/dk are shared across broadcast heads, dg is ours
        atomicAdd(&dq[(((iv3 / rq3) * T + t) * neq1 + iv1 % neq1) * S_v + j], scale * acc_dq[j]);
        atomicAdd(&dk[(((iv3 / rk3) * T + t) * nek1 + iv1 % nek1) * S_v + j], acc_dk[j]);
        if (kda) {
            dg[((iv3 * T + t) * H + iv1) * S_v + j] = acc_dg[j];
        }
        __syncthreads();
    }

    // the initial state's gradient
    for (int i = 0; i < S_v; ++i) {
        ds[(iv3 * H + iv1) * SS + j * S_v + i] = dM[j * S_v + i];
    }
}

bool ggml_cuda_gated_delta_net_back_supported(const ggml_tensor * dst) {
    const ggml_tensor * q = dst->src[0], * k = dst->src[1], * v = dst->src[2],
                      * g = dst->src[3], * b = dst->src[4], * s = dst->src[5], * gr = dst->src[6];
    for (const ggml_tensor * t : { q, k, v, g, b, s, gr }) {
        if (t == nullptr || t->type != GGML_TYPE_F32) {
            return false;
        }
    }
    const int64_t S_v = v->ne[0];
    if (S_v != 16 && S_v != 32 && S_v != 64 && S_v != 128) {
        return false;
    }
    if (q->ne[0] != S_v || k->ne[0] != S_v) {
        return false;
    }
    if (!(g->ne[0] == 1 || g->ne[0] == S_v)) {
        return false;
    }
    return dst->type == GGML_TYPE_F32 && ggml_is_contiguous(dst) &&
           ggml_is_contiguous_rows(q) && ggml_is_contiguous_rows(k) && ggml_is_contiguous_rows(v) &&
           ggml_is_contiguous(g) && ggml_is_contiguous(b) && ggml_is_contiguous(s) && ggml_is_contiguous(gr);
}

template <int S_v>
static void launch_gated_delta_net_back(
        const float * q, const float * k, const float * v, const float * g, const float * beta,
        const float * s0, const float * grad, float * states,
        float * dq, float * dk, float * dv, float * dg, float * db, float * ds,
        int64_t H, int64_t T, int64_t n_seqs,
        int64_t neq1, int64_t neq3, int64_t nek1, int64_t nek3,
        int64_t sq1, int64_t sq2, int64_t sq3, int64_t sk1, int64_t sk2, int64_t sk3,
        int64_t sv1, int64_t sv2, int64_t sv3,
        int kda, int K, float scale, cudaStream_t stream) {
    constexpr size_t smem = 2 * (size_t) S_v * S_v * sizeof(float);
    CUDA_SET_SHARED_MEMORY_LIMIT((gated_delta_net_back_cuda<S_v>), smem);
    const dim3 grid((unsigned) H, (unsigned) n_seqs, 1);
    const dim3 block(S_v, 1, 1);
    const ggml_cuda_kernel_launch_params lp(grid, block, smem, stream);
    ggml_cuda_kernel_launch(gated_delta_net_back_cuda<S_v>, lp,
        q, k, v, g, beta, s0, grad, states, dq, dk, dv, dg, db, ds,
        H, T, n_seqs, neq1, neq3, nek1, nek3, sq1, sq2, sq3, sk1, sk2, sk3, sv1, sv2, sv3,
        kda, K, scale);
}

void ggml_cuda_op_gated_delta_net_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src_q = dst->src[0], * src_k = dst->src[1], * src_v = dst->src[2],
                      * src_g = dst->src[3], * src_b = dst->src[4], * src_s = dst->src[5], * src_gr = dst->src[6];
    GGML_ASSERT(ggml_cuda_gated_delta_net_back_supported(dst));

    const int64_t S_v    = src_v->ne[0];
    const int64_t H      = src_v->ne[1];
    const int64_t T      = src_v->ne[2];
    const int64_t n_seqs = src_v->ne[3];
    const int64_t neq1 = src_q->ne[1], neq3 = src_q->ne[3];
    const int64_t nek1 = src_k->ne[1], nek3 = src_k->ne[3];
    const int     kda  = src_g->ne[0] == S_v;
    const int     K    = ggml_get_op_params_i32(dst, 0);
    const float   scale = 1.0f / sqrtf((float) S_v);
    GGML_ASSERT(n_seqs % neq3 == 0 && n_seqs % nek3 == 0);
    GGML_ASSERT(H % neq1 == 0 && H % nek1 == 0);

    cudaStream_t stream = ctx.stream();

    // the packed output: [dq | dk | dv | dg | dbeta | dstate]; dq and dk accumulate
    float * dq = (float *) dst->data;
    float * dk = dq + ggml_nelements(src_q);
    float * dv = dk + ggml_nelements(src_k);
    float * dg = dv + ggml_nelements(src_v);
    float * db = dg + ggml_nelements(src_g);
    float * ds = db + ggml_nelements(src_b);
    CUDA_CHECK(cudaMemsetAsync(dst->data, 0, ggml_nbytes(dst), stream));

    // every state of the forward, per (sequence, head): (T + 1) x S_v^2 floats
    ggml_cuda_pool_alloc<float> states(ctx.pool(), (size_t) n_seqs * H * (T + 1) * S_v * S_v);

    const int64_t sq1 = src_q->nb[1] / sizeof(float), sq2 = src_q->nb[2] / sizeof(float), sq3 = src_q->nb[3] / sizeof(float);
    const int64_t sk1 = src_k->nb[1] / sizeof(float), sk2 = src_k->nb[2] / sizeof(float), sk3 = src_k->nb[3] / sizeof(float);
    const int64_t sv1 = src_v->nb[1] / sizeof(float), sv2 = src_v->nb[2] / sizeof(float), sv3 = src_v->nb[3] / sizeof(float);

#define GDN_BACK_LAUNCH(SV) \
    launch_gated_delta_net_back<SV>((const float *) src_q->data, (const float *) src_k->data, (const float *) src_v->data, \
        (const float *) src_g->data, (const float *) src_b->data, (const float *) src_s->data, (const float *) src_gr->data, \
        states.get(), dq, dk, dv, dg, db, ds, H, T, n_seqs, neq1, neq3, nek1, nek3, \
        sq1, sq2, sq3, sk1, sk2, sk3, sv1, sv2, sv3, kda, K, scale, stream)
    switch (S_v) {
        case 16:  GDN_BACK_LAUNCH(16);  break;
        case 32:  GDN_BACK_LAUNCH(32);  break;
        case 64:  GDN_BACK_LAUNCH(64);  break;
        case 128: GDN_BACK_LAUNCH(128); break;
        default:  GGML_ABORT("gated_delta_net_back: unsupported S_v");
    }
#undef GDN_BACK_LAUNCH
    CUDA_CHECK(cudaGetLastError());
}
