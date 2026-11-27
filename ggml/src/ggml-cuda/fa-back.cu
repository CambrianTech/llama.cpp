// FLASH_ATTN_BACK: the gradients of FLASH_ATTN_EXT, recomputing the scores instead of keeping them, so
// the memory is the gradients alone. A port of Metal's per-row path (kernels/fa_back.metal,
// kernel_flash_attn_back_{scale,q,kv}) and the CPU reference (ggml-cpu ops.cpp
// ggml_compute_forward_flash_attn_back_f32): one warp per row, each lane holding every 32nd element.
//   pass Q, per query row and head: the row's log-sum-exp over the window, then in one more sweep
//          D = sum_j p_j dp_j, A = sum_j p_j dp_j k_j and B = sum_j p_j k_j, so dq = scale (A - D B);
//          lse and D go to the scratch after dV for the KV pass.
//   pass KV, per key row j >= kv0 and KV head: dk and dv summed over the query rows of every head in
//          its group, from the lse and D the Q pass wrote. Keys before kv0 are a frozen prefix: no
//          gradient is computed or stored for them (ggml.h).
//   pass scale: the power of two that brings the largest |dO| to [4, 8), recorded after the scratch so
//          every backend's result agrees (ggml_flash_attn_back_scale); the per-row path computes in F32
//          and does not apply it.

#include "fa-back.cuh"

#define FA_BACK_MAX_D 512
#define FA_BACK_SCALE_NT 1024

struct fa_back_args {
    int32_t D, Dv, N, H, M, Hkv;
    size_t  nbq1, nbq2, nbk1, nbk2, nbv1, nbv2, nbd1, nbd2, nbm1, nbm2;
    int32_t nem2;
    bool    has_mask;
    float   scale;
    size_t  offs_dk, offs_dv, offs_lse;
    int32_t kv0;
};

template <typename T> static __device__ __forceinline__ float fa_back_elem(const char * row, int c);
template <> __device__ __forceinline__ float fa_back_elem<float>(const char * row, int c) { return ((const float *) row)[c]; }
template <> __device__ __forceinline__ float fa_back_elem<half>(const char * row, int c) { return __half2float(((const half *) row)[c]); }
template <> __device__ __forceinline__ float fa_back_elem<block_q8_0>(const char * row, int c) {
    const block_q8_0 * b = (const block_q8_0 *) row + c/QK8_0;
    return __half2float(b->d) * (float) b->qs[c%QK8_0];
}

static __device__ __forceinline__ float fa_back_mask(const fa_back_args & a, const char * mask, int j, int i, int h) {
    if (!a.has_mask) {
        return 0.0f;
    }
    return __half2float(((const half *) (mask + i*a.nbm1 + (h % a.nem2)*a.nbm2))[j]);
}

template <typename T>
static __global__ void fa_back_q(const fa_back_args a, const char * q, const char * k, const char * v, const char * mask,
        const char * d, char * dst) {
    const int lane = threadIdx.x;
    const int i  = blockIdx.x % a.N;
    const int h  = blockIdx.x / a.N;
    const int hk = h / (a.H / a.Hkv);
    const int nq = a.D  / WARP_SIZE;
    const int nv = a.Dv / WARP_SIZE;

    float qr[FA_BACK_MAX_D/WARP_SIZE], dr[FA_BACK_MAX_D/WARP_SIZE], ac[FA_BACK_MAX_D/WARP_SIZE], bc[FA_BACK_MAX_D/WARP_SIZE];
    for (int c = 0; c < nq; ++c) {
        qr[c] = ((const float *) (q + i*a.nbq1 + h*a.nbq2))[lane + WARP_SIZE*c];
        ac[c] = 0.0f;
        bc[c] = 0.0f;
    }
    for (int c = 0; c < nv; ++c) {
        dr[c] = ((const float *) (d + h*a.nbd1 + i*a.nbd2))[lane + WARP_SIZE*c];
    }

    // the log-sum-exp, online
    float m = -INFINITY;
    float l = 0.0f;
    for (int j = 0; j < a.M; ++j) {
        const float mj = fa_back_mask(a, mask, j, i, h);
        if (mj == -INFINITY) {
            continue;
        }
        const char * kr = k + j*a.nbk1 + hk*a.nbk2;
        float s = 0.0f;
        for (int c = 0; c < nq; ++c) {
            s += qr[c]*fa_back_elem<T>(kr, lane + WARP_SIZE*c);
        }
        s = warp_reduce_sum(s)*a.scale + mj;
        const float mn = fmaxf(m, s);
        l = l*expf(m - mn) + expf(s - mn);
        m = mn;
    }
    const float lse = m == -INFINITY ? -INFINITY : m + logf(l);

    // D, A and B in one sweep
    float dd = 0.0f;
    if (lse != -INFINITY) {
        for (int j = 0; j < a.M; ++j) {
            const float mj = fa_back_mask(a, mask, j, i, h);
            if (mj == -INFINITY) {
                continue;
            }
            const char * kr = k + j*a.nbk1 + hk*a.nbk2;
            const char * vr = v + j*a.nbv1 + hk*a.nbv2;
            float s = 0.0f;
            for (int c = 0; c < nq; ++c) {
                s += qr[c]*fa_back_elem<T>(kr, lane + WARP_SIZE*c);
            }
            float dp = 0.0f;
            for (int c = 0; c < nv; ++c) {
                dp += dr[c]*fa_back_elem<T>(vr, lane + WARP_SIZE*c);
            }
            const float p = expf(warp_reduce_sum(s)*a.scale + mj - lse);
            dp = warp_reduce_sum(dp);
            dd += p*dp;
            for (int c = 0; c < nq; ++c) {
                const float kc = fa_back_elem<T>(kr, lane + WARP_SIZE*c);
                ac[c] += p*dp*kc;
                bc[c] += p*kc;
            }
        }
    }

    float * dq = (float *) dst + ((size_t) h*a.N + i)*a.D;
    for (int c = 0; c < nq; ++c) {
        dq[lane + WARP_SIZE*c] = a.scale*(ac[c] - dd*bc[c]);
    }
    if (lane == 0) {
        float * lsev = (float *) (dst + a.offs_lse);
        lsev[h*a.N + i] = lse;
        lsev[a.N*a.H + h*a.N + i] = dd;
    }
}

template <typename T>
static __global__ void fa_back_kv(const fa_back_args a, const char * q, const char * k, const char * v, const char * mask,
        const char * d, char * dst) {
    const int lane = threadIdx.x;
    const int L  = a.M - a.kv0;
    const int j  = a.kv0 + blockIdx.x % L;
    const int hk = blockIdx.x / L;
    const int G  = a.H / a.Hkv;
    const int nq = a.D  / WARP_SIZE;
    const int nv = a.Dv / WARP_SIZE;

    float kr[FA_BACK_MAX_D/WARP_SIZE], vr[FA_BACK_MAX_D/WARP_SIZE], gk[FA_BACK_MAX_D/WARP_SIZE], gv[FA_BACK_MAX_D/WARP_SIZE];
    const char * kp = k + j*a.nbk1 + hk*a.nbk2;
    const char * vp = v + j*a.nbv1 + hk*a.nbv2;
    for (int c = 0; c < nq; ++c) {
        kr[c] = fa_back_elem<T>(kp, lane + WARP_SIZE*c);
        gk[c] = 0.0f;
    }
    for (int c = 0; c < nv; ++c) {
        vr[c] = fa_back_elem<T>(vp, lane + WARP_SIZE*c);
        gv[c] = 0.0f;
    }
    const float * lsev = (const float *) (dst + a.offs_lse);
    for (int h = hk*G; h < (hk + 1)*G; ++h) {
        for (int i = 0; i < a.N; ++i) {
            const float lse = lsev[h*a.N + i];
            const float mj  = fa_back_mask(a, mask, j, i, h);
            if (lse == -INFINITY || mj == -INFINITY) {
                continue;
            }
            const float * qp = (const float *) (q + i*a.nbq1 + h*a.nbq2);
            const float * dp = (const float *) (d + h*a.nbd1 + i*a.nbd2);
            float s = 0.0f;
            float t = 0.0f;
            for (int c = 0; c < nq; ++c) {
                s += qp[lane + WARP_SIZE*c]*kr[c];
            }
            for (int c = 0; c < nv; ++c) {
                t += dp[lane + WARP_SIZE*c]*vr[c];
            }
            const float p  = expf(warp_reduce_sum(s)*a.scale + mj - lse);
            const float ds = p*(warp_reduce_sum(t) - lsev[a.N*a.H + h*a.N + i])*a.scale;
            for (int c = 0; c < nq; ++c) {
                gk[c] += ds*qp[lane + WARP_SIZE*c];
            }
            for (int c = 0; c < nv; ++c) {
                gv[c] += p*dp[lane + WARP_SIZE*c];
            }
        }
    }
    float * dk = (float *) (dst + a.offs_dk) + ((size_t) hk*L + j - a.kv0)*a.D;
    float * dv = (float *) (dst + a.offs_dv) + ((size_t) hk*L + j - a.kv0)*a.Dv;
    for (int c = 0; c < nq; ++c) {
        dk[lane + WARP_SIZE*c] = gk[c];
    }
    for (int c = 0; c < nv; ++c) {
        dv[lane + WARP_SIZE*c] = gv[c];
    }
}

// the dO scale every backend records (ggml_flash_attn_back_scale): one block
static __global__ void fa_back_scale(const fa_back_args a, const char * d, char * dst) {
    __shared__ float smax[FA_BACK_SCALE_NT/WARP_SIZE];
    float mx = 0.0f;
    for (int r = 0; r < a.N*a.H; ++r) {
        const float * row = (const float *) (d + (r % a.H)*a.nbd1 + (r / a.H)*a.nbd2);
        for (int c = threadIdx.x; c < a.Dv; c += FA_BACK_SCALE_NT) {
            mx = fmaxf(mx, fabsf(row[c]));
        }
    }
    mx = warp_reduce_max(mx);
    if (threadIdx.x % WARP_SIZE == 0) {
        smax[threadIdx.x / WARP_SIZE] = mx;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int s = 1; s < FA_BACK_SCALE_NT/WARP_SIZE; ++s) {
            mx = fmaxf(mx, smax[s]);
        }
        float scale = 1.0f;
        if (mx > 0.0f) {
            int e;
            frexpf(mx, &e);
            scale = ldexpf(1.0f, min(max(3 - e, -24), 40));
        }
        ((float *) (dst + a.offs_lse))[2*a.N*a.H] = scale;
    }
}

bool ggml_cuda_flash_attn_back_supported(const ggml_tensor * op) {
    // one warp per row: head sizes a multiple of 32, at most FA_BACK_MAX_D (Metal's rule exactly, so a
    // node that has either native kernel takes the same graph)
    const ggml_tensor * q = op->src[0];
    const ggml_tensor * k = op->src[1];
    const ggml_tensor * v = op->src[2];
    const ggml_tensor * mask = op->src[3];
    const ggml_tensor * d = op->src[4];
    return q->type == GGML_TYPE_F32 && d->type == GGML_TYPE_F32 && k->type == v->type &&
           (k->type == GGML_TYPE_F32 || k->type == GGML_TYPE_F16 || k->type == GGML_TYPE_Q8_0) &&
           q->ne[0] % WARP_SIZE == 0 && v->ne[0] % WARP_SIZE == 0 &&
           q->ne[0] <= FA_BACK_MAX_D && v->ne[0] <= FA_BACK_MAX_D &&
           q->nb[0] == sizeof(float) && d->nb[0] == sizeof(float) &&
           (mask == nullptr || mask->type == GGML_TYPE_F16);
}

template <typename T>
static void fa_back_launch(const fa_back_args & a, const ggml_tensor * dst, cudaStream_t stream) {
    const char * q    = (const char *) dst->src[0]->data;
    const char * k    = (const char *) dst->src[1]->data;
    const char * v    = (const char *) dst->src[2]->data;
    const char * mask = dst->src[3] ? (const char *) dst->src[3]->data : q;
    const char * d    = (const char *) dst->src[4]->data;
    char * out        = (char *) dst->data;
    const int L = a.M - a.kv0;
    fa_back_scale<<<1, FA_BACK_SCALE_NT, 0, stream>>>(a, d, out);
    // Q writes each row's lse and D; KV reads them: same stream, in order
    fa_back_q<T><<<a.N*a.H, WARP_SIZE, 0, stream>>>(a, q, k, v, mask, d, out);
    if (L > 0) {
        fa_back_kv<T><<<L*a.Hkv, WARP_SIZE, 0, stream>>>(a, q, k, v, mask, d, out);
    }
}

void ggml_cuda_flash_attn_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * q    = dst->src[0];
    const ggml_tensor * k    = dst->src[1];
    const ggml_tensor * v    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    const ggml_tensor * d    = dst->src[4];
    const fa_back_args a = {
        /*.D        =*/ (int32_t) q->ne[0],
        /*.Dv       =*/ (int32_t) v->ne[0],
        /*.N        =*/ (int32_t) q->ne[1],
        /*.H        =*/ (int32_t) q->ne[2],
        /*.M        =*/ (int32_t) k->ne[1],
        /*.Hkv      =*/ (int32_t) k->ne[2],
        /*.nbq1     =*/ q->nb[1],
        /*.nbq2     =*/ q->nb[2],
        /*.nbk1     =*/ k->nb[1],
        /*.nbk2     =*/ k->nb[2],
        /*.nbv1     =*/ v->nb[1],
        /*.nbv2     =*/ v->nb[2],
        /*.nbd1     =*/ d->nb[1],
        /*.nbd2     =*/ d->nb[2],
        /*.nbm1     =*/ mask ? mask->nb[1] : 0,
        /*.nbm2     =*/ mask ? mask->nb[2] : 0,
        /*.nem2     =*/ mask ? (int32_t) mask->ne[2] : 1,
        /*.has_mask =*/ mask != nullptr,
        /*.scale    =*/ ggml_get_op_params_f32(dst, 0),
        /*.offs_dk  =*/ ggml_flash_attn_back_offset(dst, 1),
        /*.offs_dv  =*/ ggml_flash_attn_back_offset(dst, 2),
        /*.offs_lse =*/ ggml_flash_attn_back_offset(dst, 3),
        /*.kv0      =*/ ggml_get_op_params_i32(dst, 1),
    };
    switch (k->type) {
        case GGML_TYPE_F32:  fa_back_launch<float>(a, dst, ctx.stream());      break;
        case GGML_TYPE_F16:  fa_back_launch<half>(a, dst, ctx.stream());       break;
        case GGML_TYPE_Q8_0: fa_back_launch<block_q8_0>(a, dst, ctx.stream()); break;
        default: GGML_ABORT("FLASH_ATTN_BACK: K/V type %s is refused by ggml_cuda_flash_attn_back_supported", ggml_type_name(k->type));
    }
    CUDA_CHECK(cudaGetLastError());
}
