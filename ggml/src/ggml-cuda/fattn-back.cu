#include "fattn-back.cuh"
#include "dequantize.cuh"

#include <algorithm>
#include <climits>

struct fattn_back_args {
    const char * q;
    const char * k;
    const char * v;
    const char * mask;
    const char * d;
    float * dq;
    float * dk;
    float * dv;
    float * lse;
    float * delta;
    size_t nbq[3], nbk[3], nbv[3], nbm[3], nbd[3];
    int D, Dv, N, M, H, Hkv, Hm;
    float scale;
};

template <ggml_type type>
static __device__ __forceinline__ float fattn_back_load(const char * row, int c) {
    if constexpr (type == GGML_TYPE_F32) {
        return ((const float *) row)[c];
    } else if constexpr (type == GGML_TYPE_F16) {
        return __half2float(((const half *) row)[c]);
    } else {
        float2 values;
        dequantize_q8_0(row, c / QK8_0, (c % QK8_0) & ~1, values);
        return (c & 1) ? values.y : values.x;
    }
}

static __device__ __forceinline__ float fattn_back_mask(const fattn_back_args & a, int j, int i, int h) {
    return a.mask ? __half2float(*(const half *) (a.mask + j*a.nbm[0] + i*a.nbm[1] + (h % a.Hm)*a.nbm[2])) : 0.0f;
}

template <ggml_type TK, ggml_type TV>
static __global__ void fattn_back_q(fattn_back_args a) {
    extern __shared__ float storage[];
    float * q = storage;
    float * d = q + a.D;
    float * pk = d + a.Dv;
    float * pdk = pk + a.D;
    const int lane = threadIdx.x;
    const int i = blockIdx.x % a.N;
    const int h = blockIdx.x / a.N;
    const int hk = h / (a.H / a.Hkv);
    for (int c = lane; c < a.D; c += WARP_SIZE) {
        q[c] = *(const float *) (a.q + c*a.nbq[0] + i*a.nbq[1] + h*a.nbq[2]);
        pk[c] = pdk[c] = 0.0f;
    }
    for (int c = lane; c < a.Dv; c += WARP_SIZE) {
        d[c] = *(const float *) (a.d + c*a.nbd[0] + h*a.nbd[1] + i*a.nbd[2]);
    }

    float maximum = -INFINITY;
    float sum = 0.0f;
    for (int j = 0; j < a.M; ++j) {
        const float mask = fattn_back_mask(a, j, i, h);
        if (mask == -INFINITY) { continue; }
        const char * k = a.k + j*a.nbk[1] + hk*a.nbk[2];
        float s = 0.0f;
        for (int c = lane; c < a.D; c += WARP_SIZE) {
            s += q[c]*fattn_back_load<TK>(k, c);
        }
        s = warp_reduce_sum(s)*a.scale + mask;
        if (s == -INFINITY) { continue; }
        const float next = fmaxf(maximum, s);
        sum = sum*expf(maximum - next) + expf(s - next);
        maximum = next;
    }
    const float lse = maximum == -INFINITY ? -INFINITY : maximum + logf(sum);
    float delta = 0.0f;
    if (lse != -INFINITY) {
        for (int j = 0; j < a.M; ++j) {
            const float mask = fattn_back_mask(a, j, i, h);
            if (mask == -INFINITY) { continue; }
            const char * k = a.k + j*a.nbk[1] + hk*a.nbk[2];
            float s = 0.0f;
            for (int c = lane; c < a.D; c += WARP_SIZE) {
                s += q[c]*fattn_back_load<TK>(k, c);
            }
            const float p = expf(warp_reduce_sum(s)*a.scale + mask - lse);
            if (p == 0.0f) { continue; }
            const char * v = a.v + j*a.nbv[1] + hk*a.nbv[2];
            float dp = 0.0f;
            for (int c = lane; c < a.Dv; c += WARP_SIZE) {
                dp += d[c]*fattn_back_load<TV>(v, c);
            }
            dp = warp_reduce_sum(dp);
            delta += p*dp;
            for (int c = lane; c < a.D; c += WARP_SIZE) {
                const float kc = fattn_back_load<TK>(k, c);
                pk[c] += p*kc;
                pdk[c] += p*dp*kc;
            }
        }
    }
    for (int c = lane; c < a.D; c += WARP_SIZE) {
        a.dq[(size_t) blockIdx.x*a.D + c] = a.scale*(pdk[c] - delta*pk[c]);
    }
    if (lane == 0) {
        a.lse[blockIdx.x] = lse;
        a.delta[blockIdx.x] = delta;
    }
}

template <ggml_type TK, ggml_type TV>
static __global__ void fattn_back_kv(fattn_back_args a) {
    extern __shared__ float storage[];
    float * k = storage;
    float * v = k + a.D;
    float * dk = v + a.Dv;
    float * dv = dk + a.D;
    const int lane = threadIdx.x;
    const int j = blockIdx.x % a.M;
    const int hk = blockIdx.x / a.M;
    const int group = a.H / a.Hkv;
    for (int c = lane; c < a.D; c += WARP_SIZE) {
        k[c] = fattn_back_load<TK>(a.k + j*a.nbk[1] + hk*a.nbk[2], c);
        dk[c] = 0.0f;
    }
    for (int c = lane; c < a.Dv; c += WARP_SIZE) {
        v[c] = fattn_back_load<TV>(a.v + j*a.nbv[1] + hk*a.nbv[2], c);
        dv[c] = 0.0f;
    }
    for (int h = hk*group; h < (hk + 1)*group; ++h) {
        for (int i = 0; i < a.N; ++i) {
            const float lse = a.lse[(size_t) h*a.N + i];
            const float mask = fattn_back_mask(a, j, i, h);
            if (lse == -INFINITY || mask == -INFINITY) { continue; }
            const char * q = a.q + i*a.nbq[1] + h*a.nbq[2];
            const char * d = a.d + h*a.nbd[1] + i*a.nbd[2];
            float s = 0.0f;
            for (int c = lane; c < a.D; c += WARP_SIZE) {
                s += *(const float *) (q + c*a.nbq[0])*k[c];
            }
            const float p = expf(warp_reduce_sum(s)*a.scale + mask - lse);
            if (p == 0.0f) { continue; }
            float dp = 0.0f;
            for (int c = lane; c < a.Dv; c += WARP_SIZE) {
                dp += *(const float *) (d + c*a.nbd[0])*v[c];
            }
            const float ds = p*(warp_reduce_sum(dp) - a.delta[(size_t) h*a.N + i])*a.scale;
            for (int c = lane; c < a.D; c += WARP_SIZE) {
                dk[c] += ds * *(const float *) (q + c*a.nbq[0]);
            }
            for (int c = lane; c < a.Dv; c += WARP_SIZE) {
                dv[c] += p * *(const float *) (d + c*a.nbd[0]);
            }
        }
    }
    for (int c = lane; c < a.D; c += WARP_SIZE) {
        a.dk[(size_t) blockIdx.x*a.D + c] = dk[c];
    }
    for (int c = lane; c < a.Dv; c += WARP_SIZE) {
        a.dv[(size_t) blockIdx.x*a.Dv + c] = dv[c];
    }
}

static bool fattn_back_type(ggml_type type) {
    return type == GGML_TYPE_F32 || type == GGML_TYPE_F16 || type == GGML_TYPE_Q8_0;
}

bool ggml_cuda_flash_attn_back_supported(int device, const ggml_tensor * dst) {
    const ggml_tensor * q = dst->src[0], * k = dst->src[1], * v = dst->src[2], * mask = dst->src[3], * d = dst->src[4];
    if (q->type != GGML_TYPE_F32 || d->type != GGML_TYPE_F32 || !fattn_back_type(k->type) || !fattn_back_type(v->type) ||
        k->nb[0] != ggml_type_size(k->type) || v->nb[0] != ggml_type_size(v->type) ||
        (mask && (mask->type != GGML_TYPE_F16 || mask->nb[0] != sizeof(half) || mask->ne[3] != 1))) {
        return false;
    }
    for (const ggml_tensor * t : {q, k, v, d}) {
        if (t->ne[3] != 1) { return false; }
        for (int axis = 0; axis < 3; ++axis) {
            if (t->ne[axis] <= 0 || t->ne[axis] > INT_MAX) { return false; }
        }
    }
    const auto & info = ggml_cuda_info().devices[device];
    const size_t shared = std::max(3*(size_t) q->ne[0] + v->ne[0], 2*(size_t) q->ne[0] + 2*(size_t) v->ne[0])*sizeof(float);
    return info.warp_size == WARP_SIZE && info.max_threads_per_block >= WARP_SIZE && shared <= info.smpb &&
        q->ne[1] <= info.max_grid_dim_x/q->ne[2] && k->ne[1] <= info.max_grid_dim_x/k->ne[2] &&
        (!mask || (mask->ne[2] == 1 || mask->ne[2] == q->ne[2]));
}

template <ggml_type TK, ggml_type TV>
static void fattn_back_launch(ggml_backend_cuda_context & ctx, const fattn_back_args & a) {
    fattn_back_q<TK, TV><<<a.N*a.H, WARP_SIZE, (3*(size_t) a.D + a.Dv)*sizeof(float), ctx.stream()>>>(a);
    CUDA_CHECK(cudaGetLastError());
    fattn_back_kv<TK, TV><<<a.M*a.Hkv, WARP_SIZE, (2*(size_t) a.D + 2*(size_t) a.Dv)*sizeof(float), ctx.stream()>>>(a);
    CUDA_CHECK(cudaGetLastError());
}

template <ggml_type TK>
static void fattn_back_dispatch(ggml_backend_cuda_context & ctx, const fattn_back_args & a, ggml_type tv) {
    switch (tv) {
        case GGML_TYPE_F32: fattn_back_launch<TK, GGML_TYPE_F32>(ctx, a); break;
        case GGML_TYPE_F16: fattn_back_launch<TK, GGML_TYPE_F16>(ctx, a); break;
        case GGML_TYPE_Q8_0: fattn_back_launch<TK, GGML_TYPE_Q8_0>(ctx, a); break;
        default: GGML_ABORT("unsupported flash attention backward V type");
    }
}

void ggml_cuda_flash_attn_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * q = dst->src[0], * k = dst->src[1], * v = dst->src[2], * mask = dst->src[3], * d = dst->src[4];
    fattn_back_args a = {};
    a.q = (const char *) q->data; a.k = (const char *) k->data; a.v = (const char *) v->data;
    a.mask = mask ? (const char *) mask->data : nullptr; a.d = (const char *) d->data;
    a.dq = (float *) ((char *) dst->data + ggml_flash_attn_back_offset(dst, 0));
    a.dk = (float *) ((char *) dst->data + ggml_flash_attn_back_offset(dst, 1));
    a.dv = (float *) ((char *) dst->data + ggml_flash_attn_back_offset(dst, 2));
    a.lse = (float *) ((char *) dst->data + ggml_flash_attn_back_offset(dst, 3));
    a.delta = a.lse + q->ne[1]*q->ne[2];
    for (int axis = 0; axis < 3; ++axis) {
        a.nbq[axis] = q->nb[axis]; a.nbk[axis] = k->nb[axis]; a.nbv[axis] = v->nb[axis]; a.nbd[axis] = d->nb[axis];
        a.nbm[axis] = mask ? mask->nb[axis] : 0;
    }
    a.D = q->ne[0]; a.Dv = v->ne[0]; a.N = q->ne[1]; a.M = k->ne[1]; a.H = q->ne[2]; a.Hkv = k->ne[2];
    a.Hm = mask ? mask->ne[2] : 1;
    a.scale = ggml_get_op_params_f32(dst, 0);
    switch (k->type) {
        case GGML_TYPE_F32: fattn_back_dispatch<GGML_TYPE_F32>(ctx, a, v->type); break;
        case GGML_TYPE_F16: fattn_back_dispatch<GGML_TYPE_F16>(ctx, a, v->type); break;
        case GGML_TYPE_Q8_0: fattn_back_dispatch<GGML_TYPE_Q8_0>(ctx, a, v->type); break;
        default: GGML_ABORT("unsupported flash attention backward K type");
    }
}
