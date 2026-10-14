#include "fattn-back.cuh"
#include "dequantize.cuh"
#include "mma.cuh"
#if !defined(GGML_USE_HIP)
#include <mma.h>
#endif

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

struct fattn_back_matrix {
    const char * data;
    size_t row_stride, col_stride;
    int rows, cols;
    bool transpose;
};

template <ggml_type T>
static __device__ float fattn_back_element(const fattn_back_matrix & m, int row, int col) {
    if (row >= m.rows || col >= m.cols) { return 0.0f; }
    if (m.transpose) { const int tmp = row; row = col; col = tmp; }
    if constexpr (T == GGML_TYPE_F32) {
        return *(const float *) (m.data + row*m.row_stride + col*m.col_stride);
    } else {
        return fattn_back_load<T>(m.data + row*m.row_stride, col);
    }
}

using fattn_back_fragment = ggml_cuda_mma::tile<16, 8, float>;

// Split operands at the shared MMA boundary; small residuals must survive TF32 conversion.
static __device__ void fattn_back_mma(fattn_back_fragment & result,
        const ggml_cuda_mma::tile<16, 8, float> & a, const ggml_cuda_mma::tile<8, 8, float> & b) {
#ifdef AMPERE_MMA_AVAILABLE
    auto ah = a, al = a;
    auto bh = b, bl = b;
    for (int x = 0; x < a.ne; ++x) {
        ah.x[x] = nvcuda::wmma::__float_to_tf32(a.x[x]);
        al.x[x] = nvcuda::wmma::__float_to_tf32(a.x[x] - ah.x[x]);
    }
    for (int x = 0; x < b.ne; ++x) {
        bh.x[x] = nvcuda::wmma::__float_to_tf32(b.x[x]);
        bl.x[x] = nvcuda::wmma::__float_to_tf32(b.x[x] - bh.x[x]);
    }
    ggml_cuda_mma::mma(result, al, bl);
    ggml_cuda_mma::mma(result, ah, bl);
    ggml_cuda_mma::mma(result, al, bh);
    ggml_cuda_mma::mma(result, ah, bh);
#else
    GGML_UNUSED_VARS(result, a, b);
    NO_DEVICE_CODE;
#endif
}

template <ggml_type TA, ggml_type TB>
static __device__ fattn_back_fragment fattn_back_product(
        const fattn_back_matrix & a, const fattn_back_matrix & b, int row_a, int row_b, int count) {
    fattn_back_fragment result;
    for (int base = 0; base < count; base += 8) {
        ggml_cuda_mma::tile<16, 8, float> af;
        ggml_cuda_mma::tile<8, 8, float> bf;
        for (int x = 0; x < af.ne; ++x) {
            // TF32 A registers differ from the F32 accumulator tile mapping.
            const int row = threadIdx.x/4 + (x%2)*8;
            const int feature = threadIdx.x%4 + (x/2)*4;
            af.x[x] = fattn_back_element<TA>(a, row_a + row, base + feature);
        }
        for (int x = 0; x < bf.ne; ++x) {
            bf.x[x] = fattn_back_element<TB>(b, row_b + bf.get_i(x), base + bf.get_j(x));
        }
        fattn_back_mma(result, af, bf);
    }
    return result;
}

static __device__ void fattn_back_store_tile(const fattn_back_fragment & value, float * dst, int col) {
    for (int x = 0; x < value.ne; ++x) { dst[value.get_i(x)*32 + col + value.get_j(x)] = value.x[x]; }
}

template <ggml_type TK, ggml_type TV>
static __global__ void fattn_back_q(fattn_back_args a) {
    extern __shared__ float storage[];
    float * gradient = storage;
    float * scores = gradient + 16*a.D;
    float * products = scores + 16*32;
    double * maxima = (double *) (products + 16*32);
    double * sums = maxima + 16;
    double * deltas = sums + 16;
    const int thread = threadIdx.y*WARP_SIZE + threadIdx.x;
    const int blocks = (a.N + 15)/16;
    const int row = (blockIdx.x%blocks)*16;
    const int h = blockIdx.x/blocks;
    const int hk = h/(a.H/a.Hkv);
    const int col = threadIdx.y*8;
    const fattn_back_matrix q {a.q + h*a.nbq[2], a.nbq[1], a.nbq[0], a.N, a.D, false};
    const fattn_back_matrix d {a.d + h*a.nbd[1], a.nbd[2], a.nbd[0], a.N, a.Dv, false};
    const fattn_back_matrix k {a.k + hk*a.nbk[2], a.nbk[1], a.nbk[0], a.M, a.D, false};
    const fattn_back_matrix v {a.v + hk*a.nbv[2], a.nbv[1], a.nbv[0], a.M, a.Dv, false};
    for (int x = thread; x < 16*a.D; x += 4*WARP_SIZE) { gradient[x] = 0.0f; }
    if (thread < 16) { maxima[thread] = -INFINITY; sums[thread] = deltas[thread] = 0.0; }
    __syncthreads();
    // Accumulate normalized dP with the online maximum, then form dS in F32.
    for (int phase = 0; phase < 2; ++phase) {
        for (int base = 0; base < a.M; base += 32) {
            fattn_back_store_tile(fattn_back_product<GGML_TYPE_F32, TK>(q, k, row, base + col, a.D), scores, col);
            fattn_back_store_tile(fattn_back_product<GGML_TYPE_F32, TV>(d, v, row, base + col, a.Dv), products, col);
            __syncthreads();
            for (int r = threadIdx.y; r < 16; r += blockDim.y) {
                const int j = base + threadIdx.x;
                const int i = row + r;
                const float mask = j < a.M && i < a.N ? fattn_back_mask(a, j, i, h) : -INFINITY;
                const float score = mask == -INFINITY ? -INFINITY : scores[r*32 + threadIdx.x]*a.scale + mask;
                if (phase == 0) {
                    const double next = fmax(maxima[r], (double) warp_reduce_max(score));
                    const double weight = score == -INFINITY ? 0.0 : exp((double) score - next);
                    const double total = warp_reduce_sum(weight);
                    const double weighted = warp_reduce_sum(weight ? weight*products[r*32 + threadIdx.x] : 0.0);
                    if (threadIdx.x == 0 && next != -INFINITY) {
                        const double rescale = exp(maxima[r] - next);
                        sums[r] = sums[r]*rescale + total;
                        deltas[r] = deltas[r]*rescale + weighted;
                        maxima[r] = next;
                    }
                } else {
                    const size_t index = (size_t) h*a.N + i;
                    const float lse = i < a.N ? a.lse[index] : -INFINITY;
                    const float p = score == -INFINITY || lse == -INFINITY ? 0.0f : expf(score - lse);
                    scores[r*32 + threadIdx.x] = p ? p*(products[r*32 + threadIdx.x] - a.delta[index])*a.scale : 0.0f;
                }
            }
            __syncthreads();
            if (phase == 1) {
                const fattn_back_matrix ds {(const char *) scores, 32*sizeof(float), sizeof(float), 16, 32, false};
                const fattn_back_matrix kt {k.data + base*k.row_stride, k.row_stride, k.col_stride, a.D, min(32, a.M - base), true};
                for (int feature = col; feature < a.D; feature += 32) {
                    const auto g = fattn_back_product<GGML_TYPE_F32, TK>(ds, kt, 0, feature, 32);
                    for (int x = 0; x < g.ne; ++x) {
                        const int c = feature + g.get_j(x);
                        if (c < a.D) { gradient[g.get_i(x)*a.D + c] += g.x[x]; }
                    }
                }
            }
            __syncthreads();
        }
        if (phase == 0 && thread < 16 && row + thread < a.N) {
            const size_t index = (size_t) h*a.N + row + thread;
            const float lse = maxima[thread] == -INFINITY ? -INFINITY : (float) (maxima[thread] + log(sums[thread]));
            a.lse[index] = lse;
            a.delta[index] = lse == -INFINITY ? 0.0f : (float) (deltas[thread]*exp(maxima[thread] - (double) lse));
        }
        __syncthreads();
    }
    for (int x = thread; x < 16*a.D; x += 4*WARP_SIZE) {
        const int i = row + x/a.D;
        if (i < a.N) { a.dq[((size_t) h*a.N + i)*a.D + x%a.D] = gradient[x]; }
    }
}

template <ggml_type TK, ggml_type TV>
static __global__ void fattn_back_kv(fattn_back_args a) {
    extern __shared__ float storage[];
    float * dk = storage;
    float * dv = dk + 16*a.D;
    float * scores = dv + 16*a.Dv;
    float * products = scores + 16*32;
    const int thread = threadIdx.y*WARP_SIZE + threadIdx.x;
    const int blocks = (a.M + 15)/16;
    const int row = (blockIdx.x%blocks)*16;
    const int hk = blockIdx.x/blocks;
    const int col = threadIdx.y*8;
    const int group = a.H/a.Hkv;
    const fattn_back_matrix k {a.k + hk*a.nbk[2], a.nbk[1], a.nbk[0], a.M, a.D, false};
    const fattn_back_matrix v {a.v + hk*a.nbv[2], a.nbv[1], a.nbv[0], a.M, a.Dv, false};
    for (int x = thread; x < 16*(a.D + a.Dv); x += 4*WARP_SIZE) { storage[x] = 0.0f; }
    __syncthreads();
    for (int h = hk*group; h < (hk + 1)*group; ++h) {
        const fattn_back_matrix q {a.q + h*a.nbq[2], a.nbq[1], a.nbq[0], a.N, a.D, false};
        const fattn_back_matrix d {a.d + h*a.nbd[1], a.nbd[2], a.nbd[0], a.N, a.Dv, false};
        for (int base = 0; base < a.N; base += 32) {
            fattn_back_store_tile(fattn_back_product<TK, GGML_TYPE_F32>(k, q, row, base + col, a.D), scores, col);
            fattn_back_store_tile(fattn_back_product<TV, GGML_TYPE_F32>(v, d, row, base + col, a.Dv), products, col);
            __syncthreads();
            for (int x = thread; x < 16*32; x += 4*WARP_SIZE) {
                const int j = row + x/32, i = base + x%32;
                float p = 0.0f, ds = 0.0f;
                if (j < a.M && i < a.N) {
                    const float lse = a.lse[(size_t) h*a.N + i];
                    const float mask = fattn_back_mask(a, j, i, h);
                    if (lse != -INFINITY && mask != -INFINITY) { p = expf(scores[x]*a.scale + mask - lse); }
                    if (p) { ds = p*(products[x] - a.delta[(size_t) h*a.N + i])*a.scale; }
                }
                scores[x] = ds;
                products[x] = p;
            }
            __syncthreads();
            const fattn_back_matrix ds {(const char *) scores, 32*sizeof(float), sizeof(float), 16, 32, false};
            const fattn_back_matrix p {(const char *) products, 32*sizeof(float), sizeof(float), 16, 32, false};
            const fattn_back_matrix qt {q.data + base*q.row_stride, q.row_stride, q.col_stride, a.D, min(32, a.N - base), true};
            const fattn_back_matrix dt {d.data + base*d.row_stride, d.row_stride, d.col_stride, a.Dv, min(32, a.N - base), true};
            for (int feature = col; feature < a.D; feature += 32) {
                const auto g = fattn_back_product<GGML_TYPE_F32, GGML_TYPE_F32>(ds, qt, 0, feature, 32);
                for (int x = 0; x < g.ne; ++x) {
                    const int c = feature + g.get_j(x);
                    if (c < a.D) { dk[g.get_i(x)*a.D + c] += g.x[x]; }
                }
            }
            for (int feature = col; feature < a.Dv; feature += 32) {
                const auto g = fattn_back_product<GGML_TYPE_F32, GGML_TYPE_F32>(p, dt, 0, feature, 32);
                for (int x = 0; x < g.ne; ++x) {
                    const int c = feature + g.get_j(x);
                    if (c < a.Dv) { dv[g.get_i(x)*a.Dv + c] += g.x[x]; }
                }
            }
            __syncthreads();
        }
    }
    for (int x = thread; x < 16*a.D; x += 4*WARP_SIZE) {
        const int j = row + x/a.D;
        if (j < a.M) { a.dk[((size_t) hk*a.M + j)*a.D + x%a.D] = dk[x]; }
    }
    for (int x = thread; x < 16*a.Dv; x += 4*WARP_SIZE) {
        const int j = row + x/a.Dv;
        if (j < a.M) { a.dv[((size_t) hk*a.M + j)*a.Dv + x%a.Dv] = dv[x]; }
    }
}

static bool fattn_back_type(ggml_type type) {
    return type == GGML_TYPE_F32 || type == GGML_TYPE_F16 || type == GGML_TYPE_Q8_0;
}

struct fattn_back_config {
    int rows = 0;
    size_t shared_q = 0;
    size_t shared_kv = 0;
};

static fattn_back_config fattn_back_configuration(int device, size_t d, size_t dv) {
    const auto & info = ggml_cuda_info().devices[device];
    const size_t q = (16*d + 2*16*32)*sizeof(float) + 3*16*sizeof(double);
    const size_t kv = (16*(d + dv) + 2*16*32)*sizeof(float);
#if !defined(GGML_USE_HIP)
    if (info.cc >= GGML_CUDA_CC_AMPERE && info.max_threads_per_block >= 4*WARP_SIZE &&
        std::max(q, kv) <= info.smpb) { return {16, q, kv}; }
#endif
    return {};
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
    const auto config = fattn_back_configuration(device, q->ne[0], v->ne[0]);
    return info.warp_size == WARP_SIZE && config.rows > 0 &&
        q->ne[1] <= info.max_grid_dim_x/q->ne[2] && k->ne[1] <= info.max_grid_dim_x/k->ne[2] &&
        (!mask || (mask->ne[2] == 1 || mask->ne[2] == q->ne[2]));
}

template <ggml_type TK, ggml_type TV>
static void fattn_back_launch(ggml_backend_cuda_context & ctx, const fattn_back_args & a) {
    const auto config = fattn_back_configuration(ctx.device, a.D, a.Dv);
    const dim3 threads(WARP_SIZE, 4, 1);
    const size_t nq = ((size_t) a.N + config.rows - 1)/config.rows;
    const size_t nk = ((size_t) a.M + config.rows - 1)/config.rows;
    fattn_back_q<TK, TV><<<nq*a.H, threads, config.shared_q, ctx.stream()>>>(a);
    CUDA_CHECK(cudaGetLastError());
    fattn_back_kv<TK, TV><<<nk*a.Hkv, threads, config.shared_kv, ctx.stream()>>>(a);
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
