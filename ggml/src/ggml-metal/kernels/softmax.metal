#include "common.h"

template<typename T>
kernel void kernel_soft_max(
        constant ggml_metal_kargs_soft_max & args,
        device const  char * src0,
        device const  char * src1,
        device const  char * src2,
        device        char * dst,
        threadgroup  float * buf [[threadgroup(0)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint3  tptg[[threads_per_threadgroup]]) {
    const int32_t i03 = tgpig.z;
    const int32_t i02 = tgpig.y;
    const int32_t i01 = tgpig.x;

    const int32_t i13 = i03%args.ne13;
    const int32_t i12 = i02%args.ne12;
    const int32_t i11 = i01;

    device const float * psrc0 =                (device const float *) (src0 + i01*args.nb01 + i02*args.nb02 + i03*args.nb03);
    device const     T * pmask = src1 != src0 ? (device const T *    ) (src1 + i11*args.nb11 + i12*args.nb12 + i13*args.nb13) : nullptr;
    device const float * psrc2 = src2 != src0 ? (device const float *) (src2)                                                 : nullptr;
    device       float * pdst  =                (device       float *) (dst  + i01*args.nb1  + i02*args.nb2  + i03*args.nb3);

    float slope = 1.0f;

    // ALiBi
    if (args.max_bias > 0.0f) {
        const int32_t h = i02;

        const float base = h < args.n_head_log2 ? args.m0 : args.m1;
        const int   exp  = h < args.n_head_log2 ? h + 1 : 2*(h - args.n_head_log2) + 1;

        slope = pow(base, exp);
    }

    // parallel max
    float lmax = psrc2 ? psrc2[i02] : -INFINITY;

    for (int i00 = tpitg.x; i00 < args.ne00; i00 += tptg.x) {
        lmax = MAX(lmax, psrc0[i00]*args.scale + (pmask ? slope*pmask[i00] : 0.0f));
    }

    // find the max value in the block
    float max_val = simd_max(lmax);
    if (tptg.x > N_SIMDWIDTH) {
        if (sgitg == 0) {
            buf[tiisg] = -INFINITY;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiisg == 0) {
            buf[sgitg] = max_val;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        max_val = buf[tiisg];
        max_val = simd_max(max_val);
    }

    // parallel sum
    float lsum = 0.0f;
    for (int i00 = tpitg.x; i00 < args.ne00; i00 += tptg.x) {
        const float exp_psrc0 = exp((psrc0[i00]*args.scale + (pmask ? slope*pmask[i00] : 0.0f)) - max_val);
        lsum += exp_psrc0;
        pdst[i00] = exp_psrc0;
    }

    // This barrier fixes a failing test
    // ref: https://github.com/ggml-org/ggml/pull/621#discussion_r1425156335
    threadgroup_barrier(mem_flags::mem_none);

    float sum = simd_sum(lsum);

    if (tptg.x > N_SIMDWIDTH) {
        if (sgitg == 0) {
            buf[tiisg] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiisg == 0) {
            buf[sgitg] = sum;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        sum = buf[tiisg];
        sum = simd_sum(sum);
    }

    if (psrc2) {
        sum += exp(psrc2[i02] - max_val);
    }

    const float inv_sum = 1.0f/sum;

    for (int i00 = tpitg.x; i00 < args.ne00; i00 += tptg.x) {
        pdst[i00] *= inv_sum;
    }
}

template<typename T>
kernel void kernel_soft_max_4(
        constant ggml_metal_kargs_soft_max & args,
        device const  char * src0,
        device const  char * src1,
        device const  char * src2,
        device        char * dst,
        threadgroup  float * buf [[threadgroup(0)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint3  tptg[[threads_per_threadgroup]]) {
    const int32_t i03 = tgpig.z;
    const int32_t i02 = tgpig.y;
    const int32_t i01 = tgpig.x;

    const int32_t i13 = i03%args.ne13;
    const int32_t i12 = i02%args.ne12;
    const int32_t i11 = i01;

    device const float4 * psrc4 =                (device const float4 *) (src0 + i01*args.nb01 + i02*args.nb02 + i03*args.nb03);
    device const      T * pmask = src1 != src0 ? (device const T *     ) (src1 + i11*args.nb11 + i12*args.nb12 + i13*args.nb13) : nullptr;
    device const float *  psrc2 = src2 != src0 ? (device const float * ) (src2)                                                 : nullptr;
    device       float4 * pdst4 =                (device       float4 *) (dst  + i01*args.nb1  + i02*args.nb2  + i03*args.nb3);

    float slope = 1.0f;

    if (args.max_bias > 0.0f) {
        const int32_t h = i02;

        const float base = h < args.n_head_log2 ? args.m0 : args.m1;
        const int   exp  = h < args.n_head_log2 ? h + 1 : 2*(h - args.n_head_log2) + 1;

        slope = pow(base, exp);
    }

    // parallel max
    float4 lmax4 = psrc2 ? psrc2[i02] : -INFINITY;

    for (int i00 = tpitg.x; i00 < args.ne00/4; i00 += tptg.x) {
        lmax4 = fmax(lmax4, psrc4[i00]*args.scale + (float4)((pmask ? slope*pmask[i00] : 0.0f)));
    }

    const float lmax = MAX(MAX(lmax4[0], lmax4[1]), MAX(lmax4[2], lmax4[3]));

    float max_val = simd_max(lmax);
    if (tptg.x > N_SIMDWIDTH) {
        if (sgitg == 0) {
            buf[tiisg] = -INFINITY;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiisg == 0) {
            buf[sgitg] = max_val;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        max_val = buf[tiisg];
        max_val = simd_max(max_val);
    }

    // parallel sum
    float4 lsum4 = 0.0f;
    for (int i00 = tpitg.x; i00 < args.ne00/4; i00 += tptg.x) {
        const float4 exp_psrc4 = exp((psrc4[i00]*args.scale + (float4)((pmask ? slope*pmask[i00] : 0.0f))) - max_val);
        lsum4 += exp_psrc4;
        pdst4[i00] = exp_psrc4;
    }

    const float lsum = lsum4[0] + lsum4[1] + lsum4[2] + lsum4[3];

    // This barrier fixes a failing test
    // ref: https://github.com/ggml-org/ggml/pull/621#discussion_r1425156335
    threadgroup_barrier(mem_flags::mem_none);

    float sum = simd_sum(lsum);

    if (tptg.x > N_SIMDWIDTH) {
        if (sgitg == 0) {
            buf[tiisg] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiisg == 0) {
            buf[sgitg] = sum;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        sum = buf[tiisg];
        sum = simd_sum(sum);
    }

    if (psrc2) {
        sum += exp(psrc2[i02] - max_val);
    }

    const float inv_sum = 1.0f/sum;

    for (int i00 = tpitg.x; i00 < args.ne00/4; i00 += tptg.x) {
        pdst4[i00] *= inv_sum;
    }
}

typedef decltype(kernel_soft_max<float>)    kernel_soft_max_t;
typedef decltype(kernel_soft_max_4<float4>) kernel_soft_max_4_t;

template [[host_name("kernel_soft_max_f16")]]   kernel kernel_soft_max_t   kernel_soft_max<half>;
template [[host_name("kernel_soft_max_f32")]]   kernel kernel_soft_max_t   kernel_soft_max<float>;
template [[host_name("kernel_soft_max_f16_4")]] kernel kernel_soft_max_4_t kernel_soft_max_4<half4>;
template [[host_name("kernel_soft_max_f32_4")]] kernel kernel_soft_max_4_t kernel_soft_max_4<float4>;


// CROSS_ENTROPY_LOSS: one threadgroup per row; the row's loss goes to dst[row] (a scratch of
// nrows floats the allocator reserves after the scalar dst), already divided by nrows, and
// the existing sum kernel folds the rows into dst[0]. The back kernel writes the gradient
// row: (softmax(logits) - labels) * grad[0]/nrows. Both mirror the CPU reference exactly
// (ggml-cpu ops.cpp cross_entropy_loss{,_back}_f32) so the Metal backward of a training
// graph matches the CPU oracle (Continuum card 55f314b8, S3b).
template <typename F>
static inline float ce_reduce_max(float v, threadgroup float * buf, uint sgitg, uint tiisg, uint3 tptg) {
    float r = simd_max(v);
    if (tptg.x > N_SIMDWIDTH) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgitg == 0) {
            buf[tiisg] = -INFINITY;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tiisg == 0) {
            buf[sgitg] = r;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        r = simd_max(buf[tiisg]);
    }
    return r;
}

static inline float ce_reduce_sum(float v, threadgroup float * buf, uint sgitg, uint tiisg, uint3 tptg) {
    float r = simd_sum(v);
    if (tptg.x > N_SIMDWIDTH) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgitg == 0) {
            buf[tiisg] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tiisg == 0) {
            buf[sgitg] = r;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        r = simd_sum(buf[tiisg]);
    }
    return r;
}

kernel void kernel_cross_entropy_loss_f32(
        constant ggml_metal_kargs_cross_entropy_loss & args,
        device const  char * src0,
        device const  char * src1,
        device        char * dst,
        threadgroup  float * buf [[threadgroup(0)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint3  tptg[[threads_per_threadgroup]]) {
    const int32_t row = tgpig.x;

    device const float * s0 = (device const float *) (src0 + row*args.nb01);
    device const float * s1 = (device const float *) (src1 + row*args.nb11);

    float lmax = -INFINITY;
    for (int i = tpitg.x; i < args.ne00; i += tptg.x) {
        lmax = MAX(lmax, s0[i]);
    }
    const float max_val = ce_reduce_max<float>(lmax, buf, sgitg, tiisg, tptg);

    float lsum = 0.0f;
    for (int i = tpitg.x; i < args.ne00; i += tptg.x) {
        lsum += exp(s0[i] - max_val);
    }
    const float lse = log(ce_reduce_sum(lsum, buf, sgitg, tiisg, tptg));

    float lloss = 0.0f;
    for (int i = tpitg.x; i < args.ne00; i += tptg.x) {
        lloss += (s0[i] - max_val - lse)*s1[i];
    }
    const float loss = ce_reduce_sum(lloss, buf, sgitg, tiisg, tptg);

    if (tpitg.x == 0) {
        ((device float *) dst)[row] = -loss/(float) args.nr;
    }
}

kernel void kernel_cross_entropy_loss_back_f32(
        constant ggml_metal_kargs_cross_entropy_loss & args,
        device const  char * src0,   // grad: one float
        device const  char * src1,   // logits
        device const  char * src2,   // labels
        device        char * dst,
        threadgroup  float * buf [[threadgroup(0)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint3  tptg[[threads_per_threadgroup]]) {
    const int32_t row = tgpig.x;

    const float d_by_nr = ((device const float *) src0)[0]/(float) args.nr;

    device const float * s0 = (device const float *) (src1 + row*args.nb01);
    device const float * s1 = (device const float *) (src2 + row*args.nb11);
    device       float * d  = (device       float *) (dst  + row*args.nb1);

    float lmax = -INFINITY;
    for (int i = tpitg.x; i < args.ne00; i += tptg.x) {
        lmax = MAX(lmax, s0[i]);
    }
    const float max_val = ce_reduce_max<float>(lmax, buf, sgitg, tiisg, tptg);

    float lsum = 0.0f;
    for (int i = tpitg.x; i < args.ne00; i += tptg.x) {
        lsum += exp(s0[i] - max_val);
    }
    const float inv_sum = 1.0f/ce_reduce_sum(lsum, buf, sgitg, tiisg, tptg);

    // the gradient of -sum(y log softmax(x)) is softmax(x) * sum(y) - y, which equals
    // softmax(x) - y only when the labels of a row sum to 1: a masked row (all-zero labels,
    // a prompt token under completion-only training) must contribute nothing, not softmax(x)
    float ly = 0.0f;
    for (int i = tpitg.x; i < args.ne00; i += tptg.x) {
        ly += s1[i];
    }
    const float sum_y = ce_reduce_sum(ly, buf, sgitg, tiisg, tptg);

    for (int i = tpitg.x; i < args.ne00; i += tptg.x) {
        d[i] = (exp(s0[i] - max_val)*inv_sum*sum_y - s1[i])*d_by_nr;
    }
}


// RMS_NORM_BACK: dx = (dz - x * sum(x*dz)/(sum(x*x) + eps*n)) / sqrt(sum(x*x)/n + eps)
// (the CPU reference, ggml-cpu ops.cpp rms_norm_back_f32). src0 = dz, src1 = x.
kernel void kernel_rms_norm_back_f32(
        constant ggml_metal_kargs_row_back & args,
        device const  char * src0,
        device const  char * src1,
        device        char * dst,
        threadgroup  float * buf [[threadgroup(0)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint3  tptg[[threads_per_threadgroup]]) {
    const int32_t row = tgpig.x;

    device const float * dz = (device const float *) (src0 + row*args.nb01);
    device const float * x  = (device const float *) (src1 + row*args.nb11);
    device       float * dx = (device       float *) (dst  + row*args.nb1);

    float lxx = 0.0f;
    float lxdz = 0.0f;
    for (int i = tpitg.x; i < args.ne00; i += tptg.x) {
        lxx  += x[i]*x[i];
        lxdz += x[i]*dz[i];
    }
    const float sum_xx  = ce_reduce_sum(lxx,  buf, sgitg, tiisg, tptg);
    const float sum_xdz = ce_reduce_sum(lxdz, buf, sgitg, tiisg, tptg);

    const float mean_eps = sum_xx/(float) args.ne00 + args.param;
    const float sum_eps  = sum_xx + args.param*(float) args.ne00;
    const float rrms     = 1.0f/sqrt(mean_eps);
    const float scale_x  = -sum_xdz/sum_eps;

    for (int i = tpitg.x; i < args.ne00; i += tptg.x) {
        dx[i] = (dz[i] + x[i]*scale_x)*rrms;
    }
}

// SOFT_MAX_BACK: dx = scale * y * (dy - dot(y, dy)); src0 = dy, src1 = y (the forward output).
kernel void kernel_soft_max_back_f32(
        constant ggml_metal_kargs_row_back & args,
        device const  char * src0,
        device const  char * src1,
        device        char * dst,
        threadgroup  float * buf [[threadgroup(0)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint3  tptg[[threads_per_threadgroup]]) {
    const int32_t row = tgpig.x;

    device const float * dy = (device const float *) (src0 + row*args.nb01);
    device const float * y  = (device const float *) (src1 + row*args.nb11);
    device       float * dx = (device       float *) (dst  + row*args.nb1);

    float ldot = 0.0f;
    for (int i = tpitg.x; i < args.ne00; i += tptg.x) {
        ldot += y[i]*dy[i];
    }
    const float dot_y_dy = ce_reduce_sum(ldot, buf, sgitg, tiisg, tptg);

    for (int i = tpitg.x; i < args.ne00; i += tptg.x) {
        dx[i] = (dy[i] - dot_y_dy)*y[i]*args.param;
    }
}

// REPEAT_BACK: dst[k0..k3] = sum over the nr0..nr3 repeats of src0 — the gradient of a
// broadcast. One thread per dst element, the repeats gathered in registers; src0 is read
// through its strides, so a non-contiguous view is exact.
kernel void kernel_repeat_back_f32(
        constant ggml_metal_kargs_repeat & args,
        device const char * src0,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    const int k3 = tgpig.z;
    const int k2 = tgpig.y;
    const int k1 = tgpig.x;

    const int nr0 = args.ne00/args.ne0;
    const int nr1 = args.ne01/args.ne1;
    const int nr2 = args.ne02/args.ne2;
    const int nr3 = args.ne03/args.ne3;

    device float * d = (device float *) (dst + k3*args.nb3 + k2*args.nb2 + k1*args.nb1);

    for (int k0 = tpitg.x; k0 < args.ne0; k0 += ntg.x) {
        float acc = 0.0f;
        for (int i3 = 0; i3 < nr3; i3++) {
            for (int i2 = 0; i2 < nr2; i2++) {
                for (int i1 = 0; i1 < nr1; i1++) {
                    device const char * base = src0
                        + (i3*args.ne3 + k3)*args.nb03
                        + (i2*args.ne2 + k2)*args.nb02
                        + (i1*args.ne1 + k1)*args.nb01;
                    for (int i0 = 0; i0 < nr0; i0++) {
                        acc += *(device const float *) (base + (i0*args.ne0 + k0)*args.nb00);
                    }
                }
            }
        }
        d[k0] = acc;
    }
}
