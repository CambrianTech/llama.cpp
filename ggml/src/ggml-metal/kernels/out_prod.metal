#include "common.h"
#include "dequantize.h"

// OUT_PROD: dst[m, n] = sum_k src0[m, k] * src1[n, k]. It is the backward of a matmul through
// a frozen weight (dx = out_prod(W, dy^T)) and the gradient of a LoRA factor (dA, dB).
//
// This is mul_mm with the contraction along the ROWS of both operands. Each thread reads 16
// consecutive elements of one src0 row, in the weight's own block order, dequantizes them in
// registers, and stages them transposed in threadgroup memory. Nothing is copied or transposed
// in device memory. The tiles and the simdgroup product are mul_mm's: a 64 x 32 output tile,
// 32-deep k steps, 4 simdgroups. Staging is float because gradients exceed half's range.
//
// nl == 0 selects the element path (F32/F16/BF16, any width, a LoRA rank of 8 included).
// nl > 0 dequantizes whole 16-element chunks, which needs ne00 % 16 == 0.

template<typename T0, typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread float4x4 &)>
kernel void kernel_out_prod(
        constant ggml_metal_kargs_out_prod & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    constexpr short OP_NR0 = 64; // output rows (m) per threadgroup
    constexpr short OP_NR1 = 32; // output columns (n) per threadgroup
    constexpr short OP_NK  = 32; // contraction depth per step

    threadgroup float * sa = (threadgroup float *) shmem;            // OP_NR0 x OP_NK
    threadgroup float * sb = sa + OP_NR0*OP_NK;                       // OP_NR1 x OP_NK

    const int r0 = tgpig.y*OP_NR0;
    const int r1 = tgpig.x*OP_NR1;
    const int i2 = tgpig.z % args.ne2;
    const int i3 = tgpig.z / args.ne2;

    // src0 is broadcast over dims 2-3 (grouped heads); src1 and dst are not
    src0 += (i2/args.dps2)*args.nb02 + (i3/args.dps3)*args.nb03;
    src1 += i2*args.nb12 + i3*args.nb13;

    // src0 staging: thread -> one k row, 16 consecutive m
    const short ka = tiitg/4;
    const short ma = 16*(tiitg%4);
    // src1 staging: thread -> one n, 8 consecutive k
    const short nb = tiitg/4;
    const short kb = 8*(tiitg%4);

    simdgroup_float8x8 ma8[4];
    simdgroup_float8x8 mb8[2];
    simdgroup_float8x8 mc[8];
    for (short i = 0; i < 8; i++) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }

    for (int k0 = 0; k0 < args.ne01; k0 += OP_NK) {
        float4x4 a = float4x4(0.f);
        const int k = k0 + ka;
        const int m = r0 + ma;
        if (k < args.ne01 && m < args.ne00) {
            device const char * row = src0 + (uint64_t) k*args.nb01;
            if (nl == 0) {
                device const T0 * x = (device const T0 *) row + m;
                for (short i = 0; i < 16; i++) {
                    a[i/4][i%4] = m + i < args.ne00 ? (float) x[i] : 0.f;
                }
            } else {
                dequantize_func((device const block_q *) row + m/(16*nl), (m/16) % nl, a);
            }
        }

        float b[8];
        {
            const int n = r1 + nb;
            for (short i = 0; i < 8; i++) {
                const int kk = k0 + kb + i;
                b[i] = kk < args.ne01 && n < args.ne10
                    ? *(device const float *) (src1 + (uint64_t) kk*args.nb11 + (uint64_t) n*args.nb10) : 0.f;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // the same 8x8-block layout mul_mm stages: sa block (k/8, m/8) holds [k%8][m%8],
        // sb block (k/8, n/8) holds [n%8][k%8]
        for (short i = 0; i < 16; i++) {
            const short mm = ma + i;
            sa[64*(8*(ka/8) + mm/8) + 8*(ka%8) + mm%8] = a[i/4][i%4];
        }
        for (short i = 0; i < 8; i++) {
            const short kk = kb + i;
            sb[64*(4*(kk/8) + nb/8) + 8*(nb%8) + kk%8] = b[i];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const float * lsma = sa + 4*64*(sgitg%2);
        threadgroup const float * lsmb = sb + 2*64*(sgitg/2);
        for (short ik = 0; ik < OP_NK/8; ik++) {
            for (short i = 0; i < 4; i++) {
                simdgroup_load(ma8[i], lsma + 64*i, 8, 0, false);
            }
            for (short i = 0; i < 2; i++) {
                simdgroup_load(mb8[i], lsmb + 64*i, 8, 0, false);
            }
            for (short i = 0; i < 8; i++) {
                simdgroup_multiply_accumulate(mc[i], mb8[i/4], ma8[i%4], mc[i]);
            }
            lsma += 8*64;
            lsmb += 4*64;
        }
    }

    device float * d = (device float *) dst + (uint64_t) tgpig.z*args.ne00*args.ne10;

    if (r0 + OP_NR0 <= args.ne00 && r1 + OP_NR1 <= args.ne10) {
        device float * C = d + (r0 + 32*(sgitg & 1)) + (r1 + 16*(sgitg >> 1))*args.ne00;
        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], C + 8*(i%4) + 8*args.ne00*(i/4), args.ne00, 0, false);
        }
        return;
    }

    // an edge tile: stage it, then write only the part inside dst
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float * t = (threadgroup float *) shmem + 32*(sgitg & 1) + 16*(sgitg >> 1)*OP_NR0;
    for (short i = 0; i < 8; i++) {
        simdgroup_store(mc[i], t + 8*(i%4) + 8*OP_NR0*(i/4), OP_NR0, 0, false);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const int nr0 = min((int) OP_NR0, args.ne00 - r0);
    const int nr1 = min((int) OP_NR1, args.ne10 - r1);
    for (int j = tiitg; j < nr0*nr1; j += 128) {
        const int jm = j % nr0;
        const int jn = j / nr0;
        d[(r0 + jm) + (r1 + jn)*args.ne00] = ((threadgroup float *) shmem)[jm + jn*OP_NR0];
    }
}

typedef decltype(kernel_out_prod<float, float4x4, 0, dequantize_f32>) out_prod_t;

template [[host_name("kernel_out_prod_f32")]]  kernel out_prod_t kernel_out_prod<float,  float4x4,   0,     dequantize_f32>;
template [[host_name("kernel_out_prod_f16")]]  kernel out_prod_t kernel_out_prod<half,   float4x4,   0,     dequantize_f32>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_out_prod_bf16")]] kernel out_prod_t kernel_out_prod<bfloat, float4x4,   0,     dequantize_f32>;
#endif
template [[host_name("kernel_out_prod_q4_0")]] kernel out_prod_t kernel_out_prod<float,  block_q4_0, 2,     dequantize_q4_0>;
template [[host_name("kernel_out_prod_q4_1")]] kernel out_prod_t kernel_out_prod<float,  block_q4_1, 2,     dequantize_q4_1>;
template [[host_name("kernel_out_prod_q5_0")]] kernel out_prod_t kernel_out_prod<float,  block_q5_0, 2,     dequantize_q5_0>;
template [[host_name("kernel_out_prod_q5_1")]] kernel out_prod_t kernel_out_prod<float,  block_q5_1, 2,     dequantize_q5_1>;
template [[host_name("kernel_out_prod_q8_0")]] kernel out_prod_t kernel_out_prod<float,  block_q8_0, 2,     dequantize_q8_0>;
template [[host_name("kernel_out_prod_q2_K")]] kernel out_prod_t kernel_out_prod<float,  block_q2_K, QK_NL, dequantize_q2_K>;
template [[host_name("kernel_out_prod_q3_K")]] kernel out_prod_t kernel_out_prod<float,  block_q3_K, QK_NL, dequantize_q3_K>;
template [[host_name("kernel_out_prod_q4_K")]] kernel out_prod_t kernel_out_prod<float,  block_q4_K, QK_NL, dequantize_q4_K>;
template [[host_name("kernel_out_prod_q5_K")]] kernel out_prod_t kernel_out_prod<float,  block_q5_K, QK_NL, dequantize_q5_K>;
template [[host_name("kernel_out_prod_q6_K")]] kernel out_prod_t kernel_out_prod<float,  block_q6_K, QK_NL, dequantize_q6_K>;

// MUL_MAT_ID_BACK: dst[:, i, t] = sum_r as[:, r, ids[i, t]] * grad[r, i, t], the activations'
// gradient of mul_mat_id through its frozen experts (a LoRA on attention trains on a MoE base).
// One threadgroup per routed slot (i, t); each thread owns columns of the slot's row and walks the
// expert's rows, so every dst element has one writer and no expert is copied or transposed in
// device memory. nl == 0 reads F32/F16/BF16 element by element; nl > 0 dequantizes 16
// consecutive elements of a row in registers (ne00 % 16 == 0), as kernel_out_prod does.
template<typename T0, typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread float4x4 &)>
kernel void kernel_mul_mat_id_back(
        constant ggml_metal_kargs_mul_mat_id_back & args,
        device const char * src0,
        device const char * grad,
        device const char * ids,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiitg[[thread_index_in_threadgroup]],
        ushort3 ntg  [[threads_per_threadgroup]]) {
    const int32_t i = tgpig.x;
    const int32_t t = tgpig.y;
    const int32_t e = *((device const int32_t *) (ids + i*args.nbi0 + t*args.nbi1));

    device const char * as_e = src0 + e*args.nb02;
    device const char * g    = grad + i*args.nbg1 + t*args.nbg2;
    device       char * d    = dst  + i*args.nb1  + t*args.nb2;

    if (nl == 0) {
        for (int32_t c = tiitg; c < args.ne00; c += ntg.x) {
            float acc = 0.0f;
            for (int32_t r = 0; r < args.ne01; ++r) {
                acc += (float) ((device const T0 *) (as_e + r*args.nb01))[c] * *((device const float *) (g + r*args.nbg0));
            }
            ((device float *) d)[c] = acc;
        }
    } else {
        for (int32_t ind = tiitg; ind < args.ne00/16; ind += ntg.x) {
            float4x4 acc = float4x4(0.0f);
            for (int32_t r = 0; r < args.ne01; ++r) {
                float4x4 w;
                dequantize_func(((device const block_q *) (as_e + r*args.nb01)) + ind/nl, ind%nl, w);
                acc += w * *((device const float *) (g + r*args.nbg0));
            }
            ((device float4x4 *) d)[ind] = acc;
        }
    }
}

typedef decltype(kernel_mul_mat_id_back<float, float4x4, 0, dequantize_f32>) mul_mat_id_back_t;

template [[host_name("kernel_mul_mat_id_back_f32")]]  kernel mul_mat_id_back_t kernel_mul_mat_id_back<float,  float4x4,   0,     dequantize_f32>;
template [[host_name("kernel_mul_mat_id_back_f16")]]  kernel mul_mat_id_back_t kernel_mul_mat_id_back<half,   float4x4,   0,     dequantize_f32>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_mul_mat_id_back_bf16")]] kernel mul_mat_id_back_t kernel_mul_mat_id_back<bfloat, float4x4,   0,     dequantize_f32>;
#endif
template [[host_name("kernel_mul_mat_id_back_q4_0")]] kernel mul_mat_id_back_t kernel_mul_mat_id_back<float,  block_q4_0, 2,     dequantize_q4_0>;
template [[host_name("kernel_mul_mat_id_back_q4_1")]] kernel mul_mat_id_back_t kernel_mul_mat_id_back<float,  block_q4_1, 2,     dequantize_q4_1>;
template [[host_name("kernel_mul_mat_id_back_q5_0")]] kernel mul_mat_id_back_t kernel_mul_mat_id_back<float,  block_q5_0, 2,     dequantize_q5_0>;
template [[host_name("kernel_mul_mat_id_back_q5_1")]] kernel mul_mat_id_back_t kernel_mul_mat_id_back<float,  block_q5_1, 2,     dequantize_q5_1>;
template [[host_name("kernel_mul_mat_id_back_q8_0")]] kernel mul_mat_id_back_t kernel_mul_mat_id_back<float,  block_q8_0, 2,     dequantize_q8_0>;
template [[host_name("kernel_mul_mat_id_back_q2_K")]] kernel mul_mat_id_back_t kernel_mul_mat_id_back<float,  block_q2_K, QK_NL, dequantize_q2_K>;
template [[host_name("kernel_mul_mat_id_back_q3_K")]] kernel mul_mat_id_back_t kernel_mul_mat_id_back<float,  block_q3_K, QK_NL, dequantize_q3_K>;
template [[host_name("kernel_mul_mat_id_back_q4_K")]] kernel mul_mat_id_back_t kernel_mul_mat_id_back<float,  block_q4_K, QK_NL, dequantize_q4_K>;
template [[host_name("kernel_mul_mat_id_back_q5_K")]] kernel mul_mat_id_back_t kernel_mul_mat_id_back<float,  block_q5_K, QK_NL, dequantize_q5_K>;
template [[host_name("kernel_mul_mat_id_back_q6_K")]] kernel mul_mat_id_back_t kernel_mul_mat_id_back<float,  block_q6_K, QK_NL, dequantize_q6_K>;
