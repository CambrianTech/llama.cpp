#include "out-prod.cuh"
#include "convert.cuh"

#include <algorithm>

// The F32 transient a row-blocked dequantized out_prod holds at once. 256 MiB is far below
// any device this backend serves and large enough that the per-block GEMM is not launch-bound.
static constexpr int64_t OUT_PROD_DEQUANT_BLOCK_BYTES = 256ll * 1024 * 1024;

#include <cstdint>

static __global__ void k_compute_out_prod_ptrs(
        const float * src0_d, const float * src1_d, float * dst_d,
        const float ** ptrs_a, const float ** ptrs_b, float ** ptrs_c,
        const int64_t ne2, const int64_t ne3,
        const int64_t dps2, const int64_t dps3,
        const size_t s02, const size_t s03,
        const size_t s12, const size_t s13,
        const size_t s2,  const size_t s3) {
    const int64_t i2 = blockIdx.x*blockDim.x + threadIdx.x;
    const int64_t i3 = blockIdx.y*blockDim.y + threadIdx.y;

    if (i2 >= ne2 || i3 >= ne3) {
        return;
    }

    const int64_t idx = i3*ne2 + i2;

    ptrs_a[idx] = src0_d + (i3/dps3)*s03 + (i2/dps2)*s02;
    ptrs_b[idx] = src1_d +  i3      *s13 +  i2      *s12;
    ptrs_c[idx] = dst_d  +  i3      *s3  +  i2      *s2;
}

void ggml_cuda_out_prod(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_BINARY_OP_LOCALS

    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    // A quantized (or F16/BF16) src0: the backward of mul_mat(W, x) w.r.t. x is
    // out_prod(W, grad^T), so training a LoRA on a resident quantized base needs this
    // op to read W as it is stored. The CPU backend dispatches every quantized type here
    // (ggml_compute_forward_out_prod_q_f32); this backend only took F32, which made the
    // backward through a Q4_K_M base impossible on CUDA while it worked on CPU
    // (Continuum card dd109d3c, 2026-09-26). Dequantize W to a transient F32 copy in the
    // pool through the same converters mul_mat uses, then run the F32 GEMMs below on it.
    // The copy is dense (contiguous rows), so the F32 strides are derived from ne0x,
    // not from the quantized tensor's nb0x.
    cudaStream_t stream = ctx.stream();
    ggml_cuda_pool_alloc<float> src0_f32_alloc(ctx.pool());
    const float * src0_d;
    int64_t lda, s02, s03;
    if (src0->type == GGML_TYPE_F32) {
        src0_d = (const float *) src0->data;
        lda = nb01 / sizeof(float);
        s02 = nb02 / sizeof(float);
        s03 = nb03 / sizeof(float);
    } else {
        GGML_ASSERT(ggml_is_contiguous(src0));
        const to_fp32_cuda_t to_fp32 = ggml_get_to_fp32_cuda(src0->type);
        GGML_ASSERT(to_fp32 != nullptr);
        if (ne02 == 1 && ne03 == 1 && ne2 == 1 && ne3 == 1) {
            // THE WEIGHT CASE, ROW-BLOCKED. out_prod sums over src0's rows (ne01), so the
            // dequantized transient never has to be the whole W: dequantize a block of rows,
            // GEMM it with beta = 0 for the first block and beta = 1 after, and the peak is
            // one block of F32 instead of W in F32. For a 27B's output head (5120 x 152k,
            // ~3.1 GB in F32) beside the resident model that is the difference between a
            // dream step that fits and one that does not (Cormac on llama.cpp#3).
            //
            // TENSOR CORES WHERE THE TYPE ALLOWS (Continuum card 742239ee): the forward runs
            // W x on int8 tensor cores; an F32 Sgemm here made the backward's frozen-base
            // GEMMs ~3x the forward's cost (27B ffn 5120 x 17408 at 1024 tokens: 2.9 ms).
            // When the weight type has a BF16 dequantizer, the block goes to BF16, the
            // gradient goes to BF16 once per op (dense, keeping its orientation), and each
            // block is a cublasGemmEx with F32 accumulation into dst. BF16 rather than F16:
            // the gradient operand can be tiny, and F16 would flush it to zero; BF16 keeps
            // F32's exponent range, and its rounding sits far inside the op's parity bound.
            const to_bf16_cuda_t to_bf16 = ggml_get_to_bf16_cuda(src0->type);
            const to_bf16_nc_cuda_t src1_to_bf16 = ggml_get_to_bf16_nc_cuda(GGML_TYPE_F32);
            const bool src1_T_bf = ggml_is_transposed(src1);
            if (to_bf16 != nullptr && src1_to_bf16 != nullptr &&
                (src1_T_bf ? nb11 : nb10) == sizeof(float)) {
                // src1 as stored: a fast dim of f contiguous floats, `other` rows at `ld_src`.
                const int64_t f      = src1_T_bf ? ne11 : ne10;
                const int64_t other  = src1_T_bf ? ne10 : ne11;
                const int64_t ld_src = (src1_T_bf ? nb10 : nb11) / sizeof(float);
                ggml_cuda_pool_alloc<nv_bfloat16> src1_bf(ctx.pool(), f * other);
                src1_to_bf16(src1->data, src1_bf.get(), f, other, 1, 1, ld_src, f * other, f * other, stream);
                CUDA_CHECK(cudaGetLastError());

                const int64_t row_bytes_bf = ne00 * (int64_t) sizeof(nv_bfloat16);
                int64_t rows_per_block = OUT_PROD_DEQUANT_BLOCK_BYTES / row_bytes_bf;
                rows_per_block = std::max<int64_t>(1, std::min<int64_t>(rows_per_block, ne01));
                ggml_cuda_pool_alloc<nv_bfloat16> w_bf(ctx.pool(), rows_per_block * ne00);

                float * dst_d = (float *) dst->data;
                cublasHandle_t handle = ctx.cublas_handle();
                const float alpha = 1.0f;
                const int64_t ldc = nb1 / sizeof(float);
                const cublasOperation_t src1_op = src1_T_bf ? CUBLAS_OP_N : CUBLAS_OP_T;
                for (int64_t r0 = 0; r0 < ne01; r0 += rows_per_block) {
                    const int64_t rows = std::min<int64_t>(rows_per_block, ne01 - r0);
                    to_bf16((const char *) src0->data + r0 * nb01, w_bf.get(), rows * ne00, stream);
                    CUDA_CHECK(cudaGetLastError());
                    // same k offsets as the F32 path, on the dense copy (ld = f)
                    const nv_bfloat16 * src1_block = src1_bf.get() + (src1_T_bf ? r0 : r0 * f);
                    const float beta = r0 == 0 ? 0.0f : 1.0f;
                    CUBLAS_CHECK(
                        cublasGemmEx(handle, CUBLAS_OP_N, src1_op,
                                ne0, ne1, rows,
                                &alpha, w_bf.get(), CUDA_R_16BF, ne00,
                                        src1_block, CUDA_R_16BF, f,
                                &beta,  dst_d,      CUDA_R_32F,  ldc,
                                CUBLAS_COMPUTE_32F,
                                CUBLAS_GEMM_DEFAULT_TENSOR_OP));
                }
                return;
            }
            // No BF16 dequantizer for this type (or an unusual src1 layout): the F32 path.
            const int64_t row_bytes_f32 = ne00 * (int64_t) sizeof(float);
            int64_t rows_per_block = OUT_PROD_DEQUANT_BLOCK_BYTES / row_bytes_f32;
            rows_per_block = std::max<int64_t>(1, std::min<int64_t>(rows_per_block, ne01));
            float * tmp = src0_f32_alloc.alloc(rows_per_block * ne00);
            const float * src1_d = (const float *) src1->data;
            float       *  dst_d = (float       *)  dst->data;
            cublasHandle_t handle = ctx.cublas_handle();
            const float alpha = 1.0f;
            const int64_t ldc = nb1 / sizeof(float);
            const bool src1_T = ggml_is_transposed(src1);
            const cublasOperation_t src1_cublas_op = src1_T ? CUBLAS_OP_N : CUBLAS_OP_T;
            const int64_t ldb = (src1_T ? nb10 : nb11) / sizeof(float);
            GGML_ASSERT((src1_T ? nb11 : nb10) == sizeof(float));
            for (int64_t r0 = 0; r0 < ne01; r0 += rows_per_block) {
                const int64_t rows = std::min<int64_t>(rows_per_block, ne01 - r0);
                to_fp32((const char *) src0->data + r0 * nb01, tmp, rows * ne00, stream);
                CUDA_CHECK(cudaGetLastError());
                // The k offset into src1: op(B) = B^T reads B as n x k with ldb, so k
                // advances by ldb elements; op(B) = B reads k x n, so k advances by one.
                const float * src1_block = src1_d + (src1_T ? r0 : r0 * ldb);
                const float beta = r0 == 0 ? 0.0f : 1.0f;
                CUBLAS_CHECK(
                    cublasSgemm(handle, CUBLAS_OP_N, src1_cublas_op,
                            ne0, ne1, rows,
                            &alpha, tmp,        ne00,
                                    src1_block, ldb,
                            &beta,  dst_d,      ldc));
            }
            return;
        }
        // Batched or broadcast src0 (the attention/GQA shapes): the dense copy of all of it.
        const int64_t n = ggml_nelements(src0);
        float * tmp = src0_f32_alloc.alloc(n);
        to_fp32(src0->data, tmp, n, stream);
        CUDA_CHECK(cudaGetLastError());
        src0_d = tmp;
        lda = ne00;
        s02 = ne00 * ne01;
        s03 = ne00 * ne01 * ne02;
    }

    GGML_ASSERT(ne01 == ne11);
    GGML_ASSERT(ne0 == ne00);
    GGML_ASSERT(ne1 == ne10);

    GGML_ASSERT(ne2 % src0->ne[2] == 0);
    GGML_ASSERT(ne3 % src0->ne[3] == 0);

    GGML_ASSERT(ne2 == src1->ne[2]);
    GGML_ASSERT(ne3 == src1->ne[3]);

    const float * src1_d = (const float *) src1->data;
    float       *  dst_d = (float       *)  dst->data;

    cublasHandle_t handle = ctx.cublas_handle();

    const float alpha = 1.0f;
    const float beta = 0.0f;

    const int64_t ldc = nb1  / sizeof(float);

    const bool src1_T = ggml_is_transposed(src1);
    const cublasOperation_t src1_cublas_op =  src1_T ? CUBLAS_OP_N : CUBLAS_OP_T;
    const int64_t           ldb            = (src1_T ?        nb10 :        nb11) /  sizeof(float);
    GGML_ASSERT(                             (src1_T ?        nb11 :        nb10) == sizeof(float));

    // data strides in dimensions 2/3
    // s02/s03: src0's strides in dims 2/3, set above per src0's storage (F32 view or the dense copy).
    const size_t s12 = nb12 / sizeof(float);
    const size_t s13 = nb13 / sizeof(float);
    const size_t s2  = nb2  / sizeof(float);
    const size_t s3  = nb3  / sizeof(float);

    // dps == dst per src0, used for group query attention
    const int64_t dps2 = ne2 / ne02;
    const int64_t dps3 = ne3 / ne03;

    if (dps2 == 1 && ne2 > 1) {
        // src0 has uniform stride s02 along dim 2; batch the inner loop with a strided GEMM
        GGML_ASSERT(ne2 <= std::numeric_limits<int>::max());
        const int batch_count = (int) ne2;
        for (int64_t i3 = 0; i3 < ne3; ++i3) {
            CUBLAS_CHECK(
                cublasSgemmStridedBatched(handle, CUBLAS_OP_N, src1_cublas_op,
                        ne0, ne1, ne01,
                        &alpha, src0_d + (i3/dps3)*s03, lda, s02,
                                src1_d +  i3     *s13, ldb, s12,
                        &beta,  dst_d  +  i3     *s3,  ldc, s2,
                        batch_count));
        }
    } else if (ne2 > 1 || ne3 > 1) {
        // dps2 > 1 (src0 broadcast along dim 2 with non-uniform stride) or multiple GEMMs
        // along dim 3: compute per-GEMM pointers on the device and use a single batched GEMM.
        GGML_ASSERT(ne3 > 0);
        GGML_ASSERT(ne2 <= (int64_t) std::numeric_limits<int>::max() / ne3);
        const int batch_count = (int) (ne2 * ne3);

        ggml_cuda_pool_alloc<const float *> ptrs_a(ctx.pool(), batch_count);
        ggml_cuda_pool_alloc<const float *> ptrs_b(ctx.pool(), batch_count);
        ggml_cuda_pool_alloc<      float *> ptrs_c(ctx.pool(), batch_count);

        const dim3 block_dims(16, 16);
        const dim3 grid_dims((ne2 + block_dims.x - 1)/block_dims.x, (ne3 + block_dims.y - 1)/block_dims.y);
        k_compute_out_prod_ptrs<<<grid_dims, block_dims, 0, stream>>>(
            src0_d, src1_d, dst_d,
            ptrs_a.get(), ptrs_b.get(), ptrs_c.get(),
            ne2, ne3, dps2, dps3, s02, s03, s12, s13, s2, s3);
        CUDA_CHECK(cudaGetLastError());

        CUBLAS_CHECK(
            cublasSgemmBatched(handle, CUBLAS_OP_N, src1_cublas_op,
                    ne0, ne1, ne01,
                    &alpha, ptrs_a.get(), lda,
                            ptrs_b.get(), ldb,
                    &beta,  ptrs_c.get(), ldc,
                    batch_count));
    } else {
        // ne2 == 1 && ne3 == 1: single GEMM
        CUBLAS_CHECK(
            cublasSgemm(handle, CUBLAS_OP_N, src1_cublas_op,
                    ne0, ne1, ne01,
                    &alpha, src0_d, lda,
                            src1_d, ldb,
                    &beta,  dst_d,  ldc));
    }
}
