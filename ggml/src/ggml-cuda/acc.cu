#include "acc.cuh"

static __global__ void acc_f32(const float * x, const float * y, float * dst, const int64_t ne,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const int64_t s11, const int64_t s12, const int64_t s13, const int64_t offset, const int3 order) {
    const int64_t i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= ne) {
        return;
    }

    int64_t src1_idx = i - offset;

    int64_t tmp = src1_idx;
    // A view can permute destination strides (the transposed V-cache backward does).
    // Decode physical offsets largest-stride first, then index y in its logical order.
    const int64_t strides[] = {s11, s12, s13};
    int64_t coords[3];
    coords[order.x] = tmp / strides[order.x];
    tmp -= coords[order.x] * strides[order.x];
    coords[order.y] = tmp / strides[order.y];
    tmp -= coords[order.y] * strides[order.y];
    coords[order.z] = tmp / strides[order.z];
    tmp -= coords[order.z] * strides[order.z];
    const int64_t i11 = coords[0], i12 = coords[1], i13 = coords[2];
    const int64_t i10 = tmp;

    float val = x[i];
    if (src1_idx >= 0 && i10 < ne10 && i11 < ne11 && i12 < ne12 && i13 < ne13) {
        val += y[((i13*ne12 + i12) * ne11 + i11) * ne10 + i10];
    }
    dst[i] = val;
}

static void acc_f32_cuda(const float * x, const float * y, float * dst, const int64_t n_elements,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const int64_t s1, const int64_t s2, const int64_t s3, const int64_t offset, cudaStream_t stream) {
    const int num_blocks = (n_elements + CUDA_ACC_BLOCK_SIZE - 1) / CUDA_ACC_BLOCK_SIZE;
    const int64_t strides[] = {s1, s2, s3};
    int axes[] = {2, 1, 0};
    for (int a = 0; a < 2; ++a) {
        for (int b = a + 1; b < 3; ++b) {
            if (strides[axes[a]] < strides[axes[b]]) {
                const int tmp = axes[a]; axes[a] = axes[b]; axes[b] = tmp;
            }
        }
    }
    const int3 order = make_int3(axes[0], axes[1], axes[2]);
    acc_f32<<<num_blocks, CUDA_ACC_BLOCK_SIZE, 0, stream>>>(x, y, dst, n_elements, ne10, ne11, ne12, ne13, s1, s2, s3, offset, order);
}

void ggml_cuda_op_acc(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    const float * src0_d = (const float *) src0->data;
    const float * src1_d = (const float *) src1->data;
    float       * dst_d  = (float       *)  dst->data;

    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(dst->nb[0] == ggml_element_size(dst));
    GGML_ASSERT(ggml_is_contiguously_allocated(dst));

    const int64_t s1     = dst->op_params[0] / sizeof(float);
    const int64_t s2     = dst->op_params[1] / sizeof(float);
    const int64_t s3     = dst->op_params[2] / sizeof(float);
    const int64_t offset = dst->op_params[3] / sizeof(float);

    acc_f32_cuda(src0_d, src1_d, dst_d, ggml_nelements(dst), src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3], s1, s2, s3, offset, stream);
}
