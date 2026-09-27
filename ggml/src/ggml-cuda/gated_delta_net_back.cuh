#include "common.cuh"
#include "ggml.h"

// The backward of the gated delta rule (GGML_OP_GATED_DELTA_NET_BACK): the CUDA twin of the
// CPU reference in ggml-cpu/ops.cpp, checked against it by finite differences in
// test-backend-ops (grad -o GATED_DELTA_NET).
void ggml_cuda_op_gated_delta_net_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

bool ggml_cuda_gated_delta_net_back_supported(const ggml_tensor * dst);
