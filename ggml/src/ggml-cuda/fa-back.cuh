#include "common.cuh"

// Continuum fork: FLASH_ATTN_BACK on CUDA (ggml.h, ggml_flash_attn_back). Without it a CUDA device has
// no native attention backward, so training composes the backward from tiles that keep scores, and a
// resident's in-place lesson over her served window measured 1.897 TB on a 34.19 GB 5090 (Codex's
// replay of #6292, build 2161, 2026-11-27). One warp per row, recomputing the scores: memory is the
// gradients alone, as on Metal (kernels/fa_back.metal) and the CPU reference.

void ggml_cuda_flash_attn_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

bool ggml_cuda_flash_attn_back_supported(const ggml_tensor * op);
