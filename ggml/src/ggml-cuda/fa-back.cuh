#include "common.cuh"

// FLASH_ATTN_BACK (ggml.h, ggml_flash_attn_back): the gradients of FLASH_ATTN_EXT, recomputing the
// scores instead of keeping them, so a training step's attention backward costs its gradients and a
// per-row scratch, never queries x keys. Without a native kernel a backend composes the backward from
// tiles that keep scores, whose memory grows with the window.

void ggml_cuda_flash_attn_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

bool ggml_cuda_flash_attn_back_supported(const ggml_tensor * op);
