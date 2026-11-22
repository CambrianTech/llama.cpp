// Continuum fork: where a decode step's time goes on Metal, measured, behind GGML_METAL_TIMING=1.
//
// A step is host encode, GPU work, and the gap between one graph's GPU end and the next graph's GPU
// start (the CPU round trip: logits back, sample, build and submit the next step). Batching amortizes
// GPU work; it cannot amortize that gap. This names each part so the per-step overhead is measured,
// not guessed (CONTINUUM-ENGINE 4.1b': about 10.8 ms of a 19 ms token is fixed per step).
#pragma once

#import <Metal/Metal.h>

#ifdef __cplusplus
extern "C" {
#endif

// The graph's host encode begins (call at graph compute entry).
void ggml_metal_timing_encode_begin(int n_nodes);
// The graph's host encode is done and committed (call after the last command buffer is committed).
void ggml_metal_timing_encode_end(void);
// The graph's command buffers have completed (call in synchronize, after their status is checked).
void ggml_metal_timing_completed(const id<MTLCommandBuffer> * bufs, int n);

#ifdef __cplusplus
}
#endif
