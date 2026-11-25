#pragma once

// Continuum fork (UNIFIED-PAGING.md, slices 2 and 2c): a memory's own tensors, and a mind's storage shaped exactly as
// them, which the memory views by pointer while she is bound. The one owner of the storage's layout, its reservation
// in the memory's own buffer types, its attach over caller memory (her mapped file) and the repointing, for the
// attention cache and the recurrent memory alike: a switch moves no byte of either.

#include "llama-persona-blocks.h"

#include "ggml-backend.h"
#include "ggml-cpp.h"

#include <vector>

namespace llama_persona_view {

// a tensor's own memory, recorded before its first view so a view can point it back
using memory = llama_persona_memory;

// her tensors, one per entry of the memory's own list (nullptr where the memory has none), and what they live in
using tensors = llama_persona_tensors;

// The bytes a storage of `own` takes, laid out in order at each buffer type's alignment (0: no tensor, or tensors on
// more than one device).
size_t bytes(const std::vector<ggml_tensor *> & own);

// Whether the device of `own` wraps host memory, so her storage can be the caller's (her mapped file).
bool viewable(const std::vector<ggml_tensor *> & own);

// Her storage in the memory's own buffer types (VRAM on a discrete GPU), cleared once as the memory clears its own, so
// no padding a kernel reads is NaN.
llama_persona_move reserve(const std::vector<ggml_tensor *> & own, tensors & out);

// Her storage over `base..base+n`, memory the caller owns, in one device buffer that wraps it where it lies
// (`ggml_backend_dev_buffer_from_host_ptr`; Metal: newBufferWithBytesNoCopy). Nothing is cleared: the bytes are hers.
llama_persona_move attach(const std::vector<ggml_tensor *> & own, void * base, size_t n, tensors & out);

// Point each of `own` at the matching tensor of `to` (`nullptr`: back at its own memory). `saved` records the own
// memory on first use. A built graph holds the old addresses: the context drops its graphs after every move.
void point(const std::vector<ggml_tensor *> & own, const std::vector<ggml_tensor *> * to, std::vector<memory> & saved);

// Where a second storage starts after a first of `bytes` in one run of caller memory: a page multiple every platform
// maps and wraps (Windows' allocation granularity is 64 KiB), so each part's buffer starts on a page.
size_t after(size_t bytes);

} // namespace llama_persona_view
