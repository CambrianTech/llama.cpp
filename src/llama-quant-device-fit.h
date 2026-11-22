#pragma once

// Continuum [DEVICE-FIT #40]: quantization's device-fit additions (llama-quant-device-fit.cpp)

#include "llama.h"

#include <string>

bool llama_quant_skips_tensor(const llama_model_quantize_params * params, const std::string & name);
bool llama_quant_needs_imatrix(bool upstream_requires, ggml_type target, ggml_type source);
void llama_quant_write_resident_manifest(const llama_model_quantize_params * params, const std::string & fname_out,
                                         size_t resident_bytes);
