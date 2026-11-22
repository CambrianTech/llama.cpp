// Continuum [DEVICE-FIT #40]: what quantization adds for device-fit overrides and speed-only measurement, kept
// beside llama-quant.cpp so upstream's quantizer merges without touching it.
#include "llama-quant-device-fit.h"

#include "llama-impl.h"
#include "llama-ext.h"

#include <cstdio>
#include <cstdlib>

// --resident-only: emit ONLY the resident (non-expert) tensors. The routed experts (`*_exps.*`) are served from the
// container/primary at run time (the LLAMA_RESIDENT_OVERRIDE loader takes ONLY non-`_exps.` tensors from the
// override), so a device-fit override needs just the ~25 GB resident, not a 684 GB full clone.
bool llama_quant_skips_tensor(const llama_model_quantize_params * params, const std::string & name) {
    return params->resident_only && name.find("_exps.") != std::string::npos;
}

// A same-type target is a COPY, not a quantization, so it never needs an importance matrix (copying a model whose
// tensors are already low-bit, e.g. K3's iq1_m/iq2/iq3 UD experts, would otherwise demand one).
// LLAMA_QUANT_FORCE_NO_IMATRIX: for SPEED-ONLY benchmarking, a low-bit requant proceeds WITHOUT an importance
// matrix (poor-quality weights, valid only to measure fetched-bytes -> tok/s scaling; never ship the output).
bool llama_quant_needs_imatrix(bool upstream_requires, ggml_type target, ggml_type source) {
    static const bool force_no_imatrix = getenv("LLAMA_QUANT_FORCE_NO_IMATRIX") != nullptr;
    return !force_no_imatrix && upstream_requires && target != source;
}

// the resident-only sidecar manifest: `<out>.resident.json` carries (tier_label, resident_bytes) so the governor can
// co-optimize the VRAM division (resident precision vs expert-cache size) by comparing tiers WITHOUT loading each
// GGUF. total_size_new is the resident footprint (experts were skipped from the output).
void llama_quant_write_resident_manifest(const llama_model_quantize_params * params, const std::string & fname_out,
                                         size_t resident_bytes) {
    if (!params->resident_only) {
        return;
    }
    const std::string manifest_path = fname_out + ".resident.json";
    const char * tier = ggml_type_name(llama_ftype_get_default_type(params->ftype));
    if (FILE * mf = fopen(manifest_path.c_str(), "wb")) {
        fprintf(mf, "{\"resident_only\":1,\"tier_label\":\"%s\",\"resident_bytes\":%zu}\n", tier, resident_bytes);
        fclose(mf);
        LLAMA_LOG_INFO("%s: resident-only manifest -> %s (tier_label=%s, resident_bytes=%zu)\n",
                       __func__, manifest_path.c_str(), tier, resident_bytes);
    } else {
        LLAMA_LOG_WARN("%s: resident-only: could not write manifest %s\n", __func__, manifest_path.c_str());
    }
}
