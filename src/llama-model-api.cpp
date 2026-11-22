// Continuum: the model facts our serving governor and pager read (declared in llama.h), kept beside llama-model.cpp
// so upstream's model merges without touching it.
#include "llama-model.h"

#include "ggml-backend.h"

#include <utility>
#include <vector>

// serving governor reads (declared in llama.h beside n_head_kv). Restored after
// the b10636 merge took upstream's file wholesale.
int32_t llama_model_n_head_kv_il(const llama_model * model, int32_t il) {
    return model->hparams.n_head_kv(il);
}

int32_t llama_model_n_embd_head_k_il(const llama_model * model, int32_t il) {
    return model->hparams.n_embd_head_k(il);
}

int32_t llama_model_n_embd_head_v_il(const llama_model * model, int32_t il) {
    return model->hparams.n_embd_head_v(il);
}


// existing primitive that already walks ctxs_bufs and handles the no_alloc case.
// Not a second walk of the same data.
static std::vector<std::pair<const char *, size_t>> llama_model_weight_backends(const llama_model * model) {
    std::vector<std::pair<const char *, size_t>> out;
    for (const auto & [buft, bytes] : model->memory_breakdown()) {
        out.emplace_back(ggml_backend_buft_name(buft), bytes);
    }
    return out;
}

size_t llama_model_n_weight_backends(const llama_model * model) {
    return llama_model_weight_backends(model).size();
}

llama_weight_allocation llama_model_weight_allocation(const llama_model * model) {
    llama_weight_allocation allocation = {};
    for (const auto & [buft, bytes] : model->memory_breakdown()) {
        if (ggml_backend_buft_is_host(buft)) {
            allocation.host += bytes;
        } else {
            allocation.device += bytes;
        }
    }
    return allocation;
}

const char * llama_model_weight_backend_name(const llama_model * model, size_t i) {
    const auto v = llama_model_weight_backends(model);
    return i < v.size() ? v[i].first : nullptr;
}

size_t llama_model_weight_backend_bytes(const llama_model * model, size_t i) {
    const auto v = llama_model_weight_backends(model);
    return i < v.size() ? v[i].second : 0;
}


// weight tensor, returning the MUTABLE ggml_tensor* so callers can ggml_backend_tensor_set
// a fresh expert's bytes into a sub-range of its buffer at runtime (no model reload).
// Same lookup as llama_model::get_tensor, but the vector holds non-const ggml_tensor*
// (like llama_internal_get_tensor_map), so the pointee stays writable even off a const model.
struct ggml_tensor * llama_model_get_tensor(const llama_model * model, const char * name) {
    for (const auto & it : model->tensors_by_name) {
        if (it.first == name) {
            return it.second;
        }
    }
    return nullptr;
}

