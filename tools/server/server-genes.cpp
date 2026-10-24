#include "server-genes.h"

#include "ggml.h"

json server_gene_move::to_json() const {
    return json {
        { "success",  true },
        { "id",       id },
        { "path",     path },
        { "resident", on_device ? "device" : "host" },
        { "n_bytes",  n_bytes },
        { "n_moved",  n_moved },
        { "t_ms",     t_ms },
    };
}

static server_gene_move gene_move(const common_adapter_lora_info & lora, int32_t id, bool to_device) {
    const int64_t t0    = ggml_time_us();
    const int64_t moved = to_device ? llama_adapter_lora_page_in(lora.ptr) : llama_adapter_lora_page_out(lora.ptr);
    if (moved < 0) {
        throw server_page_error(ERROR_TYPE_SERVER, "LoRA adapter " + std::to_string(id) + " could not be paged " +
                                (to_device ? "in: the device refused its buffers" : "out: no host memory for its tensors"));
    }
    server_gene_move m;
    m.id        = id;
    m.path      = lora.path;
    m.on_device = !llama_adapter_lora_is_paged_out(lora.ptr);
    m.n_bytes   = llama_adapter_lora_n_bytes(lora.ptr);
    m.n_moved   = (uint64_t) moved;
    m.t_ms      = (ggml_time_us() - t0) / 1000.0;
    return m;
}

server_gene_move server_gene_page(const std::vector<common_adapter_lora_info> & loras, int32_t id, bool to_device,
                                  int32_t learn_lora_id, const std::function<int(int32_t)> & applied_by_processing_slot) {
    if (id < 0 || (size_t) id >= loras.size() || loras[id].ptr == nullptr) {
        throw server_page_error(ERROR_TYPE_INVALID_REQUEST, "no loaded LoRA adapter with id " + std::to_string(id));
    }
    if (!to_device) {
        if (id == learn_lora_id) {
            throw server_page_error(ERROR_TYPE_INVALID_REQUEST, "LoRA adapter " + std::to_string(id) +
                                    " is bound to the learning optimizer; it stays on the device");
        }
        if (loras[id].scale != 0.0f) {
            throw server_page_error(ERROR_TYPE_INVALID_REQUEST, "LoRA adapter " + std::to_string(id) +
                                    " is active by default (scale " + std::to_string(loras[id].scale) + "); set its scale to 0 first");
        }
        const int slot = applied_by_processing_slot(id);
        if (slot >= 0) {
            throw server_page_error(ERROR_TYPE_UNAVAILABLE, "LoRA adapter " + std::to_string(id) + " is in use by slot " +
                                    std::to_string(slot) + "; retry when it finishes");
        }
    }
    return gene_move(loras[id], id, to_device);
}

std::vector<server_gene_move> server_gene_page_in_applied(const std::vector<common_adapter_lora_info> & applied) {
    std::vector<server_gene_move> moved;
    for (size_t i = 0; i < applied.size(); ++i) {
        const auto & lora = applied[i];
        if (lora.ptr != nullptr && lora.scale != 0.0f && llama_adapter_lora_is_paged_out(lora.ptr)) {
            moved.push_back(gene_move(lora, (int32_t) i, true));
        }
    }
    return moved;
}

std::pair<uint64_t, uint64_t> server_gene_bytes(const std::vector<common_adapter_lora_info> & loras) {
    uint64_t device = 0;
    uint64_t host   = 0;
    for (const auto & lora : loras) {
        if (lora.ptr == nullptr) {
            continue;
        }
        (llama_adapter_lora_is_paged_out(lora.ptr) ? host : device) += llama_adapter_lora_n_bytes(lora.ptr);
    }
    return { device, host };
}
