#include "llama-pinned-experts.h"

#include "llama-context.h"
#include "llama-cparams.h"
#include "llama-model.h"

#include <cstring>

bool llama_pinned_experts::pinned(const llama_ubatch & ubatch, uint32_t i) const {
    return n_pos > 0 && ubatch.n_seq_id[i] == 1 && ubatch.seq_id[i][0] == seq &&
           ubatch.pos[i] >= pos0 && ubatch.pos[i] < pos0 + n_pos;
}

const int32_t * llama_pinned_experts::at(int32_t il, llama_pos pos) const {
    return ids.data() + ((size_t) il * (size_t) n_pos + (size_t) (pos - pos0)) * (size_t) n_used;
}

void llm_graph_input_pinned_experts::set_input(const llama_ubatch * ubatch) {
    GGML_ASSERT(ggml_backend_buffer_is_host(ids->buffer) && ggml_backend_buffer_is_host(mask->buffer));
    int32_t * id = (int32_t *) ids->data;
    float   * m  = (float *) mask->data;
    const bool every_row = ids->ne[1] == (int64_t) ubatch->n_tokens;
    GGML_ASSERT(every_row || ubatch->output);
    int64_t row = 0;
    for (uint32_t i = 0; i < ubatch->n_tokens; ++i) {
        if (!every_row && !ubatch->output[i]) {
            continue;
        }
        const bool p = pinned.pinned(*ubatch, i);
        m[row] = p ? 1.0f : 0.0f;
        if (p) {
            std::memcpy(id + (size_t) row * pinned.n_used, pinned.at(il, ubatch->pos[i]), sizeof(int32_t) * pinned.n_used);
        } else {
            std::memset(id + (size_t) row * pinned.n_used, 0, sizeof(int32_t) * pinned.n_used);
        }
        ++row;
    }
    GGML_ASSERT(row == ids->ne[1]);
}

ggml_tensor * llama_pin_experts(const llm_graph_context & graph, int32_t il, ggml_tensor * routed) {
    const llama_pinned_experts * pinned = graph.cparams.pinned_experts;
    if (pinned == nullptr || pinned->n_used != routed->ne[0] || il >= pinned->n_layer) {
        return routed;
    }
    bool any = false;
    for (uint32_t i = 0; i < graph.ubatch.n_tokens && !any; ++i) {
        any = pinned->pinned(graph.ubatch, i);
    }
    if (!any) {
        return routed;
    }
    const int64_t n_used = routed->ne[0], n_rows = routed->ne[1];
    auto inp = std::make_unique<llm_graph_input_pinned_experts>(*pinned, il);
    inp->ids  = ggml_new_tensor_2d(graph.ctx0, GGML_TYPE_I32, n_used, n_rows);
    inp->mask = ggml_new_tensor_2d(graph.ctx0, GGML_TYPE_F32, 1, n_rows);
    ggml_set_input(inp->ids);
    ggml_set_input(inp->mask);
    ggml_tensor * ids  = inp->ids;
    ggml_tensor * mask = inp->mask;
    graph.res->add_input(std::move(inp));
    // per row: routed + mask * (pinned - routed), in F32 (expert ids are exact far below 2^24)
    ggml_context * ctx = graph.ctx0;
    ggml_tensor * r = ggml_cast(ctx, routed, GGML_TYPE_F32);
    ggml_tensor * p = ggml_cast(ctx, ids, GGML_TYPE_F32);
    ggml_tensor * mixed = ggml_add(ctx, r, ggml_mul(ctx, ggml_sub(ctx, p, r), mask));
    return ggml_cast(ctx, mixed, GGML_TYPE_I32);
}

bool llama_context::set_pinned_experts(llama_seq_id seq, int32_t n_layer, int32_t n_used, llama_pos pos0, int32_t n_pos, const int32_t * ids) {
    if (n_pos == 0) {
        pinned_experts = {};
        cparams.pinned_experts = nullptr;
        return true;
    }
    const auto & hp = model.hparams;
    if (ids == nullptr || seq < 0 || n_pos < 0 || n_layer != (int32_t) hp.n_layer() || n_used != (int32_t) hp.n_expert_used()) {
        return false;
    }
    const size_t n = (size_t) n_layer * (size_t) n_pos * (size_t) n_used;
    for (size_t k = 0; k < n; ++k) {
        if (ids[k] < 0 || ids[k] >= (int32_t) hp.n_expert) {
            return false; // not an expert of this model
        }
    }
    pinned_experts.seq     = seq;
    pinned_experts.n_layer = n_layer;
    pinned_experts.n_used  = n_used;
    pinned_experts.pos0    = pos0;
    pinned_experts.n_pos   = n_pos;
    pinned_experts.ids.assign(ids, ids + n);
    cparams.pinned_experts = &pinned_experts;
    return true;
}

bool llama_set_pinned_experts(llama_context * ctx, llama_seq_id seq, int32_t n_layer, int32_t n_used, llama_pos pos0, int32_t n_pos, const int32_t * ids) {
    return ctx && ctx->set_pinned_experts(seq, n_layer, n_used, pos0, n_pos, ids);
}
