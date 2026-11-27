#include "llama-pinned-experts.h"

#include "llama-cparams.h"

#include <cstring>

bool llama_pinned_experts::covers(const llama_ubatch & ubatch) const {
    if (n_pos == 0 || ubatch.pos == nullptr) {
        return false;
    }
    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        const llama_pos p = ubatch.pos[i];
        if (p < pos0 || p >= pos0 + n_pos) {
            return false;
        }
    }
    return true;
}

const int32_t * llama_pinned_experts::at(int32_t il, llama_pos pos) const {
    return ids.data() + ((size_t) il * (size_t) n_pos + (size_t) (pos - pos0)) * (size_t) n_used;
}

void llm_graph_input_pinned_experts::set_input(const llama_ubatch * ubatch) {
    GGML_ASSERT(ggml_backend_buffer_is_host(ids->buffer));
    int32_t * data = (int32_t *) ids->data;
    const bool every_row = ids->ne[1] == (int64_t) ubatch->n_tokens;
    GGML_ASSERT(every_row || ubatch->output);
    int64_t row = 0;
    for (uint32_t i = 0; i < ubatch->n_tokens; ++i) {
        if (every_row || ubatch->output[i]) {
            std::memcpy(data + (size_t) row++ * pinned.n_used, pinned.at(il, ubatch->pos[i]), sizeof(int32_t) * pinned.n_used);
        }
    }
    GGML_ASSERT(row == ids->ne[1]);
}

ggml_tensor * llama_build_pinned_experts(const llm_graph_context & graph, int32_t il, int64_t n_used, int64_t n_rows) {
    const llama_pinned_experts * pinned = graph.cparams.pinned_experts;
    if (pinned == nullptr || pinned->n_used != n_used || il >= pinned->n_layer || !pinned->covers(graph.ubatch)) {
        return nullptr;
    }
    auto inp = std::make_unique<llm_graph_input_pinned_experts>(*pinned, il);
    inp->ids = ggml_new_tensor_2d(graph.ctx0, GGML_TYPE_I32, n_used, n_rows);
    ggml_set_input(inp->ids);
    ggml_tensor * ids = inp->ids;
    graph.res->add_input(std::move(inp));
    return ids;
}

#include "llama-context.h"
#include "llama-model.h"

bool llama_context::set_pinned_experts(int32_t n_layer, int32_t n_used, llama_pos pos0, int32_t n_pos, const int32_t * ids) {
    if (n_pos == 0) {
        pinned_experts = {};
        cparams.pinned_experts = nullptr;
        return true;
    }
    const auto & hp = model.hparams;
    if (ids == nullptr || n_pos < 0 || n_layer != (int32_t) hp.n_layer() || n_used != (int32_t) hp.n_expert_used()) {
        return false;
    }
    pinned_experts.n_layer = n_layer;
    pinned_experts.n_used  = n_used;
    pinned_experts.pos0    = pos0;
    pinned_experts.n_pos   = n_pos;
    pinned_experts.ids.assign(ids, ids + (size_t) n_layer * (size_t) n_pos * (size_t) n_used);
    cparams.pinned_experts = &pinned_experts;
    return true;
}

bool llama_set_pinned_experts(llama_context * ctx, int32_t n_layer, int32_t n_used, llama_pos pos0, int32_t n_pos, const int32_t * ids) {
    return ctx && ctx->set_pinned_experts(n_layer, n_used, pos0, n_pos, ids);
}
