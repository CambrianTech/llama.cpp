#include "llama-lived-routing.h"

#include "llama-context.h"
#include "llama-cparams.h"
#include "llama-model.h"

#include <algorithm>
#include <cstring>

// ---- the table ----

const uint8_t * llama_lived_routing::at(llama_seq_id seq, llama_pos pos, int32_t il) const {
    if (seq < 0 || (size_t) seq >= seqs.size() || pos < 0) {
        return nullptr;
    }
    const auto & rows = seqs[seq];
    const size_t flag = (size_t) pos * (size_t) n_layer + (size_t) il;
    if (flag >= rows.lived.size() || !rows.lived[flag]) {
        return nullptr;
    }
    return rows.ids.data() + flag * (size_t) n_used;
}

void llama_lived_routing::write(llama_seq_id seq, llama_pos pos, int32_t il, const int32_t * ids) {
    if ((size_t) seq >= seqs.size()) {
        seqs.resize((size_t) seq + 1);
    }
    auto & rows = seqs[seq];
    const size_t n_flags = ((size_t) pos + 1) * (size_t) n_layer;
    if (rows.lived.size() < n_flags) {
        rows.lived.resize(n_flags, 0);
        rows.ids.resize(n_flags * (size_t) n_used, 0);
    }
    const size_t flag = (size_t) pos * (size_t) n_layer + (size_t) il;
    uint8_t * dst = rows.ids.data() + flag * (size_t) n_used;
    for (int32_t k = 0; k < n_used; ++k) {
        dst[k] = (uint8_t) ids[k];
    }
    rows.lived[flag] = 1;
}

bool llama_lived_routing::lived(llama_seq_id seq, llama_pos pos0, int32_t n) const {
    for (int32_t p = 0; p < n; ++p) {
        if (at(seq, pos0 + p, 0) == nullptr) {
            return false;
        }
    }
    return true;
}

bool llama_lived_routing::pinned(const llama_ubatch & ubatch, uint32_t i) const {
    return pin_n > 0 && ubatch.n_seq_id[i] == 1 && ubatch.seq_id[i][0] == pin_seq &&
           ubatch.pos[i] >= pin0 && ubatch.pos[i] < pin0 + pin_n;
}

size_t llama_lived_routing::bytes() const {
    size_t total = 0;
    for (const auto & rows : seqs) {
        total += rows.ids.capacity() + rows.lived.capacity();
    }
    return total;
}

// ---- the graph ----

void llm_graph_input_pinned_experts::set_input(const llama_ubatch * ubatch) {
    GGML_ASSERT(ggml_backend_buffer_is_host(ids->buffer) && ggml_backend_buffer_is_host(mask->buffer));
    int32_t * id = (int32_t *) ids->data;
    float   * m  = (float *) mask->data;
    const int32_t n_used = lived.n_used;
    const bool every_row = ids->ne[1] == (int64_t) ubatch->n_tokens;
    GGML_ASSERT(every_row || ubatch->output);
    int64_t row = 0;
    for (uint32_t i = 0; i < ubatch->n_tokens; ++i) {
        if (!every_row && !ubatch->output[i]) {
            continue;
        }
        const uint8_t * lived_ids = lived.pinned(*ubatch, i) ? lived.at(ubatch->seq_id[i][0], ubatch->pos[i], il) : nullptr;
        m[row] = lived_ids ? 1.0f : 0.0f;
        for (int32_t k = 0; k < n_used; ++k) {
            id[(size_t) row * n_used + k] = lived_ids ? (int32_t) lived_ids[k] : 0;
        }
        ++row;
    }
    GGML_ASSERT(row == ids->ne[1]);
}

bool llm_graph_input_pinned_experts::can_reuse(const llm_graph_params & params) {
    // the same pin is still set (setting or clearing one also drops the cached graphs)
    return params.cparams.lived_routing == &lived && lived.pin_n > 0;
}

ggml_tensor * llama_route_experts(const llm_graph_context & graph, int32_t il, ggml_tensor * routed) {
    const llama_lived_routing * lived = graph.cparams.lived_routing;
    if (lived == nullptr || lived->n_used != routed->ne[0] || il >= lived->n_layer) {
        return routed;
    }
    ggml_context * ctx = graph.ctx0;
    ggml_tensor * selected = routed;
    if (lived->pin_n > 0) {
        const int64_t n_used = routed->ne[0], n_rows = routed->ne[1];
        auto inp = std::make_unique<llm_graph_input_pinned_experts>(*lived, il);
        inp->ids  = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, n_rows);
        inp->mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, n_rows);
        ggml_set_input(inp->ids);
        ggml_set_input(inp->mask);
        ggml_tensor * ids  = inp->ids;
        ggml_tensor * mask = inp->mask;
        graph.res->add_input(std::move(inp));
        // per row: routed + mask * (lived - routed), in F32 (expert ids are exact far below 2^24)
        ggml_tensor * r = ggml_cast(ctx, routed, GGML_TYPE_F32);
        ggml_tensor * p = ggml_cast(ctx, ids, GGML_TYPE_F32);
        selected = ggml_cast(ctx, ggml_add(ctx, r, ggml_mul(ctx, ggml_sub(ctx, p, r), mask)), GGML_TYPE_I32);
    }
    if (lived->capture && lived->capture_dev != nullptr && !graph.cparams.training) {
        // this layer's choice into its slice of the capture buffer, a copy kernel in the step's own
        // command stream (as the K/V cache is written); one read per ubatch then carries every layer
        ggml_tensor * buf = lived->capture_dev;
        const int64_t rows = selected->ne[1];
        GGML_ASSERT(graph.n_tokens <= buf->ne[1] / lived->n_layer);
        ggml_tensor * dst = ggml_view_2d(ctx, buf, lived->n_used, rows, buf->nb[1], (size_t) il * graph.n_tokens * buf->nb[1]);
        ggml_build_forward_expand(graph.gf, ggml_cpy(ctx, selected, dst));
        graph.res->t_routing.emplace_back(il, dst);
    }
    return selected;
}

// ---- the context ----

// The most tokens a graph of this context is built for: a ubatch, rounded up to a multiple of the sequences as
// graph_reserve rounds it (three streams reserve 2049 of a 2048 ubatch; the M5, 2026-11-28 06:32Z aborted at the
// capture's bound on exactly that).
static int64_t capture_rows(const llama_cparams & cparams) {
    const int64_t seqs = std::max<int64_t>(1, cparams.n_seq_max);
    return ((int64_t) cparams.n_ubatch + seqs - 1) / seqs * seqs;
}

llama_routing_capture llama_context::set_routing_capture(bool on) {
    const auto & hp = model.hparams;
    if (on && hp.n_expert == 0) {
        return LLAMA_ROUTING_CAPTURE_NO_EXPERTS;
    }
    if (on && hp.n_expert > (uint32_t) llama_lived_routing::MAX_EXPERTS) {
        return LLAMA_ROUTING_CAPTURE_TOO_MANY;
    }
    synchronize(); // nothing queued reads the buffer this replaces
    lived_routing.capture_dev = nullptr;
    lived_routing.capture_buf.reset();
    lived_routing.capture_ctx.reset();
    lived_routing.n_layer = (int32_t) hp.n_layer();
    lived_routing.n_used  = (int32_t) hp.n_expert_used();
    if (on) {
        ggml_init_params params = { /*.mem_size =*/ ggml_tensor_overhead(), /*.mem_buffer =*/ nullptr, /*.no_alloc =*/ true };
        lived_routing.capture_ctx.reset(ggml_init(params));
        lived_routing.capture_dev = ggml_new_tensor_2d(lived_routing.capture_ctx.get(), GGML_TYPE_I32, lived_routing.n_used,
                                                       (int64_t) lived_routing.n_layer * capture_rows(cparams));
        ggml_set_name(lived_routing.capture_dev, "lived_routing_capture");
        ggml_backend_dev_t dev = model.dev_output();
        auto * buft = dev ? ggml_backend_dev_buffer_type(dev) : ggml_backend_cpu_buffer_type();
        lived_routing.capture_buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(lived_routing.capture_ctx.get(), buft));
        if (!lived_routing.capture_buf) {
            lived_routing.capture_dev = nullptr;
            lived_routing.capture_ctx.reset();
            lived_routing.capture = false;
            cparams.lived_routing = lived_routing.pin_n > 0 ? &lived_routing : nullptr;
            return LLAMA_ROUTING_CAPTURE_NO_ROOM;
        }
    }
    lived_routing.capture = on;
    cparams.lived_routing = (on || lived_routing.pin_n > 0) ? &lived_routing : nullptr;
    drop_reusable_graphs(); // a graph built the other way registers (or not) what this one needs
    return on ? LLAMA_ROUTING_CAPTURE_RECORDING : LLAMA_ROUTING_CAPTURE_OFF;
}

bool llama_context::set_pinned_experts(llama_seq_id seq, int32_t n_layer, int32_t n_used, llama_pos pos0, int32_t n_pos, const int32_t * ids) {
    if (n_pos == 0) {
        clear_pin();
        return true;
    }
    const auto & hp = model.hparams;
    if (ids == nullptr || seq < 0 || pos0 < 0 || n_pos < 0 || n_layer != (int32_t) hp.n_layer() ||
        n_used != (int32_t) hp.n_expert_used() || hp.n_expert > (uint32_t) llama_lived_routing::MAX_EXPERTS) {
        return false;
    }
    const size_t n = (size_t) n_layer * (size_t) n_pos * (size_t) n_used;
    // -1 (as llama_get_lived_routing reads it): serving never routed that layer there, and the row routes
    for (size_t k = 0; k < n; ++k) {
        if (ids[k] < -1 || ids[k] >= (int32_t) hp.n_expert) {
            return false; // not an expert of this model
        }
    }
    routing_settle(); // nothing queued may land over what is imported
    lived_routing.n_layer = n_layer;
    lived_routing.n_used  = n_used;
    for (int32_t il = 0; il < n_layer; ++il) {
        for (int32_t p = 0; p < n_pos; ++p) {
            const int32_t * row = ids + ((size_t) il * n_pos + p) * n_used;
            if (row[0] >= 0) {
                lived_routing.write(seq, pos0 + p, il, row);
            }
        }
    }
    return pin_lived(seq, pos0, n_pos);
}

bool llama_context::pin_lived(llama_seq_id seq, llama_pos pos0, int32_t n_pos) {
    synchronize(); // every serving ubatch so far has reached the table
    if (n_pos <= 0 || !lived_routing.lived(seq, pos0, n_pos)) {
        return false;
    }
    lived_routing.pin_seq = seq;
    lived_routing.pin0    = pos0;
    lived_routing.pin_n   = n_pos;
    cparams.lived_routing = &lived_routing;
    drop_reusable_graphs(); // a graph built without this pin has no input to carry it
    return true;
}

void llama_context::clear_pin() {
    if (lived_routing.pin_n == 0) {
        return;
    }
    lived_routing.pin_seq = -1;
    lived_routing.pin0    = 0;
    lived_routing.pin_n   = 0;
    cparams.lived_routing = lived_routing.capture ? &lived_routing : nullptr;
    drop_reusable_graphs();
}

void llama_context::routing_queue(const llm_graph_result * res, const llama_ubatch & ubatch) {
    if (res->t_routing.empty() || lived_routing.capture_dev == nullptr) {
        return;
    }
    const int64_t t0 = ggml_time_us();
    // entries and their buffers are reused across steps: a step allocates nothing once warm
    if (routing_used == routing_queued.size()) {
        routing_queued.emplace_back();
    }
    routing_pending & pending = routing_queued[routing_used++];
    pending.n_tokens = ubatch.n_tokens;
    pending.seqs.resize(ubatch.n_tokens);
    pending.pos.resize(ubatch.n_tokens);
    pending.out.resize(ubatch.n_tokens);
    pending.pinned.resize(ubatch.n_tokens);
    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        pending.seqs[i].assign(ubatch.seq_id[i], ubatch.seq_id[i] + ubatch.n_seq_id[i]);
        pending.pos[i]    = ubatch.pos[i];
        pending.out[i]    = ubatch.output ? ubatch.output[i] : 1;
        pending.pinned[i] = lived_routing.pinned(ubatch, i);
    }
    pending.layers.clear();
    int32_t top = 0;
    for (const auto & [il, dst] : res->t_routing) {
        pending.layers.emplace_back(il, dst->ne[1]);
        top = std::max(top, il + 1);
    }
    // one read: the first `top * n_tokens` rows hold every captured layer of this ubatch
    ggml_tensor * buf = lived_routing.capture_dev;
    const size_t bytes = (size_t) top * ubatch.n_tokens * buf->nb[1];
    pending.ids.resize(bytes / sizeof(int32_t));
    ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(sched.get(), res->t_routing.front().second);
    GGML_ASSERT(backend != nullptr);
    ggml_backend_tensor_get_async(backend, buf, pending.ids.data(), 0, bytes);
    lived_routing.host_us += ggml_time_us() - t0;
}

void llama_context::routing_settle() {
    if (routing_used == 0) {
        return;
    }
    const int64_t t0 = ggml_time_us();
    const int32_t n_used = lived_routing.n_used;
    for (size_t q = 0; q < routing_used; ++q) {
        const auto & pending = routing_queued[q];
        const uint32_t n_tokens = pending.n_tokens;
        for (const auto & [il, rows] : pending.layers) {
            const int32_t * layer = pending.ids.data() + (size_t) il * n_tokens * n_used;
            const bool every_row = rows == (int64_t) n_tokens;
            size_t row = 0;
            for (uint32_t i = 0; i < n_tokens; ++i) {
                if (!every_row && !pending.out[i]) {
                    continue; // the last layer ran only the outputs: the others were never routed there
                }
                if (!pending.pinned[i]) { // a pinned row replayed what the table already holds
                    for (const llama_seq_id seq : pending.seqs[i]) {
                        lived_routing.write(seq, pending.pos[i], il, layer + row * (size_t) n_used);
                    }
                }
                ++row;
            }
        }
    }
    routing_used = 0;
    lived_routing.host_us += ggml_time_us() - t0;
}
void llama_context::drop_reusable_graphs() {
    for (auto & res : gf_res_prev) {
        res->reset();
    }
    gf_res_prev_active = nullptr;
    graph_slots_forget(); // Continuum: each mind's kept graph too (llama-graph-slots.cpp)
}

// ---- the API ----

llama_routing_capture llama_set_routing_capture(llama_context * ctx, bool on) {
    GGML_ASSERT(ctx != nullptr);
    return ctx->set_routing_capture(on);
}

bool llama_set_pinned_experts(llama_context * ctx, llama_seq_id seq, int32_t n_layer, int32_t n_used, llama_pos pos0, int32_t n_pos, const int32_t * ids) {
    return ctx && ctx->set_pinned_experts(seq, n_layer, n_used, pos0, n_pos, ids);
}

bool llama_pin_lived_routing(llama_context * ctx, llama_seq_id seq, llama_pos pos0, int32_t n_pos) {
    return ctx && ctx->pin_lived(seq, pos0, n_pos);
}

size_t llama_lived_routing_bytes(const llama_context * ctx) {
    return ctx ? ctx->lived_routing.bytes() : 0;
}

bool llama_get_lived_routing(llama_context * ctx, llama_seq_id seq, llama_pos pos0, int32_t n_pos, int32_t * ids) {
    if (ctx == nullptr || ids == nullptr || n_pos <= 0) {
        return false;
    }
    ctx->synchronize(); // every serving ubatch so far has reached the table
    const llama_lived_routing & lived = ctx->lived_routing;
    if (!lived.lived(seq, pos0, n_pos)) {
        return false;
    }
    for (int32_t il = 0; il < lived.n_layer; ++il) {
        for (int32_t p = 0; p < n_pos; ++p) {
            int32_t * dst = ids + ((size_t) il * n_pos + p) * lived.n_used;
            const uint8_t * src = lived.at(seq, pos0 + p, il);
            for (int32_t k = 0; k < lived.n_used; ++k) {
                dst[k] = src ? (int32_t) src[k] : -1; // -1: serving never routed this layer here
            }
        }
    }
    return true;
}

int64_t llama_lived_routing_host_us(const llama_context * ctx) {
    return ctx ? ctx->lived_routing.host_us : 0;
}
