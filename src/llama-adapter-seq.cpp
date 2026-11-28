// Continuum: an adapter set per sequence (llama-adapter-seq.h).

#include "llama-adapter-seq.h"

#include "llama-context.h"
#include "llama-graph.h"
#include "llama-impl.h"
#include "llama-model.h"

#include <algorithm>
#include <mutex>
#include <shared_mutex>

//
// the sets
//

llama_adapter_set llama_adapter_set_of(const llama_adapter_loras & loras, const llama_adapter_lora * shadow) {
    llama_adapter_set set;
    set.adapters.reserve(loras.size());
    for (const auto & [adapter, scale] : loras) {
        if (adapter != shadow && scale != 0.0f) {
            set.adapters.emplace_back(adapter, scale);
        }
    }
    std::sort(set.adapters.begin(), set.adapters.end());
    return set;
}

bool llama_adapter_seq_groups::gathered() const {
    return std::any_of(groups.begin(), groups.end(), [](const llama_adapter_seq_group & group) { return !group.consecutive(); });
}

llama_adapter_set llama_adapter_seq_sets::wears(llama_seq_id seq, const llama_adapter_set & context_set) const {
    const auto it = by_seq.find(seq);
    return it == by_seq.end() ? context_set : it->second.set;
}

llama_adapter_seq_groups llama_adapter_seq_sets::group(const llama_ubatch & ubatch, const llama_adapter_set & context_set, uint64_t lora_frees) const {
    llama_adapter_seq_groups out;
    if (by_seq.empty()) {
        return out;
    }
    bool named = false;
    for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
        named = named || by_seq.count(ubatch.seq_id_unq[s]) > 0;
    }
    if (!named) {
        return out;
    }
    out.named      = true;
    out.lora_frees = lora_frees;
    // a row's set is its first sequence's: decode refused a token whose sequences wear different sets (refuses)
    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        llama_adapter_set set = wears(ubatch.seq_id[i][0], context_set);
        if (set.adapters.empty()) {
            continue; // the base: no delta, no group
        }
        auto it = std::find_if(out.groups.begin(), out.groups.end(), [&](const llama_adapter_seq_group & group) { return group.set == set; });
        if (it == out.groups.end()) {
            out.groups.push_back({ std::move(set), (int32_t) i, 1, {} });
            continue;
        }
        if (it->consecutive() && it->begin + it->n == (int32_t) i) {
            ++it->n;
            continue;
        }
        if (it->consecutive()) { // its rows interleave with another group's: list them
            for (int32_t r = it->begin; r < it->begin + it->n; ++r) {
                it->rows.push_back(r);
            }
        }
        it->rows.push_back((int32_t) i);
        ++it->n;
    }
    if (out.gathered()) {
        ++gathered;
    }
    return out;
}

const char * llama_adapter_seq_sets::refuses(const llama_batch_ext & batch, const llama_adapter_set & context_set, uint64_t lora_frees) const {
    if (by_seq.empty()) {
        return nullptr;
    }
    for (const auto & [seq, named] : by_seq) {
        if (named.lora_frees != lora_frees && !named.set.adapters.empty()) {
            return "a sequence's adapter set was named before an adapter was freed; name it again";
        }
    }
    for (const auto & token : batch.tokens) {
        if (token.seq_ids.size() < 2) {
            continue;
        }
        const llama_adapter_set first = wears(*token.seq_ids.begin(), context_set);
        for (const llama_seq_id seq : token.seq_ids) {
            if (wears(seq, context_set) != first) {
                return "a token is shared by sequences that wear different adapter sets; submit it once per set";
            }
        }
    }
    return nullptr;
}

//
// the context
//

int32_t llama_context::set_seq_adapters_lora(llama_seq_id seq, llama_adapter_lora ** adapters, size_t n_adapters, const float * scales) {
    if (seq < 0 || (uint32_t) seq >= cparams.n_seq_max) {
        LLAMA_LOG_ERROR("%s: sequence %d is outside the context's %u\n", __func__, seq, cparams.n_seq_max);
        return -1;
    }
    llama_adapter_seq_sets::named named;
    {
        std::shared_lock<std::shared_mutex> lock(model.loras_mutex); // an adapter may load or free on another thread
        for (size_t i = 0; i < n_adapters; ++i) {
            if (model.loras.count(adapters[i]) == 0) {
                LLAMA_LOG_ERROR("%s: adapter %zu is not loaded on this model\n", __func__, i);
                return -2;
            }
            if (scales[i] != 0.0f) {
                named.set.adapters.emplace_back(adapters[i], scales[i]);
            }
        }
        named.lora_frees = model.lora_frees.load(std::memory_order_acquire);
    }
    std::sort(named.set.adapters.begin(), named.set.adapters.end());
    seq_adapters.by_seq[seq] = std::move(named);
    // groups add each set's deltas: the scheduler's node budget is re-reserved only if they could exceed it
    sched_need_reserve = sched_need_reserve || graph_max_nodes() > sched_max_nodes;
    return 0;
}

void llama_context::clear_seq_adapters_lora(llama_seq_id seq) {
    seq_adapters.by_seq.erase(seq);
}

llama_adapter_set llama_context::seq_adapters_context_set() const {
    return llama_adapter_set_of(*loras, opt_shadow);
}

bool llama_context::seq_adapters_refuse(const llama_batch_ext & batch) const {
    const char * why = seq_adapters.refuses(batch, seq_adapters_context_set(), model.lora_frees.load(std::memory_order_acquire));
    if (why != nullptr) {
        LLAMA_LOG_ERROR("%s: %s\n", __func__, why);
    }
    return why != nullptr;
}

void llama_context::seq_adapters_graph_params(llm_graph_params & params, const llama_ubatch & ubatch) const {
    params.seq_groups = seq_adapters.group(ubatch, seq_adapters_context_set(), model.lora_frees.load(std::memory_order_acquire));
}

uint32_t llama_context::seq_adapters_max_nodes() const {
    if (seq_adapters.by_seq.empty()) {
        return 0;
    }
    // every group may wear every adapter, each delta adding the views of its rows and its add into them. Called by
    // graph_max_nodes under its shared lock on model.loras_mutex (a second shared lock on one thread is undefined)
    uint32_t per_set = 0;
    for (const auto & lora : model.loras) {
        per_set += lora->get_n_nodes() * 2u;
    }
    return per_set * cparams.n_seq_max;
}

//
// the graph
//

namespace {

// the rows of an interleaved group, set from its list
class llm_graph_input_seq_rows : public llm_graph_input_i {
public:
    explicit llm_graph_input_seq_rows(std::vector<int32_t> rows) : rows(std::move(rows)) {}

    void set_input(const llama_ubatch * ubatch) override {
        GGML_UNUSED(ubatch);
        ggml_backend_tensor_set(tensor, rows.data(), 0, rows.size() * sizeof(int32_t));
    }

    // the graph is reused only under the same groups (llm_graph_params::allow_reuse), so the same rows
    bool can_reuse(const llm_graph_params & params) override {
        GGML_UNUSED(params);
        return true;
    }

    std::vector<int32_t> rows;
    ggml_tensor *        tensor = nullptr;
};

// t's rows [begin, begin + n), its tokens its outermost dimension
ggml_tensor * rows_view(ggml_context * ctx, ggml_tensor * t, int64_t n_tokens, int32_t begin, int32_t n) {
    const int d = ggml_n_dims(t) - 1;
    GGML_ASSERT(t->ne[d] == n_tokens && "an adapter group's rows are its tensor's outermost dimension");
    switch (d) {
        case 0:  return ggml_view_1d(ctx, t, n, begin * t->nb[0]);
        case 1:  return ggml_view_2d(ctx, t, t->ne[0], n, t->nb[1], begin * t->nb[1]);
        default: return ggml_view_3d(ctx, t, t->ne[0], t->ne[1], n, t->nb[1], t->nb[2], begin * t->nb[2]);
    }
}

} // namespace

// one delta per adapter of each group that wears one, on that group's rows; `term` builds an adapter's delta on rows
// [begin, begin + n) of its inputs (all of them when n is the ubatch's)
template <typename Term>
ggml_tensor * llm_graph_context::seq_lora_apply(ggml_tensor * res, ggml_tensor * w, Term term) const {
    for (const auto & group : seq_groups.groups) {
        for (const auto & [adapter, scale] : group.set.adapters) {
            llama_adapter_lora_weight * lw = adapter->get_weight(w);
            if (lw == nullptr) {
                continue;
            }
            if (group.n == (int32_t) n_tokens) {
                res = ggml_add(ctx0, res, term(adapter, lw, scale, 0, group.n));
            } else if (group.consecutive()) {
                const int d = ggml_n_dims(res) - 1;
                res = ggml_acc(ctx0, res, term(adapter, lw, scale, group.begin, group.n), res->nb[1], res->nb[2], res->nb[3], group.begin * res->nb[d]);
            } else {
                res = seq_rows_add(res, term(adapter, lw, scale, 0, (int32_t) n_tokens), group.rows);
            }
        }
    }
    // the training shadow is the context's, on the labelled rows whatever set they wear
    for (const auto & [adapter, scale] : *loras) {
        llama_adapter_lora_weight * lw = adapter == shadow && lora_on(adapter) ? adapter->get_weight(w) : nullptr;
        if (lw != nullptr) {
            res = shadow_add(res, term(adapter, lw, scale, 0, (int32_t) n_tokens));
        }
    }
    return res;
}

// base plus term on the listed rows only, as shadow_add: gathered, added, written back into a copy of base, so every
// other row is base byte for byte
ggml_tensor * llm_graph_context::seq_rows_add(ggml_tensor * base, ggml_tensor * term, const std::vector<int32_t> & rows) const {
    auto inp = std::make_unique<llm_graph_input_seq_rows>(rows);
    inp->tensor = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, (int64_t) rows.size());
    ggml_set_input(inp->tensor);
    ggml_tensor * idx = inp->tensor;
    res->add_input(std::move(inp));

    GGML_ASSERT(ggml_is_contiguous(base));
    const int64_t n  = ggml_nelements(base) / n_tokens;
    ggml_tensor * b2 = ggml_n_dims(base) <= 2 ? base : ggml_reshape_2d(ctx0, base, n, n_tokens);
    ggml_tensor * t2 = ggml_n_dims(term) <= 2 ? term : ggml_reshape_2d(ctx0, ggml_cont(ctx0, term), n, n_tokens);
    ggml_tensor * sel = ggml_add(ctx0, ggml_get_rows(ctx0, b2, idx), ggml_get_rows(ctx0, t2, idx));
    ggml_tensor * updated = ggml_set_rows(ctx0, ggml_dup(ctx0, b2), sel, idx);
    return ggml_n_dims(base) <= 2 ? updated : ggml_reshape(ctx0, updated, base);
}

ggml_tensor * llm_graph_context::seq_lora_mm(ggml_tensor * w, ggml_tensor * cur, ggml_tensor * out) const {
    return seq_lora_apply(out, w, [&](llama_adapter_lora * adapter, llama_adapter_lora_weight * lw, float scale, int32_t begin, int32_t n) {
        ggml_tensor * x = n == (int32_t) n_tokens ? cur : rows_view(ctx0, cur, n_tokens, begin, n);
        ggml_tensor * ab = ggml_mul_mat(ctx0, lw->b, ggml_mul_mat(ctx0, lw->a, x));
        return ggml_scale(ctx0, ab, lw->get_scale(adapter->alpha, scale));
    });
}

ggml_tensor * llm_graph_context::seq_lora_mm_id(ggml_tensor * w, ggml_tensor * cur, ggml_tensor * ids, ggml_tensor * out) const {
    return seq_lora_apply(out, w, [&](llama_adapter_lora * adapter, llama_adapter_lora_weight * lw, float scale, int32_t begin, int32_t n) {
        const bool all = n == (int32_t) n_tokens;
        ggml_tensor * x  = all ? cur : rows_view(ctx0, cur, n_tokens, begin, n);
        ggml_tensor * id = all ? ids : rows_view(ctx0, ids, n_tokens, begin, n);
        ggml_tensor * ab = ggml_mul_mat_id(ctx0, lw->b, ggml_mul_mat_id(ctx0, lw->a, x, id), id);
        // as build_lora_mm_id: the rank is B's input width
        const float rank = (float) lw->b->ne[0];
        return ggml_scale(ctx0, ab, adapter->alpha ? scale * adapter->alpha / rank : scale);
    });
}

ggml_tensor * llm_graph_context::seq_lora_embd(ggml_tensor * tok_embd, ggml_tensor * ids, ggml_tensor * cur) const {
    return seq_lora_apply(cur, tok_embd, [&](llama_adapter_lora * adapter, llama_adapter_lora_weight * lw, float scale, int32_t begin, int32_t n) {
        ggml_tensor * id = n == (int32_t) n_tokens ? ids : rows_view(ctx0, ids, n_tokens, begin, n);
        return ggml_scale(ctx0, ggml_mul_mat(ctx0, lw->b, ggml_get_rows(ctx0, lw->a, id)), lw->get_scale(adapter->alpha, scale));
    });
}

//
// the API
//

int32_t llama_set_seq_adapters_lora(llama_context * ctx, llama_seq_id seq, llama_adapter_lora ** adapters, size_t n_adapters, const float * scales) {
    if (n_adapters > 0 && (adapters == nullptr || scales == nullptr)) {
        LLAMA_LOG_ERROR("%s: %zu adapters with no adapters or scales\n", __func__, n_adapters);
        return -1;
    }
    return ctx->set_seq_adapters_lora(seq, adapters, n_adapters, scales);
}

void llama_clear_seq_adapters_lora(llama_context * ctx, llama_seq_id seq) {
    ctx->clear_seq_adapters_lora(seq);
}

uint64_t llama_seq_adapters_gathered(const llama_context * ctx) {
    return ctx->seq_adapters.gathered;
}
