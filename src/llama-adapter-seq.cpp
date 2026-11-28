// Continuum: an adapter set per sequence (llama-adapter-seq.h).

#include "llama-adapter-seq.h"

#include "llama-context.h"
#include "llama-graph.h"
#include "llama-impl.h"
#include "llama-model.h"

#include <algorithm>
#include <atomic>
#include <mutex>
#include <shared_mutex>

//
// the sets
//

uint64_t llama_adapter_lora_next_serial() {
    static std::atomic<uint64_t> next{1};
    return next.fetch_add(1, std::memory_order_relaxed);
}

llama_adapter_set llama_adapter_set_of(const llama_adapter_loras & loras, const llama_adapter_lora * shadow) {
    llama_adapter_set set;
    set.adapters.reserve(loras.size());
    for (const auto & [adapter, scale] : loras) {
        if (adapter != shadow && scale != 0.0f) {
            set.adapters.push_back({ adapter, adapter->serial, scale });
        }
    }
    std::sort(set.adapters.begin(), set.adapters.end());
    return set;
}

void llama_adapter_seq_rows::add(int32_t row) {
    if (n == 0) {
        begin = row;
    } else if (consecutive() && row != begin + n) { // its rows interleave with another group's: list them
        for (int32_t r = begin; r < begin + n; ++r) {
            rows.push_back(r);
        }
    }
    if (!consecutive()) {
        rows.push_back(row);
    }
    ++n;
}

bool llama_adapter_seq_groups::gathered() const {
    return std::any_of(groups.begin(), groups.end(), [](const llama_adapter_seq_group & group) {
        return !group.tokens.consecutive() || !group.outputs.consecutive();
    });
}

const llama_adapter_set & llama_adapter_seq_sets::wears(llama_seq_id seq, const llama_adapter_set & context_set) const {
    const auto it = by_seq.find(seq);
    return it == by_seq.end() ? context_set : it->second;
}

llama_adapter_seq_groups llama_adapter_seq_sets::group(const llama_ubatch & ubatch, const llama_adapter_set & context_set) const {
    llama_adapter_seq_groups out;
    for (uint32_t s = 0; s < ubatch.n_seqs_unq && !by_seq.empty(); ++s) {
        out.named = out.named || by_seq.count(ubatch.seq_id_unq[s]) > 0;
    }
    if (!out.named) {
        return out;
    }
    // a row's set is its first sequence's: decode refused a token whose sequences wear different sets (refuses). Its
    // output rank is its place among the rows whose logits were asked for (inp_out_ids lists them in row order)
    int32_t output = 0;
    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        const bool outputs = ubatch.output == nullptr || ubatch.output[i];
        const llama_adapter_set & set = wears(ubatch.seq_id[i][0], context_set);
        if (!set.adapters.empty()) { // the base: no delta, no group
            auto it = std::find_if(out.groups.begin(), out.groups.end(), [&](const llama_adapter_seq_group & group) { return group.set == set; });
            if (it == out.groups.end()) {
                it = out.groups.insert(out.groups.end(), llama_adapter_seq_group{ set, {}, {} });
            }
            it->tokens.add((int32_t) i);
            if (outputs) {
                it->outputs.add(output);
            }
        }
        output += outputs;
    }
    if (out.gathered()) {
        ++gathered;
    }
    return out;
}

const char * llama_adapter_seq_sets::refuses(const llama_batch_ext & batch, const llama_adapter_set & context_set, const llama_model & model) const {
    if (by_seq.empty()) {
        return nullptr;
    }
    {
        // only the sequences this batch decodes: a parked mind whose gene was unloaded refuses her own next step, never
        // another's (Codex, #172)
        std::shared_lock<std::shared_mutex> lock(model.loras_mutex); // an adapter may load or free on another thread
        for (const auto & token : batch.tokens) {
            for (const llama_seq_id seq : token.seq_ids) {
                const auto named = by_seq.find(seq);
                if (named == by_seq.end()) {
                    continue;
                }
                for (const auto & entry : named->second.adapters) {
                    // the address must still be loaded on the model, and be the same load (read only once known alive)
                    if (model.loras.count(entry.adapter) == 0 || entry.adapter->serial != entry.serial) {
                        return "an adapter a sequence of this batch names was freed; name its set again";
                    }
                }
            }
        }
    }
    for (const auto & token : batch.tokens) {
        if (token.seq_ids.size() < 2) {
            continue;
        }
        const llama_adapter_set & first = wears(*token.seq_ids.begin(), context_set);
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
    llama_adapter_set set;
    {
        std::shared_lock<std::shared_mutex> lock(model.loras_mutex); // an adapter may load or free on another thread
        for (size_t i = 0; i < n_adapters; ++i) {
            if (model.loras.count(adapters[i]) == 0) {
                LLAMA_LOG_ERROR("%s: adapter %zu is not loaded on this model\n", __func__, i);
                return -2;
            }
            if (scales[i] != 0.0f) {
                set.adapters.push_back({ adapters[i], adapters[i]->serial, scales[i] });
            }
        }
    }
    std::sort(set.adapters.begin(), set.adapters.end());
    seq_adapters.by_seq[seq] = std::move(set);
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
    const char * why = seq_adapters.refuses(batch, seq_adapters_context_set(), model);
    if (why != nullptr) {
        LLAMA_LOG_ERROR("%s: %s\n", __func__, why);
    }
    return why != nullptr;
}

void llama_context::seq_adapters_graph_params(llm_graph_params & params, const llama_ubatch & ubatch) const {
    params.seq_groups = seq_adapters.group(ubatch, seq_adapters_context_set());
}

uint32_t llama_context::seq_adapters_max_nodes() const {
    if (seq_adapters.by_seq.empty()) {
        return 0;
    }
    // what the named sets add, never what the model holds: a genome may hold many genes loaded while each mind wears a
    // few, and the graph's budget follows the worn ones. Each worn adapter adds, per group, its delta (get_n_nodes) and
    // the views and add into its rows (as many again). A set worn by several sequences is one group: counted once.
    // Called by graph_max_nodes under its shared lock on model.loras_mutex (a second shared lock on one thread is
    // undefined); an adapter freed since it was named is skipped (decode refuses it), never read.
    uint32_t nodes = 0;
    std::vector<const llama_adapter_set *> counted;
    for (const auto & [seq, set] : seq_adapters.by_seq) {
        if (std::any_of(counted.begin(), counted.end(), [&](const llama_adapter_set * other) { return *other == set; })) {
            continue;
        }
        counted.push_back(&set);
        for (const auto & entry : set.adapters) {
            if (model.loras.count(entry.adapter) > 0 && entry.adapter->serial == entry.serial) {
                nodes += entry.adapter->get_n_nodes() * 2u;
            }
        }
    }
    return nodes;
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

// t's rows [begin, begin + n) along dimension `d`, its rows
ggml_tensor * rows_view(ggml_context * ctx, ggml_tensor * t, int d, int32_t begin, int32_t n) {
    switch (d) {
        case 0:  return ggml_view_1d(ctx, t, n, begin * t->nb[0]);
        case 1:  return ggml_view_2d(ctx, t, t->ne[0], n, t->nb[1], begin * t->nb[1]);
        default: return ggml_view_3d(ctx, t, t->ne[0], t->ne[1], n, t->nb[1], t->nb[2], begin * t->nb[2]);
    }
}

} // namespace

// The rows a tensor holds: the ubatch's tokens, or (after inp_out_ids) its outputs. A count that is neither is a shape
// this path does not know, and building on it would add a delta to the wrong rows.
const llama_adapter_seq_rows & llm_graph_context::seq_rows_of(const llama_adapter_seq_group & group, int64_t rows) const {
    if (rows == (int64_t) n_tokens) {
        return group.tokens;
    }
    if (rows == (int64_t) n_outputs) {
        return group.outputs;
    }
    GGML_ABORT("an adapter target holds %lld rows: neither the ubatch's %u tokens nor its %u outputs", (long long) rows, n_tokens, n_outputs);
}

// one delta per adapter of each group that wears one, on that group's rows of `res`, which holds `rows` rows along
// dimension `d`; `term` builds an adapter's delta on rows [begin, begin + n) of its inputs (all of them when n is `rows`)
template <typename Term>
ggml_tensor * llm_graph_context::seq_lora_apply(ggml_tensor * res, ggml_tensor * w, int d, int64_t rows, Term term) const {
    for (const auto & group : seq_groups.groups) {
        const llama_adapter_seq_rows & mine = seq_rows_of(group, rows);
        if (mine.n == 0) {
            continue; // none of this group's rows are here (none of them asked for logits)
        }
        for (const auto & entry : group.set.adapters) {
            llama_adapter_lora_weight * lw = entry.adapter->get_weight(w);
            if (lw == nullptr) {
                continue;
            }
            if (mine.n == rows) {
                res = ggml_add(ctx0, res, term(entry.adapter, lw, entry.scale, 0, mine.n));
            } else if (mine.consecutive()) {
                res = ggml_acc(ctx0, res, term(entry.adapter, lw, entry.scale, mine.begin, mine.n), res->nb[1], res->nb[2], res->nb[3], mine.begin * res->nb[d]);
            } else {
                res = seq_rows_add(res, term(entry.adapter, lw, entry.scale, 0, (int32_t) rows), mine.rows, rows);
            }
        }
    }
    // the training shadow is the context's, on the labelled rows whatever set they wear
    for (const auto & [adapter, scale] : *loras) {
        llama_adapter_lora_weight * lw = adapter == shadow && lora_on(adapter) ? adapter->get_weight(w) : nullptr;
        if (lw != nullptr) {
            res = shadow_add(res, term(adapter, lw, scale, 0, (int32_t) rows));
        }
    }
    return res;
}

// base plus term on the listed rows only, as shadow_add: gathered, added, written back into a copy of base, so every
// other row is base byte for byte
ggml_tensor * llm_graph_context::seq_rows_add(ggml_tensor * base, ggml_tensor * term, const std::vector<int32_t> & listed, int64_t rows) const {
    auto inp = std::make_unique<llm_graph_input_seq_rows>(listed);
    inp->tensor = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, (int64_t) listed.size());
    ggml_set_input(inp->tensor);
    ggml_tensor * idx = inp->tensor;
    res->add_input(std::move(inp));

    GGML_ASSERT(ggml_is_contiguous(base));
    const int64_t n  = ggml_nelements(base) / rows;
    ggml_tensor * b2 = ggml_n_dims(base) <= 2 ? base : ggml_reshape_2d(ctx0, base, n, rows);
    ggml_tensor * t2 = ggml_n_dims(term) <= 2 ? term : ggml_reshape_2d(ctx0, ggml_cont(ctx0, term), n, rows);
    ggml_tensor * sel = ggml_add(ctx0, ggml_get_rows(ctx0, b2, idx), ggml_get_rows(ctx0, t2, idx));
    ggml_tensor * updated = ggml_set_rows(ctx0, ggml_dup(ctx0, b2), sel, idx);
    return ggml_n_dims(base) <= 2 ? updated : ggml_reshape(ctx0, updated, base);
}

ggml_tensor * llm_graph_context::seq_lora_mm(ggml_tensor * w, ggml_tensor * cur, ggml_tensor * out) const {
    // a projection's input is [n_in, rows] or [n_in, x, rows]: its rows are its outermost dimension
    GGML_ASSERT(cur->ne[3] == 1 && "a LoRA target's input holds its rows in its second or third dimension");
    const int     d    = cur->ne[2] > 1 ? 2 : 1;
    const int64_t rows = cur->ne[d];
    return seq_lora_apply(out, w, d, rows, [&](llama_adapter_lora * adapter, llama_adapter_lora_weight * lw, float scale, int32_t begin, int32_t n) {
        ggml_tensor * x = n == rows ? cur : rows_view(ctx0, cur, d, begin, n);
        ggml_tensor * ab = ggml_mul_mat(ctx0, lw->b, ggml_mul_mat(ctx0, lw->a, x));
        return ggml_scale(ctx0, ab, lw->get_scale(adapter->alpha, scale));
    });
}

ggml_tensor * llm_graph_context::seq_lora_mm_id(ggml_tensor * w, ggml_tensor * cur, ggml_tensor * ids, ggml_tensor * out) const {
    const int64_t rows = ids->ne[1]; // ids [n_expert_used, rows]; cur and the result hold their rows in dimension 2
    return seq_lora_apply(out, w, 2, rows, [&](llama_adapter_lora * adapter, llama_adapter_lora_weight * lw, float scale, int32_t begin, int32_t n) {
        const bool all = n == rows;
        ggml_tensor * x  = all ? cur : rows_view(ctx0, cur, 2, begin, n);
        ggml_tensor * id = all ? ids : rows_view(ctx0, ids, 1, begin, n);
        ggml_tensor * ab = ggml_mul_mat_id(ctx0, lw->b, ggml_mul_mat_id(ctx0, lw->a, x, id), id);
        // as build_lora_mm_id: the rank is B's input width
        const float rank = (float) lw->b->ne[0];
        return ggml_scale(ctx0, ab, adapter->alpha ? scale * adapter->alpha / rank : scale);
    });
}

ggml_tensor * llm_graph_context::seq_lora_embd(ggml_tensor * tok_embd, ggml_tensor * ids, ggml_tensor * cur) const {
    const int64_t rows = ids->ne[0]; // the token ids, one per row
    return seq_lora_apply(cur, tok_embd, 1, rows, [&](llama_adapter_lora * adapter, llama_adapter_lora_weight * lw, float scale, int32_t begin, int32_t n) {
        ggml_tensor * id = n == rows ? ids : rows_view(ctx0, ids, 0, begin, n);
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
