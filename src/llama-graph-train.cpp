// Continuum: what a training graph adds to the serving graph (the walk, the shadow adapter, the training head, the
// recurrent state a walk keeps), kept beside llama-graph.cpp so upstream's graph merges without touching it. Upstream's
// functions reach this file through one-line hooks named build_attn_train, train_fa_kv and train_fa_node.
#include "llama-graph.h"

#include "llama-impl.h"
#include "llama-model.h"
#include "llama-batch.h"
#include "llama-cparams.h"
#include "ggml-opt.h"
#include "llama-kv-cache.h"
#include "llama-memory-recurrent.h"

#include <algorithm>
#include <cassert>
#include <numeric>
#include <vector>

void llm_graph_input_training_rows::set_input(const llama_ubatch * ubatch) {
    const auto map = ubatch->rows();
    if (public_rows) {
        GGML_ASSERT(public_rows->ne[0] == (int64_t) map.public_rows.size());
        ggml_backend_tensor_set(public_rows, map.public_rows.data(), 0, ggml_nbytes(public_rows));
    }
    if (label_rows) {
        GGML_ASSERT(label_rows->ne[0] == (int64_t) map.label_rows.size());
        ggml_backend_tensor_set(label_rows, map.label_rows.data(), 0, ggml_nbytes(label_rows));
    }
    if (shadow_rows && shadow_rows->buffer) {
        ggml_backend_tensor_set(shadow_rows, map.label_rows.data(), 0, ggml_nbytes(shadow_rows));
    }
    if (public_shadow_rows && public_shadow_rows->buffer) {
        std::vector<int32_t> local;
        for (size_t i = 0; i < map.public_rows.size(); ++i) {
            if (ubatch->label[map.public_rows[i]] >= 0) {
                local.push_back((int32_t) i);
            }
        }
        ggml_backend_tensor_set(public_shadow_rows, local.data(), 0, ggml_nbytes(public_shadow_rows));
    }
    for (auto * rows : slice_shadow_rows) {
        if (rows->buffer) {
            std::vector<int32_t> local(rows->ne[0]);
            std::iota(local.begin(), local.end(), 0);
            ggml_backend_tensor_set(rows, local.data(), 0, ggml_nbytes(rows));
        }
    }
}

bool llm_graph_context::uses_flash_attn(ggml_tensor * kq_b, ggml_tensor * sinks, ggml_tensor * v_mla) const {
    const bool fa_back_ok = !cparams.training ||
        (hparams.f_max_alibi_bias == 0.0f && !hparams.attn_soft_cap && sinks == nullptr && v_mla == nullptr);
    return cparams.flash_attn && kq_b == nullptr && fa_back_ok;
}

ggml_tensor * llm_graph_context::build_inp_train_rows() const {
    if (inp_train_rows == nullptr) {
        auto inp = std::make_unique<llm_graph_input_pos>(1);
        inp->pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
        ggml_set_input(inp->pos);
        ggml_set_name(inp->pos, "train_rows");
        inp_train_rows = inp->pos;
        res->add_input(std::move(inp));
    }
    return inp_train_rows;
}

bool llm_graph_context::lora_on(const llama_adapter_lora * adapter) const {
    return adapter != shadow || n_shadow_rows > 0;
}

llm_graph_input_training_rows * llm_graph_context::training_rows() const {
    if (!inp_training_rows) {
        auto inp = std::make_unique<llm_graph_input_training_rows>();
        inp_training_rows = inp.get();
        res->add_input(std::move(inp));
    }
    return inp_training_rows;
}

ggml_tensor * llm_graph_context::shadow_add(ggml_tensor * base, ggml_tensor * term) const {
    if (training_rows()->shadow_rows == nullptr) {
        auto * inp = training_rows();
        inp->shadow_rows = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_shadow_rows);
        ggml_set_input(inp->shadow_rows);
        ggml_set_name(inp->shadow_rows, "label_rows");
    }
    // one row per token: the tokens are base's outermost dimension
    GGML_ASSERT(ggml_is_contiguous(base) && (n_tokens == 1 || base->ne[ggml_n_dims(base) - 1] == n_tokens));
    const int64_t n = ggml_nelements(base) / n_tokens;
    ggml_tensor * b2  = ggml_n_dims(base) <= 2 ? base : ggml_reshape_2d(ctx0, base, n, n_tokens);
    ggml_tensor * t2  = ggml_n_dims(term) <= 2 ? term : ggml_reshape_2d(ctx0, ggml_cont(ctx0, term), n, n_tokens);
    ggml_tensor * sel = ggml_add(ctx0, ggml_get_rows(ctx0, b2, training_rows()->shadow_rows), ggml_get_rows(ctx0, t2, training_rows()->shadow_rows));
    ggml_tensor * updated = ggml_set_rows(ctx0, ggml_dup(ctx0, b2), sel, training_rows()->shadow_rows);
    return ggml_n_dims(base) <= 2 ? updated : ggml_reshape(ctx0, updated, base);
}

bool llm_graph_context::build_training_head() const {
    if (!cparams.training && n_shadow_rows == 0) {
        return true;
    }
    if (!res->t_embd || !res->t_logits || pooling_type != LLAMA_POOLING_TYPE_NONE) {
        return false;
    }
    // Upstream's models declare t_logits without expanding it (build_logits expands it after this);
    // the head is projected from the expanded graph, so expand it here first.
    ggml_build_forward_expand(gf, res->t_logits);
    const auto map = ubatch.rows();
    auto * input = training_rows();
    if (cparams.training) {
        input->label_rows = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, map.label_rows.size());
        ggml_set_input(input->label_rows);
    }
    if (!map.public_rows.empty()) {
        input->public_rows = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, map.public_rows.size());
        ggml_set_input(input->public_rows);
    }
    // Preserve the existing training head's allocation budget.
    constexpr size_t head_slice_bytes = 64u*1024*1024;
    const int64_t rows = std::max<int64_t>(1, head_slice_bytes / (sizeof(float) * res->t_logits->ne[0]));
    const int64_t loss_rows = cparams.training ? map.label_rows.size() : n_tokens;
    res->t_loss_slices.resize((loss_rows + rows - 1) / rows);
    if (input->shadow_rows) {
        size_t public_labels = 0;
        for (int32_t row : map.public_rows) {
            public_labels += ubatch.label[row] >= 0;
        }
        if (public_labels) {
            input->public_shadow_rows = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, public_labels);
            ggml_set_input(input->public_shadow_rows);
        }
        for (int64_t r0 = 0; r0 < loss_rows; r0 += rows) {
            auto * local = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, std::min(rows, loss_rows - r0));
            ggml_set_input(local);
            input->slice_shadow_rows.push_back(local);
        }
    }
    ggml_tensor * public_logits = nullptr;
    const int count = ggml_opt_project_head(ctx0, gf, res->t_embd, res->t_logits, input->shadow_rows,
            input->public_rows, input->label_rows, input->public_shadow_rows,
            input->slice_shadow_rows.empty() ? nullptr : input->slice_shadow_rows.data(), rows, &public_logits,
            res->t_loss_slices.data(), res->t_loss_slices.size());
    if (!count) {
        res->t_loss_slices.clear();
        return false;
    }
    res->t_logits = public_logits;
    if (cparams.embeddings && input->public_rows) {
        res->t_embd = ggml_get_rows(ctx0, res->t_embd, input->public_rows);
        ggml_build_forward_expand(gf, res->t_embd);
    }
    if (cparams.embeddings_nextn && cparams.embeddings_nextn_masked && input->public_rows) {
        res->t_h_nextn = ggml_get_rows(ctx0, res->t_h_nextn, input->public_rows);
        ggml_build_forward_expand(gf, res->t_h_nextn);
    }
    return true;
}

int64_t llm_graph_context::rs_keep() const {
    const bool training = cparams.training && cparams.walk_exact;
    return cparams.n_rs_seq > 0 && !training ? (int64_t) cparams.n_rs_seq + 1 : 1;
}

ggml_tensor * llm_graph_context::walk_acc_rows(ggml_tensor * acc, const ggml_tensor * like, int64_t row) const {
    return ggml_view_3d(ctx0, acc, like->ne[0], like->ne[1], like->ne[2],
            like->ne[0]*sizeof(float), acc->nb[1], row*acc->nb[1]);
}

void llm_graph_context::build_walk_surrogate(int il, ggml_tensor * k_cur, ggml_tensor * v_cur, int64_t row) const {
    ggml_tensor * gk  = walk_acc_rows(cparams.walk_acc_k[il], k_cur, row);
    ggml_tensor * gv  = walk_acc_rows(cparams.walk_acc_v[il], v_cur, row);
    ggml_tensor * k32 = k_cur->type == GGML_TYPE_F32 ? k_cur : ggml_cast(ctx0, k_cur, GGML_TYPE_F32);
    ggml_tensor * v32 = v_cur->type == GGML_TYPE_F32 ? v_cur : ggml_cast(ctx0, v_cur, GGML_TYPE_F32);
    ggml_tensor * term = ggml_add(ctx0,
            ggml_sum(ctx0, ggml_mul(ctx0, k32, gk)),
            ggml_sum(ctx0, ggml_mul(ctx0, v32, gv)));
    res->t_walk_surrogate = res->t_walk_surrogate ? ggml_add(ctx0, res->t_walk_surrogate, term) : term;
}

void llm_graph_context::build_walk_state_exit(ggml_tensor * cache, ggml_tensor * exit) const {
    if (!(cparams.training && cparams.walk_exact && cparams.walk_state_surrogate)) {
        return;
    }
    ggml_tensor * flat = ggml_reshape_1d(ctx0, ggml_cont(ctx0, exit), ggml_nelements(exit));
    if (flat->type != GGML_TYPE_F32) {
        flat = ggml_cast(ctx0, flat, GGML_TYPE_F32);
    }
    ggml_tensor * gs = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, ggml_nelements(exit));
    ggml_format_name(gs, "walk_gs-%s", cache->name);
    ggml_set_input(gs);
    ggml_tensor * term = ggml_sum(ctx0, ggml_mul(ctx0, flat, gs));
    res->t_walk_surrogate = res->t_walk_surrogate ? ggml_add(ctx0, res->t_walk_surrogate, term) : term;
    res->walk_state_of(cache).gs = gs;
}

// Continuum: the K/V a training graph attends to. Training reads her cache: the run's rows written into it in place
// (flash attention, whose backward covers plain scaled attention) or her served prefix read from it as constants in
// front of this chunk (explicit attention), so a chunk at position n_past trains against the whole window. Returns
// false when the graph is not training, leaving serving's store-and-read path to the caller. `kq_mask` arrives as the
// layer's mask and leaves trimmed to the cells this graph attends to.
bool llm_graph_context::build_attn_train(
        const llama_kv_cache_context * mctx_cur,
        llm_graph_input_attn_kv * inp,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * kq_b,
        ggml_tensor * sinks,
        ggml_tensor * v_mla,
                int   il,
        ggml_tensor *& kq_mask,
        ggml_tensor *& k,
        ggml_tensor *& v) const {
    if (!cparams.training) {
        return false;
    }
    if (cparams.walk_active && uses_flash_attn(kq_b, sinks, v_mla)) {
        // THE TRAINING GRAPH IS THE SERVING GRAPH. This chunk writes its K/V into the cache
        // exactly as serving does (SET_ROWS, the cache's own type) and attention reads the
        // cache in place: no copy of the context, at any window. The walk decoded the run
        // into the cache under the same adapter, so the write is idempotent.
        // Only the run's rows carry a K/V gradient, and they are the attention's grad sinks
        // (ggml_flash_attn_ext_set_grad_sinks): the chunk's own K/V and, in the exact walk, the
        // run's positions before it ([g0, n_past), read from the cache as F32 and TAPPED: their
        // gradient adds into the run's host accumulator, row 0 at walk_acc_base). The backward
        // computes dK/dV from g0 on and nothing for the window before it, a constant (her served
        // prefix): no window-sized gradient exists. What later chunks put on this chunk's own rows
        // reaches it through the surrogate. A quantized or f16 cache trains on the values serving
        // reads; its gradient passes straight through to the F32 K/V.
        const int64_t n_tokens = k_cur->ne[2];
        const int64_t n_past   = ubatch.pos[0];
        const int64_t n_win    = n_past + n_tokens;
        ggml_tensor * rows = build_inp_train_rows();
        const int64_t base = cparams.walk_acc_base;
        const int64_t g0   = std::min<int64_t>(std::max<int64_t>(cparams.walk_grad_from, base), n_past);
        llm_graph_result::walk_layer io = { il, (uint32_t) n_past, (uint32_t) g0 };
        // the window's cells [0, n_win) of one layer's cache with this chunk's rows written
        auto write = [&](ggml_tensor * all, ggml_tensor * cur) {
            const int64_t n_embd = cur->ne[0]*cur->ne[1];
            GGML_ASSERT(all->ne[3] == 1 && "a training context reads one stream");
            GGML_ASSERT(ggml_row_size(cur->type, cur->ne[0]) == cur->nb[1]);
            ggml_tensor * win = ggml_view_2d(ctx0, all, n_embd, n_win, all->nb[2], 0);
            ggml_tensor * w   = ggml_set_rows(ctx0, win, ggml_view_2d(ctx0, cur, n_embd, n_tokens, cur->nb[2], 0), rows);
            return ggml_reshape_3d(ctx0, w, cur->ne[0], cur->ne[1], n_win);
        };
        // the cached positions [p0, p1) as F32 [D, Hkv, p1 - p0]
        auto span = [&](ggml_tensor * all, int64_t p0, int64_t p1) {
            return ggml_cast(ctx0, ggml_view_3d(ctx0, all, all->ne[0], all->ne[1], p1 - p0, all->nb[1], all->nb[2], p0*all->nb[2]),
                    GGML_TYPE_F32);
        };
        ggml_tensor * k_all = mctx_cur->get_k(ctx0, il);
        ggml_tensor * v_all = mctx_cur->get_v(ctx0, il);
        GGML_ASSERT(v_all->nb[1] <= v_all->nb[2] && "a flash-attention cache holds V untransposed");
        GGML_ASSERT(k_cur->type == GGML_TYPE_F32 && v_cur->type == GGML_TYPE_F32);
        k = write(k_all, k_cur);
        v = write(v_all, v_cur);
        if (cparams.walk_exact && n_past > g0) {
            io.tk = span(k_all, g0, n_past);
            io.tv = span(v_all, g0, n_past);
            ggml_format_name(io.tk, "walk_tk-%d", il);
            ggml_format_name(io.tv, "walk_tv-%d", il);
            io.ak = walk_acc_rows(cparams.walk_acc_k[il], io.tk, g0 - base);
            io.av = walk_acc_rows(cparams.walk_acc_v[il], io.tv, g0 - base);
        }
        walk_grad_sinks = { io.tk, k_cur, io.tv, v_cur };
        if (cparams.walk_exact) {
            if (cparams.walk_surrogate) {
                build_walk_surrogate(il, k_cur, v_cur, n_past - base);
            }
            res->t_walk.push_back(io);
        }
        if (kq_mask->ne[0] > n_win) {
            kq_mask = ggml_cont(ctx0, ggml_view_4d(ctx0, kq_mask, n_win, kq_mask->ne[1], kq_mask->ne[2], kq_mask->ne[3],
                    kq_mask->nb[1], kq_mask->nb[2], kq_mask->nb[3], 0));
        }
    } else if (cparams.walk_active) {
        // EXPLICIT ATTENTION (a layer FLASH_ATTN_EXT's backward does not cover yet: ALiBi, a
        // logit softcap, sinks, a KQ bias, MLA) reads a copy of the context instead.
        // A backward pass cannot go through the cache: the write is an in-place SET_ROWS
        // and the read is a view of a leaf. So this ubatch's K/V are attended to directly,
        // and carry the gradient. The context before it (positions [0, n_past), decoded into
        // the cache by llama_context::opt_epoch_iter before this chunk trains) is read from
        // the cache as a CONSTANT, concatenated in front: her reply sees the whole window,
        // and memory is chunk x window instead of window x window. The prefix's own K/V get
        // no gradient (a stop-gradient at the chunk boundary).
        const int64_t n_tokens = k_cur->ne[2];
        const int64_t n_past   = ubatch.pos[0];
        k = k_cur;
        v = v_cur;
        // the cached prefix positions [p0, p1) as this chunk's K / V type
        ggml_tensor * k_all = n_past > 0 ? mctx_cur->get_k(ctx0, il) : nullptr;
        ggml_tensor * v_all = n_past > 0 ? mctx_cur->get_v(ctx0, il) : nullptr;
        auto k_span = [&](int64_t p0, int64_t p1) {
            ggml_tensor * t = ggml_view_4d(ctx0, k_all, k_all->ne[0], k_all->ne[1], p1 - p0, 1,
                    k_all->nb[1], k_all->nb[2], k_all->nb[3], p0*k_all->nb[2]);
            // the cache may hold the constants at a smaller type than the chunk's own K/V
            return t->type != k_cur->type ? ggml_cast(ctx0, t, k_cur->type) : t;
        };
        auto v_span = [&](int64_t p0, int64_t p1) {
            // a cache without flash attention stores V transposed: [n_kv, n_head_kv, n_embd_head_v]
            ggml_tensor * t = v_all->nb[1] > v_all->nb[2]
                ? ggml_cont(ctx0, ggml_permute(ctx0,
                        ggml_view_4d(ctx0, v_all, p1 - p0, v_all->ne[1], v_all->ne[2], 1,
                                v_all->nb[1], v_all->nb[2], v_all->nb[3], p0*v_all->nb[0]),
                        2, 1, 0, 3))
                : ggml_view_4d(ctx0, v_all, v_all->ne[0], v_all->ne[1], p1 - p0, 1,
                        v_all->nb[1], v_all->nb[2], v_all->nb[3], p0*v_all->nb[2]);
            return t->type != v_cur->type ? ggml_cast(ctx0, t, v_cur->type) : t;
        };
        if (cparams.walk_exact) {
            // THE EXACT WALK's chunk graph. The prefix positions [grad_from, n_past) are the
            // cached values as this chunk's (F32) K/V, TAPPED: the backward adds dL/d(cached K/V) for them straight
            // into the window's accumulator (cparams.walk_acc_k / _v, host memory), so nothing
            // window-sized outlives its layer on the device; earlier positions stay constants.
            // The chunk's own K/V meet the gradient later chunks put on them through the surrogate
            // <k_cur, gk> + <v_cur, gv>, gk / gv being the accumulator's rows for this chunk: its
            // gradient on k_cur / v_cur IS gk / gv, which the backward carries through this chunk
            // into the adapter and into its own prefix.
            // At a quantized or f16 cache the prefix's gradient is taken at the cache's values and
            // applied to the chunk's own full-precision K/V: straight-through. Measured
            // (test-walk-exact, q8_0 and f16 caches): cosine 0.998 against one graph, the same as
            // at an F32 cache, so training reads the numbers serving runs on (Joel: align the bit
            // depth to inference).
            const int64_t base = cparams.walk_acc_base;
            const int64_t g0   = std::min<int64_t>(std::max<int64_t>(cparams.walk_grad_from, base), n_past);
            llm_graph_result::walk_layer io = { il, (uint32_t) n_past, (uint32_t) g0 };
            // [const prefix][gradient prefix][this chunk]: the order the cells hold them in
            std::vector<ggml_tensor *> k_parts, v_parts;
            if (g0 > 0) {
                k_parts.push_back(k_span(0, g0));
                v_parts.push_back(v_span(0, g0));
            }
            if (n_past > g0) {
                GGML_ASSERT(k_cur->type == GGML_TYPE_F32 && v_cur->type == GGML_TYPE_F32);
                io.tk = k_span(g0, n_past);
                io.tv = v_span(g0, n_past);
                ggml_format_name(io.tk, "walk_tk-%d", il);
                ggml_format_name(io.tv, "walk_tv-%d", il);
                io.ak = walk_acc_rows(cparams.walk_acc_k[il], io.tk, g0 - base);
                io.av = walk_acc_rows(cparams.walk_acc_v[il], io.tv, g0 - base);
                k_parts.push_back(io.tk);
                v_parts.push_back(io.tv);
            }
            k_parts.push_back(k_cur);
            v_parts.push_back(v_cur);
            k = k_parts[0];
            v = v_parts[0];
            for (size_t i = 1; i < k_parts.size(); ++i) {
                k = ggml_concat(ctx0, k, k_parts[i], 2);
                v = ggml_concat(ctx0, v, v_parts[i], 2);
            }
            if (cparams.walk_surrogate) {
                build_walk_surrogate(il, k_cur, v_cur, n_past - base);
            }
            res->t_walk.push_back(io);
        } else if (n_past > 0) {
            k = ggml_concat(ctx0, k_span(0, n_past), k_cur, 2);
            v = ggml_concat(ctx0, v_span(0, n_past), v_cur, 2);
        }
        // the cache's cells hold the prefix at [0, n_past) and this ubatch right after it
        // (contiguous: soft_max_ext requires it, and a column slice of the padded mask is not)
        if (kq_mask->ne[0] > n_past + n_tokens) {
            kq_mask = ggml_cont(ctx0, ggml_view_4d(ctx0, kq_mask, n_past + n_tokens, kq_mask->ne[1], kq_mask->ne[2], kq_mask->ne[3],
                    kq_mask->nb[1], kq_mask->nb[2], kq_mask->nb[3], 0));
        }
    } else {
        // a training graph without the walk stores to the cache as serving does and reads through the write
        const auto & k_idxs = inp->get_k_idxs();
        const auto & v_idxs = inp->get_v_idxs();

        ggml_tensor * written_k = mctx_cur->cpy_k(ctx0, k_cur, k_idxs, il);
        ggml_tensor * written_v = mctx_cur->cpy_v(ctx0, v_cur, v_idxs, il);
        ggml_build_forward_expand(gf, written_k);
        ggml_build_forward_expand(gf, written_v);

        k = mctx_cur->get_k(ctx0, il, written_k);
        v = mctx_cur->get_v(ctx0, il, written_v);
    }
    return true;
}

// Continuum: the walk keeps its chunk's F32 K/V (its gradient is compared against a single graph); serving and
// direct decode use serving precision. Called where flash attention would cast F32 K/V to F16.
bool llm_graph_context::walk_keeps_f32() const {
    return cparams.training && cparams.walk_active;
}

// Continuum: a training graph reads a quantized cache as F16 through a node of its own, as the kernel would dequantize
// it anyway: inside the attention the F16 copy lives as long as the attention's output, which a training graph keeps
// for the backward, one window-sized copy per layer; the backward reads the cast's quantized source
// (ggml_flash_attn_ext_backward), so this copy dies with the forward (the M5, qwen3.5-4b, q8_0, window 32768:
// 1,888 MiB against 853 at an f16 cache)
void llm_graph_context::train_fa_kv(ggml_tensor *& k, ggml_tensor *& v) const {
    if (!walk_keeps_f32()) {
        return;
    }
    if (ggml_is_quantized(k->type)) {
        k = ggml_cast(ctx0, k, GGML_TYPE_F16);
    }
    if (ggml_is_quantized(v->type)) {
        v = ggml_cast(ctx0, v, GGML_TYPE_F16);
    }
}

// Continuum: a training graph's flash attention node: its backward tile, and in the in-place walk the rows its K/V
// gradient goes to (the run's rows only)
void llm_graph_context::train_fa_node(ggml_tensor * cur) const {
    if (cparams.training) {
        ggml_flash_attn_ext_set_back_tile(cur, cparams.fa_back_native ? GGML_FA_BACK_NATIVE : 0);
    }
    if (walk_grad_sinks[1] != nullptr) {
        ggml_flash_attn_ext_set_grad_sinks(cur, walk_grad_sinks[0], walk_grad_sinks[1], walk_grad_sinks[2], walk_grad_sinks[3]);
        walk_grad_sinks = {};
    }
}

// Continuum: the rows a walk's recurrent state is written to (s_write), the cells [head, head + n_rs) of her
// recurrent memory; set beside upstream's s_copy wherever a recurrent input is set
void llm_graph_input_rs::set_write(const llama_memory_recurrent_context * rs) {
    if (!s_write) {
        return;
    }
    const int64_t n_rs = rs->get_n_rs();
    std::vector<int64_t> rows(n_rs);
    for (uint32_t i = 0; i < n_rs; ++i) { rows[i] = rs->get_head() + i; }
    ggml_backend_tensor_set(s_write, rows.data(), 0, rows.size()*sizeof(int64_t));
}

// Continuum: the exact walk taps the state entering the chunk, so the backward adds dL/d(entry state), the gradient
// the previous chunk's exit takes, into a host accumulator the moment it is computed: no GRAD leaf, and no backward
// output held to the end of the graph for a readback
void llm_graph_context::train_rs_tap(ggml_tensor * s, ggml_tensor * output_states) const {
    if (cparams.training && cparams.walk_exact) {
        GGML_ASSERT(output_states->type == GGML_TYPE_F32 && "the exact walk carries an F32 recurrent state");
        res->walk_state_of(s).tap = output_states;
    }
}

// Continuum: the extra states (between n_seqs and n_rs), written by index when the graph carries write rows (Qwen3.5
// serving: the graph then never depends on the cache head, so it is reused across head moves), else at the head's
// offset as upstream does
void llm_graph_context::build_rs_extra(
        ggml_tensor * s,
        ggml_tensor * states,
        ggml_tensor * states_extra,
            int32_t   state_size,
            int32_t   n_seqs,
           uint32_t   n_rs,
           uint32_t   rs_head,
        ggml_tensor * state_write_extra) const {
    if (state_write_extra) {
        if (n_rs > (uint32_t) n_seqs) {
            ggml_build_forward_expand(gf, ggml_set_rows(ctx0, states, ggml_cont(ctx0, states_extra), state_write_extra));
        }
        return;
    }
    ggml_build_forward_expand(gf,
        ggml_cpy(ctx0, states_extra,
            ggml_view_2d(ctx0, s, state_size, (n_rs - n_seqs), s->nb[1], (rs_head + n_seqs)*s->nb[1])));
}

// Continuum: one sequence's last conv state written to its cell, by index when the graph carries write rows, and
// offered to the walk's exit surrogate
void llm_graph_context::build_conv_state_write(
        llm_graph_input_rs * inp,
        ggml_tensor * s,
        ggml_tensor * last,
        ggml_tensor * cell,
             size_t   offs) const {
    if (inp->s_write_main) {
        GGML_ASSERT(offs == 0 && "indexed recurrent writes store whole rows");
        ggml_tensor * rows = ggml_reshape_2d(ctx0, ggml_cont(ctx0, last), last->ne[0]*last->ne[1], last->ne[2]);
        ggml_build_forward_expand(gf, ggml_set_rows(ctx0, s, rows, inp->s_write_main));
    } else {
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, last, cell));
    }
    build_walk_state_exit(s, last);
}

// Continuum: Qwen3.5's recurrent state stores take write rows, so its serving graph never depends on the cache head;
// other families and rollback keep upstream's view-offset contract and its head equality check
void llm_graph_context::build_rs_write_rows(llm_graph_input_rs * inp_rs, const llama_memory_recurrent_context * rs) const {
    if (model.arch != LLM_ARCH_QWEN35 || cparams.training || cparams.n_rs_seq != 0) {
        return;
    }
    inp_rs->s_write = ggml_new_tensor_1d(ctx0, GGML_TYPE_I64, rs->get_n_rs());
    ggml_set_input(inp_rs->s_write);
    inp_rs->s_write_main  = ggml_view_1d(ctx0, inp_rs->s_write, ubatch.n_seqs, 0);
    inp_rs->s_write_extra = ggml_view_1d(ctx0, inp_rs->s_write, rs->get_n_rs() - ubatch.n_seqs, ubatch.n_seqs*sizeof(int64_t));
}
