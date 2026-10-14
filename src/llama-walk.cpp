#include "llama-walk.h"

#include "llama-cparams.h"
#include "llama-hparams.h"
#include "llama-memory-recurrent.h"

#include "ggml-backend.h"

#include <algorithm>

llama_walk_plan llama_walk_plan::of(const std::vector<llama_token> & labels, uint32_t n_ctx, uint32_t n_batch) {
    GGML_ASSERT(n_batch > 0 && labels.size() >= n_ctx);
    llama_walk_plan plan;
    for (int64_t i = (int64_t) n_ctx - 1; i >= 0; --i) {
        if (labels[i] >= 0) {
            plan.last_label = i;
            break;
        }
    }
    for (uint32_t p = 0; (int64_t) p <= plan.last_label; ) {
        const bool labelled = labels[p] >= 0;
        uint32_t end = p;
        while ((int64_t) end <= plan.last_label && (labels[end] >= 0) == labelled) {
            ++end;
        }
        const size_t j0 = plan.chunks.size();
        for (uint32_t c0 = p; c0 < end; c0 += n_batch) {
            const uint32_t n = std::min(c0 + n_batch, end) - c0;
            plan.chunks.push_back({ c0, n, labelled ? n : 0 });
            plan.n_labels += labelled ? n : 0;
        }
        if (labelled) {
            plan.runs.push_back({ j0, plan.chunks.size() });
        }
        p = end;
    }
    return plan;
}

uint32_t llama_walk_plan::grad_from(uint32_t c0, uint32_t horizon) {
    return horizon == 0 || c0 < horizon ? 0 : c0 - horizon;
}

void llama_walk_plan::mark_exact(uint32_t horizon) {
    for (llama_walk_chunk & c : chunks) {
        c.needed     = false;
        c.surrogate  = false;
        c.period_end = false;
    }
    for (const llama_walk_run & r : runs) {
        uint32_t reach = UINT32_MAX; // the lowest position a later chunk of the run reaches
        for (size_t j = r.j1; j-- > r.j0; ) {
            llama_walk_chunk & c = chunks[j];
            c.needed     = true;
            c.surrogate  = reach < c.c0 + c.n;
            c.grad_from  = std::max(begin(r), grad_from(c.c0, horizon));
            c.loss_scale = (float) c.n_labels / (float) n_labels;
            reach = std::min(reach, c.grad_from);
        }
    }
    GGML_ASSERT(!runs.empty());
    first = (int64_t) runs.back().j0;
    chunks[first].period_end = true;
}

uint32_t llama_walk_plan::longest_run() const {
    uint32_t n = 0;
    for (const llama_walk_run & r : runs) {
        n = std::max(n, end(r) - begin(r));
    }
    return n;
}

size_t llama_walk_plan::host_bytes(size_t stride, size_t kv_bytes, size_t state_bytes) const {
    size_t n_checkpoints = 0;
    for (const llama_walk_run & r : runs) {
        n_checkpoints = std::max(n_checkpoints, (r.j1 - r.j0 + stride - 1)/stride);
    }
    return kv_bytes + n_checkpoints*state_bytes;
}

size_t llama_walk_plan::stride_for(size_t budget, size_t kv_bytes, size_t state_bytes) const {
    size_t longest = 1;
    for (const llama_walk_run & r : runs) {
        longest = std::max(longest, r.j1 - r.j0);
    }
    size_t stride = 1;
    if (budget > 0) {
        while (stride < longest && host_bytes(stride, kv_bytes, state_bytes) > budget) {
            ++stride;
        }
    }
    return stride;
}

size_t llama_walk_acc::kv_bytes(const llama_hparams & hparams, uint32_t n_pos) {
    size_t bytes = 0;
    for (uint32_t il = 0; il < hparams.n_layer(); ++il) {
        if (!hparams.is_recr(il)) {
            bytes += (size_t) n_pos*(hparams.n_embd_k_gqa(il) + hparams.n_embd_v_gqa(il))*sizeof(float);
        }
    }
    return bytes;
}

size_t llama_walk_acc::state_bytes(const llama_memory_recurrent * recr) {
    size_t bytes = 0;
    if (recr != nullptr) {
        for (size_t il = 0; il < recr->r_l.size(); ++il) {
            bytes += recr->r_l[il] ? recr->r_l[il]->nb[1] : 0;
            bytes += recr->s_l[il] ? recr->s_l[il]->nb[1] : 0;
        }
    }
    return bytes;
}

bool llama_walk_acc::init(const llama_hparams & hparams, const llama_memory_recurrent * recr, uint32_t n_pos, llama_cparams & cparams) {
    const uint32_t n_layer = hparams.n_layer();
    k.assign(n_layer, nullptr);
    v.assign(n_layer, nullptr);
    state.clear();
    state_ready = false;
    ctx.reset(ggml_init({ 4*n_layer*ggml_tensor_overhead(), nullptr, /*no_alloc =*/ true }));
    for (uint32_t il = 0; il < n_layer; ++il) {
        if (hparams.is_recr(il)) {
            // a recurrent layer has no K/V: its state carries the chain, one sequence's row
            if (recr != nullptr) {
                state[recr->r_l[il]] = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, hparams.n_embd_r(), 1);
                state[recr->s_l[il]] = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, hparams.n_embd_s(), 1);
            }
            continue;
        }
        k[il] = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, hparams.n_embd_k_gqa(il), n_pos);
        v[il] = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, hparams.n_embd_v_gqa(il), n_pos);
    }
    buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), ggml_backend_cpu_buffer_type()));
    if (!buf) {
        return false;
    }
    ggml_backend_buffer_clear(buf.get(), 0);
    bound = &cparams;
    cparams.walk_acc_k = k.data();
    cparams.walk_acc_v = v.data();
    cparams.walk_acc_base = 0;
    return true;
}

void llama_walk_acc::start_run(uint32_t base) {
    GGML_ASSERT(bound != nullptr);
    ggml_backend_buffer_clear(buf.get(), 0);
    state_ready = false;
    bound->walk_acc_base = base;
}

llama_walk_acc::~llama_walk_acc() {
    if (bound != nullptr) {
        bound->walk_acc_k = nullptr;
        bound->walk_acc_v = nullptr;
        bound->walk_acc_base = 0;
    }
}

void llama_walk_fill(llama_batch & batch, const std::vector<llama_token> & tokens, uint32_t c0, uint32_t n) {
    batch.n_tokens = n;
    for (uint32_t i = 0; i < n; ++i) {
        batch.token   [i]    = tokens[c0 + i];
        batch.pos     [i]    = c0 + i;
        batch.n_seq_id[i]    = 1;
        batch.seq_id  [i][0] = 0;
        batch.logits  [i]    = false;
    }
}
