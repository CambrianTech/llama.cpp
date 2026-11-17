// Continuum fork (CONTINUUM-ENGINE 4.12.1): see memory_pool_llama.h.

#include "memory_pool_llama.h"

#include "ggml.h"

memory_pool_llama::memory_pool_llama(llama_context * tgt, llama_context * dft) : tgt_(tgt), dft_(dft) {
    GGML_ASSERT(tgt_ != nullptr);
}

template <typename F> edit_outcome memory_pool_llama::each(kv_contexts in, F && f) {
    if (in != kv_contexts::draft && !f(llama_get_memory(tgt_))) {
        return edit_outcome::refused;
    }
    if (in != kv_contexts::target && dft_ != nullptr && !f(llama_get_memory(dft_))) {
        return edit_outcome::refused;
    }
    return edit_outcome::done;
}

placement memory_pool_llama::place(const llama_batch & piece) {
    // llama_decode: 0 placed, 1 no memory slot for the batch (nothing placed), 2 aborted, -1 invalid, < -1 compute
    const int32_t ret = llama_decode(tgt_, piece);
    if (ret == 0) {
        return placement::placed;
    }
    if (ret == 1) {
        return placement::no_room;
    }
    if (ret == 2) {
        return placement::aborted;
    }
    if (ret == -1) {
        return placement::invalid;
    }
    return placement::compute_error;
}

edit_outcome memory_pool_llama::remove(llama_seq_id seq, llama_pos p0, llama_pos p1, kv_contexts in) {
    return each(in, [&](llama_memory_t mem) { return llama_memory_seq_rm(mem, seq, p0, p1); });
}

edit_outcome memory_pool_llama::shift(llama_seq_id seq, llama_pos p0, llama_pos p1, llama_pos delta, kv_contexts in) {
    return each(in, [&](llama_memory_t mem) { llama_memory_seq_add(mem, seq, p0, p1, delta); return true; });
}

edit_outcome memory_pool_llama::copy(llama_seq_id src, llama_seq_id dst, llama_pos p0, llama_pos p1, kv_contexts in) {
    return each(in, [&](llama_memory_t mem) { llama_memory_seq_cp(mem, src, dst, p0, p1); return true; });
}

edit_outcome memory_pool_llama::move(llama_seq_id src, llama_seq_id dst) {
    return each(kv_contexts::both, [&](llama_memory_t mem) {
        if (!llama_memory_seq_rm(mem, dst, -1, -1)) {
            return false;
        }
        llama_memory_seq_cp(mem, src, dst, -1, -1);
        return llama_memory_seq_rm(mem, src, -1, -1);
    });
}

seq_range memory_pool_llama::range(llama_seq_id seq) const {
    llama_memory_t  mem = llama_get_memory(tgt_);
    const llama_pos lo  = llama_memory_seq_pos_min(mem, seq);
    const llama_pos hi  = llama_memory_seq_pos_max(mem, seq);
    if (lo < 0 || hi < 0) {
        return {};
    }
    return { lo, hi, false };
}

seq_state_size memory_pool_llama::state_size(llama_seq_id seq, llama_state_seq_flags flags) const {
    seq_state_size n;
    n.tgt = llama_state_seq_get_size_ext(tgt_, seq, flags);
    n.dft = dft_ != nullptr ? llama_state_seq_get_size_ext(dft_, seq, flags) : 0;
    return n;
}

save_outcome memory_pool_llama::save(llama_seq_id seq, llama_state_seq_flags flags, bytes_out tgt, bytes_out dft) const {
    if (!save_pair_valid(state_size(seq, flags), tgt.size, dft.size)) {
        return save_outcome::invalid;
    }
    if (llama_state_seq_get_data_ext(tgt_, tgt.data, tgt.size, seq, flags) != tgt.size) {
        return save_outcome::failed;
    }
    if (dft_ != nullptr && llama_state_seq_get_data_ext(dft_, dft.data, dft.size, seq, flags) != dft.size) {
        return save_outcome::failed;
    }
    return save_outcome::saved;
}

load_outcome memory_pool_llama::load(llama_seq_id seq, llama_state_seq_flags flags, bytes_in tgt, bytes_in dft) {
    if (!load_pair_valid(dft_ != nullptr, tgt.size, dft.size)) {
        return load_outcome::invalid;
    }
    const bool dst_empty = llama_memory_seq_pos_max(llama_get_memory(tgt_), seq) < 0 &&
                           (dft_ == nullptr || llama_memory_seq_pos_max(llama_get_memory(dft_), seq) < 0);
    if (!dst_empty) {
        return load_outcome::invalid;
    }
    if (llama_state_seq_set_data_ext(tgt_, tgt.data, tgt.size, seq, flags) != tgt.size) {
        // llama may leave a partial target state: clear it so refused means nothing is loaded
        GGML_ASSERT(llama_memory_seq_rm(llama_get_memory(tgt_), seq, -1, -1));
        return load_outcome::refused;
    }
    if (dft_ != nullptr && llama_state_seq_set_data_ext(dft_, dft.data, dft.size, seq, flags) != dft.size) {
        // the target took its half and the draft refused: the target's half goes, so the sequence is not left half
        // loaded (both contexts hold it, or neither does); a whole-sequence removal is never refused
        GGML_ASSERT(llama_memory_seq_rm(llama_get_memory(tgt_), seq, -1, -1));
        GGML_ASSERT(llama_memory_seq_rm(llama_get_memory(dft_), seq, -1, -1));
        return load_outcome::refused;
    }
    return load_outcome::loaded;
}
