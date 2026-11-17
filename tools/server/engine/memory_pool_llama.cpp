// Continuum fork (CONTINUUM-ENGINE 4.12.1): see memory_pool_llama.h.

#include "memory_pool_llama.h"

#include "ggml.h"

memory_pool_llama::memory_pool_llama(llama_context * tgt, llama_context * dft, size_t host_budget)
    : tgt_(tgt), dft_(dft), host_budget_(host_budget) {
    GGML_ASSERT(tgt_ != nullptr);
    edits_.init(tgt_, dft_);
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

void memory_pool_llama::remove(llama_seq_id seq, llama_pos p0, llama_pos p1) {
    edits_.seq_rm(seq, p0, p1);
}

void memory_pool_llama::shift(llama_seq_id seq, llama_pos p0, llama_pos p1, llama_pos delta) {
    edits_.seq_add(seq, p0, p1, delta);
}

void memory_pool_llama::copy(llama_seq_id src, llama_seq_id dst, llama_pos p0, llama_pos p1) {
    edits_.seq_cp(src, dst, p0, p1);
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

save_outcome memory_pool_llama::save(llama_seq_id seq, llama_state_seq_flags flags, uint8_t * tgt, uint8_t * dft) const {
    const seq_state_size n = state_size(seq, flags);
    if (llama_state_seq_get_data_ext(tgt_, tgt, n.tgt, seq, flags) != n.tgt) {
        return save_outcome::failed;
    }
    if (dft_ != nullptr && llama_state_seq_get_data_ext(dft_, dft, n.dft, seq, flags) != n.dft) {
        return save_outcome::failed;
    }
    return save_outcome::saved;
}

load_outcome memory_pool_llama::load(llama_seq_id seq, llama_state_seq_flags flags, const uint8_t * tgt, size_t n_tgt,
                                     const uint8_t * dft, size_t n_dft) {
    if (llama_state_seq_set_data_ext(tgt_, tgt, n_tgt, seq, flags) != n_tgt) {
        return load_outcome::refused;
    }
    if (dft_ != nullptr && llama_state_seq_set_data_ext(dft_, dft, n_dft, seq, flags) != n_dft) {
        // the target took its half and the draft refused: the target's half goes, so the sequence is not left half
        // loaded (both contexts hold it, or neither does)
        edits_.seq_rm(seq, -1, -1);
        return load_outcome::refused;
    }
    return load_outcome::loaded;
}

host_charge memory_pool_llama::charge(size_t bytes) {
    if (bytes > host_budget_ - host_held_) {
        return host_charge::denied;
    }
    host_held_ += bytes;
    return host_charge::granted;
}

void memory_pool_llama::release(size_t bytes) {
    GGML_ASSERT(bytes <= host_held_);
    host_held_ -= bytes;
}
