#pragma once

// Continuum fork (CONTINUUM-ENGINE 4.12.1): the memory pool over a target llama_context and an optional draft context
// that share sequence ids. Edits go to the named contexts and report a refusal; a sequence's state moves for both
// together; host bytes are charged against the budget given at construction.

#include "memory_pool.h"


class memory_pool_llama final : public memory_pool {
public:
    memory_pool_llama(llama_context * tgt, llama_context * dft, size_t host_budget);

    placement place(const llama_batch & piece) override;

    edit_outcome remove(llama_seq_id seq, llama_pos p0, llama_pos p1, kv_contexts in) override;
    edit_outcome shift(llama_seq_id seq, llama_pos p0, llama_pos p1, llama_pos delta, kv_contexts in) override;
    edit_outcome copy(llama_seq_id src, llama_seq_id dst, llama_pos p0, llama_pos p1, kv_contexts in) override;
    edit_outcome move(llama_seq_id src, llama_seq_id dst) override;
    seq_range    range(llama_seq_id seq) const override;

    seq_state_size state_size(llama_seq_id seq, llama_state_seq_flags flags) const override;
    save_outcome   save(llama_seq_id seq, llama_state_seq_flags flags, uint8_t * tgt, uint8_t * dft) const override;
    load_outcome   load(llama_seq_id seq, llama_state_seq_flags flags, const uint8_t * tgt, size_t n_tgt,
                        const uint8_t * dft, size_t n_dft) override;

    host_charge charge(size_t bytes) override;
    void        release(size_t bytes) override;
    size_t      host_held() const override { return host_held_; }

private:
    // the contexts `in` names, target first; the draft only when the pool has one
    template <typename F> edit_outcome each(kv_contexts in, F && f);
    llama_context * tgt_;
    llama_context * dft_;
    size_t          host_budget_;
    size_t          host_held_ = 0;
};
