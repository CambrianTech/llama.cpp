// Continuum: what each memory needs to carry a training batch, and what it refuses, kept beside the memories so
// upstream's files merge without touching them.
#include "llama-kv-cache.h"
#include "llama-memory-hybrid-idx.h"

#include "llama-batch.h"


const char * llama_kv_cache::training_error(const llama_batch & batch, bool flash_attn) const {
    if (!supports_row_gradients()) {
        return "cache row indices exceed the backward gather's index range";
    }
    // the flash-attention backward reads one stream: a training batch may touch only one (a serving
    // context with a stream per slot learns on one slot's stream at a time)
    if (flash_attn && get_n_stream() != 1) {
        int64_t stream = -1;
        for (int32_t i = 0; i < batch.n_tokens; ++i) {
            for (int32_t k = 0; k < (batch.n_seq_id ? batch.n_seq_id[i] : 1); ++k) {
                const llama_seq_id sq = batch.seq_id ? batch.seq_id[i][k] : 0;
                const int64_t st = seq_to_stream[sq];
                if (stream >= 0 && st != stream) {
                    return "a training batch with flash attention may touch one cache stream: this one spans several";
                }
                stream = st;
            }
        }
    }
    return nullptr;
}

const char * llama_memory_hybrid_idx::training_error(const llama_batch & batch, bool flash_attn) const {
    if (const char * why = llama_memory_hybrid::training_error(batch, flash_attn)) {
        return why;
    }
    return get_mem_idx() ? "indexed hybrid cache row gradients are not supported" : nullptr;
}
