#include "llama-learn-step-plan.h"

#include "llama-kv-cells.h"

#include <algorithm>
#include <cmath>

bool llama_learn_step_plan::collect_cells(const llama_kv_cells & cells, int32_t * out) const {
    const uint32_t n = rows();
    std::fill(out, out + n, -1);
    for (uint32_t i = 0; i < cells.size(); ++i) {
        if (cells.is_empty(i) || !cells.seq_has(i, seq)) {
            continue;
        }
        const llama_pos p = cells.pos_get(i);
        if (p < 0 || (uint32_t) p >= n) {
            continue;
        }
        if (out[p] >= 0) {
            return false; // two cells claim one of her positions: not a sequence this step can read
        }
        out[p] = (int32_t) i;
    }
    for (uint32_t p = 0; p < n; ++p) {
        if (out[p] < 0) {
            return false;
        }
    }
    return true;
}

void llama_learn_step_plan::fill_mask(const llama_pos * pos, uint32_t n_tokens, ggml_fp16_t * out) const {
    const uint32_t n = rows();
    const ggml_fp16_t seen = ggml_fp32_to_fp16(0.0f);
    const ggml_fp16_t unseen = ggml_fp32_to_fp16(-INFINITY);
    for (uint32_t i = 0; i < n_tokens; ++i) {
        for (uint32_t r = 0; r < n; ++r) {
            out[(size_t) i*n + r] = (llama_pos) r <= pos[i] ? seen : unseen;
        }
    }
}
