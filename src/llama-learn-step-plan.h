#pragma once

#include "llama.h"
#include "ggml.h"

#include <cstdint>

class llama_kv_cells;

// One learning step of one sequence: her positions [0, rows()) in position order, the last
// reply_rows of them the ubatch that carries the gradient. The preflight builds the step's graph
// from a shaped plan and the live step decodes under a bound one, so the two cannot describe
// different attention: both see her rows, never the other cells of a shared cache.
struct llama_learn_step_plan {
    llama_seq_id seq = -1; // -1: shape only (the preflight's); set when the step runs
    uint32_t prefix_rows = 0;
    uint32_t reply_rows = 0;

    uint32_t rows() const { return prefix_rows + reply_rows; }
    bool bound() const { return seq >= 0; }

    static llama_learn_step_plan shaped(const llama_learn_step_geometry & step) {
        return { -1, step.prefix_rows, step.reply_rows };
    }
    static llama_learn_step_plan for_sequence(llama_seq_id seq, uint32_t prefix_rows, uint32_t reply_rows) {
        return { seq, prefix_rows, reply_rows };
    }

    // out[p] = the cell holding her position p, for p in [0, rows()); false when one is missing or
    // held twice. Cells of other sequences, wherever they sit in a shared stream, are skipped.
    bool collect_cells(const llama_kv_cells & cells, int32_t * out) const;
    // F16 [rows(), n_tokens]: query i sees her row r when r <= pos[i]
    void fill_mask(const llama_pos * pos, uint32_t n_tokens, ggml_fp16_t * out) const;
};
