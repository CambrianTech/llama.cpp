// A learning step's plan reads her rows of a shared (--kv-unified) stream: no model, no backend.
#include "../src/llama-learn-step-plan.h"
#include "../src/llama-kv-cells.h"

#include <cmath>
#include <cstdio>
#include <vector>

static int failures = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++failures; } } while (0)

static void place(llama_kv_cells & cells, uint32_t cell, llama_seq_id seq, llama_pos pos) {
    cells.pos_set(cell, pos);
    cells.seq_add(cell, seq);
}

int main() {
    // her positions 0..5 interleaved with a neighbour's 0..3 in one stream, a hole at cell 6
    llama_kv_cells cells;
    cells.resize(16);
    const llama_seq_id her = 2, neighbour = 0;
    const uint32_t her_cells[6] = { 1, 3, 4, 7, 9, 12 };
    for (llama_pos p = 0; p < 6; ++p) { place(cells, her_cells[p], her, p); }
    const uint32_t neighbour_cells[4] = { 0, 2, 5, 8 };
    for (llama_pos p = 0; p < 4; ++p) { place(cells, neighbour_cells[p], neighbour, p); }

    // what catches: the step's extent is her rows, in position order, whatever the neighbour holds
    const auto plan = llama_learn_step_plan::for_sequence(her, 4, 2);
    std::vector<int32_t> got(plan.rows());
    CHECK(plan.collect_cells(cells, got.data()));
    for (uint32_t p = 0; p < plan.rows(); ++p) { CHECK(got[p] == (int32_t) her_cells[p]); }

    // what catches: a reply longer than she holds is a refusal, never a read of a neighbour's cell
    const auto beyond = llama_learn_step_plan::for_sequence(her, 5, 2);
    std::vector<int32_t> none(beyond.rows());
    CHECK(!beyond.collect_cells(cells, none.data()));

    // what catches: a position held twice (a stale placement) is refused, not silently read once
    place(cells, 14, her, 3);
    CHECK(!plan.collect_cells(cells, got.data()));

    // what catches: the mask is causal over her positions; the frozen prefix is visible to every query
    const llama_pos pos[2] = { 4, 5 };
    std::vector<ggml_fp16_t> mask((size_t) plan.rows() * 2);
    plan.fill_mask(pos, 2, mask.data());
    for (uint32_t i = 0; i < 2; ++i) {
        for (uint32_t r = 0; r < plan.rows(); ++r) {
            const float m = ggml_fp16_to_fp32(mask[(size_t) i*plan.rows() + r]);
            CHECK((llama_pos) r <= pos[i] ? m == 0.0f : std::isinf(m) && m < 0);
        }
    }

    if (failures == 0) { std::printf("test-learn-step-plan: ok\n"); }
    return failures == 0 ? 0 : 1;
}
