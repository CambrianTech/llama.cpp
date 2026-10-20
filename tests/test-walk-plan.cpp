// The walk's plan without a model (src/llama-walk.h): what test-walk-exact proves on a model
// rests on these chunks, marks and the host budget being right.
#include "testing.h"

#include "llama.h"

#include "../src/llama-walk.h"

#include <vector>

// a window of runs: {n positions, labelled}; labelled positions take token 1, context -1
static std::vector<llama_token> window(std::initializer_list<std::pair<uint32_t, bool>> runs) {
    std::vector<llama_token> labels;
    for (const auto & [n, labelled] : runs) {
        labels.insert(labels.end(), n, labelled ? 1 : -1);
    }
    return labels;
}

int main(int argc, char ** argv) {
    testing t(std::cout);
    t.verbose = argc > 1 && std::string(argv[1]) == "-v";

    // what this catches: a chunk that mixes context and reply, or a plan that runs past the last label
    t.test("chunks_follow_runs", [](testing & t) {
        const auto labels = window({ {300, false}, {700, true}, {100, false}, {50, true}, {200, false} });
        const auto plan   = llama_walk_plan::of(labels, labels.size(), 512);
        t.assert_equal(1149, plan.last_label);
        t.assert_equal(750u, plan.n_labels);
        const std::vector<std::pair<uint32_t, uint32_t>> want = { {0, 300}, {300, 512}, {812, 188}, {1000, 100}, {1100, 50} };
        t.assert_equal(want.size(), plan.chunks.size());
        for (size_t j = 0; j < want.size() && j < plan.chunks.size(); ++j) {
            t.assert_equal(want[j].first,  plan.chunks[j].c0);
            t.assert_equal(want[j].second, plan.chunks[j].n);
            t.assert_equal(labels[want[j].first] >= 0 ? want[j].second : 0u, plan.chunks[j].n_labels);
        }
        t.assert_equal(-1, llama_walk_plan::of(window({ {64, false} }), 64, 512).last_label);
    });

    // what this catches: a step taken anywhere but the reverse pass's last chunk, a loss that is
    // not the window's mean, a gradient reaching past the horizon or before its own run (the
    // context before a run is her stable set's: the shadow never touched it), a context chunk
    // trained for nothing
    t.test("exact_marks", [](testing & t) {
        auto plan = llama_walk_plan::of(window({ {1024, false}, {1536, true}, {512, false}, {512, true} }), 3584, 512);
        t.assert_equal(2u, plan.runs.size());
        plan.mark_exact(/*horizon =*/ 512);
        float loss = 0.0f;
        int   ends = 0;
        for (const auto & c : plan.chunks) {
            t.assert_true("exactly the labelled chunks train", c.needed == (c.n_labels > 0));
            loss += c.needed ? c.loss_scale : 0.0f;
            ends += c.period_end;
        }
        // run 1 is chunks 2..4 (positions 1024..2560), run 2 is chunk 6 (3072..3584)
        t.assert_equal(1024u, plan.chunks[2].grad_from);
        t.assert_equal(1024u, plan.chunks[3].grad_from);
        t.assert_equal(1536u, plan.chunks[4].grad_from);
        t.assert_equal(3072u, plan.chunks[6].grad_from);
        t.assert_true("a chunk a later one of its run reaches", plan.chunks[2].surrogate && plan.chunks[3].surrogate);
        t.assert_true("the last chunk of a run has no later gradient", !plan.chunks[4].surrogate && !plan.chunks[6].surrogate);
        t.assert_equal(6, plan.first);
        t.assert_equal(1, ends);
        t.assert_true("loss scales sum to one", loss > 0.9999f && loss < 1.0001f);
        t.assert_equal(0u, llama_walk_plan::grad_from(1024, 0));
        t.assert_equal(512u, llama_walk_plan::grad_from(1024, 512));
    });

    // what this catches: a stride that does not shrink the checkpoints to the budget, or one
    // that grows past the window when nothing fits
    t.test("host_budget_stride", [](testing & t) {
        // a run's start is a checkpoint too: the walk restores the run to her stable self from it
        const auto plan = llama_walk_plan::of(window({ {8*512, true} }), 8*512, 512);
        t.assert_equal(180u, plan.host_bytes(1, 100, 10));
        t.assert_equal(1u, plan.stride_for(0, 100, 10));
        t.assert_equal(2u, plan.stride_for(150, 100, 10));
        t.assert_equal(140u, plan.host_bytes(2, 100, 10));
        t.assert_equal(8u, plan.stride_for(50, 100, 10));
        t.assert_equal(110u, plan.host_bytes(8, 100, 10));
    });

    return t.summary();
}
