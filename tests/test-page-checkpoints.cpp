// What a saved page keeps of a slot's context checkpoints (Continuum fork, select_page_checkpoints).
//
// what this catches: the M5 on 2026-10-25 (a hybrid model, 4 residents paging over 3 lanes) re-
// prefilled 20k+ tokens a turn and once fell to "full re-prefill: no checkpoint passed the search
// (raw lcp 9710)": a page kept the oldest checkpoint and the newest four, all near the end of a
// 35-40k prompt, so a restored slot had nothing under the point where its next prompt diverged
// (about 11k). The page must keep, within the same budget, checkpoints just under the slot's last
// raw common prefix.

#include "server-common.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK(cond) do { if (!(cond)) { std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond); std::exit(1); } } while (0)

int main() {
    // A 35-40k prompt's checkpoints at user-message starts, about 1,576 apart.
    std::vector<int64_t> pos;
    for (int64_t p = 300; p < 38000; p += 1576) {
        pos.push_back(p);
    }
    const size_t n = pos.size();

    // keep < 0: every checkpoint, as before the subset existed.
    CHECK(select_page_checkpoints(pos, -1, 11000).size() == n);

    // No divergence known (a fresh slot, no cache_prompt): the oldest and the newest four, as today.
    {
        auto k = select_page_checkpoints(pos, 4, -1);
        CHECK(k.size() == 5);
        CHECK(k[0] == 0);
        for (size_t j = 1; j < 5; ++j) {
            CHECK(k[j] == n - 5 + j);
        }
    }

    // Divergence at 11,000: the oldest, the two newest at or below 11,000, the newest two.
    {
        auto k = select_page_checkpoints(pos, 4, 11000);
        CHECK(k.size() == 5);
        CHECK(k[0] == 0);
        CHECK(pos[k[1]] <= 11000 && pos[k[2]] <= 11000);
        CHECK(pos[k[2]] + 1576 > 11000);              // the nearest under the break is kept
        CHECK(k[3] == n - 2 && k[4] == n - 1);         // and the end of the prompt
    }

    // The budget never grows: at most 1 + keep, whatever the divergence.
    for (int32_t keep = 0; keep <= 6; ++keep) {
        CHECK(select_page_checkpoints(pos, keep, 11000).size() <= (size_t) keep + 1);
    }

    // A divergence under every checkpoint but the first keeps no extra "below" points.
    {
        auto k = select_page_checkpoints(pos, 4, 100);
        CHECK(k.size() == 5 && k[0] == 0 && k[1] == n - 4);
    }

    // keep 0: the oldest only. An empty list: nothing.
    CHECK(select_page_checkpoints(pos, 0, 11000).size() == 1);
    CHECK(select_page_checkpoints({}, 4, 11000).empty());

    std::printf("test-page-checkpoints: ok\n");
    return 0;
}
