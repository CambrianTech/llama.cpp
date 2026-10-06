// what this catches (Cormac on #39): the per-turn receipt pruned the training windows BEFORE
// summing a turn's overlap, against a horizon at the turn's own end, so with no other turn in
// flight every window that ended inside the turn was erased first and the slowed turn filed
// as clean. Measured, then pruned.

#include "server-train-spans.h"

#include <cstdio>
#include <cstdlib>

#define CHECK(cond) do { if (!(cond)) { fprintf(stderr, "FAILED %s:%d: %s\n", __FILE__, __LINE__, #cond); exit(1); } } while (0)

int main() {
    // Cormac's case: a span [100, 200], one turn [50, 300], no other turn open (the server
    // passes this turn's end as the oldest open start). Overlap 100 us, then the span goes.
    {
        std::vector<std::pair<int64_t, int64_t>> spans = {{100, 200}};
        CHECK(server_train_turn_overlap_then_prune(spans, 50, 300, 300) == 100);
        CHECK(spans.empty());
    }
    // a window still running counts to the turn's end and is never pruned
    {
        std::vector<std::pair<int64_t, int64_t>> spans = {{250, 0}};
        CHECK(server_train_turn_overlap_then_prune(spans, 50, 300, 300) == 50);
        CHECK(spans.size() == 1);
    }
    // another turn in flight since 150 keeps a window that ended at 200: it overlapped her
    {
        std::vector<std::pair<int64_t, int64_t>> spans = {{100, 200}};
        CHECK(server_train_turn_overlap_then_prune(spans, 220, 300, 150) == 0);
        CHECK(spans.size() == 1);
    }
    // an empty or inverted turn measures nothing but still prunes
    {
        std::vector<std::pair<int64_t, int64_t>> spans = {{100, 200}};
        CHECK(server_train_turn_overlap_then_prune(spans, 300, 300, 300) == 0);
        CHECK(spans.empty());
    }
    printf("test-server-train-spans: OK\n");
    return 0;
}
