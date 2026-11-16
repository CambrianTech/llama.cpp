// what this catches: a request that names no view end (a short side request resending a mind's prompt)
// erasing that mind's resume point, so its saved page drops the view-end checkpoint and its next act
// re-prefills from the head (regression, M5 engine b5534b5, 2026-11-16: 83 of 95 such pages; ~100k-token
// re-prefills). A named view end always wins; an old one survives only inside the reused prefix.
#include "../tools/server/server-view-end.h"

#include <cstdlib>
#include <iostream>

static void require(bool condition, const char * message) {
    if (!condition) {
        std::cerr << "FAILED: " << message << '\n';
        std::exit(1);
    }
}

int main() {
    require(carried_view_end(120, 90, 100) == 120, "a request's own view end always wins");
    require(carried_view_end(-1, 109190, 109200) == 109190, "a side request that reused the prefix keeps the mind's view end");
    require(carried_view_end(-1, 109190, 109190) == 109190, "the view end exactly at the reused boundary survives");
    require(carried_view_end(-1, 109190, 5000) == -1, "a request that diverged before the old view end drops it");
    require(carried_view_end(-1, -1, 109200) == -1, "nothing known stays unknown");
    std::cout << "test-server-view-end: ok\n";
    return 0;
}
