#pragma once

#include <cstdint>

// Continuum fork: the view end a slot's tokens carry after a prompt pass that starts at `reused` (the
// prefix this request reused from the slot). A request that names its own view end sets it. One that
// names none (a short side request resending the mind's prompt, e.g. a one-token probe) keeps the
// previous view end while that point still lies inside the reused prefix: the mind's resume point is
// still in these tokens, and the page saved next must keep its checkpoint. Measured on the M5 (engine
// b5534b5, 2026-11-16): 83 of 95 pages that kept no view end were saved right after a <=8-token request,
// and the next act re-prefilled ~100k tokens from the head checkpoint (CONTINUUM-ENGINE 4.7.15).
// Otherwise the view end is unknown (-1).
inline int32_t carried_view_end(int32_t requested, int32_t previous, int32_t reused) {
    if (requested >= 0) {
        return requested;
    }
    return previous >= 0 && previous <= reused ? previous : -1;
}
