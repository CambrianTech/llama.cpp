#pragma once

// The per-turn receipt's one decision, pure (see server_trainer::on_turn): how much of a
// finished turn a training window overlapped, and which windows can be forgotten after it.

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

// spans: this run's training windows as [start, end) in ggml_time_us, end 0 while one runs.
// Returns the overlap of [gen_start_us, gen_end_us) with every window, a running one counted
// to the turn's end. THEN drops the windows that ended before oldest_open_us, the earliest
// start of a turn still in flight (none of them can overlap a window that ended before it,
// and turns not yet begun start after now). Measured BEFORE pruned: pruned first, a window
// that ended inside this very turn was gone before it was counted, so a slowed turn filed as
// clean (Cormac on #39).
inline int64_t server_train_turn_overlap_then_prune(std::vector<std::pair<int64_t, int64_t>> & spans,
                                                     int64_t gen_start_us, int64_t gen_end_us,
                                                     int64_t oldest_open_us) {
    int64_t overlap_us = 0;
    if (gen_end_us > gen_start_us) {
        for (const auto & [start, end] : spans) {
            const int64_t e = end == 0 ? gen_end_us : end; // the window still running
            overlap_us += std::max<int64_t>(0, std::min(e, gen_end_us) - std::max(start, gen_start_us));
        }
    }
    spans.erase(std::remove_if(spans.begin(), spans.end(),
                               [oldest_open_us](const std::pair<int64_t, int64_t> & w) { return w.second != 0 && w.second < oldest_open_us; }),
                spans.end());
    return overlap_us;
}
