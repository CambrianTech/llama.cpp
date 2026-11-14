#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

// Which slot's adapter set holds the next batch.
//
// One batch wears one adapter set (the context applies LoRA per batch, never per sequence), so
// slots wearing different sets cannot share a batch (`server_slot::can_batch_with`). The batch
// used to be held by the FIRST generating slot by index: while any slot of one set kept
// generating, a slot wearing another set was launched and never prefilled. The M5, 2026-11-14:
// three adapter-less slots sat at 0 prompt tokens processed for 50 minutes while two slots
// wearing a gene took every batch, and the mind waiting on one of them saw nothing at all.
//
// The set the LONGEST-UNSERVED slot wears holds the batch. Every slot it lets in is stamped
// served, so the next batch goes to the oldest slot of another set: with k sets waiting, every
// slot is in a batch within k batches, however long any one set's own work runs. Slots of one
// set still batch together exactly as before; only who holds the batch changed.
struct server_batch_claim {
    // The slot has work a batch can take (a prompt to prefill, or a token to decode).
    bool    waiting;
    // When its tokens last entered a batch, or when its task arrived if they never have.
    int64_t last_served_us;
};

// The slot whose adapter set holds the next batch: the waiting claim served longest ago, the
// lowest index on a tie. None when no slot is waiting.
inline std::optional<size_t> server_adapter_turn(const std::vector<server_batch_claim> & claims) {
    std::optional<size_t> oldest;
    for (size_t index = 0; index < claims.size(); ++index) {
        if (claims[index].waiting && (!oldest || claims[index].last_served_us < claims[*oldest].last_served_us)) {
            oldest = index;
        }
    }
    return oldest;
}
