#pragma once

// Continuum: an adapter set per sequence, so one step decodes minds wearing different adapters.
//
// A context applies ONE adapter set (llama_set_adapters_lora) to every row of a ubatch, so minds bound in streams of one
// context could share a step only while they wore the same adapters: on the M5 (2026-11-28, Qwen3.6-35B-A3B) every
// resident wore her own three, and the lane chose a batch 85 times in a turn and batched nothing.
//
// A sequence may name its own set (llama_set_seq_adapters_lora), once, when its owner binds a mind to it; an empty set
// is the base model. A sequence that names none wears the context's set, so a context that never names one builds the
// graph upstream builds. At graph build the ubatch's rows are grouped by the set they wear, and each adapter's delta is
// computed on its group's rows only: a group of consecutive rows is a view (no copy); rows that interleave with another
// group's are gathered and written back by index, correct and slower, and counted (llama_seq_adapters_gathered).
// Rows never move, so the caller's output indices hold. The training shadow stays the context's: it adds to the
// labelled rows whatever set they wear.

#include "llama-adapter.h"
#include "llama-batch.h"

#include <cstdint>
#include <unordered_map>
#include <utility>
#include <vector>

// an adapter set by value: each adapter with its scale, ordered by address, no zero scale
struct llama_adapter_set {
    std::vector<std::pair<llama_adapter_lora *, float>> adapters;

    bool operator==(const llama_adapter_set & other) const { return adapters == other.adapters; }
    bool operator!=(const llama_adapter_set & other) const { return !(*this == other); }
};

// the rows of one ubatch that wear one set: [begin, begin + n) when consecutive, else each row listed
struct llama_adapter_seq_group {
    llama_adapter_set    set;
    int32_t              begin = 0;
    int32_t              n     = 0;
    std::vector<int32_t> rows;  // empty when consecutive

    bool consecutive() const { return rows.empty(); }
    bool operator==(const llama_adapter_seq_group & other) const {
        return set == other.set && begin == other.begin && n == other.n && rows == other.rows;
    }
};

// a ubatch's groups, part of its graph's identity (llm_graph_params::allow_reuse): a step with the same rows under the
// same sets and scales reuses its graph, and a set rebound under another scale rebuilds it. Rows on the base model are
// in no group (they add nothing, so they never interleave). Empty when no row's sequence names a set of its own: the
// graph is upstream's.
struct llama_adapter_seq_groups {
    bool                                 named = false;   // some row's sequence names a set
    std::vector<llama_adapter_seq_group> groups;
    uint64_t                             lora_frees = 0;  // the model's adapter frees when grouped: an address freed and reused is another adapter

    bool empty() const { return !named; }
    bool gathered() const;
    bool operator==(const llama_adapter_seq_groups & other) const {
        return named == other.named && groups == other.groups && lora_frees == other.lora_frees;
    }
};

// what a context holds: each sequence's own set, as its owner named it
struct llama_adapter_seq_sets {
    struct named {
        llama_adapter_set set;
        uint64_t          lora_frees = 0; // the model's adapter frees when it was named
    };
    std::unordered_map<llama_seq_id, named> by_seq;
    mutable uint64_t                        gathered = 0; // ubatches whose groups interleaved

    // the set a row of `seq` wears: her own, or the context's (without the training shadow, which is added apart)
    llama_adapter_set wears(llama_seq_id seq, const llama_adapter_set & context_set) const;

    // a ubatch's groups; empty when none of its sequences names a set
    llama_adapter_seq_groups group(const llama_ubatch & ubatch, const llama_adapter_set & context_set, uint64_t lora_frees) const;

    // why a batch cannot be decoded, or nullptr: a token shared by sequences that wear different sets (each set needs
    // its own evaluation of the token, so the caller submits it once per set), or a set named before an adapter was freed
    const char * refuses(const llama_batch_ext & batch, const llama_adapter_set & context_set, uint64_t lora_frees) const;
};

// the context's set as a value, without the training shadow
llama_adapter_set llama_adapter_set_of(const llama_adapter_loras & loras, const llama_adapter_lora * shadow);
