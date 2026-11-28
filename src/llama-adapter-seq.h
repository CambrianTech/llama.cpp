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
// Rows never move, so the caller's output indices hold. A group knows its rows twice: among the ubatch's tokens, and
// among its outputs (after inp_out_ids keeps only the rows whose logits were asked for, a tensor holds those). The
// training shadow stays the context's: it adds to the labelled rows whatever set they wear.
//
// An adapter is named by its address and its serial (llama_adapter_lora::serial, unique per load), so an adapter freed
// and another loaded at its address is another adapter: a set naming a freed one refuses decode until named again,
// and a free of any other adapter changes nothing.

#include "llama-adapter.h"
#include "llama-batch.h"

#include <cstdint>
#include <tuple>
#include <unordered_map>
#include <vector>

// one adapter of a set: who it is, which load of it, at what scale
struct llama_adapter_set_entry {
    llama_adapter_lora * adapter = nullptr;
    uint64_t             serial  = 0;
    float                scale   = 0.0f;

    bool operator==(const llama_adapter_set_entry & other) const {
        return adapter == other.adapter && serial == other.serial && scale == other.scale;
    }
    bool operator<(const llama_adapter_set_entry & other) const {
        return std::tie(adapter, serial, scale) < std::tie(other.adapter, other.serial, other.scale);
    }
};

// an adapter set by value, ordered, no zero scale
struct llama_adapter_set {
    std::vector<llama_adapter_set_entry> adapters;

    bool operator==(const llama_adapter_set & other) const { return adapters == other.adapters; }
    bool operator!=(const llama_adapter_set & other) const { return !(*this == other); }
};

// rows of one space (the ubatch's tokens, or its outputs): [begin, begin + n) when consecutive, else each row listed
struct llama_adapter_seq_rows {
    int32_t              begin = 0;
    int32_t              n     = 0;
    std::vector<int32_t> rows;  // empty when consecutive

    bool consecutive() const { return rows.empty(); }
    void add(int32_t row);
    bool operator==(const llama_adapter_seq_rows & other) const { return begin == other.begin && n == other.n && rows == other.rows; }
};

// the rows of one ubatch that wear one set
struct llama_adapter_seq_group {
    llama_adapter_set      set;
    llama_adapter_seq_rows tokens;
    llama_adapter_seq_rows outputs;

    bool operator==(const llama_adapter_seq_group & other) const {
        return set == other.set && tokens == other.tokens && outputs == other.outputs;
    }
};

// a ubatch's groups, part of its graph's identity (llm_graph_params::allow_reuse): a step with the same rows under the
// same sets and scales reuses its graph, and a set rebound under another scale rebuilds it. Rows on the base model are
// in no group (they add nothing, so they never interleave). Empty when no row's sequence names a set of its own: the
// graph is upstream's.
struct llama_adapter_seq_groups {
    bool                                 named = false; // some row's sequence names a set
    std::vector<llama_adapter_seq_group> groups;

    bool empty() const { return !named; }
    bool gathered() const;
    bool operator==(const llama_adapter_seq_groups & other) const { return named == other.named && groups == other.groups; }
};

// what a context holds: each sequence's own set, as its owner named it
struct llama_adapter_seq_sets {
    std::unordered_map<llama_seq_id, llama_adapter_set> by_seq;
    mutable uint64_t                                    gathered = 0; // ubatches whose groups interleaved

    // the set a row of `seq` wears: her own, or the context's (without the training shadow, which is added apart)
    const llama_adapter_set & wears(llama_seq_id seq, const llama_adapter_set & context_set) const;

    // a ubatch's groups; empty when none of its sequences names a set
    llama_adapter_seq_groups group(const llama_ubatch & ubatch, const llama_adapter_set & context_set) const;

    // why a batch cannot be decoded, or nullptr: a named set whose adapter is no longer that load on `model`, or a
    // token shared by sequences that wear different sets (each set needs its own evaluation of the token, so the
    // caller submits it once per set)
    const char * refuses(const llama_batch_ext & batch, const llama_adapter_set & context_set, const llama_model & model) const;
};

// the context's set as a value, without the training shadow
llama_adapter_set llama_adapter_set_of(const llama_adapter_loras & loras, const llama_adapter_lora * shadow);

// a serial for each adapter loaded (llama_adapter_lora::serial)
uint64_t llama_adapter_lora_next_serial();
