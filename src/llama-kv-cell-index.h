#pragma once

// Continuum fork (CONTINUUM-ENGINE 4.14, the flash tier): the two indices llama_kv_cells derives from its cells, held
// in native memory. A persona's metadata then has no node containers (std::set, std::map), so her generation on flash
// can be the image of what is in memory and a load binds it, never rebuilding an index for state that did not change.
//
//  - llama_kv_used: which cells are occupied, a bitmap over the cell rows (bounded by the stream size, never by a
//    position), with its count and its extrema kept as cells change.
//  - llama_kv_seq_pos: one sequence's positions, sorted, each with how many cells hold it (duplicates are allowed: cache
//    reuse rm+add, repeated M-RoPE positions). Bounded by the cells (each cell adds at most one entry per sequence),
//    never a histogram by absolute position. Each sequence owns its own array, so a change to one never moves
//    another's.
//
// The common changes are O(1): a token appended at her newest position, and a rollback or removal of her newest
// position. Any other change (a removal inside the range, a shift, a division that merges positions) marks the
// sequence's index stale instead of editing it cell by cell, and the next query recomputes it once from the cells, the
// one truth (a scan, a sort, and a merge of equal positions): work done only because her positions actually changed.

#include "llama.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <utility>
#include <vector>

// Which cells are occupied: a bitmap over the cell rows, its count, and its extrema.
class llama_kv_used {
public:
    void resize(uint32_t n_cells) {
        words.assign((n_cells + 63) / 64, 0);
        n_used = lo = hi = 0;
    }

    void clear() {
        std::fill(words.begin(), words.end(), 0);
        n_used = lo = hi = 0;
    }

    void insert(uint32_t i) {
        uint64_t & word = words[i / 64];
        const uint64_t bit = uint64_t(1) << (i % 64);
        if (word & bit) {
            return;
        }
        word |= bit;
        if (n_used++ == 0) {
            lo = i;
            hi = i + 1;
            return;
        }
        lo = std::min(lo, i);
        hi = std::max(hi, i + 1);
    }

    void erase(uint32_t i) {
        uint64_t & word = words[i / 64];
        const uint64_t bit = uint64_t(1) << (i % 64);
        if (!(word & bit)) {
            return;
        }
        word &= ~bit;
        if (--n_used == 0) {
            lo = hi = 0;
            return;
        }
        if (i == lo) {
            lo = next_from(i + 1);
        }
        if (i + 1 == hi) {
            hi = prev_from(i) + 1;
        }
    }

    uint32_t size() const { return n_used; }
    bool empty() const { return n_used == 0; }

    // the first occupied cell, 0 when none is
    uint32_t min() const { return lo; }

    // one past the last occupied cell, 0 when none is
    uint32_t max_p1() const { return hi; }

    void swap(llama_kv_used & other) noexcept {
        words.swap(other.words);
        std::swap(n_used, other.n_used);
        std::swap(lo, other.lo);
        std::swap(hi, other.hi);
    }

    bool operator==(const llama_kv_used & other) const {
        return n_used == other.n_used && lo == other.lo && hi == other.hi && words == other.words;
    }

private:
    std::vector<uint64_t> words;
    uint32_t n_used = 0;
    uint32_t lo     = 0;
    uint32_t hi     = 0;

    // the first occupied cell at or after i; one exists (called only while n_used > 0, after an extremum at i - 1 left)
    uint32_t next_from(uint32_t i) const {
        for (size_t w = i / 64; w < words.size(); ++w) {
            uint64_t word = words[w];
            if (w == i / 64) {
                word &= ~uint64_t(0) << (i % 64);
            }
            if (word) {
                return uint32_t(w * 64 + lowest_bit(word));
            }
        }
        assert(false && "an occupied cell remains");
        return 0;
    }

    // the last occupied cell before i; one exists (called only while n_used > 0, after an extremum at i left)
    uint32_t prev_from(uint32_t i) const {
        for (size_t w = (i == 0 ? 0 : (i - 1) / 64) + 1; w-- > 0;) {
            uint64_t word = words[w];
            if (w == (i - 1) / 64 && i % 64 != 0) {
                word &= ~uint64_t(0) >> (64 - i % 64);
            }
            if (word) {
                return uint32_t(w * 64 + highest_bit(word));
            }
        }
        assert(false && "an occupied cell remains");
        return 0;
    }

    static uint32_t lowest_bit(uint64_t word) {
        uint32_t b = 0;
        while (!(word & 1)) {
            word >>= 1;
            ++b;
        }
        return b;
    }

    static uint32_t highest_bit(uint64_t word) {
        uint32_t b = 63;
        while (!(word >> b)) {
            --b;
        }
        return b;
    }
};

// One position of one sequence, and how many of its cells hold it.
struct llama_kv_pos_count {
    llama_pos pos;
    int32_t   n;
};

// One sequence's positions, sorted, each with its count: exact on its O(1) paths, else stale until recomputed.
class llama_kv_seq_pos {
public:
    // a cell of this sequence now holds position p
    void inc(llama_pos p) {
        if (stale) {
            return;
        }
        if (runs.empty() || runs.back().pos < p) {
            runs.push_back({ p, 1 });
            return;
        }
        if (runs.back().pos == p) {
            ++runs.back().n;
            return;
        }
        stale = true;
    }

    // a cell of this sequence no longer holds position p
    void dec(llama_pos p) {
        if (stale) {
            return;
        }
        assert(!runs.empty());
        if (runs.back().pos == p) {
            if (--runs.back().n == 0) {
                runs.pop_back();
            }
            return;
        }
        stale = true;
    }

    void clear() {
        runs.clear();
        stale = false;
    }

    bool is_stale() const { return stale; }

    // recompute from the positions its cells hold now, in any order: sorted, equal positions merged with their counts
    void settle(std::vector<llama_pos> & positions) {
        std::sort(positions.begin(), positions.end());
        runs.clear();
        for (const llama_pos p : positions) {
            if (!runs.empty() && runs.back().pos == p) {
                ++runs.back().n;
            } else {
                runs.push_back({ p, 1 });
            }
        }
        stale = false;
    }

    bool empty() const {
        assert(!stale);
        return runs.empty();
    }

    llama_pos min() const {
        assert(!stale && !runs.empty() && runs.front().n > 0);
        return runs.front().pos;
    }

    llama_pos max() const {
        assert(!stale && !runs.empty() && runs.back().n > 0);
        return runs.back().pos;
    }

    const std::vector<llama_kv_pos_count> & entries() const {
        assert(!stale);
        return runs;
    }

    void swap(llama_kv_seq_pos & other) noexcept {
        runs.swap(other.runs);
        std::swap(stale, other.stale);
    }

private:
    std::vector<llama_kv_pos_count> runs;
    bool stale = false;
};
