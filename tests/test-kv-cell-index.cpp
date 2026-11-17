// Continuum fork (CONTINUUM-ENGINE 4.14, the flash tier): llama_kv_cells' derived indices held natively
// (llama-kv-cell-index.h) answer exactly as the std::set / std::map indices they replace.
//
// what this catches: an occupied-cell count or extremum, or a sequence's min/max position, that differs from what the
// cells hold after any mix of the operations the KV cache performs: appends, removals inside and at the ends, shared
// sequences (seq_add / seq_keep), shifts that drop cells below zero, and divisions that merge positions (pos_div is not
// injective). A reference model with the replaced node containers is checked after every step, and also times the same
// workloads, the old-versus-new read.

#include "llama.h"

#include "../src/llama-kv-cells.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <random>
#include <set>
#include <vector>

namespace {

// The indices as they were: the oracle and the old-side baseline.
struct reference {
    std::vector<llama_pos>               pos;
    std::vector<std::set<llama_seq_id>>  seqs;
    std::set<uint32_t>                   used;
    std::map<llama_pos, int>             seq_pos[LLAMA_MAX_SEQ];

    explicit reference(uint32_t n) : pos(n, -1), seqs(n) {}

    void dec(llama_seq_id s, llama_pos p) {
        auto it = seq_pos[s].find(p);
        if (--it->second == 0) {
            seq_pos[s].erase(it);
        }
    }
    void pos_set(uint32_t i, llama_pos p) { pos[i] = p; used.insert(i); }
    void seq_add(uint32_t i, llama_seq_id s) { seqs[i].insert(s); seq_pos[s][pos[i]]++; }
    void rm(uint32_t i) {
        for (auto s : seqs[i]) { dec(s, pos[i]); }
        seqs[i].clear(); pos[i] = -1; used.erase(i);
    }
    void seq_rm(uint32_t i, llama_seq_id s) {
        seqs[i].erase(s); dec(s, pos[i]);
        if (seqs[i].empty()) { pos[i] = -1; used.erase(i); }
    }
    void seq_keep(uint32_t i, llama_seq_id s) {
        if (seqs[i].count(s)) {
            for (auto o : seqs[i]) { if (o != s) { dec(o, pos[i]); } }
            seqs[i] = { s };
        } else if (!seqs[i].empty()) {
            rm(i);
        }
    }
    void pos_add(uint32_t i, llama_pos d) {
        for (auto s : seqs[i]) { dec(s, pos[i]); }
        pos[i] += d;
        if (pos[i] < 0) { seqs[i].clear(); pos[i] = -1; used.erase(i); return; }
        for (auto s : seqs[i]) { seq_pos[s][pos[i]]++; }
    }
    void pos_div(uint32_t i, int d) {
        for (auto s : seqs[i]) { dec(s, pos[i]); }
        pos[i] /= d;
        for (auto s : seqs[i]) { seq_pos[s][pos[i]]++; }
    }
    llama_pos min(llama_seq_id s) const { return seq_pos[s].empty() ? -1 : seq_pos[s].begin()->first; }
    llama_pos max(llama_seq_id s) const { return seq_pos[s].empty() ? -1 : seq_pos[s].rbegin()->first; }
};

int failures = 0;

void expect(bool ok, const char * what, int step) {
    if (!ok) {
        fprintf(stderr, "FAIL at step %d: %s\n", step, what);
        ++failures;
    }
}

void compare(const llama_kv_cells & cells, const reference & ref, int n_seq, int step) {
    expect(cells.get_used() == ref.used.size(), "used count", step);
    expect(cells.used_min() == (ref.used.empty() ? 0 : *ref.used.begin()), "first used cell", step);
    expect(cells.used_max_p1() == (ref.used.empty() ? 0 : *ref.used.rbegin() + 1), "last used cell", step);
    for (llama_seq_id s = 0; s < n_seq; ++s) {
        expect(cells.seq_pos_min(s) == ref.min(s), "sequence min position", step);
        expect(cells.seq_pos_max(s) == ref.max(s), "sequence max position", step);
    }
}

// Every operation the cache performs, at random, on both, compared after each step.
void equivalence(uint32_t n_cells, int n_seq, int steps, uint32_t seed) {
    std::mt19937 rng(seed);
    llama_kv_cells cells;
    cells.resize(n_cells);
    reference ref(n_cells);
    llama_pos next_pos[LLAMA_MAX_SEQ] = {};
    auto pick = [&](uint32_t n) { return uint32_t(rng() % n); };

    for (int step = 0; step < steps; ++step) {
        const uint32_t i = pick(n_cells);
        const llama_seq_id s = llama_seq_id(pick(uint32_t(n_seq)));
        switch (pick(9)) {
            case 0: case 1: case 2: { // append her next position (the common case), sometimes a repeat
                if (!cells.is_empty(i)) { break; }
                const llama_pos p = pick(8) == 0 ? std::max(0, next_pos[s] - 1) : next_pos[s]++;
                cells.pos_set(i, p); ref.pos_set(i, p);
                cells.seq_add(i, s); ref.seq_add(i, s);
                break;
            }
            case 3: // a shared prefix: another sequence joins an occupied cell
                if (cells.is_empty(i) || cells.seq_has(i, s)) { break; }
                cells.seq_add(i, s); ref.seq_add(i, s);
                break;
            case 4: // remove a cell wherever it is
                if (cells.is_empty(i)) { break; }
                cells.rm(i); ref.rm(i);
                break;
            case 5: // one sequence leaves a cell
                if (cells.is_empty(i) || !cells.seq_has(i, s)) { break; }
                cells.seq_rm(i, s); ref.seq_rm(i, s);
                break;
            case 6: // keep only one sequence
                cells.seq_keep(i, s); ref.seq_keep(i, s);
                break;
            case 7: { // shift, sometimes below zero (the cell is freed)
                if (cells.is_empty(i)) { break; }
                const llama_pos d = llama_pos(pick(64)) - 40;
                cells.pos_add(i, d); ref.pos_add(i, d);
                break;
            }
            case 8: { // divide: distinct positions can merge
                if (cells.is_empty(i)) { break; }
                const int d = 2 + int(pick(3));
                cells.pos_div(i, d); ref.pos_div(i, d);
                break;
            }
        }
        compare(cells, ref, n_seq, step);
        if (failures) { return; }
    }
}

// what this catches (Codex's review of 2d54d4869): an index left stale by a genuine change (a division that merges
// positions, a removal inside the range) carried across her freeze and recomputed only at her next query after the
// bind. Her save settles every index (llama_kv_cache::persona_check_save calls settle_all); the object bound to a stream
// then answers every query with no scan and no sort, and answers as the node-container indices do.
void freeze_then_bind() {
    const uint32_t n = 64;
    llama_kv_cells cells;
    cells.resize(n);
    reference ref(n);
    for (uint32_t i = 0; i < 32; ++i) {
        cells.pos_set(i, llama_pos(i)); ref.pos_set(i, llama_pos(i));
        cells.seq_add(i, llama_seq_id(i % 2)); ref.seq_add(i, llama_seq_id(i % 2));
    }
    for (uint32_t i = 0; i < 32; ++i) {
        cells.pos_div(i, 3); ref.pos_div(i, 3);
    }
    cells.rm(5); ref.rm(5);
    const uint64_t before = cells.settles();
    cells.settle_all(); // her save's reserve phase
    expect(cells.settles() > before, "the change was paid for at the freeze", -1);

    llama_kv_cells bound;
    bound.resize(n);
    bound.swap(cells); // her blocks bound to a stream
    compare(bound, ref, 2, -1);
    expect(bound.settles() == 0, "a query after the bind recomputed nothing", -1);
}

// The same workload on both, timed: her context appended token by token, its front half dropped (a context shift's
// discard), the rest shifted down, and her extrema read after each phase.
template <typename T, typename Append, typename Remove, typename Shift, typename Read>
double timed(T & t, uint32_t n, Append append, Remove remove, Shift shift, Read read) {
    const auto t0 = std::chrono::steady_clock::now();
    for (uint32_t i = 0; i < n; ++i) { append(t, i); }
    read(t);
    for (uint32_t i = 0; i < n / 2; ++i) { remove(t, i); }
    read(t);
    for (uint32_t i = n / 2; i < n; ++i) { shift(t, i); }
    read(t);
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

void cost(uint32_t n) {
    llama_kv_cells cells;
    cells.resize(n);
    reference ref(n);
    volatile llama_pos sink = 0;
    const llama_pos d = -llama_pos(n / 2);
    const double now = timed(cells, n,
        [](llama_kv_cells & c, uint32_t i) { c.pos_set(i, llama_pos(i)); c.seq_add(i, 0); },
        [](llama_kv_cells & c, uint32_t i) { c.rm(i); },
        [d](llama_kv_cells & c, uint32_t i) { c.pos_add(i, d); },
        [&sink](llama_kv_cells & c) { sink = c.seq_pos_min(0) + c.seq_pos_max(0) + llama_pos(c.get_used()); });
    const double before = timed(ref, n,
        [](reference & r, uint32_t i) { r.pos_set(i, llama_pos(i)); r.seq_add(i, 0); },
        [](reference & r, uint32_t i) { r.rm(i); },
        [d](reference & r, uint32_t i) { r.pos_add(i, d); },
        [&sink](reference & r) { sink = r.min(0) + r.max(0) + llama_pos(r.used.size()); });
    printf("RECEIPT kv-cell-index n=%u: node containers %.3f ms, native %.3f ms\n", n, before, now);
    (void) sink;
}

} // namespace

int main(int argc, char ** /* argv: only the flag count is read */) {
    for (uint32_t seed = 1; seed <= 8 && !failures; ++seed) {
        equivalence(256, 4, 20000, seed);
    }
    freeze_then_bind();
    if (failures) {
        fprintf(stderr, "kv-cell-index: %d mismatches against the node-container indices\n", failures);
        return 1;
    }
    printf("kv-cell-index: native indices match the node-container indices over 8 x 20000 random cache operations\n");
    if (argc > 1) { // --cost: the old-versus-new read at real lengths
        for (uint32_t n : { 4096u, 32768u, 100000u }) {
            cost(n);
        }
    }
    return 0;
}
