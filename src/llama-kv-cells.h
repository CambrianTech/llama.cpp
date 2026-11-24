#pragma once

#include "llama.h"
#include "llama-cparams.h"
#include "llama-kv-cell-index.h"

#include <algorithm>
#include <atomic>
#include <bitset>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <limits>
#include <memory>
#include <vector>

#if defined(__GNUC__)
#    define LLAMA_ATTRIBUTE_NOINLINE __attribute__((noinline))
#else
#    define LLAMA_ATTRIBUTE_NOINLINE
#endif

#if defined(_MSC_VER)
#    include <intrin.h>
#endif

// index of the lowest set bit, x != 0
static inline int ctz64(uint64_t x) {
#if defined(_MSC_VER)
    unsigned long i;
#if defined(_WIN64)
    _BitScanForward64(&i, x);
    return (int) i;
#else
    if (_BitScanForward(&i, (unsigned long) x)) {
        return (int) i;
    }
    _BitScanForward(&i, (unsigned long) (x >> 32));
    return (int) i + 32;
#endif
#else
    return __builtin_ctzll(x);
#endif
}

struct llama_kv_cell_ext {
    // 2D spatial positions, typically used for M-RoPE
    llama_pos x = 0;
    llama_pos y = 0;

    // when tok = LLAMA_TOKEN_NULL when the cell is produced by embedding input (i.e. multimodal)
    // use case: n-gram embeddings hash
    llama_token tok = LLAMA_TOKEN_NULL;

    // return true if the current 2D spatial position is greater than other
    bool is_2d_gt(llama_pos ox, llama_pos oy) const {
        return (y > oy) || (y == oy && x > ox);
    }

    void reset() {
        static_assert(std::is_trivially_copyable_v<llama_kv_cell_ext>);

        *this = llama_kv_cell_ext{};
    }
};

// the data of one slice of a ubatch, used to build the cell rows of the kv_rows attention (see ggml_flash_attn_ext_rows)
struct llama_kv_rows_slice {
    std::vector<llama_seq_id> seq_ids; // the sequences whose cells form the list of the slice

    // min position of the cells that are not masked by the SWA (-1 if all cells are kept)
    llama_pos p_min = -1;
};

// meta information about KV cells that can be part of multiple sequences at the same time
// TODO: add unit tests
class llama_kv_cells {
public:
    using seq_set_t = std::bitset<LLAMA_MAX_SEQ>;

    // Continuum fork (CONTINUUM-ENGINE 4.14): exchange two cells objects in place, member by member, with each
    // container's own swap: never allocates on any standard library (a std::swap of the whole object moves through a
    // temporary, and MSVC's node containers allocate a sentinel when move-constructed). A persona's metadata moves
    // between her blocks and a stream with this.
    void swap(llama_kv_cells & other) noexcept {
        std::swap(has_shift, other.has_shift);
        used.swap(other.used);
        pos.swap(other.pos);
        ext.swap(other.ext);
        shift.swap(other.shift);
        seq.swap(other.seq);
        for (int s = 0; s < LLAMA_MAX_SEQ; ++s) {
            seq_pos[s].swap(other.seq_pos[s]);
            seq_bits[s].swap(other.seq_bits[s]);
        }
        std::swap(stale, other.stale);
        std::swap(lineage_id, other.lineage_id);
        checkpoints.swap(other.checkpoints);
    }

    // Checkpoint handles own their positions. The cells retain a weak receipt for each
    // live prefix, under the sequence it was taken for. A cut revokes just the receipts whose
    // prefix it changes: in a pool several minds share, a cut of one never revokes another's.
    using checkpoint_position = std::shared_ptr<const llama_pos>;
    uint64_t lineage() const { return lineage_id; }

    // seq_id < 0: a cut of every sequence (a match-any removal)
    void note_cut(llama_seq_id seq_id, llama_pos first) {
        checkpoints.erase(std::remove_if(checkpoints.begin(), checkpoints.end(), [seq_id, first](const auto & held) {
            const auto mark = held.second.lock();
            return !mark || ((seq_id < 0 || held.first == seq_id) && *mark >= first);
        }), checkpoints.end());
    }

    void prepare_checkpoint() {
        checkpoints.erase(std::remove_if(checkpoints.begin(), checkpoints.end(), [](const auto & held) {
            return held.second.expired();
        }), checkpoints.end());
        checkpoints.reserve(checkpoints.size() + 1);
    }

    void mark_checkpoint(llama_seq_id seq_id, const checkpoint_position & mark) { checkpoints.emplace_back(seq_id, mark); }

    bool holds_checkpoint(const checkpoint_position & mark) const {
        return mark && std::any_of(checkpoints.begin(), checkpoints.end(), [&](const auto & held) {
            return held.second.lock() == mark;
        });
    }

    // Continuum fork (CONTINUUM-ENGINE 4.3): her checkpoint receipts travel with her. Paged out of a pool, the live
    // ones move into `out` (reserved by the check to hold them); paged back in, they are hers again under the sequence
    // she is bound as. Her positions are unchanged by a move, so a checkpoint she took before still rewinds her after.
    size_t persona_marks(llama_seq_id seq_id) const {
        return (size_t) std::count_if(checkpoints.begin(), checkpoints.end(), [seq_id](const auto & held) {
            return held.first == seq_id && !held.second.expired();
        });
    }

    void persona_take_marks(llama_seq_id seq_id, std::vector<std::weak_ptr<const llama_pos>> & out) {
        out.clear();
        for (auto & held : checkpoints) {
            if (held.first == seq_id && !held.second.expired() && out.size() < out.capacity()) {
                out.push_back(std::move(held.second)); // the moved-from receipt is empty: note_cut drops it
            }
        }
        note_cut(seq_id, 0);
    }

    void persona_reserve_marks(size_t n) { checkpoints.reserve(checkpoints.size() + n); }

    void persona_give_marks(llama_seq_id seq_id, std::vector<std::weak_ptr<const llama_pos>> & in) {
        for (auto & mark : in) {
            if (!mark.expired() && checkpoints.size() < checkpoints.capacity()) {
                checkpoints.emplace_back(seq_id, std::move(mark));
            }
        }
        in.clear();
    }

    // Continuum fork (CONTINUUM-ENGINE 4.3): a pool several minds share. Her cells' sequence bits exist for every
    // sequence the pool seats from construction, so moving her in never allocates.
    void persona_reserve_seqs(uint32_t n_seq) {
        for (uint32_t s = 0; s < n_seq && s < LLAMA_MAX_SEQ; ++s) {
            if (seq_bits[s].empty()) {
                seq_bits[s].assign((pos.size() + 63)/64, 0);
            }
        }
    }

    // Her cells leave the pool: each held only her sequence (the caller checked), so each becomes empty; her index and
    // her checkpoint receipts go with them. Nothing allocates.
    void persona_release(const std::vector<uint32_t> & rows, llama_seq_id seq_id) {
        for (const uint32_t i : rows) {
            seq[i].reset();
            pos[i]   = -1;
            ext[i].reset();
            shift[i] =  0;
            used.erase(i);
            seq_bits[seq_id][i/64] &= ~(1ull << (i%64));
        }
        seq_pos[seq_id].clear();
        stale.reset(seq_id);
        note_cut(seq_id, 0);
    }

    // Her rows `src` of `from` (her blocks' cells) take the pool's empty cells `dst`, as sequence seq_id; `index` is her
    // (position, cell) index over `dst`, built and sorted by the check, swapped in. Nothing allocates.
    void persona_adopt(const std::vector<uint32_t> & src, const std::vector<uint32_t> & dst, const llama_kv_cells & from,
                       llama_seq_id seq_id, llama_kv_seq_pos & index) {
        for (size_t k = 0; k < dst.size(); ++k) {
            const uint32_t i = dst[k];
            const uint32_t j = src[k];
            assert(pos[i] == -1 && seq[i].none());
            pos[i]   = from.pos[j];
            ext[i]   = from.ext[j];
            shift[i] = 0;
            seq[i].set(seq_id);
            used.insert(i);
            seq_bits[seq_id][i/64] |= 1ull << (i%64);
        }
        seq_pos[seq_id].swap(index);
        stale.reset(seq_id);
    }

    // Continuum fork (CONTINUUM-ENGINE 4.14): recompute every stale sequence index now, on the thread that changed the
    // cells. llama_kv_cache calls this as each of its operations returns, so a query never writes an index: queries
    // arrive from other threads while a decode runs, and a persona's blocks leave a stream with valid indices.
    void settle_all() {
        if (stale.none()) {
            return;
        }
        for (llama_seq_id s = 0; s < LLAMA_MAX_SEQ; ++s) {
            if (stale.test(s)) {
                settle(s);
            }
        }
        stale.reset();
    }

    // any sequence index a change left stale and nothing has settled yet
    bool has_stale() const {
        return stale.any();
    }

    // how many index recomputations this object has made
    uint64_t settles() const {
        return n_settles;
    }

    // how many queries were answered from the cells because an index was stale: a missed settle, 0 when every change
    // was settled by its operation
    uint64_t stale_queries() const {
        return n_stale_queries.load();
    }

    void reset() {
        lineage_id = next_lineage();
        checkpoints.clear();
        stale.reset();
        for (uint32_t i = 0; i < pos.size(); ++i) {
            pos[i]   = -1;
            ext[i].reset();
            shift[i] =  0;
            seq[i].reset();
        }

        has_shift = false;

        used.clear();

        for (uint32_t s = 0; s < LLAMA_MAX_SEQ; ++s) {
            seq_pos[s].clear();
            std::fill(seq_bits[s].begin(), seq_bits[s].end(), 0);
        }
    }

    void reset_shift() {
        has_shift = false;

        for (uint32_t i = 0; i < shift.size(); ++i) {
            shift[i] = 0;
        }
    }

    uint32_t size() const {
        return pos.size();
    }

    void resize(uint32_t n) {
        pos.resize(n);
        ext.resize(n);
        shift.resize(n);
        seq.resize(n);
        used.resize(n);

        for (uint32_t s = 0; s < LLAMA_MAX_SEQ; ++s) {
            seq_bits[s].clear();
        }

        reset();
    }

    bool is_empty(uint32_t i) const {
        assert(i < pos.size());
        assert((pos[i] < 0 && pos[i] == -1) || pos[i] >= 0);

        return pos[i] == -1;
    }

    uint32_t get_used() const {
        return used.size();
    }

    // the index of the first cell that is used
    // return 0 if no cells are used
    uint32_t used_min() const {
        return used.min();
    }

    // the index of the last cell that is used + 1
    // return 0 if no cells are used
    uint32_t used_max_p1() const {
        return used.max_p1();
    }

    bool get_has_shift() const {
        return has_shift;
    }

    // move cell isrc to idst (used during defrag)
    //void mv(uint32_t isrc, uint32_t idst) {
    //    assert(isrc < pos.size());
    //    assert(idst < pos.size());

    //    assert(pos[idst] == -1);
    //    assert(pos[isrc] != -1);

    //    pos  [idst] = pos  [isrc];
    //    shift[idst] = shift[isrc];
    //    seq  [idst] = seq  [isrc];

    //    pos  [isrc] = -1;
    //    shift[isrc] =  0;
    //    seq  [isrc].reset();

    //    used.erase (isrc);
    //    used.insert(idst);
    //}

    // copy the state of cells [i, i + n) (used for save/restore the state of the cells)
    llama_kv_cells cp(uint32_t i, uint32_t n) const {
        assert(i + n <= pos.size());

        llama_kv_cells res;

        res.resize(n);

        for (uint32_t j = 0; j < n; ++j) {
            const auto idx = i + j;

            res.pos[j] = pos[idx];
            res.ext[j] = ext[idx];
            res.seq[j] = seq[idx];

            assert(shift[idx] == 0);
        }

        return res;
    }

    // copy the state of cells [idxs[0], idxs[1], ..., idxs[idxs.size() - 1])
    llama_kv_cells cp(const std::vector<uint32_t> & idxs) const {
        llama_kv_cells res;

        res.resize(idxs.size());

        for (uint32_t j = 0; j < idxs.size(); ++j) {
            const auto idx = idxs[j];

            res.pos[j] = pos[idx];
            res.ext[j] = ext[idx];
            res.seq[j] = seq[idx];

            assert(shift[idx] == 0);
        }

        return res;
    }

    // set the state of cells [i, i + other.pos.size()) (used for save/restore the state of the cells)
    void set(uint32_t i, const llama_kv_cells & other) {
        assert(i + other.pos.size() <= pos.size());

        for (uint32_t j = 0; j < other.pos.size(); ++j) {
            const auto idx = i + j;

            if (pos[idx] == -1 && other.pos[j] != -1) {
                used.insert(i + j);
            }

            if (pos[idx] != -1 && other.pos[j] == -1) {
                used.erase(i + j);
            }

            if (pos[idx] != -1) {
                seq_pos_rm(i + j);
            }

            pos[idx] = other.pos[j];
            ext[idx] = other.ext[j];
            seq[idx] = other.seq[j];

            if (pos[idx] != -1) {
                seq_pos_add(i + j);
            }

            assert(shift[idx] == 0);
        }
    }

    // set the state of cells [idxs[0], idxs[1], ..., idxs[idxs.size() - 1])
    void set(const std::vector<uint32_t> & idxs, const llama_kv_cells & other) {
        assert(idxs.size() == other.pos.size());

        for (uint32_t j = 0; j < other.pos.size(); ++j) {
            const auto idx = idxs[j];

            if (pos[idx] == -1 && other.pos[j] != -1) {
                used.insert(idx);
            }

            if (pos[idx] != -1 && other.pos[j] == -1) {
                used.erase(idx);
            }

            if (pos[idx] != -1) {
                seq_pos_rm(idx);
            }

            pos[idx] = other.pos[j];
            ext[idx] = other.ext[j];
            seq[idx] = other.seq[j];

            if (pos[idx] != -1) {
                seq_pos_add(idx);
            }

            assert(shift[idx] == 0);
        }
    }

    // clear a non-empty cell
    void rm(uint32_t i) {
        assert(i < pos.size());
        assert(pos[i] != -1);

        seq_pos_rm(i);
        seq[i].reset();

        pos[i] = -1;
        ext[i].reset();
        shift[i] = 0;

        used.erase(i);
    }

    // note: call only if the cell has seq_id
    // return true if the cell becomes empty
    bool seq_rm(uint32_t i, llama_seq_id seq_id) {
        assert(i < pos.size());
        assert(seq[i].test(seq_id));
        assert(pos[i] != -1);
        assert(seq_id >= 0);

        seq[i].reset(seq_id);
        seq_pos_dec(seq_id, i);

        if (seq[i].none()) {
            pos[i] = -1;
            ext[i].reset();
            shift[i] = 0;

            used.erase(i);

            return true;
        }

        return false;
    }

    // remove seq_id from its cells at positions in [p0, p1)
    // return the lowest index of a cell that becomes empty, or size() if none does
    uint32_t seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
        assert(seq_id >= 0);
        assert(seq_id < LLAMA_MAX_SEQ);

        uint32_t res = size();

        // Continuum: the index is a sorted array, so the cells are collected before any is removed, then removed
        // newest first: a cut of her newest positions (a rollback, a rejected draft) pops the index's tail and keeps
        // it exact. A mutator may settle a stale index; only a query may not.
        if (seq_pos[seq_id].is_stale()) {
            settle(seq_id);
            stale.reset(seq_id);
        }
        const auto & sp = seq_pos[seq_id];
        std::vector<uint32_t> cut;
        for (auto it = sp.lower_bound({ p0, 0 }); it != sp.end() && it->first < p1; ++it) {
            cut.push_back(it->second);
        }
        for (auto it = cut.rbegin(); it != cut.rend(); ++it) {
            if (seq_rm(*it, seq_id)) {
                res = std::min(res, *it);
            }
        }

        return res;
    }

    // return true if the cell becomes empty (i.e. it did not contain seq_id before the call)
    bool seq_keep(uint32_t i, llama_seq_id seq_id) {
        assert(i < pos.size());

        if (seq[i].test(seq_id)) {
            seq_pos_rm(i);
            seq[i].reset();

            seq[i].set(seq_id);
            seq_pos_inc(seq_id, i);

            return false;
        }

        if (seq[i].any()) {
            seq_pos_rm(i);
            seq[i].reset();

            pos[i] = -1;
            ext[i].reset();
            shift[i] = 0;

            used.erase(i);

            return true;
        }

        assert(pos[i] == -1);

        return false;
    }

    // number of different sequences in the cell
    int seq_count(uint32_t i) const {
        assert(i < pos.size());
        assert(pos[i] != -1);

        return seq[i].count();
    }

    // the full set of sequences this cell is visible to
    const seq_set_t & seq_get_all(uint32_t i) const {
        assert(i < pos.size());

        return seq[i];
    }

    // check if the cell contains seq_id
    bool seq_has(uint32_t i, llama_seq_id seq_id) const {
        assert(i < pos.size());
        assert(seq_id >= 0);

        return seq[i].test(seq_id);
    }

    // the (pos, cell) pairs of sequence seq_id, ordered by position
    // Continuum: a stale index here is a missed settle, refused loudly rather than read
    const llama_kv_seq_pos & seq_pos_get(llama_seq_id seq_id) const {
        assert(seq_id >= 0);
        assert(seq_id < LLAMA_MAX_SEQ);
        GGML_ASSERT(!seq_pos[seq_id].is_stale() && "a cache operation left this sequence's index unsettled");

        return seq_pos[seq_id];
    }

    // the token of the cell of sequence seq_id at the largest position <= p
    // when several cells share that position, the one with the highest index wins
    // return LLAMA_TOKEN_NULL if the sequence has no cell at or before p
    // note: used by n-gram input embeddings to recover the tokens preceding a ubatch
    llama_token seq_pos_tok_le(llama_seq_id seq_id, llama_pos p) const {
        assert(seq_id >= 0);
        assert(seq_id < LLAMA_MAX_SEQ);

        const auto & sp = seq_pos[seq_id];

        if (sp.is_stale()) {
            // answered from the cells, counted: correct, and a defect (the operation that changed them did not settle)
            n_stale_queries.add();
            int64_t best = -1;
            for (uint32_t i = used.min(); i < used.max_p1(); ++i) {
                if (seq[i].test(seq_id) && pos[i] <= p && (best < 0 || pos[i] >= pos[best])) {
                    best = i;
                }
            }
            return best < 0 ? LLAMA_TOKEN_NULL : ext[best].tok;
        }

        auto it = sp.upper_bound({ p, std::numeric_limits<uint32_t>::max() });
        if (it == sp.begin()) {
            return LLAMA_TOKEN_NULL;
        }

        return ext[(--it)->second].tok;
    }

    // note: call only if the cell is not empty and the seq_id is not in the cell
    void seq_add(uint32_t i, llama_seq_id seq_id) {
        assert(i < pos.size());
        assert(pos[i] != -1);
        assert(!seq[i].test(seq_id));

        seq[i].set(seq_id);
        seq_pos_inc(seq_id, i);
    }

    // return the sequence id of this cell
    // note: call only for cells with exactly one sequence
    llama_seq_id seq_get(uint32_t i) const {
        assert(seq[i].count() == 1);

        for (int s = 0; s < LLAMA_MAX_SEQ; ++s) {
            if (seq[i].test(s)) {
                return s;
            }
        }

        return -1;
    }

    // Continuum: relabel every cell of sequence `from` as `to`, with its position index: a persona's cells bound back
    // as the sequence she is bound as now. `to` holds no cell here (a stream holds one persona), so nothing merges.
    void seq_rename(llama_seq_id from, llama_seq_id to) {
        if (from == to) {
            return;
        }
        for (uint32_t i = used.min(); i < used.max_p1(); ++i) {
            if (seq[i].test(from)) {
                seq[i].reset(from);
                seq[i].set(to);
            }
        }
        seq_pos[to].swap(seq_pos[from]);
        seq_bits[to].swap(seq_bits[from]);
        const bool was_stale = stale.test(from);
        stale.set(from, stale.test(to));
        stale.set(to, was_stale);
    }

    // number of cells that contain seq_id
    uint32_t seq_n_cells(llama_seq_id seq_id) const {
        assert(seq_id >= 0);
        assert(seq_id < LLAMA_MAX_SEQ);

        return seq_n_cells(seq_id, std::numeric_limits<llama_pos>::min());
    }

    // number of cells that contain seq_id at a position >= p_min
    uint32_t seq_n_cells(llama_seq_id seq_id, llama_pos p_min) const {
        assert(seq_id >= 0);
        assert(seq_id < LLAMA_MAX_SEQ);

        const auto & sp = seq_pos[seq_id];

        if (sp.is_stale()) {
            n_stale_queries.add();
            uint32_t n = 0;
            for (uint32_t i = used.min(); i < used.max_p1(); ++i) {
                n += seq[i].test(seq_id) && pos[i] >= p_min;
            }
            return n;
        }

        return std::distance(sp.lower_bound({ p_min, 0 }), sp.end());
    }

    // get the lists of the cells that contain any of the sequences of each slice, in cell order
    // one list of max `n_per_seq` for each slice, padded with -1
    // the cells of slice s at a position below slices[s].p_min are skipped
    LLAMA_ATTRIBUTE_NOINLINE
    void get_seq_cells(const std::vector<llama_kv_rows_slice> & slices, int32_t * data, int64_t n_per_seq) const {
        assert(slices.size() >= 1);

        const int64_t n_slices = (int64_t) slices.size();

        const uint32_t n_words = (used_max_p1() + 63)/64;

        for (int64_t s = 0; s < n_slices; ++s) {
            const llama_pos pm = slices[s].p_min;

            int32_t * dst = data + s*n_per_seq;
            int32_t * end = dst + n_per_seq;

            for (uint32_t w = 0; w < n_words; ++w) {
                uint64_t word = 0;
                for (const llama_seq_id seq_id : slices[s].seq_ids) {
                    const auto & bits = seq_bits[seq_id];
                    if (w < bits.size()) {
                        word |= bits[w];
                    }
                }

                for (uint64_t b = word; b != 0; b &= b - 1) {
                    const int32_t i = 64*w + ctz64(b);
                    if (pos[i] >= pm) {
                        GGML_ASSERT(dst < end);
                        *dst++ = i;
                    }
                }
            }

            std::fill(dst, end, -1);
        }
    }

    // the minimum position of sequence seq_id currently present in any of the cells
    // return -1 if the sequence is not present
    llama_pos seq_pos_min(llama_seq_id seq_id) const {
        assert(seq_id >= 0);
        assert(seq_id < LLAMA_MAX_SEQ);

        const llama_kv_seq_pos & index = seq_pos[seq_id];
        if (index.is_stale()) {
            return unsettled_extremum(seq_id, false);
        }
        return index.empty() ? -1 : index.min();
    }

    // the maximum position of sequence seq_id currently present in any of the cells
    // return -1 if the sequence is not present
    llama_pos seq_pos_max(llama_seq_id seq_id) const {
        assert(seq_id >= 0);
        assert(seq_id < LLAMA_MAX_SEQ);

        const llama_kv_seq_pos & index = seq_pos[seq_id];
        if (index.is_stale()) {
            return unsettled_extremum(seq_id, true);
        }
        return index.empty() ? -1 : index.max();
    }

    // note: call only if the cell is not empty
    llama_pos pos_get(uint32_t i) const {
        assert(i < pos.size());
        assert(pos[i] != -1);

        return pos[i];
    }

    const llama_kv_cell_ext & ext_get(uint32_t i) const {
        assert(i < pos.size());
        assert(pos[i] != -1);

        return ext[i];
    }

    // note: call only if the cell is not empty
    llama_pos get_shift(uint32_t i) const {
        assert(i < pos.size());
        assert(pos[i] != -1);

        return shift[i];
    }

    // check if a cell is not empty and its position is within [p0, p1)
    bool pos_in(uint32_t i, llama_pos p0, llama_pos p1) const {
        assert(i < pos.size());

        return pos[i] >= p0 && pos[i] < p1;
    }

    // set the position of an empty cell
    // does not modify "has_shift"
    // note: call only if the cell is empty
    void pos_set(uint32_t i, llama_pos p) {
        assert(i < pos.size());
        assert(pos[i] == -1);
        assert(seq[i].none());

        pos[i] = p;

        used.insert(i);
    }

    void ext_set(uint32_t i, llama_kv_cell_ext p) {
        assert(i < ext.size());
        ext[i] = p;
    }

    // pos[i] = pos[i] + d
    // sets "has_shift" to true
    // note: call only if the cell is not empty
    bool pos_add(uint32_t i, llama_pos d) {
        assert(i < pos.size());
        assert(pos[i] != -1);

        seq_pos_rm(i);

        pos[i]   += d;
        shift[i] += d;

        has_shift = true;

        if (pos[i] < 0) {
            seq[i].reset();
            pos[i] = -1;
            shift[i] = 0;

            used.erase(i);

            return true;
        }

        seq_pos_add(i);

        return false;
    }

    // pos[i] = pos[i] / d
    // sets "has_shift" to true
    // note: call only if the cell is not empty
    void pos_div(uint32_t i, int d) {
        assert(i < pos.size());
        assert(pos[i] != -1);

        const llama_pos p_old = pos[i];

        seq_pos_rm(i);

        pos[i]   /= d;
        shift[i] += p_old - pos[i];

        seq_pos_add(i);

        has_shift = true;
    }

private:
    static uint64_t next_lineage() {
        static std::atomic<uint64_t> next { 1 };
        return next.fetch_add(1, std::memory_order_relaxed);
    }
    uint64_t  lineage_id = next_lineage();
    std::vector<std::pair<llama_seq_id, std::weak_ptr<const llama_pos>>> checkpoints;

    bool has_shift = false;

    // the occupied cells (i.e. pos[i] != -1, allowed to not have any seq_id): a bitmap with its count and extrema
    llama_kv_used used;

    std::vector<llama_pos> pos;

    // stores extra info per cell
    std::vector<llama_kv_cell_ext> ext;

    // this array accumulates any applied shifts to the pos array since the last reset_shift() call
    // this is used to queue multiple updates to the pos array, which in the end can be applied in one go:
    //
    //   cells.pos_add(x, shift_x);
    //   cells.pos_div(y, shift_y);
    //   ...
    //
    //   if (cells.has_shift()) {
    //      for (int i = 0; i < n; ++i) {
    //          auto shift_i = cells.get_shift(i);
    //          ...
    //      }
    //      cells.reset_shift();
    //   }
    //
    std::vector<llama_pos> shift;

    // the bitset seq[i] tells us which sequences are currently occupying the i-th cell
    std::vector<seq_set_t> seq;

    // seq_pos[s] holds one (pos, cell) pair per cell that carries sequence s, ordered by position
    // this way seq_pos[s].begin() and the last entry give us the min/max positions currently in the cache
    // and upper_bound() on a position finds the nearest cell of the sequence in logarithmic time
    //
    // the cell index is part of the key because a position can occur more than once for the same seq:
    //  - during performing a cache reuse via (rm + add)
    //  - some vision models have input embeddings with repeating positions
    //
    // Continuum fork: a native sorted array (llama-kv-cell-index.h), so a persona's metadata holds no node container
    // and her flash image binds without rebuilding it. Exact on its O(1) paths (her newest position appended or
    // removed); any other change marks it stale (`stale`), and the operation that made the change settles it.
    llama_kv_seq_pos seq_pos[LLAMA_MAX_SEQ];

    // the sequences whose index a change left stale, so settling visits only those
    std::bitset<LLAMA_MAX_SEQ> stale;

    // how many times an index was recomputed from the cells (the freeze and restore fixtures count them)
    uint64_t n_settles = 0;

    // queries answered from the cells because an index was stale (the only thing a query writes)
    llama_kv_query_count n_stale_queries;

    // recompute sequence seq_id's index from the cells
    void settle(llama_seq_id seq_id) {
        ++n_settles;
        std::vector<llama_kv_pos_cell> now;
        for (uint32_t i = used.min(); i < used.max_p1(); ++i) {
            if (seq[i].test(seq_id)) {
                now.emplace_back(pos[i], i);
            }
        }
        seq_pos[seq_id].settle(now);
    }

    // a stale sequence's min or max read from the cells without writing its index, counted: correct, and a defect
    // (the operation that changed the cells did not settle them)
    llama_pos unsettled_extremum(llama_seq_id seq_id, bool want_max) const {
        n_stale_queries.add();
        llama_pos found = -1;
        for (uint32_t i = used.min(); i < used.max_p1(); ++i) {
            if (seq[i].test(seq_id) && (found == -1 || (want_max ? pos[i] > found : pos[i] < found))) {
                found = pos[i];
            }
        }
        return found;
    }

    // bit i of seq_bits[s] is set if cell i carries sequence s (the transpose of `seq`), used to build the cell lists
    // allocated on the first cell of the sequence, so the copies made by cp() never allocate it
    std::vector<uint64_t> seq_bits[LLAMA_MAX_SEQ];

    // helper functions for updating `seq_pos` and `seq_bits`, once cell at a time:

    void seq_pos_dec(llama_seq_id s, uint32_t i) {
        seq_pos[s].dec(pos[i], i);
        if (seq_pos[s].is_stale()) {
            stale.set(s);
        }

        seq_bits[s][i/64] &= ~(1ull << (i%64));
    }

    void seq_pos_inc(llama_seq_id s, uint32_t i) {
        seq_pos[s].inc(pos[i], i);
        if (seq_pos[s].is_stale()) {
            stale.set(s);
        }

        if (seq_bits[s].empty()) {
            seq_bits[s].assign((pos.size() + 63)/64, 0);
        }

        seq_bits[s][i/64] |= 1ull << (i%64);
    }

    // remove cell i
    void seq_pos_rm(uint32_t i) {
        for (int s = 0; s < LLAMA_MAX_SEQ; ++s) {
            if (seq[i].test(s)) {
                seq_pos_dec(s, i);
            }
        }
    }

    // add cell i
    void seq_pos_add(uint32_t i) {
        for (int s = 0; s < LLAMA_MAX_SEQ; ++s) {
            if (seq[i].test(s)) {
                seq_pos_inc(s, i);
            }
        }
    }
};

using llama_kv_cells_vec = std::vector<llama_kv_cells>;
