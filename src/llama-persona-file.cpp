// Continuum fork (CONTINUUM-ENGINE 4.16, "restore is the norm"): a persona's saved blocks on disk and back, so a restart
// demotes her instead of dropping her. The file holds what a move already copies (her K/V rows per layer, her occupied
// cells, her recurrent rows) plus who she is and the memory layout they were taken from. Nothing is re-serialized or
// recomputed: rows are written and read as bytes.
//
// A write goes to `<path>.tmp` and is renamed into place, so a crash mid-write never leaves a torn file that reads as
// hers. A read fills temporaries and commits them to her blocks only when every check passed (owner, epoch, both layout
// fingerprints, layer and cell counts, a complete read), so a refused read leaves her as she was.

#include "llama.h"
#include "llama-persona.h"

#include <cstdio>
#include <string>
#include <vector>

namespace {

constexpr uint32_t MAGIC   = 0x31425043; // "CPB1"
constexpr uint32_t VERSION = 1;

struct writer {
    std::FILE * f;
    bool ok = true;
    template <typename T> void put(const T & v) { ok = ok && std::fwrite(&v, sizeof(T), 1, f) == 1; }
    void bytes(const std::vector<uint8_t> & b) {
        put<uint64_t>(b.size());
        ok = ok && (b.empty() || std::fwrite(b.data(), 1, b.size(), f) == b.size());
    }
    void rows(const std::vector<std::vector<uint8_t>> & layers) {
        put<uint32_t>((uint32_t) layers.size());
        for (const auto & layer : layers) {
            bytes(layer);
        }
    }
};

struct reader {
    std::FILE * f;
    bool ok = true;
    template <typename T> T get() {
        T v{};
        ok = ok && std::fread(&v, sizeof(T), 1, f) == 1;
        return v;
    }
    std::vector<uint8_t> bytes() {
        const uint64_t n = get<uint64_t>();
        std::vector<uint8_t> b;
        if (!ok) {
            return b;
        }
        b.resize(n);
        ok = n == 0 || std::fread(b.data(), 1, n, f) == n;
        return b;
    }
    // `expect` layers: the memory's count; a file with another count is not this memory's
    bool rows(std::vector<std::vector<uint8_t>> & out, size_t expect) {
        const uint32_t n = get<uint32_t>();
        if (!ok || n != expect) {
            return false;
        }
        out.resize(n);
        for (auto & layer : out) {
            layer = bytes();
        }
        return ok;
    }
};

struct closes {
    std::FILE * f;
    ~closes() { if (f) { std::fclose(f); } }
};

} // namespace

int32_t llama_persona_write_file(const llama_persona * persona, const char * path) {
    const auto & b = persona->blocks;
    if (!b.owned) {
        return (int32_t) llama_persona_move::file_mismatch; // never saved: these blocks hold nothing of hers to write
    }
    const std::string tmp = std::string(path) + ".tmp";
    std::FILE * f = std::fopen(tmp.c_str(), "wb");
    if (f == nullptr) {
        return (int32_t) llama_persona_move::io_error;
    }
    writer w { f };
    w.put(MAGIC);
    w.put(VERSION);
    w.put<uint64_t>(b.owner);
    w.put<uint64_t>(b.owner_epoch);
    w.put<int32_t>(b.seq);
    w.put<uint64_t>(b.kv_layout);
    w.put<uint64_t>(b.rs_layout);
    // attention: her rows, then her occupied cells (position, 2D extension, whether her sequence holds it)
    w.put<uint32_t>(b.kv.n_rows);
    w.rows(b.kv.k);
    w.rows(b.kv.v);
    const uint32_t n_cells = b.kv.cells.size();
    w.put<uint32_t>(n_cells);
    uint32_t occupied = 0;
    for (uint32_t i = 0; i < n_cells; ++i) {
        occupied += b.kv.cells.is_empty(i) ? 0 : 1;
    }
    w.put<uint32_t>(occupied);
    for (uint32_t i = 0; i < n_cells; ++i) {
        if (b.kv.cells.is_empty(i)) {
            continue;
        }
        const auto & ext = b.kv.cells.ext_get(i);
        w.put<uint32_t>(i);
        w.put<llama_pos>(b.kv.cells.pos_get(i));
        w.put<llama_pos>(ext.x);
        w.put<llama_pos>(ext.y);
        w.put<uint8_t>(b.seq >= 0 && b.kv.cells.seq_has(i, b.seq) ? 1 : 0);
    }
    // recurrent: her rows in every snapshot group and her cell's record
    w.put<uint8_t>(b.rs.has_state ? 1 : 0);
    w.put<llama_pos>(b.rs.pos);
    w.put<uint32_t>(b.rs.rs_idx);
    w.rows(b.rs.r);
    w.rows(b.rs.s);
    const bool flushed = w.ok && std::fflush(f) == 0;
    const bool closed = std::fclose(f) == 0;
    if (!flushed || !closed || std::rename(tmp.c_str(), path) != 0) {
        std::remove(tmp.c_str());
        return (int32_t) llama_persona_move::io_error;
    }
    return (int32_t) llama_persona_move::done;
}

int32_t llama_persona_read_file(llama_persona * persona, const char * path) {
    auto & b = persona->blocks;
    std::FILE * f = std::fopen(path, "rb");
    if (f == nullptr) {
        return (int32_t) llama_persona_move::io_error;
    }
    closes guard { f };
    reader r { f };
    const auto mismatch = (int32_t) llama_persona_move::file_mismatch;
    if (r.get<uint32_t>() != MAGIC || r.get<uint32_t>() != VERSION || !r.ok) {
        return r.ok ? mismatch : (int32_t) llama_persona_move::io_error;
    }
    const uint64_t owner = r.get<uint64_t>();
    const uint64_t epoch = r.get<uint64_t>();
    const int32_t  seq   = r.get<int32_t>();
    if (!r.ok) {
        return (int32_t) llama_persona_move::io_error;
    }
    // hers only: the persona and epoch her token history names (the identity every move checks)
    if (owner != persona->tokens->persona() || epoch != persona->tokens->epoch()) {
        return mismatch;
    }
    // this memory's only: the layouts her prepared blocks carry for this context
    if (r.get<uint64_t>() != b.kv_layout || r.get<uint64_t>() != b.rs_layout || !r.ok) {
        return r.ok ? mismatch : (int32_t) llama_persona_move::io_error;
    }
    const uint32_t n_rows = r.get<uint32_t>();
    std::vector<std::vector<uint8_t>> k, v, rs_r, rs_s;
    if (!r.rows(k, b.kv.k.size()) || !r.rows(v, b.kv.v.size())) {
        return r.ok ? mismatch : (int32_t) llama_persona_move::io_error;
    }
    const uint32_t n_cells = r.get<uint32_t>();
    const uint32_t occupied = r.get<uint32_t>();
    if (!r.ok || n_cells != b.kv.cells.size() || occupied > n_cells) {
        return r.ok ? mismatch : (int32_t) llama_persona_move::io_error;
    }
    struct cell { uint32_t i; llama_pos pos; llama_kv_cell_ext ext; bool hers; };
    std::vector<cell> cells(occupied);
    for (auto & c : cells) {
        c.i = r.get<uint32_t>();
        c.pos = r.get<llama_pos>();
        c.ext.x = r.get<llama_pos>();
        c.ext.y = r.get<llama_pos>();
        c.hers = r.get<uint8_t>() != 0;
        if (r.ok && c.i >= n_cells) {
            return mismatch;
        }
    }
    const bool has_state = r.get<uint8_t>() != 0;
    const llama_pos rs_pos = r.get<llama_pos>();
    const uint32_t rs_idx = r.get<uint32_t>();
    if (!r.ok || !r.rows(rs_r, b.rs.r.size()) || !r.rows(rs_s, b.rs.s.size())) {
        return r.ok ? mismatch : (int32_t) llama_persona_move::io_error;
    }
    // every check passed: commit. Her cells start from the empty, stream-sized object her preparation made.
    llama_kv_cells restored;
    restored.resize(n_cells);
    for (const auto & c : cells) {
        restored.pos_set(c.i, c.pos);
        restored.ext_set(c.i, c.ext);
        if (c.hers) {
            restored.seq_add(c.i, seq);
        }
    }
    b.kv.n_rows = n_rows;
    b.kv.k.swap(k);
    b.kv.v.swap(v);
    b.kv.cells.swap(restored);
    b.rs.has_state = has_state;
    b.rs.pos = rs_pos;
    b.rs.rs_idx = rs_idx;
    b.rs.r.swap(rs_r);
    b.rs.s.swap(rs_s);
    b.seq = seq;
    b.owned = true;
    b.owner = owner;
    b.owner_epoch = epoch;
    return (int32_t) llama_persona_move::done;
}
