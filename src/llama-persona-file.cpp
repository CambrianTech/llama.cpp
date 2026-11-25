// Continuum fork (CONTINUUM-ENGINE 4.16, "restore is the norm"): a persona's saved blocks on disk and back, so a restart
// demotes her instead of dropping her. The file holds what a move already copies (her K/V rows per layer, her occupied
// cells, her recurrent rows) plus who she is and the memory layout they were taken from. Nothing is re-serialized or
// recomputed: rows are written and read as bytes.
//
// A write goes to `<path>.tmp` and is renamed into place, so a crash mid-write never leaves a torn file that reads as
// hers; the rename replaces an existing file on every platform (POSIX rename, Windows MoveFileEx). A read checks every
// layer length against the geometry her preparation recorded before it allocates, fills temporaries, and commits them to
// her blocks only when every check passed (owner, epoch, both layout
// fingerprints, layer and cell counts, a complete read), so a refused read leaves her as she was.

#include "llama.h"
#include "llama-persona.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#ifdef _WIN32
#    ifndef WIN32_LEAN_AND_MEAN
#        define WIN32_LEAN_AND_MEAN
#    endif
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#endif
#include <string>
#include <vector>

namespace {

constexpr uint32_t MAGIC   = 0x31425043; // "CPB1"
// 2: the stop's generation, and where her attention rows are (in her attached storage, or in this record).
// 1 is read as a record holding her rows with no generation: the one migration of minds a format-1 stop preserved.
constexpr uint32_t VERSION = 2;
constexpr uint8_t ROWS_IN_RECORD  = 0;
constexpr uint8_t ROWS_IN_STORAGE = 1;
// one occupied cell on disk: index (u32), position, 2D extension x and y (i32 each), whether her sequence holds it (u8)
constexpr size_t CELL_BYTES = 4 + 4 + 4 + 4 + 1;

struct writer {
    std::FILE * f;
    bool ok = true;
    template <typename T> void put(const T & v) { ok = ok && std::fwrite(&v, sizeof(T), 1, f) == 1; }
    // bytes already laid out, in one write
    void raw(const std::vector<uint8_t> & b) { ok = ok && (b.empty() || std::fwrite(b.data(), 1, b.size(), f) == b.size()); }
    void bytes(const std::vector<uint8_t> & b) {
        put<uint64_t>(b.size());
        ok = ok && (b.empty() || std::fwrite(b.data(), 1, b.size(), f) == b.size());
    }
    // her rows straight from her own storage (the Core's pointer move): each layer one device-to-host transfer of
    // exactly her rows into one reused buffer, then one write. The same format as rows(), so either reader takes it.
    // The buffer is the device's pinned host memory where it has one (CUDA): the DMA lands where the write reads, with
    // no driver bounce copy through pageable memory. A device without one (CPU, Metal's shared memory) has no faster
    // host memory to offer, so the buffer is ordinary memory.
    void rows_of(const std::vector<ggml_tensor *> & layers, const std::vector<size_t> & row, uint32_t n_rows) {
        put<uint32_t>((uint32_t) layers.size());
        size_t largest = 0;
        ggml_backend_buffer_type_t pinned = nullptr;
        for (size_t i = 0; i < layers.size(); ++i) {
            if (layers[i] != nullptr) {
                largest = std::max(largest, row[i] * n_rows);
                const ggml_backend_dev_t dev = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(layers[i]->buffer));
                pinned = pinned ? pinned : (dev ? ggml_backend_dev_host_buffer_type(dev) : nullptr);
            }
        }
        ggml_backend_buffer_ptr held(pinned && largest > 0 ? ggml_backend_buft_alloc_buffer(pinned, largest) : nullptr);
        std::vector<uint8_t> ordinary(held ? 0 : largest);
        uint8_t * staging = held ? static_cast<uint8_t *>(ggml_backend_buffer_get_base(held.get())) : ordinary.data();
        for (size_t i = 0; i < layers.size() && ok; ++i) {
            const size_t n = layers[i] ? row[i] * n_rows : 0;
            put<uint64_t>(n);
            if (n > 0) {
                ggml_backend_tensor_get(layers[i], staging, 0, n);
                ok = ok && std::fwrite(staging, 1, n, f) == n;
            }
        }
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
    // exactly `n` bytes in one read (`n` from this memory's own bounds, never a length the file claims)
    bool raw(std::vector<uint8_t> & out, size_t n) {
        out.resize(n);
        ok = ok && (n == 0 || std::fread(out.data(), 1, n, f) == n);
        return ok;
    }
    // one layer's bytes, refused unless the file says exactly `expect` (from this memory's geometry): no buffer is ever
    // sized from a length the file claims, so a truncated or foreign file allocates nothing (Codex on #155)
    bool bytes(std::vector<uint8_t> & out, size_t expect) {
        const uint64_t n = get<uint64_t>();
        if (!ok || n != expect) {
            return false;
        }
        out.resize(n);
        ok = n == 0 || std::fread(out.data(), 1, n, f) == n;
        return ok;
    }
    // every layer, each the length `expect` names; a file with another layer count is not this memory's
    bool rows(std::vector<std::vector<uint8_t>> & out, const std::vector<size_t> & expect) {
        const uint32_t n = get<uint32_t>();
        if (!ok || n != expect.size()) {
            return false;
        }
        out.resize(n);
        for (uint32_t i = 0; i < n; ++i) {
            if (!bytes(out[i], expect[i])) {
                return false;
            }
        }
        return ok;
    }
};

// Replace `path` with the finished `tmp` in one step. POSIX rename replaces an existing destination; the Windows CRT's
// refuses one (Codex on #155, UCRT on BigMama: -1 with the old body intact), so a second save to her fixed path failed.
bool replace_file(const std::string & tmp, const char * path) {
#ifdef _WIN32
    return MoveFileExA(tmp.c_str(), path, MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) != 0;
#else
    return std::rename(tmp.c_str(), path) == 0;
#endif
}

// her recurrent storage's r (`which` 0) or s (1) tensor of every layer, from its layer-major r, s, p list
std::vector<ggml_tensor *> recurrent(const llama_rs_storage & storage, size_t which) {
    std::vector<ggml_tensor *> out;
    for (size_t i = which; i < storage.own.t.size(); i += 3) {
        out.push_back(storage.own.t[i]);
    }
    return out;
}

struct closes {
    std::FILE * f;
    ~closes() { if (f) { std::fclose(f); } }
};

} // namespace

int32_t llama_persona_write_file(const llama_persona * persona, const char * path, uint64_t generation) {
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
    w.put<uint64_t>(b.kv_rows); // not the stream size: a restart's window may differ
    w.put<uint64_t>(b.rs_layout);
    w.put<uint64_t>(generation);
    // attention: where her rows are, then her occupied cells (position, 2D extension, whether her sequence holds it).
    // In attached storage they are already in the caller's file: the record names its bytes and carries none of them.
    const size_t attached = b.kv.storage.attached_bytes;
    w.put<uint8_t>(attached > 0 ? ROWS_IN_STORAGE : ROWS_IN_RECORD);
    w.put<uint64_t>(attached);
    w.put<uint32_t>(b.kv.n_rows);
    if (attached > 0) {
        // nothing: her rows are her storage
    } else if (b.kv.storage.held()) {
        w.rows_of(b.kv.storage.k, b.geometry.k_row, b.kv.n_rows);
        w.rows_of(b.kv.storage.v, b.geometry.v_row, b.kv.n_rows);
    } else {
        w.rows(b.kv.k);
        w.rows(b.kv.v);
    }
    const uint32_t n_cells = b.kv.cells.size();
    w.put<uint32_t>(n_cells);
    uint32_t occupied = 0;
    for (uint32_t i = 0; i < n_cells; ++i) {
        occupied += b.kv.cells.is_empty(i) ? 0 : 1;
    }
    w.put<uint32_t>(occupied);
    // her occupied cells as one packed table (index, position, 2D extension, whether her sequence holds it), written in
    // one transfer: at 30k cells a write per field cost milliseconds of a stop
    std::vector<uint8_t> table(size_t(occupied) * CELL_BYTES);
    uint8_t * at = table.data();
    for (uint32_t i = 0; i < n_cells; ++i) {
        if (b.kv.cells.is_empty(i)) {
            continue;
        }
        const auto & ext = b.kv.cells.ext_get(i);
        const llama_pos pos = b.kv.cells.pos_get(i);
        const uint8_t hers = b.seq >= 0 && b.kv.cells.seq_has(i, b.seq) ? 1 : 0;
        std::memcpy(at, &i, 4);
        std::memcpy(at + 4, &pos, 4);
        std::memcpy(at + 8, &ext.x, 4);
        std::memcpy(at + 12, &ext.y, 4);
        at[16] = hers;
        at += CELL_BYTES;
    }
    w.raw(table);
    // recurrent: her cell's record, then her rows in every snapshot group unless they are in her attached storage
    w.put<uint8_t>(b.rs.has_state ? 1 : 0);
    w.put<llama_pos>(b.rs.pos);
    w.put<uint32_t>(b.rs.rs_idx);
    if (attached > 0 && b.rs.storage.held()) {
        // nothing: her recurrent rows are her storage too (slice 2c)
    } else if (b.rs.storage.held()) {
        const uint32_t whole = b.rs.has_state ? 1 : 0; // her storage tensor is her rows in every group: whole or none
        w.rows_of(recurrent(b.rs.storage, 0), b.geometry.r_bytes, whole);
        w.rows_of(recurrent(b.rs.storage, 1), b.geometry.s_bytes, whole);
    } else {
        w.rows(b.rs.r);
        w.rows(b.rs.s);
    }
    const bool flushed = w.ok && std::fflush(f) == 0;
    const bool closed = std::fclose(f) == 0;
    if (!flushed || !closed || !replace_file(tmp, path)) {
        std::remove(tmp.c_str());
        return (int32_t) llama_persona_move::io_error;
    }
    return (int32_t) llama_persona_move::done;
}

int32_t llama_persona_read_file(llama_persona * persona, const char * path, uint64_t generation) {
    auto & b = persona->blocks;
    std::FILE * f = std::fopen(path, "rb");
    if (f == nullptr) {
        return (int32_t) llama_persona_move::io_error;
    }
    closes guard { f };
    reader r { f };
    const auto mismatch = (int32_t) llama_persona_move::file_mismatch;
    const uint32_t magic = r.get<uint32_t>();
    const uint32_t version = r.get<uint32_t>();
    if (!r.ok || magic != MAGIC || (version != VERSION && version != 1)) {
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
    // this memory's rows only: the row format (layers, types, widths) and the recurrent layout her prepared blocks carry;
    // the stream size may differ, and she fits when her rows and every cell she occupies are inside this stream
    if (r.get<uint64_t>() != b.kv_rows || r.get<uint64_t>() != b.rs_layout || !r.ok) {
        return r.ok ? mismatch : (int32_t) llama_persona_move::io_error;
    }
    // format 2: this stop's record only, and where her rows are. A record whose rows are in storage binds only to attached
    // storage of exactly the bytes it names (her own file, mapped by the caller); one whose rows are here fills any storage.
    uint8_t rows_in = ROWS_IN_RECORD;
    if (version == VERSION) {
        const uint64_t written_in = r.get<uint64_t>();
        rows_in = r.get<uint8_t>();
        const uint64_t storage_bytes = r.get<uint64_t>();
        if (!r.ok) {
            return (int32_t) llama_persona_move::io_error;
        }
        if (written_in != generation || rows_in > ROWS_IN_STORAGE) {
            return mismatch;
        }
        if (rows_in == ROWS_IN_STORAGE && (storage_bytes == 0 || storage_bytes != b.kv.storage.attached_bytes)) {
            return mismatch;
        }
    }
    const uint32_t stream_cells = b.kv.cells.size();
    const uint32_t n_rows = r.get<uint32_t>();
    if (r.ok && n_rows > stream_cells) {
        return mismatch; // she holds more than this stream: it never fits her
    }
    // every length the file may hold, from this memory's geometry (persona_prepare), before anything is read into memory
    const auto & g = b.geometry;
    std::vector<size_t> k_bytes(g.k_row.size()), v_bytes(g.v_row.size());
    for (size_t i = 0; i < g.k_row.size(); ++i) {
        k_bytes[i] = g.k_row[i] * n_rows;
    }
    for (size_t i = 0; i < g.v_row.size(); ++i) {
        v_bytes[i] = g.v_row[i] * n_rows;
    }
    std::vector<std::vector<uint8_t>> k, v, rs_r, rs_s;
    const bool rows_here = rows_in == ROWS_IN_RECORD;
    if (!r.ok || (rows_here && (!r.rows(k, k_bytes) || !r.rows(v, v_bytes)))) {
        return r.ok ? mismatch : (int32_t) llama_persona_move::io_error;
    }
    const uint32_t n_cells = r.get<uint32_t>();
    const uint32_t occupied = r.get<uint32_t>();
    if (!r.ok || occupied > n_cells || occupied > stream_cells) {
        return r.ok ? mismatch : (int32_t) llama_persona_move::io_error;
    }
    struct cell { uint32_t i; llama_pos pos; llama_kv_cell_ext ext; bool hers; };
    // her cell table in one transfer, sized from `occupied` only once it is known to fit this stream (checked above)
    std::vector<uint8_t> table;
    if (!r.raw(table, size_t(occupied) * CELL_BYTES)) {
        return (int32_t) llama_persona_move::io_error;
    }
    std::vector<cell> cells(occupied);
    const uint8_t * at = table.data();
    for (auto & c : cells) {
        std::memcpy(&c.i, at, 4);
        std::memcpy(&c.pos, at + 4, 4);
        std::memcpy(&c.ext.x, at + 8, 4);
        std::memcpy(&c.ext.y, at + 12, 4);
        c.hers = at[16] != 0;
        at += CELL_BYTES;
        if (c.i >= stream_cells) {
            return mismatch;
        }
    }
    const bool has_state = r.get<uint8_t>() != 0;
    const llama_pos rs_pos = r.get<llama_pos>();
    const uint32_t rs_idx = r.get<uint32_t>();
    // her recurrent rows are whole when she held state and absent when she did not (persona_check_save)
    const std::vector<size_t> r_bytes = has_state ? g.r_bytes : std::vector<size_t>(g.r_bytes.size(), 0);
    const std::vector<size_t> s_bytes = has_state ? g.s_bytes : std::vector<size_t>(g.s_bytes.size(), 0);
    // her recurrent rows are here unless her record says her rows are in storage and she holds recurrent storage
    const bool rs_here = rows_here || !b.rs.storage.held();
    if (!r.ok || (rs_here && (!r.rows(rs_r, r_bytes) || !r.rows(rs_s, s_bytes)))) {
        return r.ok ? mismatch : (int32_t) llama_persona_move::io_error;
    }
    // every check passed: commit. Her cells start from the empty, stream-sized object her preparation made.
    llama_kv_cells restored;
    restored.resize(stream_cells);
    for (const auto & c : cells) {
        restored.pos_set(c.i, c.pos);
        restored.ext_set(c.i, c.ext);
        if (c.hers) {
            restored.seq_add(c.i, seq);
        }
    }
    b.kv.n_rows = n_rows;
    if (b.kv.storage.held()) {
        // the Core's pointer move: her rows land in her own storage, one transfer per layer
        for (size_t i = 0; i < k.size(); ++i) {
            if (!k[i].empty()) { ggml_backend_tensor_set(b.kv.storage.k[i], k[i].data(), 0, k[i].size()); }
            if (!v[i].empty()) { ggml_backend_tensor_set(b.kv.storage.v[i], v[i].data(), 0, v[i].size()); }
        }
    } else {
        b.kv.k.swap(k);
        b.kv.v.swap(v);
    }
    b.kv.cells.swap(restored);
    b.rs.has_state = has_state;
    b.rs.pos = rs_pos;
    b.rs.rs_idx = rs_idx;
    if (b.rs.storage.held()) {
        // her recurrent rows land in her own storage, one transfer per tensor (none when they are already there)
        const std::vector<ggml_tensor *> rs_r_to = recurrent(b.rs.storage, 0), rs_s_to = recurrent(b.rs.storage, 1);
        for (size_t i = 0; i < rs_r.size() && i < rs_r_to.size(); ++i) {
            if (!rs_r[i].empty()) { ggml_backend_tensor_set(rs_r_to[i], rs_r[i].data(), 0, rs_r[i].size()); }
            if (!rs_s[i].empty()) { ggml_backend_tensor_set(rs_s_to[i], rs_s[i].data(), 0, rs_s[i].size()); }
        }
    } else {
        b.rs.r.swap(rs_r);
        b.rs.s.swap(rs_s);
    }
    b.seq = seq;
    b.owned = true;
    b.owner = owner;
    b.owner_epoch = epoch;
    return (int32_t) llama_persona_move::done;
}
