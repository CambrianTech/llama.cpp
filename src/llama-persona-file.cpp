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

#include <cerrno>
#include <cstdio>
#ifdef _WIN32
#    ifndef WIN32_LEAN_AND_MEAN
#        define WIN32_LEAN_AND_MEAN
#    endif
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#endif
#include <algorithm>
#include <string>
#include <vector>

namespace {

constexpr uint32_t MAGIC   = 0x31425043; // "CPB1"
constexpr uint32_t VERSION = 1;

struct writer {
    std::FILE * f;
    int error = 0;
    void write(const void * data, size_t size) {
        if (!error && size && std::fwrite(data, 1, size, f) != size) {
            error = errno;
        }
    }
    template <typename T> void put(const T & v) { write(&v, sizeof(v)); }
    void bytes(const std::vector<uint8_t> & b) {
        put<uint64_t>(b.size());
        write(b.data(), b.size());
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
uint32_t replace_file(const std::string & tmp, const char * path) {
#ifdef _WIN32
    // A short-lived reader without FILE_SHARE_DELETE can hold the complete temporary
    // file across fclose. Retry only the atomic rename, never the block write.
    constexpr ULONGLONG sharing_deadline_ms = 2000;
    const auto deadline = GetTickCount64() + sharing_deadline_ms;
    DWORD delay_ms = 10;
    for (;;) {
        if (MoveFileExA(tmp.c_str(), path, MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
            return 0;
        }
        const auto error = GetLastError();
        const auto now = GetTickCount64();
        if (error != ERROR_SHARING_VIOLATION || now >= deadline) {
            return error;
        }
        Sleep(static_cast<DWORD>(std::min<ULONGLONG>(delay_ms, deadline - now)));
        delay_ms = std::min<DWORD>(2 * delay_ms, 100);
    }
#else
    return std::rename(tmp.c_str(), path) == 0 ? 0 : errno;
#endif
}

struct closes {
    std::FILE * f;
    ~closes() { if (f) { std::fclose(f); } }
};

} // namespace

llama_persona_write_result llama_persona_write_file(const llama_persona * persona, const char * path) {
    const auto & b = persona->blocks;
    if (!b.owned) {
        return {(int32_t) llama_persona_move::file_mismatch, LLAMA_PERSONA_FILE_NONE, 0}; // never saved: these blocks hold nothing of hers to write
    }
    const std::string tmp = std::string(path) + ".tmp";
    std::FILE * f = std::fopen(tmp.c_str(), "wb");
    if (f == nullptr) {
        return {(int32_t) llama_persona_move::io_error, LLAMA_PERSONA_FILE_OPEN, (uint32_t) errno};
    }
    writer w { f };
    w.put(MAGIC);
    w.put(VERSION);
    w.put<uint64_t>(b.owner);
    w.put<uint64_t>(b.owner_epoch);
    w.put<int32_t>(b.seq);
    w.put<uint64_t>(b.kv_rows); // not the stream size: a restart's window may differ
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
    llama_persona_write_result result {(int32_t) llama_persona_move::done, LLAMA_PERSONA_FILE_NONE, 0};
    const auto io_error = (int32_t) llama_persona_move::io_error;
    if (w.error) {
        result = {io_error, LLAMA_PERSONA_FILE_WRITE, (uint32_t) w.error};
    } else if (std::fflush(f) != 0) {
        result = {io_error, LLAMA_PERSONA_FILE_FLUSH, (uint32_t) errno};
    }
    if (std::fclose(f) != 0 && result.status == 0) {
        result = {io_error, LLAMA_PERSONA_FILE_CLOSE, (uint32_t) errno};
    }
    if (result.status == 0) {
        const uint32_t error = replace_file(tmp, path);
        if (error) {
            result = {io_error, LLAMA_PERSONA_FILE_RENAME, error};
        }
    }
    if (result.status != 0) {
        std::remove(tmp.c_str());
    }
    return result;
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
    // this memory's rows only: the row format (layers, types, widths) and the recurrent layout her prepared blocks carry;
    // the stream size may differ, and she fits when her rows and every cell she occupies are inside this stream
    if (r.get<uint64_t>() != b.kv_rows || r.get<uint64_t>() != b.rs_layout || !r.ok) {
        return r.ok ? mismatch : (int32_t) llama_persona_move::io_error;
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
    if (!r.ok || !r.rows(k, k_bytes) || !r.rows(v, v_bytes)) {
        return r.ok ? mismatch : (int32_t) llama_persona_move::io_error;
    }
    const uint32_t n_cells = r.get<uint32_t>();
    const uint32_t occupied = r.get<uint32_t>();
    if (!r.ok || occupied > n_cells || occupied > stream_cells) {
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
        if (r.ok && c.i >= stream_cells) {
            return mismatch;
        }
    }
    const bool has_state = r.get<uint8_t>() != 0;
    const llama_pos rs_pos = r.get<llama_pos>();
    const uint32_t rs_idx = r.get<uint32_t>();
    // her recurrent rows are whole when she held state and absent when she did not (persona_check_save)
    const std::vector<size_t> r_bytes = has_state ? g.r_bytes : std::vector<size_t>(g.r_bytes.size(), 0);
    const std::vector<size_t> s_bytes = has_state ? g.s_bytes : std::vector<size_t>(g.s_bytes.size(), 0);
    if (!r.ok || !r.rows(rs_r, r_bytes) || !r.rows(rs_s, s_bytes)) {
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
