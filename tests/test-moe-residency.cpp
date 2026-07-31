// test-moe-residency.cpp — TDD + VDD for the MoE expert-residency pager (continuum fork).
//
// WHY THIS EXISTS: the whole retention bug we hunted for a day (cross-token cache reuse == EXACTLY 0)
// is a KEYING + CACHE-LOGIC bug — pure, deterministic, and testable in microseconds instead of a
// 6-minute K3 load. WASTE (github.com/sqliteai/waste) proved the real hit rate is ~13% on K3, so
// exactly-zero is ALWAYS a bug, never physics. These tests pin that invariant.
//
// TWO LAYERS:
//   TDD  — unit tests on the REAL keying code from ggml-moe-residency.hpp: an expert's identity must
//          be STABLE across tokens and DISTINCT across (layer, tensor, expert). This is the exact
//          property whose absence caused reuse=0. Also pins canonical_name_key (M5's decoration fix).
//   VDD  — replay a CAPTURED expert-selection trace (the [RETAIN]/[NAME] probe output IS the
//          recording) through a reference LFRU cache and assert the hit-rate is >0 and in the sane
//          band WASTE measured. Catches reuse=0 end-to-end from real behavior, no model load.
//
// BUILD (Linux/CI — the Windows DirectReadFetcher is #ifdef'd out, so no windows.h needed):
//   c++ -std=c++17 -O2 -I ../ggml/src tests/test-moe-residency.cpp -o /tmp/tmoe && /tmp/tmoe
//   VDD replay of a real trace:  /tmp/tmoe path/to/trace.tsv
//
// TODO (next session / clean-architecture): give ResidencyCache a SlotAllocator seam (Ggml pinned vs
// malloc) so the PRODUCTION cache runs these same tests directly instead of the reference model below.
// The reference model here IS the spec the production cache must match.

#ifdef _WIN32
#include <windows.h>   // the header's DirectReadFetcher (under _WIN32) references Win32 types; pull them
#include <psapi.h>     // GetMappedFileNameW (declaration only — DirectReadFetcher is never instantiated)
#endif

// --- ggml stubs so the header parses standalone (ResidencyCache is never instantiated here, so these
//     are declared-not-defined: name lookup succeeds, nothing is ODR-used, nothing to link). ---
typedef struct ggml_backend_buffer      * ggml_backend_buffer_t;
typedef struct ggml_backend_buffer_type * ggml_backend_buffer_type_t;
extern "C" {
    ggml_backend_buffer_t ggml_backend_buft_alloc_buffer(ggml_backend_buffer_type_t, size_t);
    void *                ggml_backend_buffer_get_base(ggml_backend_buffer_t);
}

#include "ggml-moe-residency.hpp"
#include "ggml-moe-container.h"
#include "ggml-moe-container-fetcher.h"
#include "ggml-moe-packer.h"
#include "ggml-moe-wexp.h"
#include "ggml-moe-iq2-source.h"

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <unordered_map>
#include <list>
#include <fstream>
#include <sstream>
#include <filesystem>

// ------------------------------------------------------------------------------------------------
// tiny test harness
// ------------------------------------------------------------------------------------------------
static int g_fail = 0, g_pass = 0;
#define CHECK(cond, msg) do { \
    if (cond) { g_pass++; } \
    else { g_fail++; std::fprintf(stderr, "FAIL: %s  (%s:%d)\n", msg, __FILE__, __LINE__); } \
} while (0)

// A K3-style expert weight name for (layer, which ∈ {gate,up,down}).
static std::string wname(int layer, const char * which) {
    return "blk." + std::to_string(layer) + ".ffn_" + which + "_exps.weight";
}

// ================================================================================================
// TDD — the keying invariants (the reuse=0 root cause)
// ================================================================================================

// what this catches: the exact reuse=0 bug — an expert's identity MUST be identical when the same
// (layer, tensor, expert) is requested on a later token. If the key drifts token-to-token, the cache
// never hits and reuse collapses to 0. This is the regression for the whole day's investigation.
static void test_key_stable_across_tokens() {
    const std::string n = wname(3, "gate");
    // "token 1" and "token 2" build the id independently, exactly as prefetch and get do per forward pass.
    ggml_moe::ExpertId a = ggml_moe::expert_id_for(nullptr, n.c_str(), 42);
    ggml_moe::ExpertId b = ggml_moe::expert_id_for(nullptr, n.c_str(), 42);
    CHECK(a.key() == b.key(), "same (layer,tensor,expert) must yield the SAME key across tokens");
}

// what this catches: identity collisions — distinct experts must NOT share a slot (would serve wrong
// weights). Covers the three axes that must all participate in the key.
static void test_key_distinct_axes() {
    const ggml_moe::ExpertId gate12 = ggml_moe::expert_id_for(nullptr, wname(3, "gate").c_str(), 12);
    const ggml_moe::ExpertId gate13 = ggml_moe::expert_id_for(nullptr, wname(3, "gate").c_str(), 13);
    const ggml_moe::ExpertId up12   = ggml_moe::expert_id_for(nullptr, wname(3, "up").c_str(),   12);
    const ggml_moe::ExpertId gate12L4= ggml_moe::expert_id_for(nullptr, wname(4, "gate").c_str(), 12);
    CHECK(gate12.key() != gate13.key(), "different expert index => different key");
    CHECK(gate12.key() != up12.key(),   "different tensor (gate vs up) => different key");
    CHECK(gate12.key() != gate12L4.key(),"different layer => different key");
}

// what this catches: M5's decoration fix. At a GPU offload split ggml stamps the seam node
// "BACKEND#leaf#C" (ggml_format_name). Hashing the decorated string drifts per graph. The canonical
// key must reduce "CUDA0#blk.3.ffn_up_exps.weight#0" to the same value as the clean leaf name.
static void test_canonical_strips_decoration() {
    const uint64_t clean = ggml_moe::canonical_name_key("blk.3.ffn_up_exps.weight");
    const uint64_t dec0  = ggml_moe::canonical_name_key("CUDA0#blk.3.ffn_up_exps.weight#0");
    const uint64_t dec1  = ggml_moe::canonical_name_key("CUDA1#blk.3.ffn_up_exps.weight#7");
    CHECK(dec0 == clean, "decorated name must canonicalize to the clean leaf key");
    CHECK(dec1 == clean, "backend prefix + copy index must not affect the key");
    CHECK(ggml_moe::canonical_name_key("blk.3.ffn_up_exps.weight") ==
          ggml_moe::canonical_name_key("blk.3.ffn_up_exps.weight"), "clean name is its own canonical form");
    CHECK(clean != ggml_moe::canonical_name_key("blk.4.ffn_up_exps.weight"),
          "canonicalization must NOT collapse distinct leaves");
}

// ================================================================================================
// TDD — the streaming container format (#268): one aligned pread per expert
// ================================================================================================

static ggml_moe::ContainerHeader make_header(uint32_t layers, uint32_t experts, uint64_t stride) {
    ggml_moe::ContainerHeader h{};
    h.magic = ggml_moe::MOEC_MAGIC; h.version = ggml_moe::MOEC_VERSION;
    h.n_layers = layers; h.experts_per_layer = experts; h.align = ggml_moe::MOEC_ALIGN;
    h.record_stride = stride;
    h.data_offset = ggml_moe::moec_align_up(sizeof(ggml_moe::ContainerHeader));
    h.total_bytes = h.data_offset + (uint64_t) layers * experts * stride;
    return h;
}

// what this catches: every record offset must be ALIGNED (a valid O_DIRECT/NO_BUFFERING pread target)
// and DISTINCT+contiguous. A misaligned or overlapping offset silently corrupts direct I/O.
static void test_container_offsets_aligned_and_distinct() {
    const uint64_t stride = ggml_moe::moec_align_up(11u * 1024 * 1024);   // ~11MB expert -> aligned
    const auto h = make_header(4, 8, stride);
    CHECK(h.data_offset % ggml_moe::MOEC_ALIGN == 0, "record 0 must be aligned");
    CHECK(stride % ggml_moe::MOEC_ALIGN == 0, "record stride must be a multiple of alignment");
    uint64_t prev = 0; bool first = true, ok = true, aligned = true;
    for (uint32_t L = 0; L < h.n_layers; L++) for (uint32_t E = 0; E < h.experts_per_layer; E++) {
        const uint64_t off = ggml_moe::moec_record_offset(h, L, E);
        if (off % ggml_moe::MOEC_ALIGN != 0) aligned = false;
        if (!first && off != prev + stride) ok = false;                  // contiguous, layer-major
        prev = off; first = false;
    }
    CHECK(aligned, "every (layer,expert) record offset must be alignment-valid for direct I/O");
    CHECK(ok, "records must be contiguous & layer-major (a layer's experts stream near-sequentially)");
}

// what this catches: the container's (layer,expert) key must be stable and collision-free — this is
// the SAME single key WASTE uses, and the fix for the reuse=0 keying ambiguity once we source from a
// container (no tensor-name, no per-eval pointer).
static void test_container_key_stable_and_distinct() {
    CHECK(ggml_moe::moec_expert_key(3, 12) == ggml_moe::moec_expert_key(3, 12), "same (layer,expert) => same key");
    CHECK(ggml_moe::moec_expert_key(3, 12) != ggml_moe::moec_expert_key(3, 13), "different expert => different key");
    CHECK(ggml_moe::moec_expert_key(3, 12) != ggml_moe::moec_expert_key(4, 12), "different layer => different key");
}

// what this catches: the derived-budget policy — thresholds MUST scale off system capabilities, never
// a constant, and the cliff must be encoded. Below one-token WS the box can't retain (budget 0 => route
// to grid/smaller container); above it the budget scales with free RAM but always clears the cliff and
// never exceeds RAM. This is the "no magic GB number" invariant made testable.
static void test_derive_budget_from_system_capabilities() {
    const uint64_t GB = 1024ull * 1024 * 1024;
    ggml_moe::BudgetInputs in{};
    in.expert_bytes = 16ull * 1024 * 1024;   // 16 MiB/expert
    in.experts_per_token = 512;              // one-token WS = 16MiB * 512 = 8 GiB exactly
    in.ram_fraction = 0.6;
    in.headroom = 2.0;                       // need >= 16GB to clear the cliff
    const uint64_t ws = ggml_moe::moec_one_token_working_set(in);
    CHECK(ws == 8ull * GB, "one-token working set = expert_bytes * experts_per_token");

    in.free_ram_bytes = 12 * GB;                                   // below 2x WS (16GB) -> cliff
    CHECK(ggml_moe::moec_derive_budget(in) == 0, "below the cliff: cannot retain, budget 0");
    CHECK(!ggml_moe::moec_can_retain(in), "below the cliff: can_retain must be false (route to grid)");

    in.free_ram_bytes = 64 * GB;                                   // generous -> fraction rules
    CHECK(ggml_moe::moec_derive_budget(in) == (uint64_t)(0.6 * 64 * GB), "above cliff: budget = fraction * free RAM");
    CHECK(ggml_moe::moec_can_retain(in), "above cliff: can_retain true");

    in.free_ram_bytes = 20 * GB; in.ram_fraction = 0.1;           // enough RAM but stingy fraction
    CHECK(ggml_moe::moec_derive_budget(in) == 16ull * GB, "stingy policy bumped up to clear the cliff (2x WS)");

    in.free_ram_bytes = 17 * GB; in.ram_fraction = 0.99;          // fraction would exceed RAM
    CHECK(ggml_moe::moec_derive_budget(in) <= 17ull * GB, "budget must never exceed free RAM");
}

// what this catches: the container round-trips with REAL file I/O. Write synthetic experts at the
// format's offsets (the "packer" half), then read each (layer,expert) back at moec_record_offset (the
// "fetcher" half) and verify the payload. Proves both ends agree on the one-pread-per-expert layout
// before either is built for real. A stride/offset mismatch corrupts the read here, loudly.
static void test_container_roundtrip_real_io() {
    const uint64_t stride = ggml_moe::moec_align_up(8192);        // small aligned record for the test
    ggml_moe::ContainerHeader h = make_header(3, 4, stride);      // 3 layers x 4 experts
    char path[L_tmpnam]; std::tmpnam(path);
    std::string p = std::string(path) + ".moec";

    // marker byte for (layer, expert) — distinct per record, written at both ends of the payload.
    auto marker = [](uint32_t L, uint32_t E) -> unsigned char { return (unsigned char)(1 + L * 4 + E); };

    // PACK: header, pad to data_offset, then each record filled with its marker.
    {
        std::ofstream out(p, std::ios::binary);
        out.write(reinterpret_cast<const char *>(&h), sizeof(h));
        std::vector<char> pad(h.data_offset - sizeof(h), 0);
        out.write(pad.data(), pad.size());
        for (uint32_t L = 0; L < h.n_layers; L++)
            for (uint32_t E = 0; E < h.experts_per_layer; E++) {
                std::vector<char> rec(stride, (char) marker(L, E));
                out.write(rec.data(), rec.size());
            }
    }
    // FETCH: read each record back at its computed offset; verify the marker at first & last byte.
    bool ok = true, offset_ok = true;
    {
        std::ifstream in(p, std::ios::binary);
        std::vector<char> buf(stride);
        for (uint32_t L = 0; L < h.n_layers; L++)
            for (uint32_t E = 0; E < h.experts_per_layer; E++) {
                const uint64_t off = ggml_moe::moec_record_offset(h, L, E);
                if (off + stride > h.total_bytes) offset_ok = false;
                in.seekg((std::streamoff) off);
                in.read(buf.data(), (std::streamsize) stride);
                if ((unsigned char) buf.front() != marker(L, E) ||
                    (unsigned char) buf.back()  != marker(L, E)) ok = false;
            }
    }
    std::remove(p.c_str());
    CHECK(offset_ok, "every record must lie within total_bytes");
    CHECK(ok, "container round-trip: each expert reads back its own payload at moec_record_offset");
}

// what this catches: the ContainerFetcher reads the RIGHT expert. Pack a synthetic container, then
// fetch each (layer,expert) through the real ContainerFetcher with src = moec_record_offset, and verify
// the payload. This is the fetcher half of the packer<->fetcher contract, exercised through the actual
// adapter that plugs into the ResidencyCache. A stride/offset drift serves the wrong expert here.
static void test_container_fetcher_reads_right_expert() {
    const uint64_t stride = ggml_moe::moec_align_up(8192);
    ggml_moe::ContainerHeader h = make_header(3, 5, stride);
    char path[L_tmpnam]; std::tmpnam(path);
    std::string p = std::string(path) + ".moec";
    auto marker = [](uint32_t L, uint32_t E) -> unsigned char { return (unsigned char)(7 + L * 5 + E); };
    {   // pack
        std::ofstream out(p, std::ios::binary);
        out.write(reinterpret_cast<const char *>(&h), sizeof(h));
        std::vector<char> pad(h.data_offset - sizeof(h), 0); out.write(pad.data(), pad.size());
        for (uint32_t L = 0; L < h.n_layers; L++)
            for (uint32_t E = 0; E < h.experts_per_layer; E++) {
                std::vector<char> rec(stride, (char) marker(L, E)); out.write(rec.data(), rec.size());
            }
    }
    ggml_moe::ContainerFetcher cf(p.c_str());
    CHECK(cf.ok(), "container fetcher must open the packed file");
    bool ok = true;
    std::vector<char> buf(stride);
    for (uint32_t L = 0; L < h.n_layers; L++)
        for (uint32_t E = 0; E < h.experts_per_layer; E++) {
            const uint64_t off = ggml_moe::moec_record_offset(h, L, E);
            const bool got = cf.fetch(buf.data(), (const void *) (uintptr_t) off, (size_t) stride);
            if (!got || (unsigned char) buf.front() != marker(L, E) ||
                        (unsigned char) buf.back()  != marker(L, E)) ok = false;
        }
    std::remove(p.c_str());
    CHECK(ok, "ContainerFetcher must return each expert's own payload for its record offset");
}

// what this catches: THE packer<->fetcher contract, through BOTH real components. moec_pack writes a
// container from a synthetic interior; ContainerFetcher reads each record back by offset; the payload
// must match. This is the write-side proof — if the packer's ordering/stride ever drifts from the
// fetcher's offset math, this fails loudly. (The GGUF/RVQ interior plugs into ExpertSource unchanged.)
struct SyntheticSource : ggml_moe::ExpertSource {
    bool write_record(uint32_t L, uint32_t E, void * dst, size_t stride) override {
        std::memset(dst, (int) (unsigned char) (11 + L * 7 + E), stride);   // a marker per (layer,expert)
        return true;
    }
};
static void test_packer_fetcher_end_to_end() {
    const uint64_t stride = ggml_moe::moec_align_up(4096);
    const uint32_t layers = 4, experts = 6;
    char path[L_tmpnam]; std::tmpnam(path);
    std::string p = std::string(path) + ".moec";

    SyntheticSource src;
    const bool packed = ggml_moe::moec_pack(p.c_str(), layers, experts, stride, ggml_moe::MOEC_Q_RVQ3, src);
    CHECK(packed, "moec_pack must write the container successfully");

    // read the header back and confirm the packer authored it as the fetcher/offset-math expects.
    ggml_moe::ContainerHeader h{};
    { std::ifstream in(p, std::ios::binary); in.read(reinterpret_cast<char *>(&h), sizeof(h)); }
    CHECK(h.magic == ggml_moe::MOEC_MAGIC && h.n_layers == layers && h.experts_per_layer == experts &&
          h.record_stride == stride, "packed header must round-trip the shape");

    ggml_moe::ContainerFetcher cf(p.c_str());
    CHECK(cf.ok(), "fetcher opens the packed container");
    bool ok = true;
    std::vector<char> buf((size_t) stride);
    for (uint32_t L = 0; L < layers; L++)
        for (uint32_t E = 0; E < experts; E++) {
            const uint64_t off = ggml_moe::moec_record_offset(h, L, E);
            const unsigned char want = (unsigned char) (11 + L * 7 + E);
            if (!cf.fetch(buf.data(), (const void *) (uintptr_t) off, (size_t) stride) ||
                (unsigned char) buf.front() != want || (unsigned char) buf.back() != want) ok = false;
        }
    std::remove(p.c_str());
    CHECK(ok, "packer output and fetcher offsets agree byte-for-byte, every expert");
}

// what this catches: the WEXP record header must be BYTE-EXACT — it's a wire format shared with M5's
// Rust reader, so every field must land at its exact offset, little-endian, and the fmt trap must be
// 4/5 not 0/1. Assert the raw bytes at fixed positions, not just a round-trip (a symmetric writer+reader
// bug would pass a round-trip but still mismatch the Rust reader). This is the confirmed-contract pin.
static void test_wexp_header_byte_exact() {
    // the fmt trap, pinned as a constant so a regression to 0/1 fails to compile-intent here.
    CHECK(ggml_moe::WEXP_VQ3R == 4 && ggml_moe::WEXP_VQ2R == 5, "fmt is VQ3R=4 / VQ2R=5, never 0/1");

    ggml_moe::WexpRecord r;
    r.layer = 0x1234; r.expert_id = 0x5678; r.fmt = ggml_moe::WEXP_VQ2R; r.codebook_id = 0x9ABC;
    r.gate_off = 32; r.up_off = 0x00010000; r.down_off = 0x00020000; r.correction_off = 0x00030000;
    r.record_4k_blocks = 7;
    uint8_t b[ggml_moe::WEXP_HEADER_BYTES] = {0};
    ggml_moe::wexp_write_header(b, r);

    CHECK(std::memcmp(b, "WEXP", 4) == 0, "bytes 0..4 are ASCII 'WEXP'");
    CHECK(b[4] == 0x34 && b[5] == 0x12, "layer is little-endian u16 at offset 4");
    CHECK(b[6] == 0x78 && b[7] == 0x56, "expert_id is little-endian u16 at offset 6");
    CHECK(b[8] == 5, "fmt (VQ2R) is the raw byte 5 at offset 8");
    CHECK(b[9] == 0, "flags byte at offset 9 is 0");
    CHECK(b[10] == 0xBC && b[11] == 0x9A, "codebook_id LE u16 at offset 10");
    CHECK(ggml_moe::wexp_get_u32(b + 12) == 32u,          "gate_off u32 at offset 12");
    CHECK(ggml_moe::wexp_get_u32(b + 16) == 0x00010000u,  "up_off u32 at offset 16");
    CHECK(ggml_moe::wexp_get_u32(b + 20) == 0x00020000u,  "down_off u32 at offset 20");
    CHECK(ggml_moe::wexp_get_u32(b + 24) == 0x00030000u,  "correction_off u32 at offset 24");
    CHECK(ggml_moe::wexp_get_u32(b + 28) == 7u,           "record_4k_blocks u32 at offset 28");
    CHECK(ggml_moe::wexp_record_bytes(r) == 7ull * 4096,  "record bytes = record_4k_blocks * 4096");

    ggml_moe::WexpRecord back;
    CHECK(ggml_moe::wexp_read_header(b, back), "header parses (magic ok)");
    CHECK(back.layer == r.layer && back.expert_id == r.expert_id && back.fmt == r.fmt &&
          back.codebook_id == r.codebook_id && back.gate_off == r.gate_off && back.up_off == r.up_off &&
          back.down_off == r.down_off && back.correction_off == r.correction_off &&
          back.record_4k_blocks == r.record_4k_blocks, "all fields round-trip");
    uint8_t bad[ggml_moe::WEXP_HEADER_BYTES] = {0};
    ggml_moe::WexpRecord tmp;
    CHECK(!ggml_moe::wexp_read_header(bad, tmp), "bad magic must be rejected (read-side validation)");
}

// what this catches: a WEXP ExpertSource packs a real container whose records M5's reader accepts.
// The source writes a proper WEXP header + payload into each record; we pack, then read each record
// back and confirm the header is byte-exact for its (layer,expert). This is the GgufRvqSource shape
// minus the RVQ encoding — the record framing is proven; only the codebook bytes remain.
struct WexpSource : ggml_moe::ExpertSource {
    uint32_t blocks;
    explicit WexpSource(uint32_t b) : blocks(b) {}
    bool write_record(uint32_t L, uint32_t E, void * dst, size_t stride) override {
        if (stride < ggml_moe::WEXP_HEADER_BYTES) { return false; }
        ggml_moe::WexpRecord r;
        r.layer = (uint16_t) L; r.expert_id = (uint16_t) E; r.fmt = ggml_moe::WEXP_VQ3R;
        r.gate_off = (uint32_t) ggml_moe::WEXP_HEADER_BYTES;   // payload starts right after the header
        r.up_off = r.gate_off + 64; r.down_off = r.up_off + 64; r.correction_off = r.down_off + 64;
        r.record_4k_blocks = (uint32_t) (stride / 4096);
        wexp_write_header(static_cast<uint8_t *>(dst), r);
        return true;
    }
};
static void test_wexp_source_packs_valid_records() {
    const uint64_t stride = ggml_moe::moec_align_up(4096);
    const uint32_t layers = 3, experts = 4;
    char path[L_tmpnam]; std::tmpnam(path);
    std::string p = std::string(path) + ".moec";
    WexpSource src((uint32_t) (stride / 4096));
    CHECK(ggml_moe::moec_pack(p.c_str(), layers, experts, stride, ggml_moe::MOEC_Q_RVQ3, src),
          "pack a container of real WEXP records");
    ggml_moe::ContainerHeader h = make_header(layers, experts, stride);
    std::ifstream in(p, std::ios::binary);
    bool ok = true;
    std::vector<uint8_t> buf((size_t) stride);
    for (uint32_t L = 0; L < layers; L++)
        for (uint32_t E = 0; E < experts; E++) {
            in.seekg((std::streamoff) ggml_moe::moec_record_offset(h, L, E));
            in.read(reinterpret_cast<char *>(buf.data()), (std::streamsize) stride);
            ggml_moe::WexpRecord r;
            if (!ggml_moe::wexp_read_header(buf.data(), r) || r.layer != L || r.expert_id != E ||
                r.fmt != ggml_moe::WEXP_VQ3R) ok = false;
        }
    in.close(); std::remove(p.c_str());
    CHECK(ok, "each packed record is a valid WEXP header for its (layer,expert) — reader-ready");
}

// what this catches: the DIRECTORY emitter (M5's reader format) — manifest.json v1 with the exact
// fields, per-layer experts-L{n}.bin banks whose size == record_bytes*experts_per_layer, and records
// laid out so offset = expert_id*record_bytes (one pread). This is the envelope her Rust reader opens;
// a geometry drift (wrong bank size, unsorted records, bad offset) is what her open()/fetch rejects.
static void test_pack_dir_matches_reader_geometry() {
    const uint64_t record_bytes = 4096;                        // must be a 4KiB multiple
    const uint32_t layers = 3, experts = 5, top_k = 16, activated = top_k * layers;   // TOTAL, not per-layer
    namespace fs = std::filesystem;
    const fs::path dir = fs::temp_directory_path() / "moec_pack_test";
    fs::remove_all(dir); fs::create_directories(dir);

    WexpSource src((uint32_t) (record_bytes / 4096));
    const bool ok = ggml_moe::moec_pack_dir(dir.string().c_str(), "kimi-k3", ggml_moe::WEXP_VQ3R,
                                            layers, experts, record_bytes, activated, top_k, src);
    CHECK(ok, "moec_pack_dir writes the directory");
    CHECK(fs::exists(dir / "manifest.json"), "manifest.json exists");

    // manifest carries the pinned v1 fields
    { std::ifstream m(dir / "manifest.json"); std::stringstream ss; ss << m.rdbuf(); const std::string j = ss.str();
      CHECK(j.find("\"version\": 1") != std::string::npos,          "manifest version 1");
      CHECK(j.find("\"model\": \"kimi-k3\"") != std::string::npos,  "manifest model");
      CHECK(j.find("\"fmt\": 4") != std::string::npos,              "manifest fmt = VQ3R(4)");
      CHECK(j.find("\"record_bytes\": 4096") != std::string::npos,  "manifest record_bytes");
      CHECK(j.find("\"n_layers\": 3") != std::string::npos,         "manifest n_layers");
      CHECK(j.find("\"experts_per_layer\": 5") != std::string::npos,"manifest experts_per_layer");
      CHECK(j.find("\"activated_per_token\": 48") != std::string::npos, "manifest activated_per_token = TOTAL (top_k*layers)");
      CHECK(j.find("\"top_k_per_layer\": 16") != std::string::npos,     "manifest top_k_per_layer (audit field)");
    }
    // each bank: exact size, and offset = expert_id*record_bytes reads the right WEXP record
    bool geom_ok = true, offset_ok = true;
    for (uint32_t L = 0; L < layers; L++) {
        const fs::path bank = dir / ("experts-L" + std::to_string(L) + ".bin");
        if (!fs::exists(bank) || fs::file_size(bank) != record_bytes * experts) geom_ok = false;
        std::ifstream in(bank, std::ios::binary);
        std::vector<uint8_t> buf((size_t) record_bytes);
        for (uint32_t E = 0; E < experts; E++) {
            in.seekg((std::streamoff) (E * record_bytes));     // her offset math: expert_id * record_bytes
            in.read(reinterpret_cast<char *>(buf.data()), (std::streamsize) record_bytes);
            ggml_moe::WexpRecord r;
            if (!ggml_moe::wexp_read_header(buf.data(), r) || r.layer != L || r.expert_id != E) offset_ok = false;
        }
    }
    fs::remove_all(dir);
    CHECK(geom_ok, "each bank file size == record_bytes*experts_per_layer (no truncated tail)");
    CHECK(offset_ok, "offset=expert_id*record_bytes reads the correct WEXP record in every bank");
}

// what this catches: THE activated_per_token trap (M5, reader 9c85a4f88). It must be the TOTAL across
// all MoE layers, not per-layer top-k. Feeding per-layer understates the working set ~n_layers-fold, so
// the budget policy green-lights a cache that can't retain -> silently recreates reuse=0, the exact
// failure this lane exists to kill. Pin: (a) the working set scales with the TOTAL; (b) the packer
// rejects a manifest where activated_per_token != top_k_per_layer * n_layers.
static void test_activated_per_token_is_total_not_per_layer() {
    const uint64_t GB = 1024ull * 1024 * 1024;
    const uint32_t n_layers = 61, top_k = 8;                    // K3-ish
    ggml_moe::BudgetInputs in{};
    in.expert_bytes = 12ull * 1024 * 1024;                     // ~12MB VQ3 record

    in.experts_per_token = top_k;                              // WRONG: per-layer
    const uint64_t ws_per_layer = ggml_moe::moec_one_token_working_set(in);
    in.experts_per_token = top_k * n_layers;                   // RIGHT: total
    const uint64_t ws_total = ggml_moe::moec_one_token_working_set(in);
    CHECK(ws_total == ws_per_layer * n_layers, "total working set is n_layers x the per-layer misreading");
    CHECK(ws_total > 5ull * GB && ws_per_layer < 128ull * 1024 * 1024,
          "per-layer understates the cliff ~61x (GBs vs ~100MB) — the silent reuse=0 recreator");

    // the packer refuses a manifest whose activated_per_token isn't the total (top_k*layers).
    namespace fs = std::filesystem;
    const fs::path dir = fs::temp_directory_path() / "moec_trap_test";
    fs::remove_all(dir); fs::create_directories(dir);
    WexpSource src(1);
    const bool rejected = !ggml_moe::moec_pack_dir(dir.string().c_str(), "k3", ggml_moe::WEXP_VQ3R,
        n_layers, /*experts*/ 8, /*record_bytes*/ 4096, /*activated=per-layer WRONG*/ top_k, top_k, src);
    const bool accepted = ggml_moe::moec_pack_dir(dir.string().c_str(), "k3", ggml_moe::WEXP_VQ3R,
        n_layers, 8, 4096, /*activated=total RIGHT*/ top_k * n_layers, top_k, src);
    fs::remove_all(dir);
    CHECK(rejected, "packer REJECTS activated_per_token != top_k*n_layers (the per-layer trap)");
    CHECK(accepted, "packer accepts the correct total");
}

// what this catches: rung-1 IQ2 packing — Iq2ExpertSource byte-copies gate/up/down into a record with
// the WEXP ident header + fmt threaded (not guessed), correction_off=0. Pack via moec_pack_dir, read
// each record back through ContainerFetcher, and verify the header identity AND that gate/up/down bytes
// land at their declared offsets. This is the fast path's framing proven end-to-end; only the GGUF byte
// source (real IQ2 tensors) plugs into the provider next. fmt is a placeholder here (M5 owns the value).
static void test_iq2_source_packs_and_reads_back() {
    const uint8_t IQ2_FMT_PLACEHOLDER = 6;                 // M5 confirms the real WEXP-enum value
    const size_t gate_len = 300, up_len = 300, down_len = 300;   // small IQ2-ish blocks
    // provider fills each matrix with a distinct per-(layer,expert,which) marker byte
    auto provider = [&](uint32_t L, uint32_t E, ggml_moe::Iq2Bytes & b) -> bool {
        static thread_local std::vector<uint8_t> g, u, d;
        g.assign(gate_len, (uint8_t)(0x10 + L * 3 + E));
        u.assign(up_len,   (uint8_t)(0x40 + L * 3 + E));
        d.assign(down_len, (uint8_t)(0x70 + L * 3 + E));
        b = { g.data(), g.size(), u.data(), u.size(), d.data(), d.size() };
        return true;
    };
    ggml_moe::Iq2ExpertSource src(IQ2_FMT_PLACEHOLDER, provider);

    const uint64_t record_bytes = 4096;                    // header(32)+900 payload fits, 4KiB-aligned
    const uint32_t layers = 3, experts = 4, top_k = 8;
    namespace fs = std::filesystem;
    const fs::path dir = fs::temp_directory_path() / "moec_iq2_test";
    fs::remove_all(dir); fs::create_directories(dir);
    CHECK(ggml_moe::moec_pack_dir(dir.string().c_str(), "kimi-k3-iq2", ggml_moe::MOEC_Q_IQ2,
          layers, experts, record_bytes, top_k * layers, top_k, src), "pack IQ2 container");

    ggml_moe::ContainerHeader h = make_header(layers, experts, record_bytes);
    bool ok = true;
    for (uint32_t L = 0; L < layers && ok; L++) {
        std::ifstream in(dir / ("experts-L" + std::to_string(L) + ".bin"), std::ios::binary);
        std::vector<uint8_t> buf((size_t) record_bytes);
        for (uint32_t E = 0; E < experts; E++) {
            in.seekg((std::streamoff) (E * record_bytes));
            in.read(reinterpret_cast<char *>(buf.data()), (std::streamsize) record_bytes);
            ggml_moe::WexpRecord r;
            if (!ggml_moe::wexp_read_header(buf.data(), r) || r.layer != L || r.expert_id != E ||
                r.fmt != IQ2_FMT_PLACEHOLDER || r.correction_off == 0) { ok = false; break; }
            if (buf[r.gate_off] != (uint8_t)(0x10 + L * 3 + E) ||    // gate/up/down at declared offsets
                buf[r.up_off]   != (uint8_t)(0x40 + L * 3 + E) ||
                buf[r.down_off] != (uint8_t)(0x70 + L * 3 + E)) { ok = false; break; }
        }
    }
    fs::remove_all(dir);
    CHECK(ok, "IQ2 records: ident + fmt + gate/up/down bytes read back correct at declared offsets");
}

// what this catches: the GGUF expert-slice math — the crux of extracting one expert's IQ2 bytes from a
// blk.N.ffn_*_exps.weight tensor. Slices must be equal-size, contiguous, non-overlapping, and cover the
// whole tensor (or the byte-copy grabs a neighbour's weights — a silent wrong-expert bug). Also rejects
// a tensor size that isn't divisible by n_expert (a layout assumption violated).
static void test_gguf_expert_slice_math() {
    const uint32_t n_expert = 256;
    const uint64_t total = (uint64_t) n_expert * 2600;     // 2600 bytes/expert, evenly divisible
    ggml_moe::ExpertSlice s0, s1, slast, bad;
    CHECK(ggml_moe::moec_expert_slice(total, n_expert, 0, s0) && s0.offset == 0 && s0.len == 2600,
          "expert 0 at offset 0");
    CHECK(ggml_moe::moec_expert_slice(total, n_expert, 1, s1) && s1.offset == 2600,
          "expert 1 contiguous right after expert 0");
    CHECK(ggml_moe::moec_expert_slice(total, n_expert, n_expert - 1, slast) &&
          slast.offset + slast.len == total, "last expert ends exactly at tensor end (full coverage)");
    CHECK(!ggml_moe::moec_expert_slice(total, n_expert, n_expert, bad), "out-of-range expert rejected");
    CHECK(!ggml_moe::moec_expert_slice(total + 1, n_expert, 0, bad),
          "non-divisible tensor size rejected (layout assumption guard)");
    // contiguity + non-overlap across the whole tensor
    bool tiled = true; uint64_t expect = 0;
    for (uint32_t E = 0; E < n_expert; E++) {
        ggml_moe::ExpertSlice s;
        if (!ggml_moe::moec_expert_slice(total, n_expert, E, s) || s.offset != expect) { tiled = false; break; }
        expect += s.len;
    }
    CHECK(tiled && expect == total, "experts tile the tensor exactly — contiguous, no gaps, no overlap");
}

// ================================================================================================
// VDD — reference LFRU cache + replay of a captured expert-selection trace
// ================================================================================================

// Reference cache — the SPEC the production ResidencyCache must match. Dependency-free, so it runs
// anywhere and doubles as the oracle for the replay. Bounded, LFRU eviction (WASTE's policy: evict
// least-frequently-then-least-recently used), persistent across "tokens" by construction.
struct RefCache {
    size_t capacity;
    uint64_t clock = 0, hits = 0, misses = 0;
    struct Slot { uint64_t key; uint64_t last; uint64_t freq; };
    std::vector<Slot> slots;
    std::unordered_map<uint64_t, size_t> key2slot;

    explicit RefCache(size_t cap) : capacity(cap) {}

    // returns true on HIT (already resident from a prior token), false on MISS (admitted now).
    bool get(uint64_t key) {
        ++clock;
        auto it = key2slot.find(key);
        if (it != key2slot.end()) { hits++; slots[it->second].last = clock; slots[it->second].freq++; return true; }
        misses++;
        size_t slot;
        if (slots.size() < capacity) { slot = slots.size(); slots.push_back({key, clock, 1}); }
        else {                                            // LFRU victim: min freq, ties -> oldest last
            slot = 0;
            for (size_t i = 1; i < slots.size(); i++) {
                if (slots[i].freq <  slots[slot].freq ||
                   (slots[i].freq == slots[slot].freq && slots[i].last < slots[slot].last)) slot = i;
            }
            key2slot.erase(slots[slot].key);
            slots[slot] = {key, clock, 1};
        }
        key2slot[key] = slot;
        return false;
    }
    double hit_rate() const { const uint64_t t = hits + misses; return t ? (double) hits / t : 0.0; }
};

// what this catches: reuse=0 as a LOGIC bug. Synthesize a routing trace with KNOWN locality — a small
// pool of "hot" experts reused across tokens plus cold churn — and assert the cache actually reuses
// them. A cache that keys unstably (the bug) scores 0 here; a correct one scores well above it.
static void test_refcache_reuse_with_locality() {
    RefCache c(64);
    const int LAYERS = 8, HOT = 4, COLD_PER = 4, TOKENS = 40;
    int cold = 1000;
    for (int t = 0; t < TOKENS; t++) {
        for (int L = 0; L < LAYERS; L++) {
            for (int h = 0; h < HOT; h++)                       // hot experts recur every token
                c.get(ggml_moe::expert_id_for(nullptr, wname(L, "gate").c_str(), h).key());
            for (int k = 0; k < COLD_PER; k++)                  // cold experts never repeat
                c.get(ggml_moe::expert_id_for(nullptr, wname(L, "gate").c_str(), cold++).key());
        }
    }
    CHECK(c.hits > 0, "REGRESSION: cache with real locality must reuse experts (reuse=0 is the bug)");
    // hot fraction is HOT/(HOT+COLD_PER)=50% of accesses, reused every token after the first ->
    // steady-state hit-rate approaches that fraction; assert it clears a floor well above zero.
    CHECK(c.hit_rate() > 0.25, "hit-rate must reflect the injected locality, not collapse to ~0");
}

// what this catches: LFRU's specific value ABOVE the cliff — when the working set fits but there is
// churn, LFRU must keep the FREQUENTLY-used hot experts resident and evict the cold scan. Cache is
// modestly larger than one token's working set (16 > 4 hot + 8 cold = 12), so it's above the cliff;
// the invariant is that the hot subset keeps hitting despite continuous cold eviction pressure.
// (Below the cliff no policy retains anything — that case is pinned by the cliff test.)
static void test_refcache_lfru_retains_hot_above_cliff() {
    RefCache c(16);                                            // above one-token WS (12), with headroom
    const int TOKENS = 50;
    int cold = 5000;
    for (int t = 0; t < TOKENS; t++) {
        for (int h = 0; h < 4; h++)                             // 4 hot, reused every token -> freq climbs
            c.get(ggml_moe::expert_id_for(nullptr, wname(0, "gate").c_str(), h).key());
        for (int k = 0; k < 8; k++)                             // 8 cold/token churn -> eviction pressure
            c.get(ggml_moe::expert_id_for(nullptr, wname(0, "gate").c_str(), cold++).key());
    }
    // hot are 4 of 12 accesses/token; once resident they hit every token => steady-state ~33%.
    CHECK(c.hit_rate() > 0.20, "LFRU must keep the frequently-used hot set resident above the cliff");
}

// what this catches: WASTE's Gate-5 CLIFF — the likely REAL cause of reuse=0. Reuse stays ~0 while the
// cache is smaller than ONE TOKEN's working set (each token fully churns the cache, so the hot subset
// is evicted before the next token can reuse it — LFRU freq can't even build), then climbs sharply
// once the cache exceeds the working set. This is WHY the container re-pack (#268), which shrinks the
// per-token working set ~48x, is what makes retention viable at all — not (only) a keying fix.
// Regression: below one-token WS => ~0; above => reuse the hot subset; and the step between is sharp.
static void test_reuse_cliff_around_one_token_working_set() {
    const int WS = 100, HOT = 30, TOKENS = 40;          // WS distinct/token; HOT recur next token
    auto run = [&](size_t cap) {
        RefCache c(cap);
        int cold = 0;
        for (int t = 0; t < TOKENS; t++) {
            for (int h = 0; h < HOT; h++)                              // hot subset, reused every token
                c.get(ggml_moe::expert_id_for(nullptr, wname(0, "gate").c_str(), h).key());
            for (int k = 0; k < WS - HOT; k++)                         // fresh cold, one token's churn
                c.get(ggml_moe::expert_id_for(nullptr, wname(0, "gate").c_str(), 1000 + cold++).key());
        }
        return c.hit_rate();
    };
    const double below = run(WS / 2);                   // cache holds HALF a token -> below the cliff
    const double above = run(WS * 2);                   // cache holds TWO tokens  -> above the cliff
    CHECK(below < 0.05, "CLIFF: cache below one-token working set reuses ~nothing (the reuse=0 regime)");
    CHECK(above > 0.20, "CLIFF: cache above one-token working set reuses the hot subset");
    CHECK(above - below > 0.15, "reuse must CLIMB sharply across the working-set boundary (the cliff)");
}

// VDD replay: feed a REAL captured trace through the reference cache and assert the hit-rate lands in
// the band WASTE measured on K3 (~13%) — proving the pager reuses experts exactly as the reference
// implementation does. Trace format (one selection per line, TSV): <layer> <which> <expert_idx>.
// Produce it from a live run's [RETAIN]/[NAME] probe, or any router log.
static int replay_trace(const char * path) {
    std::ifstream in(path);
    if (!in) { std::fprintf(stderr, "VDD: cannot open trace %s\n", path); return 2; }
    RefCache c(2048);                                           // ~one K3 token's working set per pool
    std::string line;
    size_t n = 0;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        int layer, idx; std::string which;
        if (!(ss >> layer >> which >> idx)) continue;
        c.get(ggml_moe::expert_id_for(nullptr, wname(layer, which.c_str()).c_str(), idx).key());
        n++;
    }
    const double hr = c.hit_rate();
    std::printf("VDD replay: %zu selections, hit_rate=%.1f%% (hits=%llu miss=%llu)\n",
                n, 100.0 * hr, (unsigned long long) c.hits, (unsigned long long) c.misses);
    // WASTE's measured band on K3 is ~13%; exactly-0 is always a bug. Assert a sane nonzero floor.
    CHECK(c.hits > 0, "VDD REGRESSION: real trace produced ZERO reuse — the exact bug we chased");
    CHECK(hr > 0.03 && hr < 0.60, "VDD: replayed hit-rate should sit in the plausible K3 band (~13%)");
    return g_fail ? 1 : 0;
}

int main(int argc, char ** argv) {
    // VDD mode: replay a captured trace if a path is given.
    if (argc > 1) { int rc = replay_trace(argv[1]); std::printf("%d passed, %d failed\n", g_pass, g_fail); return rc; }

    // TDD suite
    test_key_stable_across_tokens();
    test_key_distinct_axes();
    test_canonical_strips_decoration();
    test_container_offsets_aligned_and_distinct();
    test_container_key_stable_and_distinct();
    test_derive_budget_from_system_capabilities();
    test_container_roundtrip_real_io();
    test_container_fetcher_reads_right_expert();
    test_packer_fetcher_end_to_end();
    test_wexp_header_byte_exact();
    test_wexp_source_packs_valid_records();
    test_pack_dir_matches_reader_geometry();
    test_iq2_source_packs_and_reads_back();
    test_gguf_expert_slice_math();
    test_activated_per_token_is_total_not_per_layer();
    test_refcache_reuse_with_locality();
    test_refcache_lfru_retains_hot_above_cliff();
    test_reuse_cliff_around_one_token_working_set();

    std::printf("%d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
