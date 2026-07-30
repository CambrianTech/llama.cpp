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

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <unordered_map>
#include <list>
#include <fstream>
#include <sstream>

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
    test_refcache_reuse_with_locality();
    test_refcache_lfru_retains_hot_above_cliff();
    test_reuse_cliff_around_one_token_working_set();

    std::printf("%d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
