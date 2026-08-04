#pragma once
// ggml-moe-container-fetcher.h — ExpertFetcher that reads from a packed streaming container (#268).
//
// Drops into the ResidencyCache's existing fetcher seam. Unlike the raw-GGUF DirectReadFetcher (which
// scrapes a file offset out of a scattered mmap address and needs a sector-aligned BOUNCE buffer),
// the container guarantees every record is 16KiB-aligned and contiguous — so here `src` simply carries
// the record's FILE OFFSET (moec_record_offset), each fetch is ONE positional read of the record, and
// the production direct-I/O path needs NO bounce at all. That alignment-by-construction is the whole
// point of #268: it turns scattered ~200 MB/s random reads into aligned ~GB/s streaming.
//
// This portable form uses 64-bit buffered positional reads (correct on huge containers). The Windows
// direct-I/O variant (FILE_FLAG_NO_BUFFERING + the sliding-window/handle-pool from DirectReadFetcher)
// is the drop-in for the serving path; the offset contract is identical, so it swaps without touching
// the cache — the same adapter discipline the module was built for.

#include "ggml-moe-residency.hpp"   // ExpertFetcher, FetchItem
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <functional>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#if defined(_WIN32)
#  define MOEC_FSEEK64(f, off) (_fseeki64((f), (long long)(off), SEEK_SET))
#else
#  define MOEC_FSEEK64(f, off) (fseeko((f), (off_t)(off), SEEK_SET))
#endif

namespace ggml_moe {

// Reads packed experts by file offset. `src` (the ResidencyCache's opaque "where") is reinterpreted as
// the record's byte offset in the container — the cache never inspects it, so no interface change.
class ContainerFetcher final : public ExpertFetcher {
    FILE * f_ = nullptr;
public:
    explicit ContainerFetcher(const char * path) { f_ = std::fopen(path, "rb"); }
    ~ContainerFetcher() override { if (f_) std::fclose(f_); }

    bool         ok()   const { return f_ != nullptr; }
    const char * name() const override { return "container"; }

    // One positional read of `bytes` at the record offset encoded in `src`. false => caller falls back.
    bool fetch(void * dst, const void * src, size_t bytes) override {
        if (!f_) { return false; }
        const uint64_t off = (uint64_t) (uintptr_t) src;   // src encodes moec_record_offset(h, layer, expert)
        if (MOEC_FSEEK64(f_, off) != 0) { return false; }
        return std::fread(dst, 1, bytes, f_) == bytes;
    }
    // Batch: sequential positional reads. The serving direct-I/O variant overlaps these (queue depth);
    // the offset contract is unchanged, which is why that swap never touches the ResidencyCache.
    void fetch_many(const FetchItem * items, size_t n) override {
        for (size_t i = 0; i < n; i++) {
            if (!fetch(items[i].dst, items[i].src, items[i].bytes)) {
                std::memset(items[i].dst, 0, items[i].bytes);   // loud-by-absence: a zero record fails validation downstream
            }
        }
    }
};

// V1 MULTI-FILE container fetcher: the packer (moec_pack_dir) writes a DIRECTORY of one bank per MoE
// layer — `experts-L{n}.bin` — with expert e of layer L at byte offset `e * record_bytes` (contiguous,
// 16KiB-aligned by construction). Unlike single-file ContainerFetcher (one FILE, `src` = a flat offset)
// a multi-file container needs the LAYER too, so the caller packs BOTH into the opaque `src`:
//
//     src = ((uint64_t) layer << SHIFT) | byte_offset      // byte_offset = expert * record_bytes
//
// SHIFT = 48: layer < 2^16 (K3 has 92), offset < 2^48 (256 TiB) — no real container overflows it. The
// ResidencyCache still passes `src` opaquely (NO interface change); only the SERVING caller in
// ggml-backend computes this packed value instead of an mmap address when a container fetcher is active
// (that caller-side offset is the serving-graph seam — M5's half). Banks open LAZILY and are cached per
// layer (a node holding only some layers' shards never stats the rest — the grid shard unit), exactly
// like TieredContainerFetcher minus the tier dimension.
class DirContainerFetcher final : public ExpertFetcher {
public:
    static const int      LAYER_SHIFT = 48;
    static const uint64_t OFF_MASK    = (uint64_t(1) << LAYER_SHIFT) - 1;
    // Pack a (layer, byte_offset) pair into the opaque `src` the serving caller hands the cache.
    static uint64_t pack_src(uint32_t layer, uint64_t byte_offset) {
        return (uint64_t(layer) << LAYER_SHIFT) | (byte_offset & OFF_MASK);
    }

private:
    std::string dir_;
    std::mutex  mtx_;
    std::unordered_map<uint32_t, FILE *> banks_;   // layer -> lazily-opened experts-L{layer}.bin (nullptr cached)

    FILE * bank(uint32_t layer) {
        std::lock_guard<std::mutex> lk(mtx_);
        auto it = banks_.find(layer);
        if (it != banks_.end()) { return it->second; }
        const std::string bp = dir_ + "/experts-L" + std::to_string(layer) + ".bin";
        FILE * f = std::fopen(bp.c_str(), "rb");    // nullptr cached too, so a missing shard isn't re-stat'd
        banks_.emplace(layer, f);
        return f;
    }

public:
    explicit DirContainerFetcher(const char * dir) : dir_(dir) {}
    ~DirContainerFetcher() override {
        for (auto & kv : banks_) { if (kv.second) { std::fclose(kv.second); } }
    }
    const char * name() const override { return "container-dir"; }

    // Decode `src` -> (layer, offset); one positional read of the record from that layer's bank.
    // false => bank missing or short read (the cache then streams the fallback source).
    bool fetch(void * dst, const void * src, size_t bytes) override {
        const uint64_t s   = (uint64_t) (uintptr_t) src;
        const uint32_t L   = (uint32_t) (s >> LAYER_SHIFT);
        const uint64_t off = s & OFF_MASK;
        FILE * f = bank(L);
        if (!f) { return false; }
        if (MOEC_FSEEK64(f, off) != 0) { return false; }
        return std::fread(dst, 1, bytes, f) == bytes;
    }
    // Batch: sequential positional reads (the serving direct-I/O variant overlaps these — same offset
    // contract, swaps in without touching the cache). A failed record zeroes so it fails validation LOUD.
    void fetch_many(const FetchItem * items, size_t n) override {
        for (size_t i = 0; i < n; i++) {
            if (!fetch(items[i].dst, items[i].src, items[i].bytes)) {
                std::memset(items[i].dst, 0, items[i].bytes);
            }
        }
    }
};

// DEVICE-RESIDENT expert upload adapter — the LiveUploadPager mechanism (#23), the H2D KILL.
// A ResidencyCache built on a DEVICE (VRAM) buffer type turns its slots into VRAM. This fetcher fills a
// MISS: it reads the record HOST-side via an inner ExpertFetcher (a DirContainerFetcher), then does ONE
// host->device copy into the VRAM slot. The win is what does NOT happen — on a cache HIT the ResidencyCache
// returns the resident device slot with NO fetch, so the serving loop's per-op H2D (ggml-backend.cpp:1857,
// ~11GB/token over PCIe) is SKIPPED for every recency-resident expert. A miss pays one PCIe copy; a hot
// token pays ~none. That is the 0.32-tok/s (H2D-bound) -> 1-10-tok/s lever: UMA gets residency for free, a
// discrete GPU only by keeping experts on the card. The bandit's pin_list (plan file) drives WHICH experts
// stay resident (M5's policy via ResidencyCache::pinned_); THIS is the pure upload mechanism.
//
// Backend-neutral by construction: the host->device copy is INJECTED — the CUDA-aware serving caller
// supplies cudaMemcpyAsync / a ggml_backend device copy — so this header keeps ZERO CUDA dependency, the
// same adapter discipline as every other fetcher. `dst` in fetch() is a DEVICE pointer (a VRAM slot).
class DeviceUploadFetcher final : public ExpertFetcher {
    ExpertFetcher & inner_;                                                    // reads the record HOST-side
    std::function<bool(void * dst_dev, const void * src_host, size_t)> h2d_;   // ONE host->device copy, injected
    std::vector<uint8_t> bounce_;                                             // host staging, grows to the largest record
    std::mutex mtx_;

public:
    DeviceUploadFetcher(ExpertFetcher & inner,
                        std::function<bool(void *, const void *, size_t)> h2d)
        : inner_(inner), h2d_(std::move(h2d)) {}
    const char * name() const override { return "device-upload"; }

    // MISS path: read the record host-side via `inner_`, then ONE H2D into the VRAM slot `dst`. false =>
    // the inner read or the device copy failed (cache leaves the slot unfilled; container mode fails loud
    // downstream). Serialized by `mtx_` so the single `bounce_` is reused, never reallocated per fetch —
    // the same churn-avoidance the DirectReadFetcher learned the hard way. (Overlapped multi-buffer H2D
    // is the v2 optimization; correctness first.)
    bool fetch(void * dst, const void * src, size_t bytes) override {
        std::lock_guard<std::mutex> lk(mtx_);
        if (bounce_.size() < bytes) { bounce_.resize(bytes); }
        if (!inner_.fetch(bounce_.data(), src, bytes)) { return false; }
        return h2d_ && h2d_(dst, bounce_.data(), bytes);
    }
    void fetch_many(const FetchItem * items, size_t n) override {
        for (size_t i = 0; i < n; i++) { fetch(items[i].dst, items[i].src, items[i].bytes); }
    }
};

// The container's record stride, read ONCE from `<dir>/manifest.json` (the moec_pack_dir
// manifest — tolerant key scan, same no-JSON-dep style as PagerPlan). This is the number the
// SERVING CALLER needs to compute byte_offset = expert * record_bytes for pack_src(); the
// fetcher itself never needs it (offsets arrive pre-computed). 0 => no/unreadable manifest —
// the caller must then treat the container as ABSENT (fall back to the mmap path wholesale),
// never guess a stride: a wrong stride reads the wrong expert's bytes silently.
//
// v2 (TIERED) manifests: this scan finds the FIRST "record_bytes", which the packer emits as
// tiers[0]'s — and tier 0's stride IS the one the caller must pack with (TieredDirFetcher derives
// expert_id = offset / stride0, then reads the selected tier at its own record_bytes). That makes
// v2 work here without a special case, but it is LOAD-BEARING, not luck: if the manifest ever
// stops emitting tier 0 first, this must gain an explicit v2 branch or every offset silently
// decodes to the wrong expert.
static inline uint64_t dir_container_record_bytes(const char * dir) {
    if (!dir) { return 0; }
    const std::string mp = std::string(dir) + "/manifest.json";
    FILE * f = std::fopen(mp.c_str(), "rb");
    if (!f) { return 0; }
    char buf[4096];
    const size_t n = std::fread(buf, 1, sizeof(buf) - 1, f);
    std::fclose(f);
    buf[n] = 0;
    const char * k = std::strstr(buf, "\"record_bytes\"");
    if (!k) { return 0; }
    k = std::strchr(k, ':');
    if (!k) { return 0; }
    unsigned long long v = 0;
    if (std::sscanf(k + 1, " %llu", &v) != 1) { return 0; }
    // The packer refuses non-4KiB-multiple records; mirror that here so a hand-edited
    // manifest can't smuggle an unaligned stride past the aligned-read contract.
    if (v == 0 || v % 4096 != 0) { return 0; }
    return (uint64_t) v;
}

// TIERED read primitive: reads an expert from a specific PRECISION TIER of a tiered container (manifest
// v2, per-(layer,tier) banks). The pager selects the tier per expert (all-star … cruft, rate-distortion
// under budget); this reads that tier's bank at offset = expert_id * tier_record_bytes (one positional
// read, alignment-by-construction). Bank handles open LAZILY and are cached per (layer,tier) — a node
// holding only some (layer,tier) shards never stats the rest (the grid shard unit). This is the read
// side that pairs with moec_pack_dir_tiered; the serving direct-I/O variant swaps in the same way as
// ContainerFetcher (identical offset contract). record_bytes comes from the manifest tier row.
// [TIERED WIRE] One tier row as the serving side needs it. Mirrors TierSpec in the packer; kept
// here so the READ path has no dependency on the packer header.
struct DirTier {
    uint32_t id;
    uint32_t quant;
    uint64_t record_bytes;
};

// Parse the tiers[] array out of a v2 manifest. Returns how many were read (0 => not a v2
// container, or unreadable — caller falls back to the v1 single-tier path). Same tolerant
// no-JSON-dep scan as dir_container_record_bytes: a hand-edited manifest cannot smuggle an
// unaligned stride past the aligned-read contract, because each row is validated.
static inline uint32_t dir_container_tiers(const char * dir, DirTier * out, uint32_t max_out) {
    if (!dir || !out || max_out == 0) { return 0; }
    const std::string mp = std::string(dir) + "/manifest.json";
    FILE * f = std::fopen(mp.c_str(), "rb");
    if (!f) { return 0; }
    std::vector<char> buf(1 << 16);
    const size_t n = std::fread(buf.data(), 1, buf.size() - 1, f);
    std::fclose(f);
    buf[n] = 0;
    const char * p = buf.data();
    // v2 only: the reader gates on version==2 exactly as the packer's doc specifies.
    const char * v = std::strstr(p, "\"version\"");
    if (!v) { return 0; }
    { const char * c = std::strchr(v, ':'); if (!c || std::strtol(c + 1, nullptr, 10) != 2) { return 0; } }
    const char * arr = std::strstr(p, "\"tiers\"");
    if (!arr) { return 0; }
    const char * end = std::strchr(arr, ']');
    uint32_t got = 0;
    for (const char * q = arr; q && end && q < end && got < max_out; ) {
        const char * idk = std::strstr(q, "\"id\"");
        if (!idk || idk > end) { break; }
        const char * qk = std::strstr(idk, "\"quant\"");
        const char * rk = std::strstr(idk, "\"record_bytes\"");
        if (!qk || !rk || qk > end || rk > end) { break; }
        auto num = [](const char * k, const char * key) -> unsigned long long {
            const char * c = std::strchr(k + std::strlen(key), ':');
            return c ? std::strtoull(c + 1, nullptr, 10) : 0ull;
        };
        const unsigned long long rb = num(rk, "\"record_bytes\"");
        // Same invariant the packer enforces: 4KiB-multiple records, so offset = expert*record
        // is a single aligned positional read. A bad row invalidates the whole table (fail loud,
        // never serve a wrong-stride record silently — the #268 safety invariant).
        if (rb == 0 || rb % 4096 != 0) { return 0; }
        out[got].id           = (uint32_t) num(idk, "\"id\"");
        out[got].quant        = (uint32_t) num(qk,  "\"quant\"");
        out[got].record_bytes = (uint64_t) rb;
        got++;
        q = rk + 1;
    }
    return got;
}

class TieredContainerFetcher {
    std::string dir_;
    std::mutex  mtx_;
    std::unordered_map<uint64_t, FILE *> banks_;   // key = (layer<<16 | tier) -> lazily-opened bank

    FILE * bank(uint32_t layer, uint32_t tier) {
        const uint64_t key = ((uint64_t) layer << 16) | tier;
        std::lock_guard<std::mutex> lk(mtx_);
        auto it = banks_.find(key);
        if (it != banks_.end()) { return it->second; }
        const std::string bp = dir_ + "/experts-L" + std::to_string(layer) + "-T" + std::to_string(tier) + ".bin";
        FILE * f = std::fopen(bp.c_str(), "rb");    // nullptr cached too, so a missing shard isn't re-stat'd
        banks_.emplace(key, f);
        return f;
    }
public:
    explicit TieredContainerFetcher(const char * dir) : dir_(dir) {}
    ~TieredContainerFetcher() { for (auto & kv : banks_) { if (kv.second) { std::fclose(kv.second); } } }

    // Read expert `expert` of layer `layer` at precision `tier` (record is `record_bytes`, from manifest).
    // false => bank missing or short read (caller streams the fallback tier or a full-precision source).
    bool read(uint32_t layer, uint32_t expert, uint32_t tier, uint64_t record_bytes, void * dst) {
        FILE * f = bank(layer, tier);
        if (!f) { return false; }
        const uint64_t off = (uint64_t) expert * record_bytes;
        if (MOEC_FSEEK64(f, off) != 0) { return false; }
        return std::fread(dst, 1, (size_t) record_bytes, f) == (size_t) record_bytes;
    }
};

// [TIERED WIRE] The serving adapter: makes a v2 tiered container readable through the ordinary
// ExpertFetcher seam, so the ResidencyCache and the whole offload path need to know NOTHING about
// tiers. Decodes the same packed (layer, byte_offset) `src` DirContainerFetcher uses — the offset
// was packed with tier 0's stride, so expert_id = offset / stride0 — then reads that expert from
// the SELECTED tier's bank at expert_id * tier.record_bytes.
//
// TIER SELECTION IS DELIBERATELY NOT HERE. This reads tier 0 for everything: a dumb, honest
// default that proves the read path end to end. Choosing a tier per expert is a POLICY (a
// rate-distortion call under budget) and belongs in the governor beside the residency budget —
// as a DivisionBandit arm gated on a quality guard, M5's lane. When that lands it replaces
// `select_tier` and nothing else in the serving path changes.
class TieredDirFetcher final : public ExpertFetcher {
    TieredContainerFetcher inner_;
    std::vector<DirTier>   tiers_;
    uint64_t               stride0_ = 0;   // tier 0 record bytes == the stride the caller packed with

public:
    TieredDirFetcher(const char * dir, const DirTier * tiers, uint32_t n_tiers)
        : inner_(dir), tiers_(tiers, tiers + n_tiers) {
        if (!tiers_.empty()) { stride0_ = tiers_[0].record_bytes; }
    }
    const char * name() const override { return "container-tiered"; }
    bool ok() const { return stride0_ != 0; }

    // The policy seam. Constant today (see class doc); the governor replaces it.
    uint32_t select_tier(uint32_t /*layer*/, uint32_t /*expert*/) const { return 0; }

    bool fetch(void * dst, const void * src, size_t bytes) override {
        if (stride0_ == 0) { return false; }
        const uint64_t s   = (uint64_t) (uintptr_t) src;
        const uint32_t L   = (uint32_t) (s >> DirContainerFetcher::LAYER_SHIFT);
        const uint64_t off = s & DirContainerFetcher::OFF_MASK;
        const uint32_t E   = (uint32_t) (off / stride0_);
        const uint32_t T   = select_tier(L, E);
        if (T >= tiers_.size()) { return false; }
        const uint64_t rb = tiers_[T].record_bytes;
        // The cache sized this slot for the CALLER's stride; a tier record must fit it. A coarser
        // tier is smaller (that is the point), so this only rejects a malformed table.
        if (rb > (uint64_t) bytes) { return false; }
        if (!inner_.read(L, E, tiers_[T].id, rb, dst)) { return false; }
        // Zero the tail when a coarser tier under-fills the slot, so no stale bytes from a prior
        // occupant are read as weights (the #43 class of bug: never hand the kernel garbage).
        if (rb < (uint64_t) bytes) {
            std::memset((uint8_t *) dst + rb, 0, (size_t) ((uint64_t) bytes - rb));
        }
        return true;
    }
    void fetch_many(const FetchItem * items, size_t n) override {
        for (size_t i = 0; i < n; i++) {
            if (!fetch(items[i].dst, items[i].src, items[i].bytes)) {
                std::memset(items[i].dst, 0, items[i].bytes);   // fail LOUD downstream, never silent wrong weights
            }
        }
    }
};

} // namespace ggml_moe
