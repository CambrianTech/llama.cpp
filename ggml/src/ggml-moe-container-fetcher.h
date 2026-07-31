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
#include <mutex>
#include <string>
#include <unordered_map>

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

// TIERED read primitive: reads an expert from a specific PRECISION TIER of a tiered container (manifest
// v2, per-(layer,tier) banks). The pager selects the tier per expert (all-star … cruft, rate-distortion
// under budget); this reads that tier's bank at offset = expert_id * tier_record_bytes (one positional
// read, alignment-by-construction). Bank handles open LAZILY and are cached per (layer,tier) — a node
// holding only some (layer,tier) shards never stats the rest (the grid shard unit). This is the read
// side that pairs with moec_pack_dir_tiered; the serving direct-I/O variant swaps in the same way as
// ContainerFetcher (identical offset contract). record_bytes comes from the manifest tier row.
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

} // namespace ggml_moe
