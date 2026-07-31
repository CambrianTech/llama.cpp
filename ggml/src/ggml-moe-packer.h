#pragma once
// ggml-moe-packer.h — writes a streaming expert container (#268), the WRITE side of the format.
//
// Splits cleanly into ENVELOPE (here, tested) and INTERIOR (pluggable):
//   ENVELOPE  — the container header, 16KiB-aligned layer-major record layout, and the guarantee that
//               record R lands at exactly moec_record_offset(h, layer, expert). This is what the
//               ContainerFetcher reads back with one pread; the round-trip test pins it.
//   INTERIOR  — how one expert's bytes are produced (GGUF dequant -> RVQ codebooks, crc). Supplied by
//               an ExpertSource, because the byte-exact record layout is M5's WASTE-EXTRACT.md
//               authority; the packer must not guess it. A GgufRvqSource plugs in without touching the
//               envelope. That separation is why we can ship + test the packer TODAY and slot the RVQ
//               interior in the moment the layout is confirmed.

#include "ggml-moe-container.h"
#include <cstdio>
#include <cstdint>
#include <string>
#include <vector>
#include <algorithm>

namespace ggml_moe {

// The pluggable INTERIOR: fill `dst` (record_stride bytes) with the packed record for (layer, expert).
// Return false to abort the pack. Implementations: GgufRvqSource (real), a synthetic source (tests).
struct ExpertSource {
    virtual ~ExpertSource() = default;
    virtual bool write_record(uint32_t layer, uint32_t expert, void * dst, size_t stride) = 0;
};

// Author a container header for the given shape. record_stride must already be alignment-rounded so
// every record offset is a valid direct-I/O target.
static inline ContainerHeader moec_make_header(uint32_t layers, uint32_t experts, uint64_t stride,
                                               uint32_t quant) {
    ContainerHeader h{};
    h.magic = MOEC_MAGIC; h.version = MOEC_VERSION;
    h.n_layers = layers; h.experts_per_layer = experts;
    h.align = MOEC_ALIGN; h.record_stride = stride;
    h.data_offset = moec_align_up(sizeof(ContainerHeader));
    h.quant = quant; h.crc_algo = 0;
    h.total_bytes = h.data_offset + (uint64_t) layers * experts * stride;
    return h;
}

// Write the whole container to `path`: header, alignment padding, then every (layer,expert) record in
// LAYER-MAJOR order — which makes the sequential file position match moec_record_offset exactly, so a
// layer's experts are contiguous on disk (near-sequential prefetch) and the fetcher's offset math is
// correct by construction. Returns true on success. `stride` must be alignment-rounded by the caller.
static inline bool moec_pack(const char * path, uint32_t layers, uint32_t experts, uint64_t stride,
                             uint32_t quant, ExpertSource & src) {
    if (stride == 0 || stride % MOEC_ALIGN != 0) { return false; }   // records must be aligned
    const ContainerHeader h = moec_make_header(layers, experts, stride, quant);

    FILE * f = std::fopen(path, "wb");
    if (!f) { return false; }

    bool ok = std::fwrite(&h, 1, sizeof(h), f) == sizeof(h);
    if (ok) {                                                        // pad header up to data_offset
        const size_t pad_n = (size_t) (h.data_offset - sizeof(h));
        std::vector<char> pad(pad_n, 0);
        ok = std::fwrite(pad.data(), 1, pad_n, f) == pad_n;
    }
    std::vector<char> rec((size_t) stride);
    for (uint32_t L = 0; L < layers && ok; L++) {
        for (uint32_t E = 0; E < experts && ok; E++) {
            std::fill(rec.begin(), rec.end(), 0);                    // zero-pad the record tail
            if (!src.write_record(L, E, rec.data(), (size_t) stride)) { ok = false; break; }
            if (std::fwrite(rec.data(), 1, (size_t) stride, f) != (size_t) stride) { ok = false; break; }
        }
    }
    std::fclose(f);
    return ok;
}

// Emit M5's Rust-reader format: a DIRECTORY of manifest.json + per-layer experts-L{n}.bin banks.
// (capacity/expert_container.rs opens banks lazily so a node holding 2/60 shards never stats the rest
// — the grid shard unit.) Contract, pinned on #k3-serving: record_bytes is a 4KiB multiple (her open()
// refuses otherwise); within a bank, records are sorted by expert_id so offset = expert_id*record_bytes
// (one pread); bank file size == record_bytes*experts_per_layer (she refuses a truncated tail). `dir`
// must already exist. This is the ENVELOPE that replaces the single-file container; the WEXP record and
// budget governor are unchanged — activated_per_token is exactly the field the cliff policy consumes.
// activated_per_token MUST be the TOTAL activated experts across ALL MoE layers (K3: top_k * n_layers
// ~= 8*61 ~= 488), NOT the per-layer top-k. The budget governor uses it as the per-token working set
// (expert_bytes * activated_per_token); a per-layer value understates the cliff ~n_layers-fold and
// silently green-lights a cache that CANNOT retain — recreating the reuse=0 failure this lane exists to
// kill (semantics pinned by M5, reader commit 9c85a4f88). top_k_per_layer is the auditable per-layer
// value so the total is verifiable: activated_per_token == top_k_per_layer * n_layers.
static inline bool moec_pack_dir(const char * dir, const char * model, uint32_t fmt,
                                 uint32_t layers, uint32_t experts, uint64_t record_bytes,
                                 uint32_t activated_per_token, uint32_t top_k_per_layer, ExpertSource & src) {
    if (record_bytes == 0 || record_bytes % 4096 != 0) { return false; }   // her open() refuses otherwise
    if (top_k_per_layer != 0 && activated_per_token != top_k_per_layer * layers) { return false; } // total, not per-layer

    {   // manifest.json (v1) — her reader gates on version; fields exactly as pinned.
        const std::string mp = std::string(dir) + "/manifest.json";
        FILE * f = std::fopen(mp.c_str(), "wb");
        if (!f) { return false; }
        std::fprintf(f,
            "{\n"
            "  \"version\": 1,\n"
            "  \"model\": \"%s\",\n"
            "  \"fmt\": %u,\n"
            "  \"record_bytes\": %llu,\n"
            "  \"n_layers\": %u,\n"
            "  \"experts_per_layer\": %u,\n"
            "  \"activated_per_token\": %u,\n"
            "  \"top_k_per_layer\": %u\n"
            "}\n",
            model, fmt, (unsigned long long) record_bytes, layers, experts, activated_per_token, top_k_per_layer);
        std::fclose(f);
    }

    std::vector<char> rec((size_t) record_bytes);
    for (uint32_t L = 0; L < layers; L++) {                    // one bank per layer
        const std::string bp = std::string(dir) + "/experts-L" + std::to_string(L) + ".bin";
        FILE * f = std::fopen(bp.c_str(), "wb");
        if (!f) { return false; }
        bool ok = true;
        for (uint32_t E = 0; E < experts && ok; E++) {         // sorted by expert_id => offset = E*record_bytes
            std::fill(rec.begin(), rec.end(), 0);
            if (!src.write_record(L, E, rec.data(), (size_t) record_bytes)) { ok = false; break; }
            if (std::fwrite(rec.data(), 1, (size_t) record_bytes, f) != (size_t) record_bytes) { ok = false; break; }
        }
        std::fclose(f);
        if (!ok) { return false; }                             // bank size == record_bytes*experts by construction
    }
    return true;
}

// ---------------------------------------------------------------------------------------------------
// TIERED container (manifest v2) — the dynamic-quant precision axis: "all-stars sharp, decaying quant at
// the cruft". Each expert is stored at N precomputed precision TIERS (descending fidelity); the pager
// SELECTS a tier per expert (rate-distortion under budget) and the fetcher reads that tier's bank — pure
// selection, no runtime re-quant. GENERIC path (here): per-(layer,tier) banks, offset = expert_id *
// tiers[t].record_bytes — works for any quant incl. non-residual IQ2/IQ1. PROGRESSIVE path (RVQ VQ3R,
// M5's moat move): one stage-major bank, tiers carry prefix_bytes = read-lengths — added when
// GgufRvqSource lands. Schema pinned with M5 (7711fe60): tiers[] of {id, quant, record_bytes} DESCENDING
// fidelity; v1 (single-tier) is the serde-default degenerate case. Reader gates on version==2.
struct TierSpec {
    uint32_t id;             // tier index, 0 = highest fidelity (all-star), ascending = cruft
    uint32_t quant;          // MoecQuant of THIS tier
    uint64_t record_bytes;   // per-expert record bytes at this tier (4KiB multiple; one pread)
};

// Emit a tiered container: manifest.json (v2) + per-(layer,tier) bank experts-L{n}-T{t}.bin.
// `source_for(t)` returns the ExpertSource that produces tier `tiers[t]`'s records (e.g. a full-fidelity
// GGUF source for T0 and a coarser/requantized or RVQ-prefix source for the cruft tiers). `dir` must
// exist. Same invariants as moec_pack_dir per bank: record_bytes % 4096 == 0, sorted by expert_id so
// offset = expert_id * record_bytes, bank size == record_bytes * experts by construction.
static inline bool moec_pack_dir_tiered(const char * dir, const char * model, uint32_t fmt,
                                        uint32_t layers, uint32_t experts,
                                        uint32_t activated_per_token, uint32_t top_k_per_layer,
                                        const TierSpec * tiers, uint32_t n_tiers,
                                        ExpertSource & (*source_for)(uint32_t tier, void * user), void * user) {
    if (n_tiers == 0) { return false; }
    if (top_k_per_layer != 0 && activated_per_token != top_k_per_layer * layers) { return false; }
    for (uint32_t t = 0; t < n_tiers; t++) {
        if (tiers[t].record_bytes == 0 || tiers[t].record_bytes % 4096 != 0) { return false; }
    }

    {   // manifest.json (v2) — adds the tiers[] array; reader gates on version==2.
        const std::string mp = std::string(dir) + "/manifest.json";
        FILE * f = std::fopen(mp.c_str(), "wb");
        if (!f) { return false; }
        std::fprintf(f,
            "{\n"
            "  \"version\": 2,\n"
            "  \"model\": \"%s\",\n"
            "  \"fmt\": %u,\n"
            "  \"n_layers\": %u,\n"
            "  \"experts_per_layer\": %u,\n"
            "  \"activated_per_token\": %u,\n"
            "  \"top_k_per_layer\": %u,\n"
            "  \"tiers\": [\n",
            model, fmt, layers, experts, activated_per_token, top_k_per_layer);
        for (uint32_t t = 0; t < n_tiers; t++) {
            std::fprintf(f, "    { \"id\": %u, \"quant\": %u, \"record_bytes\": %llu }%s\n",
                         tiers[t].id, tiers[t].quant, (unsigned long long) tiers[t].record_bytes,
                         t + 1 < n_tiers ? "," : "");
        }
        std::fprintf(f, "  ]\n}\n");
        std::fclose(f);
    }

    for (uint32_t t = 0; t < n_tiers; t++) {
        ExpertSource & src = source_for(t, user);
        std::vector<char> rec((size_t) tiers[t].record_bytes);
        for (uint32_t L = 0; L < layers; L++) {
            const std::string bp = std::string(dir) + "/experts-L" + std::to_string(L)
                                 + "-T" + std::to_string(tiers[t].id) + ".bin";
            FILE * f = std::fopen(bp.c_str(), "wb");
            if (!f) { return false; }
            bool ok = true;
            for (uint32_t E = 0; E < experts && ok; E++) {
                std::fill(rec.begin(), rec.end(), 0);
                if (!src.write_record(L, E, rec.data(), (size_t) tiers[t].record_bytes)) { ok = false; break; }
                if (std::fwrite(rec.data(), 1, (size_t) tiers[t].record_bytes, f)
                        != (size_t) tiers[t].record_bytes) { ok = false; break; }
            }
            std::fclose(f);
            if (!ok) { return false; }
        }
    }
    return true;
}

} // namespace ggml_moe
