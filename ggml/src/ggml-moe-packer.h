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
static inline bool moec_pack_dir(const char * dir, const char * model, uint32_t fmt,
                                 uint32_t layers, uint32_t experts, uint64_t record_bytes,
                                 uint32_t activated_per_token, ExpertSource & src) {
    if (record_bytes == 0 || record_bytes % 4096 != 0) { return false; }   // her open() refuses otherwise

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
            "  \"activated_per_token\": %u\n"
            "}\n",
            model, fmt, (unsigned long long) record_bytes, layers, experts, activated_per_token);
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

} // namespace ggml_moe
