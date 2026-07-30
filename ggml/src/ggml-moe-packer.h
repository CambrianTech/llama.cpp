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

} // namespace ggml_moe
