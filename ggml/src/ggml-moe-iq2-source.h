#pragma once
// ggml-moe-iq2-source.h — pack existing IQ2 experts into aligned container records (rung 1, #33).
//
// The fast path to beat WASTE: our IQ2 experts are already quantized (~2.6MB, smaller than their VQ3
// 12.4MB), so we BYTE-COPY gate|up|down into aligned records — no re-quant, no RVQ codebooks. Alignment
// is the fetch win; the small footprint is ours to keep. This is the FRAMING (record header + payload
// layout), tested now; the byte SOURCE (reading real IQ2 tensors from the K3 GGUF) plugs in via the
// provider callback. The record `fmt` is a PARAMETER, not a baked constant — M5 owns the WEXP fmt enum,
// so we thread her confirmed IQ2 value through instead of guessing a shared byte.

#include "ggml-moe-packer.h"   // ExpertSource
#include "ggml-moe-wexp.h"     // WEXP record header (M5's tested framing)
#include <cstdint>
#include <cstring>
#include <functional>

namespace ggml_moe {

// Raw byte spans for one expert's three matrices (already IQ2-quantized GGUF blocks). Provided by the
// GGUF reader in production, or a synthetic filler in tests.
struct Iq2Bytes {
    const uint8_t * gate; size_t gate_len;
    const uint8_t * up;   size_t up_len;
    const uint8_t * down; size_t down_len;
};

// Composes an IQ2 expert record: WEXP ident header (magic|layer|expert — M5's per-fetch drift check),
// fmt = the caller-supplied IQ2 value, gate|up|down adjacent right after the header, correction_off=0
// (IQ2 has no residual correction), zero-pad to the 4KiB-block record. The provider yields the bytes.
class Iq2ExpertSource : public ExpertSource {
    uint8_t  fmt_;                                        // M5's confirmed IQ2 fmt value (threaded, not guessed)
    std::function<bool(uint32_t, uint32_t, Iq2Bytes &)> provider_;
public:
    Iq2ExpertSource(uint8_t iq2_fmt, std::function<bool(uint32_t, uint32_t, Iq2Bytes &)> provider)
        : fmt_(iq2_fmt), provider_(std::move(provider)) {}

    bool write_record(uint32_t layer, uint32_t expert, void * dst, size_t stride) override {
        Iq2Bytes b{};
        if (!provider_(layer, expert, b)) { return false; }
        const size_t hdr = WEXP_HEADER_BYTES;
        const size_t need = hdr + b.gate_len + b.up_len + b.down_len;
        if (need > stride) { return false; }              // record too small for this expert — caller sizes stride

        WexpRecord r;
        r.layer = (uint16_t) layer; r.expert_id = (uint16_t) expert; r.fmt = fmt_;
        r.gate_off       = (uint32_t) hdr;
        r.up_off         = (uint32_t) (r.gate_off + b.gate_len);
        r.down_off       = (uint32_t) (r.up_off + b.up_len);
        r.correction_off = 0;                             // IQ2: no residual-correction section
        r.record_4k_blocks = (uint32_t) (stride / 4096);
        wexp_write_header(static_cast<uint8_t *>(dst), r);

        uint8_t * p = static_cast<uint8_t *>(dst);
        std::memcpy(p + r.gate_off, b.gate, b.gate_len);
        std::memcpy(p + r.up_off,   b.up,   b.up_len);
        std::memcpy(p + r.down_off, b.down, b.down_len);
        // stride > need: the tail is already zeroed by moec_pack before write_record.
        return true;
    }
};

} // namespace ggml_moe
