#pragma once
// ggml-moe-wexp.h — byte-exact WEXP expert-record header (#268 record INTERIOR).
//
// Authority: WASTE docs/FORMAT.md, confirmed by M5 into docs/reference/WASTE-EXTRACT.md (9f55ae765).
// Each container record begins with this 32-byte little-endian header, then the payload
// (gate | up | down | per-channel corrections), zero-padded to record_4k_blocks * 4096. No header CRC.
//
// Serialized field-by-field (NOT a struct memcpy) so the on-disk bytes are exact regardless of
// compiler packing/endianness — this is a wire format shared with M5's Rust reader, so it must match
// to the byte. The `fmt` field is the one M5 refused to guess: VQ3R=4, VQ2R=5 (NOT 0/1) — a wrong
// value packs a structurally-valid container that both readers accept and decode as garbage.

#include <cstdint>
#include <cstring>

namespace ggml_moe {

enum WexpFmt : uint8_t { WEXP_VQ3R = 4, WEXP_VQ2R = 5 };   // per the confirmed contract — never 0/1
static const uint32_t WEXP_MAGIC_LEN    = 4;               // "WEXP"
static const size_t   WEXP_HEADER_BYTES = 32;

// little-endian field access
static inline void     wexp_put_u16(uint8_t * p, uint16_t v) { p[0] = uint8_t(v); p[1] = uint8_t(v >> 8); }
static inline void     wexp_put_u32(uint8_t * p, uint32_t v) { p[0]=uint8_t(v); p[1]=uint8_t(v>>8); p[2]=uint8_t(v>>16); p[3]=uint8_t(v>>24); }
static inline uint16_t wexp_get_u16(const uint8_t * p) { return uint16_t(p[0] | (uint16_t(p[1]) << 8)); }
static inline uint32_t wexp_get_u32(const uint8_t * p) { return uint32_t(p[0]) | (uint32_t(p[1])<<8) | (uint32_t(p[2])<<16) | (uint32_t(p[3])<<24); }

// Field byte offsets (single source of truth for both the writer and any reader).
enum WexpOff : size_t {
    WEXP_O_MAGIC = 0, WEXP_O_LAYER = 4, WEXP_O_EXPERT = 6, WEXP_O_FMT = 8, WEXP_O_FLAGS = 9,
    WEXP_O_CODEBOOK = 10, WEXP_O_GATE = 12, WEXP_O_UP = 16, WEXP_O_DOWN = 20, WEXP_O_CORR = 24,
    WEXP_O_BLOCKS = 28
};

struct WexpRecord {
    uint16_t layer = 0, expert_id = 0;
    uint8_t  fmt = WEXP_VQ3R;
    uint16_t codebook_id = 0;
    uint32_t gate_off = 0, up_off = 0, down_off = 0, correction_off = 0;   // relative to record start
    uint32_t record_4k_blocks = 0;                                          // record bytes = value * 4096
};

// Write the 32-byte header at dst (>= 32 bytes). flags is always 0.
static inline void wexp_write_header(uint8_t * dst, const WexpRecord & r) {
    std::memcpy(dst + WEXP_O_MAGIC, "WEXP", 4);
    wexp_put_u16(dst + WEXP_O_LAYER,    r.layer);
    wexp_put_u16(dst + WEXP_O_EXPERT,   r.expert_id);
    dst[WEXP_O_FMT]   = r.fmt;
    dst[WEXP_O_FLAGS] = 0;
    wexp_put_u16(dst + WEXP_O_CODEBOOK, r.codebook_id);
    wexp_put_u32(dst + WEXP_O_GATE,     r.gate_off);
    wexp_put_u32(dst + WEXP_O_UP,       r.up_off);
    wexp_put_u32(dst + WEXP_O_DOWN,     r.down_off);
    wexp_put_u32(dst + WEXP_O_CORR,     r.correction_off);
    wexp_put_u32(dst + WEXP_O_BLOCKS,   r.record_4k_blocks);
}

// Parse a header; returns false if the magic is wrong (the read-side validation M5's reader also does).
static inline bool wexp_read_header(const uint8_t * src, WexpRecord & r) {
    if (std::memcmp(src + WEXP_O_MAGIC, "WEXP", 4) != 0) { return false; }
    r.layer          = wexp_get_u16(src + WEXP_O_LAYER);
    r.expert_id      = wexp_get_u16(src + WEXP_O_EXPERT);
    r.fmt            = src[WEXP_O_FMT];
    r.codebook_id    = wexp_get_u16(src + WEXP_O_CODEBOOK);
    r.gate_off       = wexp_get_u32(src + WEXP_O_GATE);
    r.up_off         = wexp_get_u32(src + WEXP_O_UP);
    r.down_off       = wexp_get_u32(src + WEXP_O_DOWN);
    r.correction_off = wexp_get_u32(src + WEXP_O_CORR);
    r.record_4k_blocks = wexp_get_u32(src + WEXP_O_BLOCKS);
    return true;
}

static inline uint64_t wexp_record_bytes(const WexpRecord & r) { return uint64_t(r.record_4k_blocks) * 4096u; }

} // namespace ggml_moe
