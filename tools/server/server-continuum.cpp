// Continuum fork: the engine edge of the contract (server-continuum.h). Frame sizes, validation and UUIDv7 minting,
// the same rules the core's Rust edge applies. Pure: nothing here touches the server's state.

#include "server-continuum.h"

#include <chrono>
#include <cstring>
#include <random>

size_t ctm_frame_size(uint16_t kind) {
    switch (kind) {
        case CTM_KIND_ATTACH:            return sizeof(ctm_attach);
        case CTM_KIND_ENGINE_DESCRIPTOR: return sizeof(ctm_engine_descriptor);
        case CTM_KIND_ACT_SUBMIT:        return sizeof(ctm_act_submit);
        case CTM_KIND_ACT_DONE:          return sizeof(ctm_act_done);
        case CTM_KIND_CANCEL:            return sizeof(ctm_cancel);
        case CTM_KIND_PREFETCH:          return sizeof(ctm_prefetch);
        case CTM_KIND_PLACED:            return sizeof(ctm_placed);
        case CTM_KIND_PIN_RELEASE:       return sizeof(ctm_pin_release);
        case CTM_KIND_RELEASED:          return sizeof(ctm_released);
        case CTM_KIND_GENOME_SET:        return sizeof(ctm_genome_set);
        case CTM_KIND_LEARN_DUTY:        return sizeof(ctm_learn_duty);
        case CTM_KIND_REFUSED:           return sizeof(ctm_refused);
        default:                         return 0; // ACT_TOKENS and LEARN_STEP stream frames are defined with the act vertical (slice 4/5)
    }
}

namespace {

// a frame's bytes, read without assuming alignment (a ring slot or a socket buffer need not be aligned)
template <typename T> T read_frame(const void * bytes) {
    T out;
    std::memcpy(&out, bytes, sizeof(T));
    return out;
}

bool in_range(uint32_t v, uint32_t lo, uint32_t count) { return v >= lo && v < count; }

bool buffer_ok(const ctm_buffer_ref & b) { return b.offset <= UINT64_MAX - b.length; }

int check_tags(uint16_t kind, const void * bytes) {
    switch (kind) {
        case CTM_KIND_ENGINE_DESCRIPTOR: {
            const auto f = read_frame<ctm_engine_descriptor>(bytes);
            if (f.reserved0 != 0) { return CTM_INVALID_RESERVED; }
            return buffer_ok(f.resources) ? CTM_VALID : CTM_INVALID_BUFFER_BOUNDS;
        }
        case CTM_KIND_ACT_SUBMIT: {
            const auto f = read_frame<ctm_act_submit>(bytes);
            if (f.reserved0 != 0) { return CTM_INVALID_RESERVED; }
            if (!in_range(f.mode, 1, CTM_ACT_MODE_COUNT) || (f.flags & ~(uint32_t) CTM_ACT_FLAGS_ALL) != 0) { return CTM_INVALID_TAG; }
            for (const auto & b : { f.prompt, f.labels, f.view, f.sampling }) {
                if (!buffer_ok(b)) { return CTM_INVALID_BUFFER_BOUNDS; }
            }
            return CTM_VALID;
        }
        case CTM_KIND_ACT_DONE: {
            const auto f = read_frame<ctm_act_done>(bytes);
            if (!in_range(f.outcome, 1, CTM_OUTCOME_COUNT) || !in_range(f.reason, 0, CTM_REASON_COUNT)
                || !in_range(f.mind_from, 0, CTM_MIND_FROM_COUNT) || (f.flags & ~(uint32_t) CTM_DONE_FLAGS_ALL) != 0) {
                return CTM_INVALID_TAG;
            }
            return buffer_ok(f.observations) ? CTM_VALID : CTM_INVALID_BUFFER_BOUNDS;
        }
        case CTM_KIND_PLACED: {
            const auto f = read_frame<ctm_placed>(bytes);
            if (f.reserved0 != 0 || f.reserved1 != 0) { return CTM_INVALID_RESERVED; }
            return in_range(f.place, 1, CTM_PLACE_COUNT) ? CTM_VALID : CTM_INVALID_TAG;
        }
        case CTM_KIND_RELEASED: {
            const auto f = read_frame<ctm_released>(bytes);
            if (f.reserved0 != 0 || f.reserved1 != 0) { return CTM_INVALID_RESERVED; }
            return in_range(f.outcome, 1, CTM_PIN_COUNT) ? CTM_VALID : CTM_INVALID_TAG;
        }
        case CTM_KIND_GENOME_SET: {
            const auto f = read_frame<ctm_genome_set>(bytes);
            return buffer_ok(f.layers) ? CTM_VALID : CTM_INVALID_BUFFER_BOUNDS;
        }
        case CTM_KIND_LEARN_DUTY: {
            const auto f = read_frame<ctm_learn_duty>(bytes);
            if (f.reserved0 != 0) { return CTM_INVALID_RESERVED; }
            if (!in_range(f.mode, 1, CTM_LEARN_MODE_COUNT) || f.share_ppm > 1000000u) { return CTM_INVALID_TAG; }
            return CTM_VALID;
        }
        case CTM_KIND_REFUSED: {
            const auto f = read_frame<ctm_refused>(bytes);
            if (f.reserved0 != 0) { return CTM_INVALID_RESERVED; }
            if (!in_range(f.refused_kind, 0, CTM_KIND_KIND_COUNT) || !in_range(f.reason, 0, CTM_REASON_COUNT)) { return CTM_INVALID_TAG; }
            return CTM_VALID;
        }
        default:
            return CTM_VALID; // attach, cancel, prefetch, pin_release: ids only, nothing further to range-check
    }
}

} // namespace

int ctm_validate(const void * bytes, size_t len) {
    if (bytes == nullptr || len < sizeof(ctm_frame_header)) {
        return CTM_INVALID_SHORT;
    }
    const auto h = read_frame<ctm_frame_header>(bytes);
    if (h.magic != CTM_MAGIC) {
        return CTM_INVALID_MAGIC;
    }
    if (h.schema_version != CTM_SCHEMA_VERSION) {
        return CTM_INVALID_VERSION;
    }
    if (std::memcmp(h.schema.bytes, ctm_schema_id.bytes, sizeof(h.schema.bytes)) != 0) {
        return CTM_INVALID_SCHEMA;
    }
    const size_t expected = ctm_frame_size(h.kind);
    if (expected == 0) {
        return CTM_INVALID_KIND;
    }
    if (h.size != expected) {
        return CTM_INVALID_SIZE;
    }
    if (len < expected) {
        return CTM_INVALID_SHORT;
    }
    if (h.reserved != 0) {
        return CTM_INVALID_RESERVED;
    }
    return check_tags(h.kind, bytes);
}

ctm_id ctm_uuid_v7(void) {
    // 48-bit Unix milliseconds, version 7, then 74 random bits (RFC 9562 section 5.7); one generator per process
    static thread_local std::mt19937_64 rng{ std::random_device{}() ^ (uint64_t) std::chrono::steady_clock::now().time_since_epoch().count() };
    const uint64_t ms = (uint64_t) std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    const uint64_t r1 = rng();
    const uint64_t r2 = rng();
    ctm_id id;
    for (int i = 0; i < 6; ++i) {
        id.bytes[i] = (uint8_t) (ms >> (40 - 8 * i));
    }
    id.bytes[6] = (uint8_t) (0x70 | ((r1 >> 8) & 0x0F)); // version 7
    id.bytes[7] = (uint8_t) r1;
    id.bytes[8] = (uint8_t) (0x80 | ((r2 >> 56) & 0x3F)); // variant 10
    for (int i = 9; i < 16; ++i) {
        id.bytes[i] = (uint8_t) (r2 >> (8 * (15 - i)));
    }
    return id;
}
