// what this catches: the engine edge of the core<->engine contract (tools/server/server-continuum.h, CONTINUUM-ENGINE
// 4.11) accepting a frame it must refuse, or refusing a good one. Model-free. The core's Rust edge runs the same
// cases on the same bytes (the cross-language fixture), so the two edges cannot drift.
#include "../tools/server/server-continuum.h"

#include <cstdlib>
#include <cstring>
#include <iostream>

static void require(bool condition, const char * message) {
    if (!condition) {
        std::cerr << "FAILED: " << message << '\n';
        std::exit(1);
    }
}

template <typename F> static F frame(uint16_t kind) {
    F f;
    std::memset(&f, 0, sizeof(f));
    f.header.magic          = CTM_MAGIC;
    f.header.schema_version = CTM_SCHEMA_VERSION;
    f.header.kind           = kind;
    f.header.size           = (uint32_t) sizeof(F);
    f.header.schema         = ctm_schema_id;
    return f;
}

static void a_valid_frame_of_every_kind_passes() {
    auto p = frame<ctm_prefetch>(CTM_KIND_PREFETCH);
    require(ctm_validate(&p, sizeof(p)) == CTM_VALID, "prefetch");
    auto a = frame<ctm_act_submit>(CTM_KIND_ACT_SUBMIT);
    a.mode = CTM_ACT_GENERATE;
    a.flags = CTM_ACT_LEARN_PIN;
    require(ctm_validate(&a, sizeof(a)) == CTM_VALID, "act_submit");
    auto d = frame<ctm_act_done>(CTM_KIND_ACT_DONE);
    d.outcome = CTM_OUTCOME_DONE;
    d.mind_from = CTM_MIND_FROM_PARKED;
    require(ctm_validate(&d, sizeof(d)) == CTM_VALID, "act_done");
}

static void every_corruption_is_refused_by_name() {
    auto p = frame<ctm_prefetch>(CTM_KIND_PREFETCH);
    require(ctm_validate(&p, sizeof(ctm_frame_header) - 1) == CTM_INVALID_SHORT, "short");
    auto m = p; m.header.magic ^= 1;
    require(ctm_validate(&m, sizeof(m)) == CTM_INVALID_MAGIC, "magic");
    auto v = p; v.header.schema_version = 2;
    require(ctm_validate(&v, sizeof(v)) == CTM_INVALID_VERSION, "version");
    auto s = p; s.header.schema.bytes[0] ^= 1;
    require(ctm_validate(&s, sizeof(s)) == CTM_INVALID_SCHEMA, "a frame from another schema");
    auto k = p; k.header.kind = 999;
    require(ctm_validate(&k, sizeof(k)) == CTM_INVALID_KIND, "unknown kind");
    auto z = p; z.header.size += 8;
    require(ctm_validate(&z, sizeof(z)) == CTM_INVALID_SIZE, "size disagrees with the kind");
    auto r = p; r.header.reserved = 1;
    require(ctm_validate(&r, sizeof(r)) == CTM_INVALID_RESERVED, "reserved header bytes");
    auto a = frame<ctm_act_submit>(CTM_KIND_ACT_SUBMIT);
    a.mode = 7;
    require(ctm_validate(&a, sizeof(a)) == CTM_INVALID_TAG, "an out-of-range enum is refused, never cast");
    a.mode = CTM_ACT_LEARN; a.flags = 1u << 9;
    require(ctm_validate(&a, sizeof(a)) == CTM_INVALID_TAG, "an unknown flag bit");
    a.flags = 0; a.prompt.offset = UINT64_MAX; a.prompt.length = 2;
    require(ctm_validate(&a, sizeof(a)) == CTM_INVALID_BUFFER_BOUNDS, "offset + length overflows");
    auto l = frame<ctm_learn_duty>(CTM_KIND_LEARN_DUTY);
    l.mode = CTM_LEARN_DREAM; l.share_ppm = 1000001;
    require(ctm_validate(&l, sizeof(l)) == CTM_INVALID_TAG, "a share past the whole");
}

static void uuid_v7_is_versioned_and_distinct() {
    const ctm_id a = ctm_uuid_v7();
    const ctm_id b = ctm_uuid_v7();
    require((a.bytes[6] >> 4) == 7 && (a.bytes[8] >> 6) == 2, "version 7, variant 10");
    require(std::memcmp(a.bytes, b.bytes, 16) != 0, "two mints differ");
}

int main() {
    a_valid_frame_of_every_kind_passes();
    every_corruption_is_refused_by_name();
    uuid_v7_is_versioned_and_distinct();
    std::cout << "test-server-continuum: ok\n";
    return 0;
}
