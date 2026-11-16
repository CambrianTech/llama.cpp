#pragma once

// Continuum fork: THE contract between the continuum core and this engine (CONTINUUM-ENGINE.md 4.11, accepted by
// Codex on #6092 at 6830b410a). Every frame either side sends is defined here and nowhere else; the core's Rust
// types are generated from this file by bindgen and wrapped once in safe newtypes. Plain C, so both sides read the
// same layout.
//
// Rules this file encodes (each checked at both edges, never assumed):
// - Byte order: every integer field is little-endian. An id is 16 bytes in RFC 9562 order (most significant first),
//   never text.
// - Every frame begins with `ctm_frame_header`: magic, schema version, kind, total size, and the schema identity
//   (`ctm_schema_id`, a digest of this file compiled into both sides). A mismatch refuses the link at attach.
// - Tags (`kind`, enums) travel as raw unsigned integers and are validated on receipt; an unknown value is a typed
//   refusal, never a cast into an enum and never a default.
// - Reserved bytes are zero, and a receiver refuses a frame whose reserved bytes are not.
// - Variable-length data is never inside a frame: it is a `ctm_buffer_ref` naming bytes the sender keeps until the
//   receiver's terminal frame for that request (4.11.4).
// - Identity is exact equality of 16 bytes; UUIDv7 time order is for reading logs only.
//
// Sizes and offsets are asserted at the end of this file; the Rust side asserts the same numbers.

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CTM_MAGIC          0x4D544E43u // "CNTM" read as little-endian bytes
#define CTM_SCHEMA_VERSION 1u

// ---- ids --------------------------------------------------------------------------------------------------------

// A UUID in its native 16 bytes. Kinds (all exact-equality identities):
//   engine instance  minted by the engine once at start (UUIDv7)
//   act / turn       minted by the core once per act; a retry keeps it (UUIDv7); a learn pin IS its act
//   mind             her persona's id; a fork's is UUIDv5 of (persona, room)
//   layer            one LoRA layer of her genome, minted by the core (UUIDv7)
//   grant            a learning authority, minted by the core's governor (UUIDv7)
typedef struct ctm_id {
    uint8_t bytes[16];
} ctm_id;

// The schema's identity: a digest of this file, compiled into both sides by their builds.
extern const ctm_id ctm_schema_id;

// ---- the header every frame starts with -------------------------------------------------------------------------

typedef struct ctm_frame_header {
    uint32_t magic;          // CTM_MAGIC
    uint16_t schema_version; // CTM_SCHEMA_VERSION
    uint16_t kind;           // ctm_kind
    uint32_t size;           // the whole frame, header included
    uint32_t reserved;       // zero
    ctm_id   schema;         // ctm_schema_id
} ctm_frame_header;

// Frame kinds (ctm_frame_header.kind).
enum {
    CTM_KIND_ATTACH            = 1,
    CTM_KIND_ENGINE_DESCRIPTOR = 2,
    CTM_KIND_ACT_SUBMIT        = 3,
    CTM_KIND_ACT_TOKENS        = 4,
    CTM_KIND_LEARN_STEP        = 5,
    CTM_KIND_ACT_DONE          = 6,
    CTM_KIND_CANCEL            = 7,
    CTM_KIND_PREFETCH          = 8,
    CTM_KIND_PLACED            = 9,
    CTM_KIND_PIN_RELEASE       = 10,
    CTM_KIND_RELEASED          = 11,
    CTM_KIND_GENOME_SET        = 12,
    CTM_KIND_LEARN_DUTY        = 13,
    CTM_KIND_REFUSED           = 14,
    CTM_KIND_KIND_COUNT        = 15 // one past the last; validation bound only
};

// ---- bytes by reference -----------------------------------------------------------------------------------------

// Bytes NAMED, never carried. The receiver checks `instance`, `generation` (a buffer freed and reallocated has a new
// one) and `offset + length` within the allocation without overflow; the sender keeps the bytes until the
// receiver's terminal frame for the request that names them.
typedef struct ctm_buffer_ref {
    ctm_id   instance;   // the engine instance whose pool holds it
    uint64_t pool;
    uint64_t buffer;
    uint64_t generation; // allocation generation
    uint64_t offset;
    uint64_t length;
} ctm_buffer_ref;

// ---- the one memory standard (4.11.8) ---------------------------------------------------------------------------

enum { // ctm_memory_class
    CTM_MEMORY_KV          = 1, // a mind's sequence
    CTM_MEMORY_CHECKPOINT  = 2, // her checkpoints
    CTM_MEMORY_LAYER       = 3, // a layer of her genome
    CTM_MEMORY_LEARNING    = 4, // backprop workspace and optimizer state, while learning is on
    CTM_MEMORY_PROMPT_CACHE = 5,
    CTM_MEMORY_CLASS_COUNT = 6
};

enum { // ctm_tier
    CTM_TIER_DEVICE = 1,
    CTM_TIER_HOST   = 2,
    CTM_TIER_FLASH  = 3,
    CTM_TIER_COUNT  = 4
};

typedef struct ctm_memory_resource {
    ctm_id         mind;
    uint16_t       memory_class; // ctm_memory_class
    uint16_t       tier;         // ctm_tier
    uint32_t       reserved;     // zero
    uint64_t       bytes;        // the engine's own allocator's measurement
    ctm_buffer_ref where;
} ctm_memory_resource;

// ---- attach: the first frame, before any instance is known ------------------------------------------------------

typedef struct ctm_attach {
    ctm_frame_header header; // kind CTM_KIND_ATTACH; header.schema is the core's schema identity
} ctm_attach;

typedef struct ctm_engine_descriptor {
    ctm_frame_header header;              // kind CTM_KIND_ENGINE_DESCRIPTOR
    ctm_id           instance;            // every stateful frame after this carries exactly this
    // process evidence: the proof a process died, separate from the instance id (4.11.2)
    uint32_t         process_id;
    uint16_t         port;
    uint16_t         reserved0;           // zero
    uint64_t         process_started_s;   // OS start time, seconds since the Unix epoch
    uint64_t         pool_tokens;         // the unified KV pool
    uint64_t         sequence_tokens;     // the most one sequence may hold
    uint32_t         lanes;
    uint32_t         parked_sequences;    // resident minds kept beside the lanes
    uint64_t         host_budget_bytes;   // charged separately from the device (4.11.8)
    uint64_t         device_budget_bytes;
    ctm_buffer_ref   resources;           // ctm_memory_resource[] the engine holds now
} ctm_engine_descriptor;

// ---- acts: generate or learn, on the one path (4.11.6) ----------------------------------------------------------

enum { // ctm_act_mode
    CTM_ACT_GENERATE = 1,
    CTM_ACT_LEARN    = 2,
    CTM_ACT_MODE_COUNT = 3
};

enum { // ctm_act_flags (bits)
    CTM_ACT_LEARN_PIN   = 1u << 0, // keep the checkpoint at this act's prompt end until learned or released
    CTM_ACT_STATE_EXACT = 1u << 1, // recompute any KV stamped with another layer step (evaluations, A/B)
    CTM_ACT_FLAGS_ALL   = (1u << 2) - 1u
};

typedef struct ctm_act_submit {
    ctm_frame_header header;   // kind CTM_KIND_ACT_SUBMIT
    ctm_id           instance;
    ctm_id           act;      // a retry keeps it; a duplicate is answered with the existing outcome
    ctm_id           mind;
    uint16_t         mode;     // ctm_act_mode
    uint16_t         reserved0;
    uint32_t         flags;    // ctm_act_flags
    ctm_buffer_ref   prompt;   // generate: her prompt's bytes; learn: the example's tokens
    ctm_buffer_ref   labels;   // learn only (zero length for generate)
    ctm_buffer_ref   view;     // her view spans (fixed records), zero length when none
    ctm_buffer_ref   sampling; // fixed sampling record, zero length for defaults
} ctm_act_submit;

// What an act saw of one layer, for a stretch of her output (4.11.6: an act spanning commits reports each).
typedef struct ctm_layer_observation {
    ctm_id   layer;
    uint64_t step;
    uint64_t first_token;
    uint64_t last_token;
} ctm_layer_observation;

enum { // ctm_mind_from: where her state came from for this act
    CTM_MIND_FROM_NONE            = 0,
    CTM_MIND_FROM_LANE            = 1,
    CTM_MIND_FROM_PARKED          = 2,
    CTM_MIND_FROM_FLASH           = 3,
    CTM_MIND_FROM_RAM             = 4,
    CTM_MIND_FROM_NEW             = 5,
    CTM_MIND_FROM_ADOPTED         = 6,
    CTM_MIND_FROM_FLASH_POOL_FULL = 7,
    CTM_MIND_FROM_COUNT           = 8
};

enum { // ctm_outcome: an act's terminal outcome
    CTM_OUTCOME_DONE      = 1,
    CTM_OUTCOME_CANCELLED = 2,
    CTM_OUTCOME_REFUSED   = 3,
    CTM_OUTCOME_DEFERRED  = 4,
    CTM_OUTCOME_COUNT     = 5
};

enum { // ctm_reason: why an act was refused or deferred (typed; detail text never carries the signal)
    CTM_REASON_NONE                 = 0,
    CTM_REASON_STALE_ENGINE         = 1,
    CTM_REASON_BACKPRESSURE         = 2,
    CTM_REASON_ALL_PINNED           = 3,
    CTM_REASON_EXCEEDS_GRANT        = 4,
    CTM_REASON_TIER_IO_ERROR        = 5,
    CTM_REASON_LAYER_UNCERTAIN      = 6,
    CTM_REASON_NO_CHECKPOINT        = 7,
    CTM_REASON_PREFIX_MISMATCH      = 8,
    CTM_REASON_NO_SIDE_SEQUENCE     = 9,
    CTM_REASON_OPTIMIZER_FAILED     = 10,
    CTM_REASON_UNSUPPORTED          = 11,
    CTM_REASON_INVALID_FRAME        = 12,
    CTM_REASON_COUNT                = 13
};

typedef struct ctm_act_done {
    ctm_frame_header header;   // kind CTM_KIND_ACT_DONE
    ctm_id           instance;
    ctm_id           act;
    uint16_t         outcome;  // ctm_outcome
    uint16_t         reason;   // ctm_reason (CTM_REASON_NONE when done)
    uint16_t         mind_from;
    uint16_t         flags;    // bit 0: pinned; bit 1: live sequence preserved (a learn act)
    uint64_t         switch_us;
    uint64_t         generation;
    uint64_t         cached_tokens;
    uint64_t         retry_after_us; // deferred only
    ctm_buffer_ref   observations;   // ctm_layer_observation[]
} ctm_act_done;

enum { // ctm_act_done.flags (bits)
    CTM_DONE_PINNED             = 1u << 0,
    CTM_DONE_SEQUENCE_PRESERVED = 1u << 1,
    CTM_DONE_FLAGS_ALL          = (1u << 2) - 1u
};

typedef struct ctm_cancel {
    ctm_frame_header header; // kind CTM_KIND_CANCEL; answered by the act's terminal frame
    ctm_id           instance;
    ctm_id           act;
} ctm_cancel;

// ---- minds ------------------------------------------------------------------------------------------------------

enum { // ctm_mind_place: where a prefetch left her
    CTM_PLACE_LANE                  = 1,
    CTM_PLACE_PARKED                = 2,
    CTM_PLACE_UNKNOWN               = 3,
    CTM_PLACE_DEFERRED_NO_PARKED_ID = 4,
    CTM_PLACE_DEFERRED_POOL_FULL    = 5,
    CTM_PLACE_MINDS_OFF             = 6,
    CTM_PLACE_COUNT                 = 7
};

typedef struct ctm_prefetch {
    ctm_frame_header header; // kind CTM_KIND_PREFETCH
    ctm_id           instance;
    ctm_id           mind;
} ctm_prefetch;

typedef struct ctm_placed {
    ctm_frame_header header; // kind CTM_KIND_PLACED
    ctm_id           instance;
    ctm_id           mind;
    uint16_t         place;  // ctm_mind_place
    uint16_t         reserved0;
    uint32_t         reserved1;
    uint64_t         elapsed_us;
} ctm_placed;

enum { // ctm_pin_release
    CTM_PIN_RELEASED          = 1,
    CTM_PIN_RELEASED_ON_FLASH = 2,
    CTM_PIN_NOT_PINNED        = 3,
    CTM_PIN_UNKNOWN_MIND      = 4,
    CTM_PIN_COUNT             = 5
};

typedef struct ctm_pin_release {
    ctm_frame_header header; // kind CTM_KIND_PIN_RELEASE; idempotent
    ctm_id           instance;
    ctm_id           mind;
    ctm_id           act;    // the pin is its act
} ctm_pin_release;

typedef struct ctm_released {
    ctm_frame_header header; // kind CTM_KIND_RELEASED
    ctm_id           instance;
    ctm_id           act;
    uint16_t         outcome; // ctm_pin_release
    uint16_t         reserved0;
    uint32_t         reserved1;
} ctm_released;

// ---- the genome (4.11.6): always on; learning a per-layer switch ------------------------------------------------

typedef struct ctm_genome_layer {
    ctm_id   layer;
    float    scale;
    uint32_t learning;  // 0 off, 1 on
} ctm_genome_layer;

typedef struct ctm_genome_set {
    ctm_frame_header header; // kind CTM_KIND_GENOME_SET; an explicit, authorized genome change
    ctm_id           instance;
    ctm_id           mind;
    ctm_buffer_ref   layers; // ctm_genome_layer[]
} ctm_genome_set;

enum { // ctm_learn_mode
    CTM_LEARN_TRICKLE = 1, // awake
    CTM_LEARN_DREAM   = 2, // asleep
    CTM_LEARN_MODE_COUNT = 3
};

typedef struct ctm_learn_duty {
    ctm_frame_header header; // kind CTM_KIND_LEARN_DUTY
    ctm_id           instance;
    ctm_id           grant;
    uint32_t         share_ppm; // the engine time learning may take, parts per million
    uint16_t         mode;      // ctm_learn_mode
    uint16_t         reserved0;
} ctm_learn_duty;

// A request refused before it was accepted (bad frame, stale engine, backpressure).
typedef struct ctm_refused {
    ctm_frame_header header; // kind CTM_KIND_REFUSED
    ctm_id           instance;
    ctm_id           request; // the act or mind the refused frame named (zero when it could not be read)
    uint16_t         refused_kind;
    uint16_t         reason;  // ctm_reason
    uint32_t         reserved0;
} ctm_refused;

// ---- validation, the same at both edges -------------------------------------------------------------------------

enum { // ctm_validation
    CTM_VALID                 = 0,
    CTM_INVALID_SHORT         = 1, // fewer bytes than a header, or than the kind's size
    CTM_INVALID_MAGIC         = 2,
    CTM_INVALID_VERSION       = 3,
    CTM_INVALID_SCHEMA        = 4,
    CTM_INVALID_KIND          = 5,
    CTM_INVALID_SIZE          = 6, // header.size disagrees with the kind's fixed size
    CTM_INVALID_RESERVED      = 7,
    CTM_INVALID_TAG           = 8, // an enum field out of range
    CTM_INVALID_BUFFER_BOUNDS = 9  // offset + length overflows
};

// The fixed size of a frame kind, or 0 for an unknown kind.
size_t ctm_frame_size(uint16_t kind);

// Validate the header and every tag of the frame at `bytes` (`len` bytes long) against this build's schema. Pure:
// reads only the frame. Returns a ctm_validation value.
int ctm_validate(const void * bytes, size_t len);

// A new UUIDv7 (48-bit Unix milliseconds, version 7, 74 random bits, RFC 9562).
ctm_id ctm_uuid_v7(void);

// ---- the layout, asserted (the Rust side asserts the same numbers) ----------------------------------------------

#ifdef __cplusplus
#define CTM_ASSERT(cond, msg) static_assert(cond, msg)
#else
#define CTM_ASSERT(cond, msg) _Static_assert(cond, msg)
#endif

CTM_ASSERT(sizeof(ctm_id) == 16, "ctm_id");
CTM_ASSERT(sizeof(ctm_frame_header) == 32, "ctm_frame_header");
CTM_ASSERT(offsetof(ctm_frame_header, kind) == 6 && offsetof(ctm_frame_header, schema) == 16, "ctm_frame_header offsets");
CTM_ASSERT(sizeof(ctm_buffer_ref) == 56, "ctm_buffer_ref");
CTM_ASSERT(sizeof(ctm_memory_resource) == 88, "ctm_memory_resource");
CTM_ASSERT(sizeof(ctm_attach) == 32, "ctm_attach");
CTM_ASSERT(sizeof(ctm_engine_descriptor) == 160, "ctm_engine_descriptor");
CTM_ASSERT(offsetof(ctm_engine_descriptor, resources) == 104, "ctm_engine_descriptor.resources");
CTM_ASSERT(sizeof(ctm_act_submit) == 312, "ctm_act_submit");
CTM_ASSERT(offsetof(ctm_act_submit, prompt) == 88, "ctm_act_submit.prompt");
CTM_ASSERT(sizeof(ctm_layer_observation) == 40, "ctm_layer_observation");
CTM_ASSERT(sizeof(ctm_act_done) == 160, "ctm_act_done");
CTM_ASSERT(offsetof(ctm_act_done, observations) == 104, "ctm_act_done.observations");
CTM_ASSERT(sizeof(ctm_cancel) == 64, "ctm_cancel");
CTM_ASSERT(sizeof(ctm_prefetch) == 64, "ctm_prefetch");
CTM_ASSERT(sizeof(ctm_placed) == 80, "ctm_placed");
CTM_ASSERT(sizeof(ctm_pin_release) == 80, "ctm_pin_release");
CTM_ASSERT(sizeof(ctm_released) == 72, "ctm_released");
CTM_ASSERT(sizeof(ctm_genome_layer) == 24, "ctm_genome_layer");
CTM_ASSERT(sizeof(ctm_genome_set) == 120, "ctm_genome_set");
CTM_ASSERT(sizeof(ctm_learn_duty) == 72, "ctm_learn_duty");
CTM_ASSERT(sizeof(ctm_refused) == 72, "ctm_refused");

#ifdef __cplusplus
}
#endif
