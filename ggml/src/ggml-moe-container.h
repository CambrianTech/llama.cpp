#pragma once
// ggml-moe-container.h — streaming expert container format (#268 foundry re-pack).
//
// THE PROBLEM IT SOLVES: raw GGUF experts are 32-byte-aligned and scattered, so streaming them is a
// storm of misaligned random reads capped at ~200 MB/s (measured, BigMama PM9A1). WASTE proved that
// repacking each expert into ONE aligned, contiguous record — gate/up/down adjacent — turns that into
// ~GB/s sequential-ish streaming (50x) AND, by shrinking the per-token working set, makes the reuse
// CLIFF crossable at a sane RAM budget. Same fix closes the fetch gap and the retention cliff.
//
// THE CONTRACT: the foundry PACKS experts into this container; the fetcher READS exactly one record
// per (layer, expert) with ONE pread at a deterministic aligned offset — no index walk, no bounce.
//
// Byte-exact record layout (RVQ codebooks, crc trailer, quant tags) is the authority in
// docs/reference/WASTE-EXTRACT.md (M5). This header is the FORK-SIDE consumer interface + the offset
// math both ends must agree on. Keep the two reconciled; this is the seam.

#include <cstdint>

namespace ggml_moe {

// "MOEC" (little-endian): 'M'|'O'<<8|'E'<<16|'C'<<24
static const uint32_t MOEC_MAGIC   = 0x43454F4Du;
static const uint32_t MOEC_VERSION = 1u;

// 16 KiB alignment: clears the 4 KiB direct-I/O floor on every drive AND lands on the Metal page so
// the same record double-serves as a zero-copy GPU upload on Apple silicon (WASTE-EXTRACT note).
static const uint64_t MOEC_ALIGN   = 16384u;

// Quant/format tag for the packed matrices (kept generic — the container is model-agnostic).
enum MoecQuant : uint32_t { MOEC_Q_UNKNOWN = 0, MOEC_Q_RVQ3 = 1, MOEC_Q_IQ2 = 2, MOEC_Q_MXFP4 = 3 };

// File header, itself padded to MOEC_ALIGN so record 0 starts aligned.
struct ContainerHeader {
    uint32_t magic;             // MOEC_MAGIC
    uint32_t version;           // MOEC_VERSION
    uint32_t n_layers;          // MoE layers packed
    uint32_t experts_per_layer; // routed experts per layer
    uint64_t align;             // MOEC_ALIGN (record + header alignment actually used)
    uint64_t record_stride;     // bytes per expert record (multiple of align) — one pread reads this
    uint64_t data_offset;       // byte offset where record 0 begins (multiple of align)
    uint64_t total_bytes;       // whole container size

    // per-record interior layout: gate|up|down ADJACENT within the record (the streaming win).
    uint64_t gate_off,  gate_bytes;
    uint64_t up_off,    up_bytes;
    uint64_t down_off,  down_bytes;

    uint32_t quant;             // MoecQuant of the packed matrices
    uint32_t crc_algo;          // 0 = none, 1 = crc32c over the record payload (validated on read)
};
static_assert(sizeof(ContainerHeader) <= MOEC_ALIGN, "header must fit in one aligned block");

// Linear record index for (layer, expert) — layer-major so a layer's experts are contiguous on disk
// (prefetching a layer's whole selected set is then near-sequential).
static inline uint64_t moec_record_index(const ContainerHeader & h, uint32_t layer, uint32_t expert) {
    return (uint64_t) layer * h.experts_per_layer + expert;
}

// Byte offset of the (layer, expert) record. Aligned by construction (data_offset and record_stride
// are both multiples of align), so it is a valid FILE_FLAG_NO_BUFFERING / O_DIRECT read target: the
// fetcher issues exactly ONE pread of record_stride bytes here — no VirtualQuery, no bounce buffer.
static inline uint64_t moec_record_offset(const ContainerHeader & h, uint32_t layer, uint32_t expert) {
    return h.data_offset + moec_record_index(h, layer, expert) * h.record_stride;
}

// Round a size up to the container alignment (used by the packer to size the header and stride).
static inline uint64_t moec_align_up(uint64_t n) { return (n + (MOEC_ALIGN - 1)) & ~(MOEC_ALIGN - 1); }

// The cache identity for a container-resident expert is (layer, expert) directly — the SAME single
// key WASTE uses, stable across tokens by construction (no tensor-name, no per-eval pointer). This is
// what the ResidencyCache keys on once the fetcher sources from a container instead of the mmap.
static inline uint64_t moec_expert_key(uint32_t layer, uint32_t expert) {
    return ((uint64_t) layer << 32) | (uint64_t) expert;
}

// ------------------------------------------------------------------------------------------------
// Derived cache-budget policy — thresholds come from SYSTEM CAPABILITIES, never a constant.
// ------------------------------------------------------------------------------------------------
// The reuse CLIFF (WASTE Gate-5): retention is impossible until the cache exceeds ONE TOKEN's working
// set. So the budget is a FUNCTION of the model (per-token working set) and the box (free RAM), not a
// magic GB number. The governor (Rust SubstrateGovernor) supplies free_ram + policy; this is the pure
// math both sides agree on. Returns 0 when the box physically cannot clear the cliff — the signal to
// route to the grid or serve a smaller container (#268 re-pack) instead of paging that can't retain.
struct BudgetInputs {
    uint64_t expert_bytes;        // one expert's resident size (== record_stride for a container)
    uint32_t experts_per_token;   // routed experts summed over all layers/tensor-classes per token
    uint64_t free_ram_bytes;      // measured available RAM (from the governor)
    double   ram_fraction;        // policy: fraction of free RAM the cache may use (e.g. 0.6)
    double   headroom;            // K >= ~2: budget must exceed one-token WS by this to clear the cliff
};

static inline uint64_t moec_one_token_working_set(const BudgetInputs & in) {
    return in.expert_bytes * (uint64_t) in.experts_per_token;
}

// Cache budget in bytes, or 0 if this box cannot retain (below the cliff even using all free RAM).
static inline uint64_t moec_derive_budget(const BudgetInputs & in) {
    const uint64_t ws    = moec_one_token_working_set(in);
    const uint64_t floor = (uint64_t) ((double) ws * in.headroom);   // must clear the cliff
    if (in.free_ram_bytes < floor) { return 0; }                     // physically can't retain here
    uint64_t budget = (uint64_t) (in.ram_fraction * (double) in.free_ram_bytes);
    if (budget < floor)               { budget = floor; }            // stingy policy -> bump to clear cliff
    if (budget > in.free_ram_bytes)   { budget = in.free_ram_bytes; }// never exceed RAM
    return budget;
}

// True when paging can retain on this box; false => route to grid or smaller container.
static inline bool moec_can_retain(const BudgetInputs & in) { return moec_derive_budget(in) > 0; }

} // namespace ggml_moe
