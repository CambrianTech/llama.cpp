#pragma once
// ggml-moe-gguf-source.h — GGUF MoE expert reader ADAPTER (model/format-specific, separated).
//
// The general pieces are elsewhere: Iq2ExpertSource composes the WEXP record framing, moec_pack_dir
// writes the container. This adapter owns the parts that are specific to reading experts OUT of a
// (possibly sharded) GGUF: the standard llama.cpp MoE tensor naming (blk.N.ffn_{gate,up,down}_exps.
// weight), split resolution across shards, per-expert slicing, and the UD-quant MAX-scan (expert byte
// sizes vary per layer, so the record must be sized to the largest). Anything model/deployment-specific
// (leading dense layers, expert count, top_k) enters as PARAMETERS, never hardcoded — so this reader
// serves any GGUF MoE, and a different weight source (safetensors, a remote store) is a sibling adapter.
//
// Requires the gguf API => real-build only (not the standalone unit-test harness, which tests the pure
// framing/slice math directly). Provides the Iq2Bytes provider that Iq2ExpertSource consumes.

#include "ggml.h"
#include "gguf.h"
#include "ggml-moe-container.h"    // moec_align_up
#include "ggml-moe-wexp.h"         // WEXP_HEADER_BYTES
#include "ggml-moe-iq2-source.h"   // Iq2Bytes, ExpertSlice, moec_expert_slice

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

#if defined(_WIN32)
#  define GGUFSRC_FSEEK64(f, off) (_fseeki64((f), (long long)(off), SEEK_SET))
#else
#  define GGUFSRC_FSEEK64(f, off) (fseeko((f), (off_t)(off), SEEK_SET))
#endif

namespace ggml_moe {

class GgufMoeSource {
    struct Shard { gguf_context * ctx = nullptr; FILE * f = nullptr; };
    struct Loc   { int shard = -1; uint64_t off = 0; uint64_t size = 0; };

    std::vector<Shard> shards_;
    uint32_t experts_ = 0, layer_base_ = 0;
    uint64_t max_gate_ = 0, max_up_ = 0, max_down_ = 0, stride_ = 0;
    std::vector<uint8_t> gbuf_, ubuf_, dbuf_;
    bool ok_ = false;

    // Standard llama.cpp MoE expert tensor name (Mixtral/Qwen/DeepSeek/Kimi all share it).
    static std::string tname(uint32_t blk, const char * which) {
        return "blk." + std::to_string(blk) + ".ffn_" + which + "_exps.weight";
    }
    bool locate(const std::string & name, Loc & loc) const {
        for (int si = 0; si < (int) shards_.size(); si++) {
            const int64_t id = gguf_find_tensor(shards_[si].ctx, name.c_str());
            if (id < 0) { continue; }
            loc.shard = si;
            loc.off   = gguf_get_data_offset(shards_[si].ctx) + gguf_get_tensor_offset(shards_[si].ctx, id);
            loc.size  = gguf_get_tensor_size(shards_[si].ctx, id);
            return true;
        }
        return false;
    }
    static bool read_at(FILE * f, uint64_t off, void * dst, size_t len) {
        if (GGUFSRC_FSEEK64(f, off) != 0) { return false; }
        return std::fread(dst, 1, len, f) == len;
    }
    bool open_shards(const std::string & first) {
        const std::string pat = "-of-";
        const size_t of = first.rfind(pat);
        std::vector<std::string> files;
        if (of == std::string::npos) { files.push_back(first); }
        else {
            const int total = std::atoi(first.substr(of + pat.size(), 5).c_str());
            const size_t idx0 = first.rfind('-', of - 1);
            const std::string prefix = first.substr(0, idx0 + 1), suffix = first.substr(of);
            for (int i = 1; i <= total; i++) { char n[8]; std::snprintf(n, sizeof(n), "%05d", i); files.push_back(prefix + n + suffix); }
        }
        for (const auto & p : files) {
            Shard s; gguf_init_params gp{}; gp.no_alloc = true; gp.ctx = nullptr;
            s.ctx = gguf_init_from_file(p.c_str(), gp);
            if (!s.ctx) { std::fprintf(stderr, "gguf-source: parse fail %s\n", p.c_str()); return false; }
            s.f = std::fopen(p.c_str(), "rb");
            if (!s.f) { std::fprintf(stderr, "gguf-source: open fail %s\n", p.c_str()); return false; }
            shards_.push_back(s);
        }
        return true;
    }

public:
    // first_shard: "*-00001-of-000NN.gguf" or a single file. layer_base = leading dense blocks (K3: 1).
    GgufMoeSource(const std::string & first_shard, uint32_t layers, uint32_t experts, uint32_t layer_base)
        : experts_(experts), layer_base_(layer_base) {
        if (!open_shards(first_shard)) { return; }
        for (uint32_t L = 0; L < layers; L++) {                        // MAX-scan (UD variable sizes)
            const uint32_t blk = layer_base_ + L;
            Loc g, u, d;
            if (!locate(tname(blk, "gate"), g) || !locate(tname(blk, "up"), u) || !locate(tname(blk, "down"), d)) {
                std::fprintf(stderr, "gguf-source: could not locate expert tensors for blk %u\n", blk); return;
            }
            if (g.size % experts_ || u.size % experts_ || d.size % experts_) {
                std::fprintf(stderr, "gguf-source: blk %u tensor not divisible by %u experts\n", blk, experts_); return;
            }
            if (g.size / experts_ > max_gate_) max_gate_ = g.size / experts_;
            if (u.size / experts_ > max_up_)   max_up_   = u.size / experts_;
            if (d.size / experts_ > max_down_) max_down_ = d.size / experts_;
        }
        stride_ = moec_align_up(WEXP_HEADER_BYTES + max_gate_ + max_up_ + max_down_);
        gbuf_.resize(max_gate_); ubuf_.resize(max_up_); dbuf_.resize(max_down_);
        ok_ = true;
    }
    ~GgufMoeSource() { for (auto & s : shards_) { if (s.f) std::fclose(s.f); if (s.ctx) gguf_free(s.ctx); } }

    bool     ok()            const { return ok_; }
    uint64_t record_stride() const { return stride_; }
    uint64_t max_gate()      const { return max_gate_; }
    uint64_t max_up()        const { return max_up_; }
    uint64_t max_down()      const { return max_down_; }

    // The provider Iq2ExpertSource consumes: read (container-layer L, expert E)'s three slices into
    // the reusable buffers. Container layer L maps to GGUF block layer_base + L.
    bool read(uint32_t L, uint32_t E, Iq2Bytes & b) {
        const uint32_t blk = layer_base_ + L;
        Loc g, u, d;
        if (!locate(tname(blk, "gate"), g) || !locate(tname(blk, "up"), u) || !locate(tname(blk, "down"), d)) {
            std::fprintf(stderr, "gguf-source: LOCATE FAIL blk %u exp %u\n", blk, E); return false; }
        ExpertSlice gs, us, ds;
        if (!moec_expert_slice(g.size, experts_, E, gs) || !moec_expert_slice(u.size, experts_, E, us) ||
            !moec_expert_slice(d.size, experts_, E, ds)) { std::fprintf(stderr, "gguf-source: SLICE FAIL blk %u exp %u\n", blk, E); return false; }
        if (!read_at(shards_[g.shard].f, g.off + gs.offset, gbuf_.data(), gs.len) ||
            !read_at(shards_[u.shard].f, u.off + us.offset, ubuf_.data(), us.len) ||
            !read_at(shards_[d.shard].f, d.off + ds.offset, dbuf_.data(), ds.len)) {
            std::fprintf(stderr, "gguf-source: READ FAIL blk %u exp %u\n", blk, E); return false; }
        b = { gbuf_.data(), (size_t) gs.len, ubuf_.data(), (size_t) us.len, dbuf_.data(), (size_t) ds.len };
        return true;
    }
};

} // namespace ggml_moe
