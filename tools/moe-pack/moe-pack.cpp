// moe-pack — pack a (possibly sharded) IQ2 MoE GGUF into an aligned streaming container (K3 #268/#33).
//
// The rung-1 fast path: byte-copy the already-IQ2-quantized experts into 16KiB-aligned records so the
// serving fetcher does one aligned pread per expert (GB/s) instead of scattered mmap faults (~200MB/s).
// No re-quant, no RVQ — smaller footprint than WASTE + alignment. Output is M5's reader format
// (manifest.json + per-layer experts-L{n}.bin banks). All the framing/offset/slice math is unit-tested
// in tests/test-moe-residency.cpp; this tool just wires the gguf API to those proven pieces.
//
// Usage:
//   llama-moe-pack --gguf model-00001-of-00016.gguf --out DIR --layers N --experts M --top-k K [--fmt V]

#include "ggml.h"
#include "gguf.h"
#include "ggml-moe-packer.h"
#include "ggml-moe-iq2-source.h"
#include "ggml-moe-container.h"
#include "ggml-moe-wexp.h"

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <filesystem>

#if defined(_WIN32)
#  define MOEC_FSEEK64(f, off) (_fseeki64((f), (long long)(off), SEEK_SET))
#else
#  define MOEC_FSEEK64(f, off) (fseeko((f), (off_t)(off), SEEK_SET))
#endif

namespace fs = std::filesystem;

// K3 expert weight tensor name for (layer, which in {gate,up,down}).
static std::string wname(int layer, const char * which) {
    return "blk." + std::to_string(layer) + ".ffn_" + which + "_exps.weight";
}

// One shard: its gguf metadata context + an open FILE for reading tensor bytes.
struct Shard { gguf_context * ctx = nullptr; FILE * f = nullptr; std::string path; };

// A resolved tensor: which shard, and its absolute byte offset + size in that shard's file.
struct TensorLoc { int shard = -1; uint64_t file_off = 0; uint64_t size = 0; };

// Derive the shard set from a "*-00001-of-000NN.gguf" path (or a single file). Opens all, builds a
// name->location map. Returns false on any open/parse failure (fail loud — no silent partial packs).
static bool open_shards(const std::string & first, std::vector<Shard> & shards) {
    // find "-%05d-of-%05d" — if absent, single file.
    const std::string pat = "-of-";
    const size_t of = first.rfind(pat);
    std::vector<std::string> files;
    if (of == std::string::npos) {
        files.push_back(first);
    } else {
        // parse total count from the 5 digits after "-of-"
        const int total = std::atoi(first.substr(of + pat.size(), 5).c_str());
        const size_t idx0 = first.rfind('-', of - 1);   // start of the NNNNN before "-of-"
        const std::string prefix = first.substr(0, idx0 + 1);
        const std::string suffix = first.substr(of);    // "-of-000NN.gguf"
        for (int i = 1; i <= total; i++) {
            char num[8]; std::snprintf(num, sizeof(num), "%05d", i);
            files.push_back(prefix + num + suffix);
        }
    }
    for (const auto & p : files) {
        Shard s; s.path = p;
        gguf_init_params gp{}; gp.no_alloc = true; gp.ctx = nullptr;
        s.ctx = gguf_init_from_file(p.c_str(), gp);
        if (!s.ctx) { std::fprintf(stderr, "moe-pack: failed to parse gguf shard %s\n", p.c_str()); return false; }
        s.f = std::fopen(p.c_str(), "rb");
        if (!s.f) { std::fprintf(stderr, "moe-pack: failed to open shard %s\n", p.c_str()); return false; }
        shards.push_back(s);
    }
    return true;
}

// Resolve a tensor by name across all shards to its file offset + size.
static bool locate(const std::vector<Shard> & shards, const std::string & name, TensorLoc & loc) {
    for (int si = 0; si < (int) shards.size(); si++) {
        const int64_t id = gguf_find_tensor(shards[si].ctx, name.c_str());
        if (id < 0) { continue; }
        loc.shard    = si;
        loc.file_off = gguf_get_data_offset(shards[si].ctx) + gguf_get_tensor_offset(shards[si].ctx, id);
        loc.size     = gguf_get_tensor_size(shards[si].ctx, id);
        return true;
    }
    return false;
}

static bool read_at(FILE * f, uint64_t off, void * dst, size_t len) {
    if (MOEC_FSEEK64(f, off) != 0) { return false; }
    return std::fread(dst, 1, len, f) == len;
}

int main(int argc, char ** argv) {
    std::string gguf, out;
    uint32_t layers = 0, experts = 0, top_k = 0, layer_base = 0;   // layer_base = leading dense blocks (K3: 1)
    uint8_t  fmt = 6;   // IQ2 placeholder until M5 confirms the WEXP-enum value (metadata only for CUDA serve)
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() { return (i + 1 < argc) ? argv[++i] : ""; };
        if      (a == "--gguf")    gguf = next();
        else if (a == "--out")     out = next();
        else if (a == "--layers")  layers = (uint32_t) std::atoi(next());
        else if (a == "--experts") experts = (uint32_t) std::atoi(next());
        else if (a == "--top-k")   top_k = (uint32_t) std::atoi(next());
        else if (a == "--layer-base") layer_base = (uint32_t) std::atoi(next());
        else if (a == "--fmt")     fmt = (uint8_t) std::atoi(next());
    }
    if (gguf.empty() || out.empty() || !layers || !experts || !top_k) {
        std::fprintf(stderr, "usage: llama-moe-pack --gguf FIRST_SHARD --out DIR --layers N --experts M --top-k K [--fmt V]\n");
        return 2;
    }

    std::vector<Shard> shards;
    if (!open_shards(gguf, shards)) { return 1; }
    std::fprintf(stderr, "moe-pack: %zu shard(s), %u layers x %u experts, top_k=%u\n",
                 shards.size(), layers, experts, top_k);

    // Determine the record stride from layer 0's gate/up/down sizes (uniform across K3 layers). Each
    // expert record = WEXP header + gate|up|down slice, aligned up to 4KiB (also satisfies 16KiB).
    TensorLoc g0, u0, d0;
    if (!locate(shards, wname((int) layer_base, "gate"), g0) ||
        !locate(shards, wname((int) layer_base, "up"),   u0) ||
        !locate(shards, wname((int) layer_base, "down"), d0)) {
        std::fprintf(stderr, "moe-pack: could not locate layer-0 expert tensors\n"); return 1;
    }
    const uint64_t gate_per = g0.size / experts, up_per = u0.size / experts, down_per = d0.size / experts;
    const uint64_t payload  = ggml_moe::WEXP_HEADER_BYTES + gate_per + up_per + down_per;
    const uint64_t stride   = ggml_moe::moec_align_up(payload);
    std::fprintf(stderr, "moe-pack: gate/up/down per-expert = %llu/%llu/%llu, record stride = %llu (%.1fMB)\n",
                 (unsigned long long) gate_per, (unsigned long long) up_per, (unsigned long long) down_per,
                 (unsigned long long) stride, stride / (1024.0 * 1024.0));

    fs::create_directories(out);

    // Per-expert scratch buffers reused across the pack.
    std::vector<uint8_t> gbuf(gate_per), ubuf(up_per), dbuf(down_per);

    // The provider: for (layer, expert) read the three IQ2 slices from the source shards. Uses the
    // unit-tested moec_expert_slice for the in-tensor offset — no layout guessing.
    auto provider = [&](uint32_t L, uint32_t E, ggml_moe::Iq2Bytes & b) -> bool {
        TensorLoc gl, ul, dl;
        if (!locate(shards, wname((int)(layer_base + L), "gate"), gl) ||
            !locate(shards, wname((int)(layer_base + L), "up"),   ul) ||
            !locate(shards, wname((int)(layer_base + L), "down"), dl)) { return false; }
        ggml_moe::ExpertSlice gs, us, ds;
        if (!ggml_moe::moec_expert_slice(gl.size, experts, E, gs) ||
            !ggml_moe::moec_expert_slice(ul.size, experts, E, us) ||
            !ggml_moe::moec_expert_slice(dl.size, experts, E, ds)) { return false; }
        if (gs.len > gbuf.size() || us.len > ubuf.size() || ds.len > dbuf.size()) { return false; }
        if (!read_at(shards[gl.shard].f, gl.file_off + gs.offset, gbuf.data(), gs.len) ||
            !read_at(shards[ul.shard].f, ul.file_off + us.offset, ubuf.data(), us.len) ||
            !read_at(shards[dl.shard].f, dl.file_off + ds.offset, dbuf.data(), ds.len)) { return false; }
        b = { gbuf.data(), (size_t) gs.len, ubuf.data(), (size_t) us.len, dbuf.data(), (size_t) ds.len };
        return true;
    };

    ggml_moe::Iq2ExpertSource src(fmt, provider);
    std::fprintf(stderr, "moe-pack: packing -> %s ...\n", out.c_str());
    const bool ok = ggml_moe::moec_pack_dir(out.c_str(), "kimi-k3-iq2", ggml_moe::MOEC_Q_IQ2,
                                            layers, experts, stride, top_k * layers, top_k, src);

    for (auto & s : shards) { if (s.f) std::fclose(s.f); if (s.ctx) gguf_free(s.ctx); }
    if (!ok) { std::fprintf(stderr, "moe-pack: PACK FAILED\n"); return 1; }
    std::fprintf(stderr, "moe-pack: done. container in %s (manifest.json + experts-L{n}.bin)\n", out.c_str());
    return 0;
}
