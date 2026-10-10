// moe-pack — pack a (possibly sharded) IQ2 MoE GGUF into an aligned streaming container (K3 #268/#33).
//
// Thin CLI over the pieces: the model/format-specific reading lives in the GgufMoeSource ADAPTER
// (ggml-moe-gguf-source.h); the record framing (Iq2ExpertSource) and container write (moec_pack_dir)
// are general and unit-tested. This file only parses args and wires them together.
//
// Usage:
//   llama-moe-pack --gguf FIRST_SHARD --out DIR --layers N --experts M --top-k K [--layer-base B] [--fmt V]

#include "ggml-moe-gguf-source.h"   // GgufMoeSource adapter (pulls gguf api)
#include "ggml-moe-iq2-source.h"    // Iq2ExpertSource (general record framing)
#include "ggml-moe-packer.h"        // moec_pack_dir (general container write)
#include "ggml-moe-container.h"     // MOEC_Q_IQ2

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <filesystem>

int main(int argc, char ** argv) {
    std::string gguf, out;
    uint32_t layers = 0, experts = 0, top_k = 0, layer_base = 0;
    uint8_t  fmt = 2;   // MOEC_Q_IQ2 — confirmed against MoecQuant in ggml-moe-container.h (M5, 2026-08-03)
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> const char * { return (i + 1 < argc) ? argv[++i] : ""; };
        if      (a == "--gguf")       gguf = next();
        else if (a == "--out")        out = next();
        else if (a == "--layers")     layers = (uint32_t) std::atoi(next());
        else if (a == "--experts")    experts = (uint32_t) std::atoi(next());
        else if (a == "--top-k")      top_k = (uint32_t) std::atoi(next());
        else if (a == "--layer-base") layer_base = (uint32_t) std::atoi(next());
        else if (a == "--fmt")        fmt = (uint8_t) std::atoi(next());
    }
    if (gguf.empty() || out.empty() || !layers || !experts || !top_k) {
        std::fprintf(stderr, "usage: llama-moe-pack --gguf FIRST_SHARD --out DIR --layers N --experts M --top-k K [--layer-base B] [--fmt V]\n");
        return 2;
    }

    // ADAPTER: read experts out of the (sharded) GGUF; MAX-scans for the UD-variable record size.
    ggml_moe::GgufMoeSource gguf_src(gguf, layers, experts, layer_base);
    if (!gguf_src.ok()) { std::fprintf(stderr, "moe-pack: source init failed\n"); return 1; }
    std::fprintf(stderr, "moe-pack: %u layers (base %u) x %u experts, top_k=%u | MAX gate/up/down = %llu/%llu/%llu, record = %llu (%.1fMB)\n",
                 layers, layer_base, experts, top_k,
                 (unsigned long long) gguf_src.max_gate(), (unsigned long long) gguf_src.max_up(),
                 (unsigned long long) gguf_src.max_down(), (unsigned long long) gguf_src.record_stride(),
                 gguf_src.record_stride() / (1024.0 * 1024.0));

    // GENERAL: frame each expert into a WEXP record (Iq2ExpertSource) and write the container.
    // moec_pack_dir writes into `out` assuming it exists — the tool owns creating the output dir.
    std::error_code ec;
    std::filesystem::create_directories(out, ec);
    if (ec) { std::fprintf(stderr, "moe-pack: cannot create out dir %s: %s\n", out.c_str(), ec.message().c_str()); return 1; }
    ggml_moe::Iq2ExpertSource src(fmt, [&](uint32_t L, uint32_t E, ggml_moe::Iq2Bytes & b) { return gguf_src.read(L, E, b); });
    std::fprintf(stderr, "moe-pack: packing -> %s ...\n", out.c_str());
    const bool ok = ggml_moe::moec_pack_dir(out.c_str(), "kimi-k3-iq2", ggml_moe::MOEC_Q_IQ2,
                                            layers, experts, gguf_src.record_stride(), top_k * layers, top_k, src);
    if (!ok) { std::fprintf(stderr, "moe-pack: PACK FAILED\n"); return 1; }
    std::fprintf(stderr, "moe-pack: done. container in %s (manifest.json + experts-L{n}.bin)\n", out.c_str());
    return 0;
}
