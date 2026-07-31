// fit-device — the repeatable "fit the model to the hardware" primitive (misfit-design foundry step).
//
// Given a (possibly sharded) GGUF and a target VRAM budget, it classifies every tensor as RESIDENT
// (non-expert: attention / embeddings / output / dense / norms — must live in VRAM for fast compute) vs
// PAGED (the *_exps* experts — streamed from disk by the pager, left as-is), sums the resident footprint
// at its current quant, and if it exceeds the VRAM budget emits a per-tensor DOWN-QUANT plan (largest
// resident tensors first) as a --tensor-type-file for llama-quantize. Repeatable: one command per
// (model, device) — no hand-picked flags. This is the device-targeted-compaction planner; the actual
// requant is a separate llama-quantize invocation fed the emitted file (and, done properly, sources the
// full-precision model + a compensation-LoRA — this planner is quant-source-agnostic).
//
// Usage: llama-fit-device --gguf FIRST_SHARD --vram-gb N [--target-type q4_K] [--out plan.txt]

#include "ggml.h"
#include "gguf.h"
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>

static bool is_expert(const std::string & n) { return n.find("_exps.") != std::string::npos; }

struct Shard { gguf_context * ctx = nullptr; };

static std::vector<std::string> shard_paths(const std::string & first) {
    const std::string pat = "-of-";
    const size_t of = first.rfind(pat);
    std::vector<std::string> files;
    if (of == std::string::npos) { files.push_back(first); return files; }
    const int total = std::atoi(first.substr(of + pat.size(), 5).c_str());
    const size_t idx0 = first.rfind('-', of - 1);
    const std::string prefix = first.substr(0, idx0 + 1), suffix = first.substr(of);
    for (int i = 1; i <= total; i++) { char b[8]; std::snprintf(b, sizeof(b), "%05d", i); files.push_back(prefix + b + suffix); }
    return files;
}

int main(int argc, char ** argv) {
    std::string gguf, out, target = "q4_K";
    double vram_gb = 0.0;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto nx = [&]{ return (i + 1 < argc) ? argv[++i] : ""; };
        if      (a == "--gguf")        gguf = nx();
        else if (a == "--vram-gb")     vram_gb = std::atof(nx());
        else if (a == "--target-type") target = nx();
        else if (a == "--out")         out = nx();
    }
    if (gguf.empty() || vram_gb <= 0) { std::fprintf(stderr, "usage: --gguf FIRST_SHARD --vram-gb N [--target-type q4_K] [--out plan.txt]\n"); return 2; }

    // one row per RESIDENT tensor: name, current type, bytes
    struct T { std::string name; ggml_type type; uint64_t bytes; };
    std::vector<T> resident;
    uint64_t resident_bytes = 0, paged_bytes = 0;
    uint64_t by_type_bytes[GGML_TYPE_COUNT] = {0};

    for (const auto & p : shard_paths(gguf)) {
        gguf_init_params gp{}; gp.no_alloc = true; gp.ctx = nullptr;
        gguf_context * ctx = gguf_init_from_file(p.c_str(), gp);
        if (!ctx) { std::fprintf(stderr, "fit-device: parse fail %s\n", p.c_str()); return 1; }
        const int64_t n = gguf_get_n_tensors(ctx);
        for (int64_t t = 0; t < n; t++) {
            const char * nm = gguf_get_tensor_name(ctx, t);
            const uint64_t sz = gguf_get_tensor_size(ctx, t);
            if (is_expert(nm)) { paged_bytes += sz; continue; }
            const ggml_type ty = gguf_get_tensor_type(ctx, t);
            resident.push_back({ nm, ty, sz });
            resident_bytes += sz;
            by_type_bytes[ty] += sz;
        }
        gguf_free(ctx);
    }

    const double GB = 1024.0 * 1024.0 * 1024.0;
    std::printf("model: %s\n", gguf.c_str());
    std::printf("PAGED experts   : %.1f GB (streamed by the pager, left as-is)\n", paged_bytes / GB);
    std::printf("RESIDENT (must fit VRAM): %.1f GB across %zu tensors\n", resident_bytes / GB, resident.size());
    std::printf("  by current quant type:\n");
    for (int ty = 0; ty < GGML_TYPE_COUNT; ty++) {
        if (by_type_bytes[ty]) std::printf("    %-10s %.2f GB\n", ggml_type_name((ggml_type) ty), by_type_bytes[ty] / GB);
    }

    const double budget = vram_gb * GB;
    std::printf("\nVRAM budget: %.1f GB  =>  resident %s by %.1f GB\n",
                vram_gb, resident_bytes <= budget ? "FITS, under" : "OVER, must shrink",
                (resident_bytes - budget) / GB);
    if (resident_bytes <= budget) { std::printf("no down-quant needed.\n"); return 0; }

    // greedy plan: down-quant the largest resident tensors first to `target` until under budget.
    // bytes at target ~ elements * bits(target)/8; approximate elements from current bytes/type.
    std::sort(resident.begin(), resident.end(), [](const T & a, const T & b){ return a.bytes > b.bytes; });
    const double tgt_bpw =  // approx bits-per-weight of the target type (coarse; the requant is exact)
        target == "q8_0" ? 8.5 : target == "q6_K" ? 6.6 : target == "q5_K" ? 5.5 :
        target == "q4_K" ? 4.5 : target == "q3_K" ? 3.4 : target == "iq3_xxs" ? 3.1 :
        target == "iq2_xxs" ? 2.1 : 4.5;
    std::vector<std::string> plan;
    uint64_t saved = 0;
    for (const auto & t : resident) {
        if (resident_bytes - saved <= budget) break;
        const double cur_bpw = (double) ggml_type_size(t.type) * 8.0 / (double) ggml_blck_size(t.type);  // exact bpw of the type
        if (tgt_bpw >= cur_bpw) continue;   // target isn't smaller than current — skip
        const uint64_t new_bytes = (uint64_t) (t.bytes * (tgt_bpw / cur_bpw));
        saved += (t.bytes - new_bytes);
        plan.push_back(std::string(t.name) + "=" + target);
    }
    std::printf("\nplan: down-quant %zu resident tensors to %s, saving ~%.1f GB -> resident ~%.1f GB\n",
                plan.size(), target.c_str(), saved / GB, (resident_bytes - saved) / GB);
    if (resident_bytes - saved > budget) std::printf("  WARNING: still over budget at %s; need a smaller target-type\n", target.c_str());

    if (!out.empty()) {
        FILE * f = std::fopen(out.c_str(), "wb");
        if (f) { for (auto & l : plan) std::fprintf(f, "%s\n", l.c_str()); std::fclose(f);
                 std::printf("wrote llama-quantize --tensor-type-file: %s (%zu overrides)\n", out.c_str(), plan.size()); }
    } else {
        std::printf("\n--tensor-type-file lines (feed to llama-quantize):\n");
        for (auto & l : plan) std::printf("  %s\n", l.c_str());
    }
    return 0;
}
