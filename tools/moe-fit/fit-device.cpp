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

    // SENSITIVITY-AWARE MIXED PRECISION — "some high fidelity, others low" (Joel). NOT a uniform
    // down-quant: keep the SENSITIVE tensors sharp and spend the bit-savings on the insensitive bulk.
    // Rate-distortion / the all-star-cruft principle applied to the RESIDENT tier. v1 = a NAME heuristic
    // (the known sensitivity ordering); v2 feeds MEASURED per-tensor sensitivity (perturb -> perplexity/KL
    // delta — the same sensor the expert allocator uses). --target-type is now just the LOW-class floor.
    (void) target;
    auto sens_class = [](const std::string & n) -> int {   // 2=HIGH keep sharp, 1=MED, 0=LOW = the bulk
        if (n.find("norm") != std::string::npos) return 2;                          // tiny + very sensitive
        if (n.find("token_embd") != std::string::npos || n.find("output") != std::string::npos) return 2;
        if (n.find("attn_output") != std::string::npos || n.find("attn_v") != std::string::npos) return 2;
        if (n.find("attn_q") != std::string::npos || n.find("attn_k") != std::string::npos ||
            n.find("ffn_down") != std::string::npos) return 1;
        return 0;                                                                    // ffn_gate/up + misc
    };
    auto bpw_of = [](const std::string & ty) -> double {
        return ty == "keep" ? 99.0 : ty == "q8_0" ? 8.5 : ty == "q6_K" ? 6.6 : ty == "q5_K" ? 5.5 :
               ty == "q4_K" ? 4.5 : ty == "q3_K" ? 3.4 : ty == "iq3_xxs" ? 3.1 : ty == "iq2_xxs" ? 2.1 : 4.5;
    };
    // per-class target LADDERS (sharpest first). Successive steps squeeze the LOW class first, then MED,
    // and only nudge HIGH as a last resort — so fidelity is preserved where it matters.
    const char * ladders[3][4] = {
        { "q4_K", "q3_K", "iq3_xxs", "iq2_xxs" },   // LOW  (0): the bulk, most aggressive
        { "q5_K", "q4_K", "q3_K",    "q3_K"    },   // MED  (1)
        { "keep", "q6_K", "q6_K",    "q5_K"    },   // HIGH (2): keep sharp; only nudge if desperate
    };
    for (int step = 0; step < 4; step++) {
        std::vector<std::string> plan;
        uint64_t projected = 0, kept_hi = 0;
        for (const auto & t : resident) {
            const int c = sens_class(t.name);
            const std::string tgt = ladders[c][step];
            const double cur_bpw = (double) ggml_type_size(t.type) * 8.0 / (double) ggml_blck_size(t.type);
            if (tgt == "keep" || bpw_of(tgt) >= cur_bpw) { projected += t.bytes; if (c == 2) kept_hi += t.bytes; continue; }
            projected += (uint64_t) (t.bytes * (bpw_of(tgt) / cur_bpw));
            plan.push_back(t.name + "=" + tgt);
        }
        const bool fits = projected <= budget;
        if (fits || step == 3) {
            std::printf("\nMIXED plan (step %d, %s): %zu tensors down-quant, ~%.1f GB HIGH-class kept sharp, resident ~%.1f GB (budget %.1f GB)%s\n",
                        step, fits ? "FITS" : "best-effort", plan.size(), kept_hi / GB, projected / GB, vram_gb,
                        fits ? "" : "  [still over — raise --vram-gb or lower the HIGH class]");
            if (!out.empty()) {
                FILE * f = std::fopen(out.c_str(), "wb");
                if (f) { for (auto & l : plan) std::fprintf(f, "%s\n", l.c_str()); std::fclose(f);
                         std::printf("wrote --tensor-type-file: %s (%zu overrides; unlisted tensors keep their type)\n", out.c_str(), plan.size()); }
            } else {
                std::printf("\n--tensor-type-file lines:\n");
                for (auto & l : plan) std::printf("  %s\n", l.c_str());
            }
            break;
        }
    }
    return 0;
}
