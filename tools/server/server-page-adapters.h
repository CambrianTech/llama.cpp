#pragma once

#include <cmath>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

// Continuum fork: whether a saved page's KV belongs to this engine's adapters.
//
// A page's `.lora` sidecar records every adapter its slot carried, with its scale and the
// file revision (size, mtime) it was computed under. An adapter at scale 0 adds nothing to
// the KV, so the page's identity is the base model plus its ACTIVE adapters (scale != 0).
// Refusing on the full loaded list made every relaunch discard valid pages: an engine that
// had a training adapter attached at 0.0 saved pages a relaunched engine, without it, could
// never restore, and each turn re-prefilled its whole prompt (IntelMac, 2026-11-03).

// One adapter as the page recorded it.
struct server_page_adapter {
    std::string path;
    float       scale;
    uint64_t    size;
    int64_t     mtime;
};

// One adapter as this engine has it loaded: its path and its file's revision now
// (size 0 when the file cannot be read, which proves nothing about its weights).
struct server_loaded_adapter {
    std::string path;
    uint64_t    size;
    int64_t     mtime;
};

// The scale the restored slot runs each LOADED adapter at, in load order, or nullopt with
// `why` when the page's KV is not this engine's. Each active page adapter claims one loaded
// adapter at the same path and revision and keeps the page's scale, so the slot computes
// exactly what the KV holds; every other loaded adapter runs at 0.0, never at its default.
inline std::optional<std::vector<float>> server_page_adapter_scales(
        const std::vector<server_page_adapter> & page,
        const std::vector<server_loaded_adapter> & loaded,
        std::string & why) {
    std::vector<float> scales(loaded.size(), 0.0f);
    std::vector<bool>  claimed(loaded.size(), false);
    for (const auto & p : page) {
        if (!std::isfinite(p.scale)) {
            why = "a recorded scale is not finite: " + p.path;
            return std::nullopt;
        }
        if (p.scale == 0.0f) {
            continue;
        }
        bool matched = false;
        for (size_t i = 0; i < loaded.size() && !matched; ++i) {
            const auto & l = loaded[i];
            if (claimed[i] || l.path != p.path) {
                continue;
            }
            if (l.size == 0 || l.size != p.size || l.mtime != p.mtime) {
                why = "an active adapter's file changed since the page was saved: " + p.path;
                return std::nullopt;
            }
            claimed[i] = true;
            scales[i]  = p.scale;
            matched    = true;
        }
        if (!matched) {
            why = "an active adapter is not loaded: " + p.path;
            return std::nullopt;
        }
    }
    return scales;
}
