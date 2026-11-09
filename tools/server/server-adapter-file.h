#pragma once

#include <filesystem>
#include <functional>
#include <string>

// One directory rename publishes every component of an immutable generation.
// A failed write leaves no public generation; a process crash retains its reservation.
inline std::string server_publish_adapter_generation(const std::string & path,
        const std::function<bool(const std::string &)> & write) {
    namespace fs = std::filesystem;
    const fs::path staging = path + ".pending";
    std::error_code ec;
    if (!fs::create_directory(staging, ec)) {
        return "checkpoint publication already reserved or unavailable: " + staging.string();
    }
    struct cleanup {
        fs::path dir;
        ~cleanup() {
            std::error_code ignored;
            for (const char * name : {"adapter.gguf", "optimizer.gguf", "manifest.json"}) {
                fs::remove(dir / name, ignored);
            }
            fs::remove(dir, ignored);
        }
    } guard{staging};
    if (fs::exists(path, ec) || ec) { return "immutable checkpoint already exists or is unavailable"; }
    if (!write(staging.string())) { return "checkpoint serialization failed"; }
    fs::rename(staging, path, ec);
    return ec ? "checkpoint publication refused: " + ec.message() : std::string();
}

// Publish a complete adapter without replacing an existing snapshot. The private
// directory reserves this destination across concurrent fresh/save requests.
// A crash leaves a named reservation for diagnosis, never a partial public GGUF.
inline std::string server_publish_adapter(const std::string & path,
                                         const std::function<bool(const std::string &)> & write) {
    namespace fs = std::filesystem;
    const fs::path staging = path + ".pending";
    std::error_code ec;
    if (!fs::create_directory(staging, ec)) {
        return "adapter publication already reserved or unavailable: " + staging.string();
    }
    struct cleanup {
        fs::path dir;
        ~cleanup() {
            std::error_code ignored;
            fs::remove(dir / "adapter.gguf", ignored);
            fs::remove(dir, ignored);
        }
    } guard{staging};
    const auto temporary = staging / "adapter.gguf";
    if (!write(temporary.string())) {
        return "adapter serialization failed";
    }
    // Unlike rename on POSIX, hard-link creation cannot replace a destination.
    // Both paths live on the train-dir filesystem. Unsupported filesystems refuse.
    fs::create_hard_link(temporary, path, ec);
    if (ec) {
        return "immutable adapter publication refused: " + ec.message();
    }
    return {};
}
