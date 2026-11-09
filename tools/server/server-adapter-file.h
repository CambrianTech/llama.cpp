#pragma once

#include <filesystem>
#include <functional>
#include <string>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

inline bool server_flush_checkpoint_path(const std::filesystem::path & path) {
#ifdef _WIN32
    HANDLE file = CreateFileW(path.c_str(), GENERIC_WRITE, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE) { return false; }
    const bool flushed = FlushFileBuffers(file) != 0;
    const bool closed = CloseHandle(file) != 0;
    return flushed && closed;
#else
    const int file = open(path.c_str(), O_RDONLY);
    if (file < 0) { return false; }
    const bool flushed = fsync(file) == 0;
    const bool closed = close(file) == 0;
    return flushed && closed;
#endif
}

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
    for (const char * name : {"adapter.gguf", "optimizer.gguf", "manifest.json"}) {
        if (!server_flush_checkpoint_path(staging / name)) { return "checkpoint component flush failed"; }
    }
#ifdef _WIN32
    // Same-volume directory publication, with no replacement of an existing generation.
    if (!MoveFileExW(staging.c_str(), fs::path(path).c_str(), MOVEFILE_WRITE_THROUGH)) {
        return "checkpoint publication refused: " + std::to_string(GetLastError());
    }
#else
    if (!server_flush_checkpoint_path(staging)) { return "checkpoint directory flush failed"; }
    fs::rename(staging, path, ec);
    if (ec) { return "checkpoint publication refused: " + ec.message(); }
    if (!server_flush_checkpoint_path(fs::path(path).parent_path())) {
        return "checkpoint published but parent flush unconfirmed; retain prior cursor";
    }
#endif
    return {};
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
