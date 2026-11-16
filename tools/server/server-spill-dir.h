#pragma once

// Continuum fork: the prompt cache's disk tier keeps its files in a directory this engine instance owns
// (Codex's review of llama.cpp#124: two engines sharing one --cache-disk DIR deleted each other's files).
//
// - Each instance writes only under DIR/instance-<pid>-<start ns>/, and holds an exclusive lock on its
//   `.owner` file there for as long as it lives (the OS releases it if the process dies).
// - At start, an instance removes only instance directories whose `.owner` lock it can take: their owner
//   is gone. A live owner's directory, and anything else in DIR, is never touched.
// - Spill files hold length-prefixed blocks; a read checks every length against the bytes left in the
//   file BEFORE allocating, so a corrupt or truncated file is refused, never an out-of-memory.
//
// The OS lock is the only platform-specific part (server_owner_lock below).

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <string>
#include <system_error>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#endif

// An exclusive, non-blocking lock on one file, released when this object or its process ends.
class server_owner_lock {
public:
    server_owner_lock() = default;
    server_owner_lock(const server_owner_lock &) = delete;
    server_owner_lock & operator=(const server_owner_lock &) = delete;
    ~server_owner_lock() { release(); }

    // true when this process now holds the lock on `path` (created if absent)
    bool try_take(const std::filesystem::path & path) {
        release();
#if defined(_WIN32)
        // no sharing: a second open of a file another process holds open fails
        handle = CreateFileW(path.wstring().c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr, OPEN_ALWAYS,
                             FILE_ATTRIBUTE_NORMAL, nullptr);
        return handle != INVALID_HANDLE_VALUE;
#else
        fd = ::open(path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0600);
        if (fd < 0) {
            return false;
        }
        if (::flock(fd, LOCK_EX | LOCK_NB) != 0) {
            ::close(fd);
            fd = -1;
            return false;
        }
        return true;
#endif
    }

    void release() {
#if defined(_WIN32)
        if (handle != INVALID_HANDLE_VALUE) {
            CloseHandle(handle);
            handle = INVALID_HANDLE_VALUE;
        }
#else
        if (fd >= 0) {
            ::close(fd); // closing releases the flock
            fd = -1;
        }
#endif
    }

private:
#if defined(_WIN32)
    HANDLE handle = INVALID_HANDLE_VALUE;
#else
    int fd = -1;
#endif
};

class server_spill_dir {
public:
    // Claims an instance directory under `base` and sweeps instance directories whose owner is gone.
    // Returns the number swept; `ok()` says whether this instance holds its directory.
    size_t claim(const std::filesystem::path & base, std::error_code & ec) {
        std::filesystem::create_directories(base, ec);
        if (ec) {
            return 0;
        }
        const auto ns = std::chrono::steady_clock::now().time_since_epoch().count();
#if defined(_WIN32)
        const unsigned long pid = GetCurrentProcessId();
#else
        const long pid = (long) ::getpid();
#endif
        dir = base / ("instance-" + std::to_string(pid) + "-" + std::to_string(ns));
        std::filesystem::create_directories(dir, ec);
        if (ec || !owner.try_take(dir / ".owner")) {
            dir.clear();
            return 0;
        }
        size_t swept = 0;
        for (const auto & entry : std::filesystem::directory_iterator(base, ec)) {
            const auto name = entry.path().filename().string();
            if (!entry.is_directory() || name.rfind("instance-", 0) != 0 || entry.path() == dir) {
                continue; // anything that is not an instance directory is never ours to touch
            }
            server_owner_lock probe;
            if (!probe.try_take(entry.path() / ".owner")) {
                continue; // a live owner holds it
            }
            probe.release();
            std::error_code rm;
            std::filesystem::remove_all(entry.path(), rm);
            swept += rm ? 0 : 1;
        }
        return swept;
    }

    bool ok() const { return !dir.empty(); }
    const std::filesystem::path & path() const { return dir; }

    ~server_spill_dir() {
        if (ok()) {
            owner.release();
            std::error_code ec;
            std::filesystem::remove_all(dir, ec); // this instance's files leave with it
        }
    }

private:
    std::filesystem::path dir;
    server_owner_lock     owner;
};

// Length-prefixed blocks. `remaining` is the bytes left in the file; a length beyond it is refused
// before anything is allocated.
inline bool server_spill_write_block(FILE * f, const std::vector<uint8_t> & data) {
    const uint64_t n = data.size();
    return fwrite(&n, sizeof(n), 1, f) == 1 && (n == 0 || fwrite(data.data(), 1, n, f) == n);
}

inline bool server_spill_read_block(FILE * f, std::vector<uint8_t> & data, uint64_t & remaining) {
    uint64_t n = 0;
    if (remaining < sizeof(n) || fread(&n, sizeof(n), 1, f) != 1) {
        return false;
    }
    remaining -= sizeof(n);
    if (n > remaining) {
        return false; // a corrupt or truncated file: refused, never sized from its own claim
    }
    data.resize(n);
    if (n != 0 && fread(data.data(), 1, n, f) != n) {
        return false;
    }
    remaining -= n;
    return true;
}
