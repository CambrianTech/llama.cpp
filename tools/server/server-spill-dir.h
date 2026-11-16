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


// An exclusive, non-blocking lock on one file, released when this object or its process ends. The OS calls
// live in server-spill-dir.cpp: this header includes no platform header (Codex on #124: windows.h here
// pulled winsock.h in ahead of Winsock2 for every includer).
class server_owner_lock {
public:
    server_owner_lock() = default;
    server_owner_lock(const server_owner_lock &) = delete;
    server_owner_lock & operator=(const server_owner_lock &) = delete;
    ~server_owner_lock() { release(); }

    // true when this process now holds the lock on `path`. `create`: make the file (exclusively: it
    // must not exist yet); otherwise open only an existing file, never creating a marker for anyone.
    bool try_take(const std::filesystem::path & path, bool create);
    void release();

private:
    intptr_t handle = -1; // a POSIX descriptor or a Windows HANDLE; -1 = none
};

// This process's id (server-spill-dir.cpp).
uint64_t server_process_id();

// The instance directory's protocol. The marker is published before the directory and only while
// locked, so a sweeper can never see a live owner unlocked (Codex on #124: a claim raced a sweep):
//   owner:   create <name>.owner.tmp exclusively, lock it, rename it to <name>.owner (atomic; the lock
//            stays on the open file), then create <name>/;
//   sweeper: for each final <name>.owner (never a .tmp, never anything else), open it WITHOUT creating
//            and try the lock; taken means its owner is gone: remove <name>/, then the marker. A
//            directory without a marker, a .tmp, or any other file is never touched.
class server_spill_dir {
public:
    // Claims an instance directory under `base` and sweeps instances whose owner is gone. Returns the
    // number swept; `ok()` says whether this instance holds its directory.
    size_t claim(const std::filesystem::path & base, std::error_code & ec) {
        std::filesystem::create_directories(base, ec);
        if (ec) {
            return 0;
        }
        const auto ns = std::chrono::steady_clock::now().time_since_epoch().count();
        const std::string name = "instance-" + std::to_string(server_process_id()) + "-" + std::to_string(ns);
        const auto tmp    = base / (name + ".owner.tmp");
        const auto marker = base / (name + ".owner");
        if (!owner.try_take(tmp, /*create=*/true)) {
            return 0;
        }
        std::filesystem::rename(tmp, marker, ec);
        if (ec) {
            owner.release();
            std::error_code rm;
            std::filesystem::remove(tmp, rm);
            return 0;
        }
        dir = base / name;
        std::filesystem::create_directories(dir, ec);
        if (ec) {
            owner.release();
            std::error_code rm;
            std::filesystem::remove(marker, rm);
            dir.clear();
            return 0;
        }
        owner_marker = marker;
        // collect first: removing entries while iterating the same directory is unspecified
        std::vector<std::filesystem::path> markers;
        for (const auto & entry : std::filesystem::directory_iterator(base, ec)) {
            markers.push_back(entry.path());
        }
        size_t swept = 0;
        for (const auto & path : markers) {
            const auto file = path.filename().string();
            const std::string suffix = ".owner";
            if (file.rfind("instance-", 0) != 0 || file.size() <= suffix.size() ||
                file.compare(file.size() - suffix.size(), suffix.size(), suffix) != 0 || path == marker) {
                continue; // only a final owner marker names an instance a sweeper may judge
            }
            server_owner_lock probe;
            if (!probe.try_take(path, /*create=*/false)) {
                continue; // a live owner holds it (or it vanished meanwhile)
            }
            const auto instance = base / file.substr(0, file.size() - suffix.size());
            std::error_code rm;
            std::filesystem::remove_all(instance, rm);
            if (!rm) {
                std::filesystem::remove(path, rm); // the marker last: a crash in between is swept again
                swept += rm ? 0 : 1;
            }
        }
        return swept;
    }

    bool ok() const { return !dir.empty(); }
    const std::filesystem::path & path() const { return dir; }

    ~server_spill_dir() {
        if (ok()) {
            std::error_code ec;
            std::filesystem::remove_all(dir, ec); // this instance's files leave with it
            std::filesystem::remove(owner_marker, ec);
            owner.release();
        }
    }

private:
    std::filesystem::path dir;
    std::filesystem::path owner_marker;
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
