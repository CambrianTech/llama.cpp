// what this catches: the prompt cache's disk tier deleting another engine's files (Codex's review of
// llama.cpp#124: two owners of one --cache-disk DIR deleted each other's spills), sweeping a live owner's
// directory, touching files that are not instance directories, and sizing a read from a corrupt file's own
// length claim (an out-of-memory). Model-free.
#include "../tools/server/server-spill-dir.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>

namespace fs = std::filesystem;

static void require(bool condition, const char * message) {
    if (!condition) {
        std::cerr << "FAILED: " << message << '\n';
        std::exit(1);
    }
}

static void two_owners_never_touch_each_other() {
    const fs::path base = fs::temp_directory_path() / ("spill-dir-test-" + std::to_string(std::rand()));
    fs::remove_all(base);
    fs::create_directories(base);
    { std::ofstream(base / "foreign.state") << "not ours"; }      // a file in DIR that is no instance's
    fs::create_directories(base / "notes");                         // a directory that is no instance's
    // an abandoned instance: its .owner exists but nobody holds the lock
    fs::create_directories(base / "instance-1-1");
    { std::ofstream(base / "instance-1-1" / ".owner") << ""; }
    { std::ofstream(base / "instance-1-1" / "0.state") << "stale"; }

    std::error_code ec;
    auto a = std::make_unique<server_spill_dir>();
    require(a->claim(base, ec) == 1 && a->ok(), "the first owner claims its directory and sweeps the abandoned one");
    { std::ofstream(a->path() / "0.state") << "a's spill"; }
    server_spill_dir b;
    require(b.claim(base, ec) == 0 && b.ok(), "a second live owner sweeps nothing");
    require(a->path() != b.path(), "each owner has its own directory");
    require(fs::exists(a->path() / "0.state"), "the second owner never deletes the first owner's spill");
    require(fs::exists(base / "foreign.state") && fs::exists(base / "notes"), "nothing that is not an instance directory is touched");
    require(!fs::exists(base / "instance-1-1"), "an abandoned instance directory is swept");
    const fs::path a_dir = a->path();
    a.reset();
    require(!fs::exists(a_dir), "an owner's directory leaves with it");
    require(b.ok() && fs::exists(b.path()), "and the other owner's stays");
    fs::remove_all(base);
}

static void a_corrupt_length_is_refused_before_allocation() {
    const fs::path p = fs::temp_directory_path() / ("spill-block-" + std::to_string(std::rand()));
    {
        FILE * f = std::fopen(p.string().c_str(), "wb");
        require(server_spill_write_block(f, std::vector<uint8_t>(1000, 7)), "write a block");
        const uint64_t lie = 1ull << 62; // a corrupt length: far beyond the file
        std::fwrite(&lie, sizeof(lie), 1, f);
        std::fclose(f);
    }
    FILE * f = std::fopen(p.string().c_str(), "rb");
    uint64_t remaining = fs::file_size(p);
    std::vector<uint8_t> block;
    require(server_spill_read_block(f, block, remaining) && block.size() == 1000 && block[999] == 7, "a sound block reads back");
    require(!server_spill_read_block(f, block, remaining), "a length beyond the file is refused");
    require(block.size() == 1000, "and nothing was resized from its claim");
    std::fclose(f);
    fs::remove(p);
}

int main() {
    two_owners_never_touch_each_other();
    a_corrupt_length_is_refused_before_allocation();
    std::cout << "test-server-spill-dir: ok\n";
    return 0;
}
