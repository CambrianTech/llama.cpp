// what this catches (Continuum fork, server-pages.h + server-page-files.h, no model needed):
// - a page admitted on a count that leaves out part of what it holds (its speculative state), so the
//   store passes its ceiling after a successful page_out;
// - a page out of the store but still held by a spill that stops counting, so the live RAM passes the ceiling;
// - a refused page_out that has already moved the slot's state out (the incumbent is lost);
// - a publication that removes or overwrites the committed files before its own are whole: a failed
//   sidecar write, or a crash between files, that loses the last durable page;
// - a directory sync that fails (stage, commit record, destination) yet reports a publication or drops
//   the commit record before the moved names are durable;
// - a repeated publication over existing page files (rename onto an existing name) that leaves the old one.

#include "server-page-files.h"
#include "server-page-os.h"
#include "server-pages.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <string>


using page_os::current_pid;

#define CHECK(cond) do { if (!(cond)) { fprintf(stderr, "FAILED %s:%d: %s\n", __FILE__, __LINE__, #cond); exit(1); } } while (0)

namespace fs = std::filesystem;

// a page holding `n` bytes in each part, the speculative tail state included
static server_page_ptr make_page(size_t n) {
    auto page = std::make_shared<server_page>();
    page->data.main.assign(n, 1);
    page->data.drft.assign(n, 2);
    page->spec.assign(n, 3);
    return page;
}

// a slot's prompt: tokens and one checkpoint of `n` bytes
static server_prompt make_prompt(size_t n) {
    server_prompt prompt;
    prompt.tokens = server_tokens(llama_tokens{ 1, 2, 3 }, false);
    auto & c = prompt.checkpoints.emplace_back();
    c.data_tgt.assign(n, 4);
    return prompt;
}

static error_type refusal(const std::function<void()> & fn) {
    try {
        fn();
    } catch (const server_page_error & err) {
        return err.type;
    }
    return ERROR_TYPE_SERVER; // stands for "not refused"; every refusal checked here is UNAVAILABLE
}

static void test_admission_counts_what_the_store_charges() {
    const size_t n = 64;
    const size_t bytes = server_page_bytes(2 * n, n, n); // main + draft, speculative, one checkpoint
    {
        server_page_store store;
        store.limit_bytes = bytes; // fits exactly
        auto page = make_page(n);
        auto from = make_prompt(n);
        store.admit("a", bytes);
        store.put("a", page, &from);
        CHECK(page->n_bytes() == bytes);
        CHECK(store.n_bytes() == bytes);
        CHECK(from.checkpoints.empty() && page->prompt.checkpoints.size() == 1);
    }
    {
        server_page_store store;
        store.limit_bytes = bytes - 1; // one byte short: the speculative state alone tips it
        CHECK(refusal([&] { store.admit("a", bytes); }) == ERROR_TYPE_UNAVAILABLE);
        // a put is checked on the same count, so a short pre-count cannot get a page past the ceiling
        auto page = make_page(n);
        auto from = make_prompt(n);
        CHECK(refusal([&] { store.put("a", page, &from); }) == ERROR_TYPE_UNAVAILABLE);
        // the refusal leaves the incumbent: the slot's prompt is where it was, and nothing is stored
        CHECK(from.tokens.size() == 3 && from.checkpoints.size() == 1 && from.checkpoints.front().data_tgt.size() == n);
        CHECK(page->prompt.tokens.size() == 0);
        CHECK(store.n_bytes() == 0 && store.n_pages() == 0 && store.find("a") == nullptr);
    }
}

static void test_spill_held_pages_stay_counted() {
    const size_t n = 64;
    const size_t bytes = server_page_bytes(2 * n, n, n);
    server_page_store store;
    store.limit_bytes = 2 * bytes - 1; // room for one page, not two

    auto from = make_prompt(n);
    store.put("a", make_page(n), &from);
    server_page_ptr spilling = store.find("a"); // a spill's reference on its HTTP thread

    // replaced under its name while spilling: the old page is still live, so the new one does not fit
    auto from2 = make_prompt(n);
    CHECK(refusal([&] { store.put("a", make_page(n), &from2); }) == ERROR_TYPE_UNAVAILABLE);
    CHECK(from2.checkpoints.size() == 1);

    // left the store (page_in, page_drop) while spilling: still counted
    CHECK(store.erase("a"));
    CHECK(store.n_pages() == 0 && store.n_bytes() == bytes);
    CHECK(refusal([&] { store.put("b", make_page(n), &from2); }) == ERROR_TYPE_UNAVAILABLE);

    // the spill finishes: its bytes are released and the page fits
    spilling.reset();
    CHECK(store.n_bytes() == 0);
    store.put("b", make_page(n), &from2);
    CHECK(store.n_bytes() == bytes);

    // a replacement the store solely owns frees the page it replaces, so it fits
    auto from3 = make_prompt(n);
    store.put("b", make_page(n), &from3);
    CHECK(store.n_bytes() == bytes && store.n_pages() == 1);

    // a cleared store (the model unloaded) still counts a page a spill holds
    spilling = store.find("b");
    store.clear();
    CHECK(store.n_bytes() == bytes);
    spilling.reset();
    CHECK(store.n_bytes() == 0);
}

static std::string read_file(const fs::path & p) {
    std::ifstream in(p, std::ios::binary);
    return in ? std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()) : std::string("<absent>");
}

static void write_file(const fs::path & p, const std::string & bytes) {
    std::ofstream out(p, std::ios::binary | std::ios::trunc);
    out << bytes;
}

static server_page_file text_file(const std::string & suffix, const std::string & text) {
    return server_page_sidecar(suffix, [text](const server_page_put & put) { put(text.data(), text.size()); });
}

static const char * const SUFFIXES[] = { "", ".ckpt", ".lora", ".dft" };

static void check_generation(const fs::path & file, const std::string & tag) {
    for (const char * s : SUFFIXES) {
        CHECK(read_file(file.string() + s) == tag + s);
    }
}

// no stage and no commit record is left beside the page
static void check_no_leftovers(const fs::path & dir) {
    const fs::path staging = dir / "..page-staging";
    if (fs::exists(staging)) {
        CHECK(fs::is_empty(staging));
    }
}

static void publish_generation(const fs::path & file, const std::string & tag) {
    std::vector<server_page_file> files;
    for (const char * s : SUFFIXES) {
        files.push_back(text_file(s, tag + s));
    }
    server_page_publish(file.string(), files);
}

// `op` fails on every path ending in `suffix` (and, when given, containing `within`) while this lives
struct page_fault {
    page_fault(server_page_fault op, std::string suffix, std::string within = "") {
        server_page_files_inject_fault([=](server_page_fault o, const std::string & path) {
            return o == op && path.find(within) != std::string::npos && path.size() >= suffix.size() &&
                path.compare(path.size() - suffix.size(), suffix.size(), suffix) == 0;
        });
    }
    ~page_fault() { server_page_files_inject_fault(nullptr); }
};

static bool publish_refused(const fs::path & file, const std::string & tag) {
    try {
        publish_generation(file, tag);
    } catch (const server_page_error &) {
        return true;
    }
    return false;
}

static bool reader_refused(const fs::path & file) {
    try {
        server_page_reader reader(file.string());
    } catch (const server_page_error &) {
        return true;
    }
    return false;
}

[[noreturn]] static void disk_full(const server_page_put &) {
    throw std::runtime_error("disk full");
}

static void test_a_failed_publication_keeps_the_committed_generation(const fs::path & dir) {
    const fs::path file = dir / "mind-a.bin";
    publish_generation(file, "one");
    check_generation(file, "one");

    // the .lora sidecar fails after the state and .ckpt of the new generation were written
    const auto failing = server_page_file{ ".lora", [](const std::string &) -> size_t { return 0; } };
    bool refused = false;
    try {
        server_page_publish(file.string(), { text_file("", "two"), text_file(".ckpt", "two.ckpt"), failing, text_file(".dft", "two.dft") });
    } catch (const server_page_error &) {
        refused = true;
    }
    CHECK(refused);
    { server_page_reader reader(file.string()); check_generation(file, "one"); }
    check_no_leftovers(dir);

    // a writer that throws is the same failure
    refused = false;
    try {
        server_page_publish(file.string(), { text_file("", "two"),
            server_page_sidecar(".ckpt", disk_full) });
    } catch (const server_page_error &) {
        refused = true;
    }
    CHECK(refused);
    check_generation(file, "one");
    check_no_leftovers(dir);

    // a generation without a sidecar removes the older one: a stale .dft is never read beside new state
    server_page_publish(file.string(), { text_file("", "three"), text_file(".ckpt", "three.ckpt"), text_file(".lora", "three.lora") });
    CHECK(read_file(file) == "three" && read_file(file.string() + ".dft") == "<absent>");
    check_no_leftovers(dir);

    // a staged member's status cannot be read after the commit: never taken for already moved, so
    // the record and stage keep custody until a reader can read it, then the generation is whole
    const fs::path record = dir / "..page-staging" / "mind-a.bin.commit";
    {
        page_fault fault(server_page_fault::status, "mind-a.bin.ckpt", "..page-staging");
        CHECK(publish_refused(file, "four"));
        CHECK(fs::exists(record));
        CHECK(reader_refused(file) && fs::exists(record));
    }
    { server_page_reader reader(file.string()); check_generation(file, "four"); }
    check_no_leftovers(dir);
}

static void test_an_interrupted_publication_restores_whole(const fs::path & dir) {
    const fs::path file    = dir / "mind-b.bin";
    const fs::path staging = dir / "..page-staging";
    publish_generation(file, "one");

    // a crash after staging, before the commit record: this process's pid, a stage no publication owns
    const fs::path stage = staging / (std::to_string(current_pid()) + "-999999");
    fs::create_directories(stage);
    write_file(stage / "mind-b.bin", "two");
    write_file(stage / "mind-b.bin.ckpt", "two.ckpt");
    { server_page_reader reader(file.string()); check_generation(file, "one"); }
    check_no_leftovers(dir);

    // a crash after the commit record, with one file moved: the reader completes the generation,
    // and the sidecar the generation lacks (.dft) is removed
    fs::create_directories(stage);
    for (const char * s : { "", ".ckpt", ".lora" }) {
        write_file(stage / (std::string("mind-b.bin") + s), std::string("two") + s);
    }
    fs::rename(stage / "mind-b.bin", file);
    write_file(staging / "mind-b.bin.commit", stage.filename().string() + "\nf\nf.ckpt\nf.lora\n");
    {
        server_page_reader reader(file.string());
        CHECK(read_file(file) == "two" && read_file(file.string() + ".ckpt") == "two.ckpt" && read_file(file.string() + ".lora") == "two.lora");
        CHECK(read_file(file.string() + ".dft") == "<absent>");
    }
    check_no_leftovers(dir);
}

static void test_a_failed_dir_sync_never_reports_or_loses_a_generation(const fs::path & dir) {
    const fs::path file    = dir / "mind-d.bin";
    const fs::path staging = dir / "..page-staging";
    publish_generation(file, "one");

    // the stage's sync fails (every sync fails; the stage's is the first reached): refused before
    // the commit, the committed generation as it was
    {
        page_fault fault(server_page_fault::sync, "");
        CHECK(publish_refused(file, "two"));
    }
    { server_page_reader reader(file.string()); check_generation(file, "one"); }
    check_no_leftovers(dir);

    // the commit record's dir sync fails: refused, but the record keeps custody; the next reader completes it
    {
        page_fault fault(server_page_fault::sync, "..page-staging");
        CHECK(publish_refused(file, "three"));
        CHECK(fs::exists(staging / "mind-d.bin.commit"));
    }
    { server_page_reader reader(file.string()); check_generation(file, "three"); }
    check_no_leftovers(dir);

    // the destination's sync fails after the files moved: refused, the record kept until a sync succeeds
    {
        page_fault fault(server_page_fault::sync, dir.filename().string());
        CHECK(publish_refused(file, "four"));
        CHECK(fs::exists(staging / "mind-d.bin.commit"));
        CHECK(reader_refused(file) && fs::exists(staging / "mind-d.bin.commit"));
    }
    { server_page_reader reader(file.string()); check_generation(file, "four"); }
    check_no_leftovers(dir);
}

// every publication after the first renames onto existing page files: each must replace them
static void test_repeated_publication_replaces_every_file(const fs::path & dir) {
    const fs::path file = dir / "mind-e.bin";
    for (const char * tag : { "one", "two", "three", "four" }) {
        publish_generation(file, tag);
        { server_page_reader reader(file.string()); check_generation(file, tag); }
        check_no_leftovers(dir);
    }
}

static void test_spill_writes_a_page_restore_reads(const fs::path & dir) {
    // a sequence state as llama_state_seq_get_data writes it: [io magic][seq id][state]
    auto page = std::make_shared<server_page>();
    const uint32_t io_magic = 0xaf143cd8;
    page->data.main.resize(sizeof(uint32_t) + sizeof(llama_seq_id) + 16, 7);
    memcpy(page->data.main.data(), &io_magic, sizeof(io_magic));
    page->data.drft.assign(8, 5);
    page->spec.assign(4, 6);
    page->prompt = make_prompt(32);

    const fs::path file = dir / "mind-c.bin";
    const size_t n = server_page_spill(*page, file.string());
    size_t on_disk = 0;
    for (const char * s : SUFFIXES) {
        on_disk += fs::file_size(file.string() + s);
    }
    CHECK(n == on_disk);
    std::vector<uint8_t> dft, spec;
    std::string why;
    CHECK(server_page_read_draft(file.string() + ".dft", dft, spec, why));
    CHECK(dft == page->data.drft && spec == page->spec);
    check_no_leftovers(dir);
}

int main() {
    test_admission_counts_what_the_store_charges();
    test_spill_held_pages_stay_counted();

    const fs::path dir = fs::temp_directory_path() / ("test-server-pages-" + std::to_string(current_pid()));
    fs::remove_all(dir);
    fs::create_directories(dir);
    test_a_failed_publication_keeps_the_committed_generation(dir);
    test_an_interrupted_publication_restores_whole(dir);
    test_a_failed_dir_sync_never_reports_or_loses_a_generation(dir);
    test_repeated_publication_replaces_every_file(dir);
    test_spill_writes_a_page_restore_reads(dir);
    fs::remove_all(dir);

    printf("test-server-pages: OK\n");
    return 0;
}
