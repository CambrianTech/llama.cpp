#pragma once

// Continuum fork: a page on disk, `<file>` (sequence state) and its sidecars `.ckpt` (context
// checkpoints), `.lora` (the adapters the state was computed under) and `.dft` (draft state),
// shared by save, spill and restore.
//
// The files of a page are one generation, published whole: a writer stages every file beside the
// page dir and syncs them, writes a commit record, then moves them over the committed files. A
// failure before the record leaves the committed generation as it was; a crash after it is
// completed by the next reader. Restore reads only through a server_page_reader, so it sees one
// committed generation, never a mix. Two servers publishing the SAME name into one dir at once are
// not ordered against each other: a page is seated in one server at a time.

#include "server-pages.h"

#include <cstdint>
#include <functional>
#include <shared_mutex>
#include <string>
#include <utility>
#include <vector>

using server_page_put = std::function<void(const void *, size_t)>;

// An adapter file's revision (size, mtime in ns since the epoch; (0, 0) when it cannot be read),
// stored in the `.lora` sidecar so a restore refuses a page computed under different weights.
std::pair<uint64_t, int64_t> lora_file_revision(const std::string & path);

void server_page_write_checkpoints(const std::vector<const common_prompt_checkpoint *> & kept, const server_page_put & put); // .ckpt
void server_page_write_adapters(const std::vector<common_adapter_lora_info> & lora, const server_page_put & put);           // .lora
// `dft` may be null when `put` only counts bytes
void server_page_write_draft(const uint8_t * dft, size_t n_dft, const std::vector<uint8_t> & spec, const server_page_put & put); // .dft

// Reads a `.dft` sidecar. Returns false with `why` set when it is absent or malformed.
bool server_page_read_draft(const std::string & path, std::vector<uint8_t> & dft, std::vector<uint8_t> & spec, std::string & why);

// One file of a generation: its suffix ("" is the sequence state; one of "", ".ckpt", ".lora",
// ".dft") and the writer that fills the staged `path`, returning the bytes written (0 = failed).
struct server_page_file {
    std::string suffix;
    std::function<size_t(const std::string & path)> write;
};

// A sidecar written by one of the serializers above.
server_page_file server_page_sidecar(std::string suffix, std::function<void(const server_page_put &)> body);

// Publishes `files` as the generation of the page at `filepath`, whole or not at all; a page file
// the generation lacks is removed at the commit. Throws server_page_error; on any failure before
// the commit record the committed generation is untouched and nothing staged is left. Returns the
// bytes written.
size_t server_page_publish(const std::string & filepath, const std::vector<server_page_file> & files);

// Test seam: `fault(dir)` returning true makes that directory sync fail, as a full or failing disk
// would. Empty (the default) syncs for real.
void server_page_files_inject_sync_fault(std::function<bool(const std::string & dir)> fault);

// Writes `page` to `filepath` (+ .ckpt, .lora, .dft) with every checkpoint, as one generation.
// Touches no context, so it runs off the task thread. Returns the bytes written.
size_t server_page_spill(const server_page & page, const std::string & filepath);

// Holds the committed generation at `filepath` still while it is read. Constructing one completes a
// publication a crash cut short after its commit record and removes staged files no live publication
// owns (a crash before the commit), then keeps publication off until it is destroyed. Throws
// server_page_error when an interrupted commit cannot be completed.
class server_page_reader {
public:
    explicit server_page_reader(const std::string & filepath);

private:
    std::shared_lock<std::shared_mutex> lock;
};
