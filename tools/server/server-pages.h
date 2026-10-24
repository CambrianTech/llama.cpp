#pragma once

// Continuum fork: a slot's sequence state held as a named page in host RAM.
//
// A mind's switch costs a memcpy instead of a disk save (66-100 s at 27B) or a re-prefill
// (70-160 s): `page_out` moves a slot's target KV, draft (MTP) KV, speculative tail state
// and context checkpoints into the store and frees the slot; `page_in` puts them back into
// any slot with its tokens, so the next request's prefix matches. `spill` writes a page in
// the slot-save format (.bin + .ckpt + .lora + .dft) that `restore` reads.
// Design: continuum docs/architecture/SERVING-ENGINE.md, slice S1.

#include "server-common.h"
#include "server-task.h"

#include "common.h"
#include "llama.h"

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

struct common_speculative;

// A page operation that failed, with the HTTP error class the caller sees.
struct server_page_error : std::runtime_error {
    error_type type;
    server_page_error(error_type type, const std::string & msg) : std::runtime_error(msg), type(type) {}
};

struct server_page {
    uint64_t gen = 0; // identity within the store: a page replaced under the same name is another page

    server_prompt      prompt;    // tokens and context checkpoints
    server_prompt_data data;      // target (main) and draft (drft) sequence state
    std::vector<uint8_t> spec;    // speculative tail state (common_speculative_get_state); may be empty

    std::vector<common_adapter_lora_info> lora; // the slot's adapter list the state was computed under

    size_t n_checkpoint_bytes() const;
    size_t n_bytes() const;
};

using server_page_ptr = std::shared_ptr<server_page>;

// The sequence state bytes a page of `seq` will hold (target, draft), before it is taken.
std::pair<size_t, size_t> server_page_state_size(llama_context * ctx_tgt, llama_context * ctx_dft, llama_seq_id seq);

// Copies `seq`'s state out of both contexts (each get_data synchronizes its context first) and
// the speculative tail state; the caller moves in the prompt and adapters. Throws on a short read.
server_page_ptr server_page_capture(llama_context * ctx_tgt, llama_context * ctx_dft, common_speculative * spec, llama_seq_id seq);

// Writes a page's state into `seq` of both contexts. The page is left intact (it may be spilling).
// Throws server_page_error; the caller clears the slot so a failure is never a silent half-state.
void server_page_restore(const server_page & page, llama_context * ctx_tgt, llama_context * ctx_dft, common_speculative * spec, llama_seq_id seq);

// The named host-RAM page store. Owned by the server's task thread.
struct server_page_store {
    uint64_t limit_bytes = 0; // --page-ram-bytes; 0 = no ceiling

    // Refuses (server_page_error, UNAVAILABLE) a page of `n_bytes` under `name` that would take the
    // store past its ceiling; a page it replaces under the same name is not counted.
    void admit(const std::string & name, size_t n_bytes) const;

    // Stores `page` under `name`, replacing any page there; returns its generation.
    uint64_t put(const std::string & name, server_page_ptr page);

    server_page_ptr find(const std::string & name) const;

    // Removes the page under `name` when it is generation `gen` (0 = any); returns the bytes freed.
    size_t erase(const std::string & name, uint64_t gen = 0);

    // Drops every page computed with `adapter` active: the adapter is about to be freed.
    void evict_lora(const llama_adapter_lora * adapter);

    size_t n_bytes() const;
    size_t n_pages() const { return pages.size(); }

private:
    std::map<std::string, server_page_ptr> pages;
    uint64_t next_gen = 1;
};

// A page name: a non-empty opaque key of at most 255 bytes without control characters.
bool server_page_name_valid(const std::string & name);

//
// the slot page files: `<file>` (sequence state) and its sidecars, shared by save, spill and restore
//

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

// Writes `page` to `filepath` (+ .ckpt, .lora, .dft) with every checkpoint. Touches no context, so it
// runs off the task thread. All four files are written or none are left. Returns the bytes written.
size_t server_page_spill(const server_page & page, const std::string & filepath);
