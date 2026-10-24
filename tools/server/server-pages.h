#pragma once

// Continuum fork: a slot's sequence state held as a named page in host RAM.
//
// A mind's switch costs a memcpy instead of a disk save (66-100 s at 27B) or a re-prefill
// (70-160 s): `page_out` moves a slot's target KV, draft (MTP) KV, speculative tail state
// and context checkpoints into the store and frees the slot; `page_in` puts them back into
// any slot with its tokens, so the next request's prefix matches. `spill` writes a page in
// the slot-save format (server-page-files.h) that `restore` reads.
// Design: continuum docs/architecture/SERVING-ENGINE.md, slice S1.

#include "server-common.h"
#include "server-task.h"

#include "common.h"
#include "llama.h"

#include <atomic>
#include <cstdint>
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

// The host RAM a page holds: its target + draft sequence state, its speculative tail state and its
// context checkpoints. The one count: page_out admits a page with it before copying anything, and
// the store charges a stored page with it.
size_t server_page_bytes(size_t n_state, size_t n_spec, size_t n_checkpoint_bytes);

// The bytes the store counts for one page, released when the page's LAST owner drops it: a spill
// writing on an HTTP thread keeps a page alive after it left the store, and its RAM is still held.
struct server_page_charge {
    std::shared_ptr<std::atomic<size_t>> held; // the store's live-bytes ledger; null until stored
    size_t n = 0;

    server_page_charge() = default;
    server_page_charge(const server_page_charge &) = delete;
    server_page_charge & operator=(const server_page_charge &) = delete;
    ~server_page_charge();
};

struct server_page {
    // declared first so it is destroyed LAST: the bytes stay counted until every buffer below is freed
    server_page_charge charge;

    uint64_t gen = 0; // identity within the store: a page replaced under the same name is another page

    server_prompt      prompt;    // tokens and context checkpoints
    server_prompt_data data;      // target (main) and draft (drft) sequence state
    std::vector<uint8_t> spec;    // speculative tail state (common_speculative_get_state); may be empty

    std::vector<common_adapter_lora_info> lora; // the slot's adapter list the state was computed under

    size_t n_checkpoint_bytes() const;
    size_t n_bytes() const; // server_page_bytes of what the page holds
};

using server_page_ptr = std::shared_ptr<server_page>;

// A page of `seq` sized before anything is copied: what admission counts, then what capture takes.
struct server_page_shape {
    size_t n_tgt = 0;
    size_t n_dft = 0;
    std::vector<uint8_t> spec; // the speculative tail state (common_speculative_get_state), small: taken now

    size_t n_bytes(size_t n_checkpoint_bytes) const { return server_page_bytes(n_tgt + n_dft, spec.size(), n_checkpoint_bytes); }
};

server_page_shape server_page_measure(llama_context * ctx_tgt, llama_context * ctx_dft, common_speculative * spec, llama_seq_id seq);

// Copies `seq`'s state out of both contexts (each get_data synchronizes its context first): exactly
// the bytes `shape` measured. The caller moves in the prompt and adapters. Throws on a short read.
server_page_ptr server_page_capture(llama_context * ctx_tgt, llama_context * ctx_dft, llama_seq_id seq, server_page_shape shape);

// Writes a page's state into `seq` of both contexts. The page is left intact (it may be spilling).
// Throws server_page_error; the caller clears the slot so a failure is never a silent half-state.
void server_page_restore(const server_page & page, llama_context * ctx_tgt, llama_context * ctx_dft, common_speculative * spec, llama_seq_id seq);

// The named host-RAM page store. Owned by the server's task thread: only that thread finds pages and
// so makes new owners; a spill's thread only drops its own. The ceiling counts every live page,
// stored or still held by a spill.
struct server_page_store {
    uint64_t limit_bytes = 0; // --page-ram-bytes; 0 = no ceiling

    // Refuses (server_page_error, UNAVAILABLE) a page of `n_bytes` under `name` that would take the
    // live pages past the ceiling. The page it replaces is not counted when the store is its only
    // owner (the replacement frees it); a spill-held one stays live and is counted.
    void admit(const std::string & name, size_t n_bytes) const;

    // Stores `page` under `name`, replacing any page there; returns its generation. The ceiling is
    // checked on exactly what is stored. `from` (a slot's prompt), when given, moves into the page
    // first; a refused or failed put moves it back and throws, leaving the caller's state as it was.
    uint64_t put(const std::string & name, const server_page_ptr & page, server_prompt * from = nullptr);

    server_page_ptr find(const std::string & name) const;

    // Removes the page under `name` when it is generation `gen` (0 = any); returns whether it did.
    // Its bytes stay counted until its last owner drops it.
    bool erase(const std::string & name, uint64_t gen = 0);

    // Drops every page computed with `adapter` active: the adapter is about to be freed.
    void evict_lora(const llama_adapter_lora * adapter);

    // Drops every stored page; pages a spill still holds stay counted until it finishes.
    void clear() { pages.clear(); }

    size_t n_bytes() const { return held->load(); } // every live page, stored or spilling
    size_t n_pages() const { return pages.size(); }

private:
    std::map<std::string, server_page_ptr> pages;
    std::shared_ptr<std::atomic<size_t>> held = std::make_shared<std::atomic<size_t>>(0);
    uint64_t next_gen = 1;
};

// A page name: a non-empty opaque key of at most 255 bytes without control characters.
bool server_page_name_valid(const std::string & name);
