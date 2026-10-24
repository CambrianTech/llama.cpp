#pragma once

// Continuum fork: a running server changes its context geometry without reloading the model.
//
// POST /reconfigure {"n_ctx", "n_parallel", "cache_type_k", "cache_type_v", "force"} is a pause, not
// a relaunch: admission stops, the slots reach a ubatch boundary, every occupied slot's sequence state
// (target KV, draft KV, speculative tail, checkpoints) moves to host RAM (server-pages.h), the target
// and draft contexts are freed and rebuilt on the SAME llama_model (weights and mmap stay), the slots
// are rebuilt for the new n_parallel, the states go back, and serving resumes. A request in flight
// keeps its slot and continues where it paused; nothing is truncated and nothing is lost.
// Design: continuum docs/architecture/SERVING-ENGINE.md §3 ("a relaunch is a pause"), slice S2.
//
// What the core sees: the receipt (server_reconfigure_receipt::to_json) and /props context_geometry.

#include "server-common.h"
#include "server-pages.h"
#include "server-task.h"

#include "common.h"
#include "llama.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct server_context_meta;

// The context geometry a reconfigure may change. Everything else (the model, adapters, batch
// sizes, speculative types, checkpoint budget) stays as launched.
struct server_geometry {
    int32_t   n_ctx      = 0; // KV cells of the whole context (llama_n_ctx)
    int32_t   n_parallel = 0; // slots, one sequence each (llama_n_seq_max)
    ggml_type type_k     = GGML_TYPE_F16;
    ggml_type type_v     = GGML_TYPE_F16;

    // a sequence state is written in its KV types: across a type change no page can be carried
    bool same_kv_format(const server_geometry & other) const {
        return type_k == other.type_k && type_v == other.type_v;
    }

    json to_json() const;
};

// The tokens one sequence may hold under `geo`: libllama's rule (llama-context.cpp: n_ctx padded to
// 256; a split cache gives each sequence n_ctx / n_seq_max, padded) and the server's cap at the
// model's training context. Used before the new context exists, so a refusal costs nothing.
int32_t server_geometry_n_ctx_seq(const server_geometry & geo, bool kv_unified, int32_t n_ctx_train);

struct server_reconfigure_request {
    server_geometry geo;
    // a KV type change is an intentional cold start: refused unless `force`, then every resident
    // state, page and prompt-cache entry is dropped (logged loudly and counted in the receipt)
    bool force = false;

    // Absent fields keep `cur`. Throws server_page_error (ERROR_TYPE_INVALID_REQUEST).
    static server_reconfigure_request parse(const json & body, const server_geometry & cur);
};

// One slot as the reconfigure sees it on the task thread.
struct server_reconfigure_slot {
    int     id          = -1;
    bool    processing  = false; // a request is in flight
    bool    at_boundary = true;  // no sampling or verification is half done (see server-context.cpp)
    int32_t n_tokens    = 0;     // tokens its KV holds
};

// Where an occupied slot's state goes.
enum server_reconfigure_fate {
    SERVER_RECONFIGURE_RESTORED,      // into a slot of the new geometry
    SERVER_RECONFIGURE_KEPT_IN_STORE, // a named RAM page (/slots/{id}?action=page_in brings it back)
    SERVER_RECONFIGURE_DROPPED,       // cold start (KV type change with force)
};

struct server_reconfigure_move {
    int     id_from    = -1;
    int     id_to      = -1; // -1 unless restored
    bool    processing = false;
    int32_t n_tokens   = 0;
    size_t  n_bytes    = 0;

    server_reconfigure_fate fate = SERVER_RECONFIGURE_RESTORED;
    std::string page;   // the store name when kept
    std::string reason; // why it was not restored

    json to_json() const;
};

// The decision before anything is touched.
struct server_reconfigure_plan {
    bool wait = false;       // a slot must first finish (it sits past the new n_parallel) or reach a boundary
    std::string wait_reason;

    bool cold_start = false; // KV type change with force
    int32_t n_ctx_seq = 0;   // predicted tokens per sequence after
    std::vector<server_reconfigure_move> moves; // one per occupied slot, fate decided
};

// Decides the reconfigure. Throws server_page_error for a refusal: a KV type change without force,
// an in-flight request whose tokens do not fit the new window (shrinking under a live act would
// truncate it), or a geometry libllama would reject. `name_prefix` names pages that stay in the store.
server_reconfigure_plan server_reconfigure_decide(
        const server_reconfigure_request & req,
        const server_geometry & cur,
        const std::vector<server_reconfigure_slot> & slots,
        bool kv_unified,
        int32_t n_ctx_train,
        const std::string & name_prefix);

struct server_reconfigure_receipt {
    server_geometry before;
    server_geometry after;
    int32_t n_ctx_seq_before = 0;
    int32_t n_ctx_seq_after  = 0;

    bool cold_start  = false;
    bool rolled_back = false; // the new geometry could not be built; the old one was rebuilt and every state restored
    std::string rollback_reason;

    double t_drain_ms    = 0.0; // request posted -> every slot at a boundary (and the ones past n_parallel done)
    double t_page_out_ms = 0.0;
    double t_free_ms     = 0.0;
    double t_init_ms     = 0.0;
    double t_page_in_ms  = 0.0;
    double t_total_ms    = 0.0; // the pause serving saw: page out through page in

    size_t n_bytes         = 0; // paged out and back
    size_t n_bytes_dropped = 0; // cold start: resident states, pages and prompt-cache entries dropped
    size_t n_tasks_deferred = 0; // requests that arrived during the pause, queued and resumed after

    std::vector<server_reconfigure_move> moves;

    json to_json() const;
};

// What a reconfigure needs from the server and nothing more; server-context.cpp implements it over
// its private slots and contexts. Slot ids are dense: 0 .. geometry().n_parallel - 1.
struct server_reconfigure_host {
    virtual ~server_reconfigure_host() = default;

    virtual server_geometry geometry() const = 0;
    virtual int32_t n_ctx_seq() const = 0; // tokens one slot holds now

    virtual llama_context      * target() const = 0;
    virtual llama_context      * draft() const = 0;
    virtual common_speculative * speculative() const = 0;

    virtual server_prompt & slot_prompt(int id_slot) = 0;
    virtual std::vector<common_adapter_lora_info> & slot_lora(int id_slot) = 0;
    virtual bool slot_processing(int id_slot) const = 0;
    // ends the slot's request with a typed error and empties the slot
    virtual void slot_fail(int id_slot, const std::string & why) = 0;

    // frees the draft and target contexts and the speculative state; the model(s) stay loaded
    virtual void free_contexts() = 0;
    // builds both contexts for `geo` on the loaded model(s); false leaves no context
    virtual bool build_contexts(const server_geometry & geo) = 0;
    // `n` slots bound to the current contexts; slots past `n` are idle and go
    virtual void set_slots(int n) = 0;

    virtual server_page_store & pages() = 0;
    virtual size_t drop_prompt_cache() = 0; // the bytes it held
};

// Runs a decided plan: page out, free, build (or roll back to the old geometry), page in. Throws
// server_page_error only before anything is touched. `page_prefix` names pages that stay in the store.
void server_reconfigure_run(
        server_reconfigure_host & host,
        const server_reconfigure_request & req,
        server_reconfigure_plan & plan,
        server_reconfigure_receipt & rc,
        const std::string & page_prefix);

struct server_task_result_reconfigure : server_task_result {
    server_reconfigure_receipt receipt;
    // the routes' view of the server after the reconfigure, built on the task thread
    std::shared_ptr<const server_context_meta> meta;

    virtual json to_json() override {
        return receipt.to_json();
    }
};
