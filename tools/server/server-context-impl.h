#pragma once

// Continuum fork: struct server_context_impl (its methods in the server-context-*.inc files), the response
// generator and the response helpers the route families share; out of server-context.cpp (pure move) so each route
// family compiles in its own translation unit (server-routes-*.cpp).

#include "server-slot.h"
#include "engine/responder_http.h"
#include "engine/turn_results.h"

struct server_context_impl {
    friend struct server_context;

public:
    // only use these pointers outside of this class:
    //  - when not in sleeping state
    //  - and, with thread-safe APIs (e.g., tokenizer calls)
    llama_model * model_tgt = nullptr;

    // the in-process trainer on model_tgt (created once the model is loaded)
    std::unique_ptr<server_trainer> trainer;
    // serving slots working right now, refreshed by the loop at the start and end of every
    // update_slots pass; the trainer reads it between batches (see server-train.h)
    std::atomic<int> n_busy_slots{0};

    // The most one sequence may hold: llama_n_ctx_seq capped at the model's training context.
    // ONE value, set at load: every slot is sized from it and context_geometry reports it, so
    // the report never names a size the slots refuse (it said 524288 while each slot capped at
    // 262144, the M5 2026-11-11).
    int n_ctx_slot = 0;
    void publish_busy_slots() {
        int n = 0;
        for (const auto & slot : slots) {
            n += slot.is_processing() ? 1 : 0;
        }
        n_busy_slots.store(n);
        if (n > 0) {
            t_last_busy_us.store(ggml_time_us());
        }
    }
    std::atomic<int64_t> t_last_busy_us{0};
    int64_t idle_ms() const {
        return n_busy_slots.load() > 0 ? 0 : (ggml_time_us() - t_last_busy_us.load()) / 1000;
    }
    // every token a slot has accepted since the server started: the trainer's slowdown bound
    // measures her decode rate with it (see server-train.h, yield_for_slowdown)
    std::atomic<int64_t> n_tokens_generated{0};

    // the adapter set the main loop last published (publish_lora_report): GET /lora-adapters reads
    // it here, never through the task queue
    server_residency_mirror & residencies() { return residency_mirror; }
    std::string claim_weight_residency(const std::string & id, const json & subject) {
        try { return residency_mirror.claim(id, residency_subject(subject)); }
        catch (const std::exception &) { return "invalid"; }
    }
    const std::string & learning_engine_identity() const { return learn_engine; }
    // the ledger as the loop last published it; readable from any thread (card 7fc7e2ce)
    json ledger_as_published() const { return published_ledger(); }

    json lora_report_snapshot() const {
        std::lock_guard<std::mutex> lock(lora_report_mutex);
        return lora_report;
    }

    mtmd_context * mctx = nullptr;
    const llama_vocab * vocab = nullptr;

    // the host bytes in-flight KV page switches hold, admitted against the budget the core sets at
    // runtime (POST /page-io/budget; server-page-io.h); thread-safe, read by the routes directly.
    // Declared before queue_tasks so it outlives every task holding a reservation.
    server_page_admission page_io_bytes;

    server_queue    queue_tasks;
    server_response queue_results;
    // where every turn result goes (CONTINUUM-ENGINE 4.12.1); built by turn_results.h
    std::unique_ptr<responder> turn_sink = std::make_unique<responder_http>(queue_results);

    // note: chat_params must not be refreshed upon existing sleeping state
    server_chat_params chat_params;

    server_state_callback_t callback_state = [](server_state, json) -> void {};

    server_context_impl() {
        mtmd_helper_log_set(common_log_default_callback, nullptr);
    }

    ~server_context_impl() {
        // page IO first: its last jobs release into the budget and post to the live queue; then no
        // waiting switch is admitted into a queue being torn down
        page_io.reset();
        page_io_bytes.close();
        if (!sleeping) {
            // destroy() is already called when entering sleeping state
            // we don't call it again here to avoid double free
            destroy();
        }
    }

    server_metrics get_metrics() const {
        return metrics;
    }

    void reset_metrics_bucket() {
        metrics.reset_bucket();
    }

private:
    // note: accessing these fields outside of this class is not thread-safe
    // use server_context methods instead

    common_params params_base;

    // note: keep these alive - they determine the lifetime of the model, context, etc.
    common_init_result_ptr llama_init;

    llama_context * ctx_tgt = nullptr;

    server_batch batch;

    llama_model   * model_dft = nullptr;
    llama_context * ctx_dft   = nullptr;

    common_speculative_init_result_ptr spec_init;

    common_context_seq_rm_type ctx_tgt_seq_rm_type = COMMON_CONTEXT_SEQ_RM_TYPE_NO;

    // LEARNING IN THE SERVING CONTEXT (SERVER_TASK_TYPE_LEARN): the learning rate the next step reads
    // (each mind's shadow has its own optimizer in the context: llama_opt_init selects it)
    float   learn_lr      = 1e-4f;
    const std::string learn_engine = [] {
        std::random_device rd;
        const char * hex = "0123456789abcdef";
        std::string id;
        for (int i = 0; i < 16; ++i) {
            const auto byte = rd() & 255;
            id += hex[byte >> 4]; id += hex[byte & 15];
        }
        return id;
    }();
    server_learn_ledger learn_ledger{learn_engine};
    server_residency_mirror residency_mirror{learn_engine};
    // Include the served tail: restoring a long generated answer is not the same geometry.
    using learn_geometry = std::tuple<int32_t, int32_t, int32_t, size_t>;
    struct learn_phase_cost {
        int64_t gradient = 0, entry_cold = 0, entry_warm = 0, exit = 0;
    };
    std::map<learn_geometry, learn_phase_cost> learn_chunk_us;

    static ggml_opt_optimizer_params learn_opt_pars(void * ud) {
        ggml_opt_optimizer_params p = ggml_opt_get_default_optimizer_params(nullptr);
        p.adamw.alpha = *(const float *) ud;
        p.adamw.wd    = 0.0f;
        return p;
    }
    struct pending_learn {
        // The step's identity; every other field starts at its stated default (an aggregate with
        // only these set trips -Wmissing-field-initializers on the ARM CI build).
        pending_learn(server_slot * slot, const common_prompt_checkpoint * checkpoint, std::vector<common_adapter_lora_info> adapters,
                      ggml_opt_result_t result, llama_token last_prompt, int32_t prefix, int32_t n_labels,
                      const llama_adapter_lora * shadow, int64_t started)
            : slot(slot), checkpoint(checkpoint), adapters(std::move(adapters)), result(result), last_prompt(last_prompt),
              prefix(prefix), n_labels(n_labels), shadow(shadow), started(started) {}
        server_slot * slot;
        const common_prompt_checkpoint * checkpoint;
        std::vector<common_adapter_lora_info> adapters;
        ggml_opt_result_t result;
        llama_token last_prompt;
        int32_t prefix, n_labels;
        // the shadow this learn trains: held by the optimizer from begin through restoration
        const llama_adapter_lora * shadow = nullptr;
        int64_t started;
        int64_t compute_us = 0;
        bool separate_request = false;
        bool began = false;
        std::string grant;
        int32_t lora_id = -1;
        llama_seq_id side = -1; // learning on a side sequence: her own sequence is never touched
        std::string refusal = "none";
        bool drained = false;
        uint64_t restoration_reservation = 0;
        int64_t restoration_us = 0;
        int64_t restoration_steps = 0;
        int64_t committed_steps = 0;
        int64_t max_chunk_us = 0;
        int64_t entry_us = 0;
        int64_t max_graph_us = 0;
        int64_t max_restore_us = 0;
        size_t served_tokens = 0;
        int64_t max_restoration_chunk_us = 0;
        std::optional<bool> finished;
        std::string failure;
        ~pending_learn() { ggml_opt_result_free(result); }
    };
    std::unique_ptr<pending_learn> learning;
    common_context_seq_rm_type ctx_dft_seq_rm_type = COMMON_CONTEXT_SEQ_RM_TYPE_NO;

    common_speculative_ptr spec;

    bool add_bos_token = true;

    int32_t n_ctx; // total context for all clients / slots

    // set to llama_model_n_swa(model)
    // if swa_full is enabled, this is set to 0 to simulate a non-SWA model
    int32_t n_swa;

    // slots / clients
    std::vector<server_slot> slots;

    int trace = 0;        // env: LLAMA_TRACE
    int slots_debug = 0;  // env: LLAMA_SERVER_SLOTS_DEBUG
    int slots_n_diff = 0; // env: LLAMA_SERVER_SLOTS_N_DIFF
    server_test_faults test_faults; // the server test build only (LLAMA_SERVER_TEST_FAULTS)

    // KV page file IO off the task loop (server-page-io.h), a worker per slot so every slot's page
    // can move at once within the page IO budget. Declared after queue_tasks and page_io_bytes so it
    // is destroyed first: its jobs post to the one and release into the other.
    std::unique_ptr<server_page_io_pool> page_io;
    // the restores read for each slot and not yet applied, which a later task or erase on it drops
    std::map<int, std::vector<std::weak_ptr<server_page_job>>> pending_restores;

    int n_empty_consecutive = 0;

    std::unique_ptr<server_prompt_cache> prompt_cache;

    server_metrics metrics;

    // queued prompt stats - llama_decode() is async, so the timing is only valid after a sync
    // note: kept out of server_metrics, which is copied as-is into the task result
    int64_t  t_decode_start  = 0; // start of the last submitted decode
    int64_t  t_prompt_start  = 0; // start of the oldest queued prompt decode
    uint64_t n_prompt_queued = 0;

    json json_ui_settings = json::object();

    // Necessary similarity of prompt for slot selection
    float slot_prompt_similarity = 0.0f;

    std::string model_name; // name of the loaded model, to be used by API
    std::set<std::string> model_aliases; // additional names for the model
    std::set<std::string> model_tags;    // informational tags

    bool sleeping = false;

    // Continuum fork: the adapter set as GET /lora-adapters reports it, republished by the main loop
    // every time the set changes (load, set, retire), so the route reads it under a lock and never
    // waits behind decoding slots (on a loaded CPU engine the queued read took over 60 s, card d765a668).
    mutable std::mutex lora_report_mutex;
    json lora_report = json::array();

    // the one place the report is built: every adapter still loaded, by the id requests address it by
    void publish_lora_report() {
        json report = json::array();
        const auto & loras = params_base.lora_adapters;
        for (size_t i = 0; i < loras.size(); ++i) {
            const auto & lora = loras[i];
            if (lora.ptr == nullptr) {
                continue; // retired
            }
            json entry = {
                {"id",            i},
                {"path",          lora.path},
                {"scale",         lora.scale},
                {"task_name",     lora.task_name},
                {"prompt_prefix", lora.prompt_prefix},
            };
            const uint64_t n_alora_tokens = llama_adapter_get_alora_n_invocation_tokens(lora.ptr);
            if (n_alora_tokens) {
                const llama_token * alora_tokens = llama_adapter_get_alora_invocation_tokens(lora.ptr);
                std::string  invocation_string;
                llama_tokens invocation_tokens;
                for (uint64_t j = 0; j < n_alora_tokens; ++j) {
                    invocation_string += common_token_to_piece(vocab, alora_tokens[j]);
                    invocation_tokens.push_back(alora_tokens[j]);
                }
                entry["alora_invocation_string"] = invocation_string;
                entry["alora_invocation_tokens"] = invocation_tokens;
            }
            report.push_back(std::move(entry));
        }
        std::lock_guard<std::mutex> lock(lora_report_mutex);
        lora_report = std::move(report);
    }

    int64_t t_last_load_progress_ms = 0;

    void destroy() {
        spec.reset();
        spec_init.reset();

        ctx_dft   = nullptr;
        model_dft = nullptr;

        llama_init.reset();

        ctx_tgt = nullptr;
        model_tgt = nullptr;

        mtmd_free(mctx);
        mctx = nullptr;
    }

    void handle_sleeping_state(bool new_state) {
        GGML_ASSERT(sleeping != new_state);
        if (new_state) {
            if (callback_state) {
                callback_state(SERVER_STATE_SLEEPING, {});
                // note: for sleeping == false, event is emitted by load_model()
            }
            SRV_INF("%s", "server is entering sleeping state\n");
            destroy();
        } else {
            SRV_INF("%s", "server is exiting sleeping state\n");
            if (!load_model(params_base)) {
                GGML_ABORT("failed to reload model after sleeping");
            }
        }
        sleeping = new_state;
    }

    struct load_progress_data {
        server_context_impl * ctx;
        std::string stage;
        std::vector<std::string> stages;
        int64_t t_last_load_progress_ms = 0;
        load_progress_data(server_context_impl * ctx, const std::string & stage) : ctx(ctx), stage(stage) {}
    };
    static bool load_progress_callback(float progress, void * user_data) {
        auto * d = static_cast<load_progress_data *>(user_data);
        GGML_ASSERT(d);
        // always emit the first and final sample; throttle the rest to one per 200ms
        {
            auto & t_last = d->t_last_load_progress_ms;
            const int64_t t_now = ggml_time_ms();
            const bool first = t_last == 0;
            const bool done  = progress >= 1.0f;
            const bool throttled = !first && !done && (t_now - t_last) < 200;
            if (throttled) {
                return true;
            }
            t_last = t_now;
        }
        if (d->ctx->callback_state) {
            d->ctx->callback_state(SERVER_STATE_LOADING, {
                {"stages", d->stages},
                {"current", d->stage},
                {"value", progress},
            });
        }
        return true;
    }

#include "server-context-load.inc"

#include "server-context-slots.inc"

#include "server-context-learn-residency.inc"
#include "server-context-learn.inc"
#include "server-context-learn-checkpoint.inc"
#include "server-context-page.inc"
#include "server-context-minds.inc"

#include "server-context-checkpoint.inc"

#include "server-context-tasks.inc"

    void iterate(std::vector<server_slot> & slots, std::function<void(server_slot &)> callback) {
        for (auto & slot : slots) {
            try {
                callback(slot);
            } catch (const std::exception & e) {
                SLT_ERR(slot, "got exception: %s\n", e.what());
                turn_sink->deliver(turn_error(slot, std::string("got exception: ") + e.what(), ERROR_TYPE_SERVER));
                slot.release();
            }
        }
    }

    void iterate(std::vector<server_slot *> & slots, std::function<void(server_slot &)> callback) {
        for (auto & slot : slots) {
            try {
                callback(*slot);
            } catch (const std::exception & e) {
                SLT_ERR(*slot, "got exception: %s\n", e.what());
                turn_sink->deliver(turn_error(*slot, std::string("got exception: ") + e.what(), ERROR_TYPE_SERVER));
                slot->release();
            }
        }
    }

    void abort_all_slots(const std::string & reason) {
        for (auto & slot : slots) {
            if (slot.is_processing()) {
                turn_sink->deliver(turn_error(slot, reason, ERROR_TYPE_SERVER));
                slot.release();
            }
        }
    }

    // @ngxson : for debugging only
    int64_t t_pre_decode  = 0;
    int64_t t_decode      = 0;
    int64_t t_post_decode = 0;
    int64_t t_sampl       = 0;
    int64_t n_pre_decode  = 0;
    int64_t n_decode      = 0;
    int64_t n_post_decode = 0;
    int64_t n_sampl       = 0;

    // time-fair interleave: after a prefill step, generating slots are owed decode-only time equal to that step
    int64_t t_decode_owed_us = 0;
    int32_t n_batch_gen      = 0; // batch size before prompt tokens are added
    server_prefill_order prefill_order;
// #define DEBUG_TIMINGS
#ifdef DEBUG_TIMINGS
    struct scoped_timer {
        int64_t & t;
        int64_t & n;
        int64_t t_start;
        scoped_timer(int64_t & t_, int64_t & n_) : t(t_), n(n_) {
            t_start = ggml_time_us();
        }
        ~scoped_timer() {
            t += ggml_time_us() - t_start;
            n++;
        }
    };
#else
    struct scoped_timer {
        scoped_timer(int64_t &, int64_t &) {}
        ~scoped_timer() {}
    };
#endif

    void update_slots() {
        if (minds_on()) {
            minds_reap(); // a landed flash write frees her RAM copy; a failed one is written again
        }
        const bool learning_at_start = bool(learning);
        publish_busy_slots();
        struct publish_on_exit { server_context_impl * s; ~publish_on_exit() { s->publish_busy_slots(); } } publish_guard{this};
#ifdef DEBUG_TIMINGS
        static int64_t t_prev = 0;
        int64_t t_start = ggml_time_us();
        if (t_start - t_prev > 5 * 1000 * 1000) { // every 5 seconds
            t_prev = t_start;
            SRV_INF("n_pre_decode      = %" PRId64 "\n", n_pre_decode);
            SRV_INF("avg t_pre_decode  = %f ms\n", (double) t_pre_decode / n_pre_decode / 1000.0);
            SRV_INF("avg t_decode      = %f ms\n", (double) t_decode / n_decode / 1000.0);
            SRV_INF("avg t_post_decode = %f ms\n", (double) t_post_decode / n_post_decode / 1000.0);
            SRV_INF("avg t_sampl       = %f ms\n", (double) t_sampl / n_sampl / 1000.0);
        }
#endif

        // check if all slots are idle
        {
            bool all_idle = true;

            for (auto & slot : slots) {
                if (slot.is_processing()) {
                    all_idle = false;
                    break;
                }
            }

            if (all_idle) {
                SRV_TRC("%s", "all slots are idle\n");

                metrics_flush_idle();

                return; // skip further processing

            } else {
                SRV_DBG("%s", "posting NEXT_RESPONSE\n");

                server_task task(SERVER_TASK_TYPE_NEXT_RESPONSE);
                task.id = queue_tasks.get_new_id();
                queue_tasks.post(std::move(task));
            }
        }

        try {
            scoped_timer t(t_pre_decode, n_pre_decode);
            pre_decode();
            batch.render();
        } catch (const std::exception & e) {
            SRV_ERR("pre_decode() failed: %s\n", e.what());
            abort_all_slots("pre_decode() failed: " + std::string(e.what()));

            // the batch is half-built and not rendered, skip now to avoid UB
            return;
        }

        GGML_ASSERT(batch.slot_batched || batch.size() == 0);

        if (batch.slot_batched) {
            auto & slot_batched      = batch.slot_batched;
            auto & alora_scale       = batch.alora_scale;
            auto & alora_disabled_id = batch.alora_disabled_id;

            // TODO @ngxson : alora handling is too messy, need to refactor it to be more clear and maintainable
            // apply lora, only need to do it once per batch
            common_set_adapter_lora(ctx_tgt, slot_batched->lora);

            // if the lora is temporarily disabled for an alora, re-enable it
            // for next time
            if (alora_scale > 0.0f) {
                SRV_DBG("re-enabling alora with scale %f\n", alora_scale);
                slot_batched->lora[alora_disabled_id].scale = alora_scale;
            }

            llama_set_embeddings(ctx_tgt, slot_batched->need_embd());
        }

        llama_batch batch_view;
        int32_t off_next = 0;
        int32_t n_batch = llama_n_batch(ctx_tgt);
        const bool    has_prompt   = batch.size() > n_batch_gen;
        const int64_t t_step_start = ggml_time_us();
        for (int32_t off = 0; off < batch.size(); off = off_next) {
            const int32_t n_tokens = std::min(n_batch, batch.size() - off);
            try {
                scoped_timer t(t_decode, n_decode);
                // TODO @ngxson : maybe handle n_batch == 1 here instead of inside decode()

                batch_view = batch.get_view(off, n_tokens);
                bool ok = decode(n_batch, off, batch_view);
#ifdef DEBUG_TIMINGS
                llama_synchronize(ctx_tgt);
#endif

                if (ok) {
                    // move the head of the batch forward with the number of tokens we just processed
                    off_next = off + n_tokens;

                    // on successful decode, restore the original batch size
                    n_batch = llama_n_batch(ctx_tgt);
                } else {
                    // try again with the updated n_batch
                    continue;
                }
            } catch (const std::exception & e) {
                SRV_ERR("decode() failed: %s\n", e.what());
                abort_all_slots("decode() failed: " + std::string(e.what()));
                break; // stop any further processing
            }

            try {
                scoped_timer t(t_post_decode, n_post_decode);
                post_decode(n_tokens, off, batch_view);
            } catch (const std::exception & e) {
                SRV_ERR("post_decode() failed: %s\n", e.what());
                abort_all_slots("post_decode() failed: " + std::string(e.what()));
                break; // stop any further processing
            }
        }

        // a prefill step adds its duration to the decode time owed; a decode-only step pays it back
        const int64_t t_step = ggml_time_us() - t_step_start;
        t_decode_owed_us = has_prompt ? t_decode_owed_us + t_step : std::max<int64_t>(0, t_decode_owed_us - t_step);
        // Completion-triggered learning can begin inside post_decode. Its entry
        // preparation is already one measured blocking segment; do not append a
        // graph segment before the queued NEXT_RESPONSE lets serving run again.
        if (learning_at_start) { advance_learning(); }
    }

#include "server-context-prefill.inc"

#include "server-context-decode.inc"

};

// generator-like API for HTTP response generation
// may have bypass_sleep = true if the task does not use ctx_server
struct server_res_generator : server_res_spipe {
    server_response_reader rd;
    server_res_generator(server_queue & queue_tasks, server_response & queue_results, int sleep_idle_seconds, bool bypass_sleep = false)
            : rd(queue_tasks, queue_results, HTTP_POLLING_SECONDS) {
        // fast path in case sleeping is disabled
        bypass_sleep |= sleep_idle_seconds < 0;
        if (!bypass_sleep) {
            queue_tasks.wait_until_no_sleep();
        }
    }
    void ok(const json & response_data) {
        status = 200;
        data = safe_json_to_str(response_data);
    }
    void error(const json & error_data) {
        status = json_value(error_data, "code", 500);
        data = safe_json_to_str({{ "error", error_data }});
    }
};

// The tokens a parsed request is served as: a chat request with a projector loaded goes through
// mtmd (its media markers and files), everything else through tokenize_input_prompts. One place, so
// what /learn checks a slot against is what serving put there by construction.
inline std::vector<server_tokens> serving_prompt_tokens(const llama_vocab * vocab, mtmd_context * mctx, const json & prompt,
                                                        const std::vector<raw_buffer> & files, bool chat) {
    if (chat && mctx != nullptr) {
        // This is the case used by OAI compatible chat path with MTMD. TODO It can be moved to the path below.
        std::vector<server_tokens> out;
        out.push_back(process_mtmd_prompt(mctx, prompt.get<std::string>(), files));
        return out;
    }
    // Everything else, including multimodal completions.
    return tokenize_input_prompts(vocab, mctx, prompt, true, true);
}

inline json get_res_model_info(const server_context_meta & meta) {
    // note: do NOT use ctx_server here, otherwise it's not possible to use this during sleep

    return {
        {"id",       meta.model_name},
        {"aliases",  meta.model_aliases},
        {"tags",     meta.model_tags},
        {"object",   "model"},
        {"created",  std::time(0)},
        {"owned_by", "llamacpp"},
        {"meta",     {
            {"vocab_type",  meta.model_vocab_type},
            {"n_vocab",     meta.model_vocab_n_tokens},
            {"n_ctx",       meta.slot_n_ctx},
            {"n_ctx_train", meta.model_n_ctx_train},
            {"n_embd",      meta.model_n_embd_inp},
            {"n_params",    meta.model_n_params},
            {"size",        meta.model_size},
            {"ftype",       meta.model_ftype},
        }},
    };
}

inline json get_res_models(const server_context_meta & meta) {
    // note: do NOT use ctx_server here, otherwise it's not possible to use this during sleep

    return json{
        {"models", json::array({
            {
                {"name",  meta.model_name},
                {"model", meta.model_name},
                {"modified_at", ""},
                {"size", ""},
                {"digest", ""}, // dummy value, llama.cpp does not support managing model file's hash
                {"type", "model"},
                {"description", ""},
                {"tags", json::array({""})},
                {"capabilities", meta.has_mtmd ? json::array({"completion","multimodal"}) : json::array({"completion"})},
                {"parameters", ""},
                {"details", {
                    {"parent_model", ""},
                    {"format", "gguf"},
                    {"family", ""},
                    {"families", json::array({""})},
                    {"parameter_size", ""},
                    {"quantization_level", ""}
                }}
            }
        })},
        {"object", "list"},
        {"data", json::array({
            get_res_model_info(meta),
        })}
    };
}

inline json get_res_props(const server_context_meta & meta, const common_params & params, bool is_sleeping) {
    // note: do NOT use ctx_server here, otherwise it's not possible to use this during sleep

    task_params tparams;
    tparams.sampling = params.sampling;
    json default_generation_settings_for_props = json {
        { "params", tparams.to_json(true) },
        { "n_ctx",  meta.slot_n_ctx },
    };

    std::string tmpl_default = common_chat_templates_source(meta.chat_params.tmpls.get(), "");
    std::string tmpl_tools   = common_chat_templates_source(meta.chat_params.tmpls.get(), "tool_use");

    json props = {
        { "default_generation_settings", default_generation_settings_for_props },
        { "total_slots",                 params.n_parallel },
        { "learn_ledger",                {{"engine", meta.learn_engine}, {"clock_us", ggml_time_us()},
            {"max_period_ms", server_learn_ledger::kMaxPeriodMs}, {"max_expiry_ms", server_learn_ledger::kMaxPeriodMs}} },
        { "slot_save_size_version",      1 },
        { "expert_container_custody_version", 1 },
        { "model_alias",                 meta.model_name },
        { "model_ftype",                 meta.model_ftype },
        { "model_path",                  meta.model_path },
        { "model_weight_buffers",        meta.model_weight_buffers },
        { "memory_breakdown",            meta.memory_breakdown },
        { "context_geometry",            meta.context_geometry },
        { "modalities",                  json {
            {"vision", meta.has_inp_image},
            {"video",  meta.has_inp_video},
            {"audio",  meta.has_inp_audio},
        } },
        { "media_marker",                get_media_marker() },
        { "endpoint_slots",              params.endpoint_slots },
        { "endpoint_props",              params.endpoint_props },
        { "endpoint_metrics",            params.endpoint_metrics },
        { "ui",                          params.ui },
        { "ui_settings",                 meta.json_ui_settings },
        { "chat_template",               tmpl_default },
        { "chat_template_caps",          meta.chat_template_caps },
        { "bos_token",                   meta.bos_token_str },
        { "eos_token",                   meta.eos_token_str },
        { "build_info",                  meta.build_info },
        { "is_sleeping",                 is_sleeping },
        { "cors_proxy_enabled",          params.ui_mcp_proxy },
    };
    if (params.use_jinja) {
        if (!tmpl_tools.empty()) {
            props["chat_template_tool_use"] = tmpl_tools;
        }
    }

    return props;
}

