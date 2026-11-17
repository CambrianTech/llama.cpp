#pragma once

// Continuum fork: the server's includes, small helpers and struct server_slot, out of server-context.cpp (pure move;
// the 1,000-line rule). Included by server-context-impl.h.

#include <limits>
#include "engine/memory_pool.h"
#include "server-context.h"
#include "server-checkpoint-policy.h"
#include "server-view-end.h"
#include "server-mapped-file.h"
#include <filesystem>
#include <thread>
#include "../../src/llama-ext.h"
#include "server-train.h"
#include "server-adapter-file.h"
#include "server-chat.h"
#include "server-common.h"
#include "server-http.h"
#include "server-task.h"
#include "server-queue.h"
#include "server-schema.h"
#include "server-stream.h"
#include "server-page-adapters.h"
#include "server-prefill.h"
#include "server-adapter-turn.h"
#include "server-learn-ledger.h"
#include "server-learn-residency.h"
#include "server-test-faults.h"
#include "server-page-io.h"
#include "server-page-job.h"

#include "build-info.h"
#include "common.h"
#include "fit.h"
#include "llama.h"
#include "gguf.h"
#include "log.h"
#include "sampling.h"
#include "speculative.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <cinttypes>
#include <exception>
#include <memory>
#include <filesystem>
#include <utility>
#include <fstream>
#include <random>
#include <sstream>

// Continuum fork: an adapter file's revision (size, mtime in ns since the epoch; (0, 0) when it
// cannot be read), stored in a saved slot's `.lora` sidecar so a restore refuses a page computed
// under different weights at the same path.
inline std::pair<uint64_t, int64_t> lora_file_revision(const std::string & path) {
    std::error_code ec;
    const auto size = std::filesystem::file_size(path, ec);
    if (ec) {
        return {0, 0};
    }
    const auto mtime = std::filesystem::last_write_time(path, ec);
    if (ec) {
        return {0, 0};
    }
    return {(uint64_t) size, (int64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(mtime.time_since_epoch()).count()};
}

// fix problem with std::min and std::max
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#include <windows.h>
#endif

constexpr int HTTP_POLLING_SECONDS = 1;

inline common_speculative_output_limits server_output_limits(const common_params & params) {
    if (params.embedding ||
            (params.pooling_type != LLAMA_POOLING_TYPE_UNSPECIFIED && params.pooling_type != LLAMA_POOLING_TYPE_NONE)) {
        return { params.n_batch, 1 };
    }

    auto result = common_speculative_get_output_limits(
            params.n_batch, params.n_parallel, common_speculative_n_max(&params.speculative));

    result.total   = std::max<int32_t>(1, result.total);
    result.per_seq = std::max<int32_t>(1, result.per_seq);
    return result;
}

// state diagram: https://github.com/ggml-org/llama.cpp/pull/9283
enum slot_state {
    SLOT_STATE_IDLE,
    SLOT_STATE_WAIT_OTHER, // after assigning a task, but waiting for parent slot to process prompt
    SLOT_STATE_STARTED,    // after assigning a task and about to process prompt
    SLOT_STATE_PROCESSING_PROMPT,
    SLOT_STATE_DONE_PROMPT,
    SLOT_STATE_GENERATING,
    SLOT_STATE_LEARNING, // admitted optimizer transaction owns the sequence until restoration
};

struct server_slot; // forward declaration

struct server_batch {
    llama_batch batch;
    bool batch_rendered = false;

    struct token {
        int32_t id_slot;
        llama_token token;
        llama_pos pos;
        bool output;
        bool is_prompt; // for stats tracking
    };
    std::vector<token> tokens;
    int32_t n_tokens_alloc = 0;
    int32_t n_embd = 0;

    // track if given slot can be batched with slots already in the batch
    server_slot * slot_batched = nullptr;

    // in embd mode, we temporarily swap out the tokens arr and restore it on clear()
    bool has_embd = false;
    llama_token * tokens_ptr = nullptr;
    std::vector<float> embd;

    float  alora_scale       = -1.0f;
    size_t alora_disabled_id = 0;

    server_batch() {
        batch.pos = nullptr; // sentinel: uninitialized batch
    }

    ~server_batch() {
        if (batch.pos != nullptr) {
            clear();
            llama_batch_free(batch);
        }
    }

    void init(int32_t n_tokens_alloc, int32_t n_embd) {
        this->n_tokens_alloc = n_tokens_alloc;
        this->n_embd = n_embd;
        batch = llama_batch_init(n_tokens_alloc, 0, 1);
        tokens_ptr = batch.token;
        tokens.reserve(n_tokens_alloc);
    }

    bool add(int32_t id_slot, llama_token token, llama_pos pos, bool output, bool is_prompt) {
        GGML_ASSERT(!has_embd); // cannot mix tokens + embd in same batch
        GGML_ASSERT(batch.pos != nullptr);
        if ((int32_t)tokens.size() >= n_tokens_alloc) {
            return false;
        }
        tokens.push_back({ id_slot, token, pos, output, is_prompt });
        return true;
    }

    bool add(int32_t id_slot, const std::vector<float> & embd_in, llama_pos pos, bool output, bool is_prompt) {
        GGML_ASSERT(batch.pos != nullptr);
        if ((int32_t)tokens.size() >= n_tokens_alloc) {
            return false;
        }
        tokens.push_back({ id_slot, LLAMA_TOKEN_NULL, pos, output, is_prompt });
        has_embd = true;
        embd.insert(embd.end(), embd_in.begin(), embd_in.end());
        return true;
    }

    void clear() {
        tokens.clear();
        embd.clear();
        common_batch_clear(batch);
        slot_batched      = nullptr;
        alora_scale       = -1.0f;
        alora_disabled_id = 0;
        batch_rendered    = false;
        has_embd          = false;
        if (batch.token == nullptr) {
            batch.token = tokens_ptr;
            batch.embd  = nullptr;
        }
    }

    int32_t size() const {
        return (int32_t)tokens.size();
    }

    // what a slot finds in the batch before it adds to it
    struct mark {
        int32_t       n_tokens;
        size_t        n_embd;
        bool          has_embd;
        server_slot * slot_batched;
        float         alora_scale;
        size_t        alora_disabled_id;
    };

    mark get_mark() const {
        return { size(), embd.size(), has_embd, slot_batched, alora_scale, alora_disabled_id };
    }

    // undo everything added since the mark, so a slot that fails leaves the batch as it found it
    void rollback(const mark & m) {
        GGML_ASSERT(!batch_rendered);
        GGML_ASSERT(m.n_tokens <= size() && m.n_embd <= embd.size());
        tokens.resize(m.n_tokens);
        embd.resize(m.n_embd);
        has_embd          = m.has_embd;
        slot_batched      = m.slot_batched;
        alora_scale       = m.alora_scale;
        alora_disabled_id = m.alora_disabled_id;
    }

    void set_output(int32_t idx, bool output) {
        GGML_ASSERT(idx >= 0 && idx < (int32_t)tokens.size());
        tokens[idx].output = output;
    }

    void render() {
        GGML_ASSERT(!batch_rendered);
        GGML_ASSERT(batch.pos != nullptr);
        common_batch_clear(batch);
        for (int32_t i = 0; i < size(); i++) {
            const auto & t = tokens[i];
            common_batch_add(batch, t.token, t.pos, { t.id_slot }, t.output);
        }
        if (has_embd) {
            batch.token = nullptr; // will be restored on clear()
            batch.embd  = embd.data();
        }
        batch_rendered = true;
    }

    llama_batch get_view(int32_t off, int32_t n_tokens) const {
        GGML_ASSERT(batch.pos != nullptr);
        GGML_ASSERT(batch_rendered);
        GGML_ASSERT(off >= 0 && off < size());
        GGML_ASSERT(n_tokens > 0 && off + n_tokens <= size());

        auto * token = batch.token ? batch.token + off          : nullptr;
        auto * embd  = batch.embd  ? batch.embd  + off * n_embd : nullptr;

        llama_batch view = {
            n_tokens,
            token,
            embd,
            batch.pos      + off,
            batch.n_seq_id + off,
            batch.seq_id   + off,
            batch.logits   + off,
            batch.label ? batch.label + off : nullptr,
            batch.label && batch.label_weight ? batch.label_weight + off : nullptr,
        };

        return view;
    }
};

struct server_slot {
    int id;

    llama_context * ctx_tgt = nullptr;
    llama_context * ctx_dft = nullptr;

    memory_pool * mem = nullptr; // the engine's allocator (engine/memory_pool.h); every KV edit of this slot goes through it

    // multimodal
    mtmd_context * mctx = nullptr;
    mtmd::batch_ptr mbatch = nullptr;

    // speculative decoding
    common_speculative * spec;

    llama_tokens spec_draft;
    llama_tokens spec_prompt;
    std::vector<int32_t> spec_i_batch;
    common_prompt_checkpoint spec_ckpt;
    bool spec_is_replay = false;

    // TODO: move members that belong to the task (such as `generated_text`, `has_new_line`) to task_results_state
    //       see https://github.com/ggml-org/llama.cpp/pull/18283#issuecomment-3710175837
    std::unique_ptr<const server_task> task;
    std::unique_ptr<const server_task> task_prev; // used for debugging

    // used to determine the slot that has been used the longest
    int64_t t_last_used = -1;

    // Continuum fork: counts every task given to this slot and every erase; a page restore records
    // it when asked and is refused, typed, if it moved before the page was read (server-context-page.inc)
    uint64_t n_occupancy = 0;

    // generation props
    int32_t n_ctx   = 0;  // context size per slot
    int32_t n_keep  = 0;
    int32_t i_batch = -1;

    // effective generation limit for the current task, -1 means unlimited
    int32_t n_predict_max = -1;

    size_t last_nl_pos = 0;

    std::string  generated_text;
    std::string  debug_generated_text;
    llama_tokens generated_tokens;
    size_t n_sent_text = 0; // number of sent text character (i.e. handle partial UTF-8 on streaming)

    std::vector<completion_token_output> generated_token_probs;

    bool has_next_token = true;
    bool has_new_line   = false;
    bool truncated      = false;

    stop_type stop;

    std::string stopping_word;

    // state
    slot_state state = SLOT_STATE_IDLE;

    server_prompt prompt;

    // Continuum fork: the mind whose sequence this lane holds (engine/mind_table_resident.cpp); invalid: anonymous
    mind_page page;
    uint64_t  page_generation = 0; // her state's generation on this lane

    bool prompt_save(server_prompt_cache & prompt_cache) const {
        if (prompt.tokens.size() == 0) {
            return false;
        }

        const size_t cur_size_tgt =           llama_state_seq_get_size_ext(ctx_tgt, id, LLAMA_STATE_SEQ_FLAGS_NONE);
        const size_t cur_size_dft = ctx_dft ? llama_state_seq_get_size_ext(ctx_dft, id, LLAMA_STATE_SEQ_FLAGS_NONE) : 0;

        const size_t cur_size = cur_size_tgt + cur_size_dft;

        SRV_TRC(" - saving prompt with length %d, total state size = %.3f MiB (draft: %.3f MiB)\n",
                (int) prompt.tokens.size(), cur_size / (1024.0 * 1024.0), cur_size_dft / (1024.0 * 1024.0));

        auto * cur = prompt_cache.alloc(prompt, lora_active(lora), cur_size_tgt, cur_size_dft);
        if (cur == nullptr) {
            return false;
        }

        llama_state_seq_get_data_ext(ctx_tgt, cur->data.main.data(), cur_size_tgt, id, LLAMA_STATE_SEQ_FLAGS_NONE);
        if (ctx_dft) {
            llama_state_seq_get_data_ext(ctx_dft, cur->data.drft.data(), cur_size_dft, id, LLAMA_STATE_SEQ_FLAGS_NONE);
        }

        return true;
    }

    // `task_lora` is the adapter list the incoming task will run under: only a state
    // computed under the same active set is restored, and the slot then records it.
    bool prompt_load(server_prompt_cache & prompt_cache, const server_tokens & tokens,
                     const std::vector<common_adapter_lora_info> & task_lora) {
        bool res = prompt_cache.load(prompt, lora_active(task_lora), tokens, ctx_tgt, ctx_dft, id);
        if (res) {
            lora = task_lora;
        } else {
            SLT_WRN(*this, "%s", "failed to load prompt from cache\n");
        }

        return res;
    }

    void prompt_clear() {
        SLT_TRC(*this, "clearing prompt with %zu tokens\n", prompt.tokens.size());

        GGML_ASSERT(mem->remove(id, -1, -1, kv_contexts::both) == edit_outcome::done);

        prompt.clear();
    }

    std::vector<common_adapter_lora_info> lora;
    int32_t alora_invocation_start = -1;
    // when its tokens last entered a batch, or its task arrived: who holds the batch (server-adapter-turn.h)
    int64_t t_last_batched_us = 0;

    // sampling
    json json_schema;

    common_sampler_ptr smpl;

    llama_token sampled; // in speculative mode, this is the last accepted token

    // for TTS models, this is the embd generated from prev step, decode this to generate next hidden state
    // corresponding to one token position (size = n_embd)
    std::vector<float> inp_embd;

    server_slot_stats stats;

    // accepted tokens per draft position
    // not in server_slot_stats to avoid copying to every task result
    std::vector<uint64_t> n_accepted_per_pos;

    std::function<void(int /* id_slot */)>   callback_on_release;
    std::function<void(const server_slot &)> callback_on_reset; // called before reset()

    // this is for printing timings with slot progress, not part of metrics
    int64_t t_print_last = 0;
    int32_t n_gen_last = 0;

    void reset() {
        learned.reset();
        SLT_DBG(*this, "%s", "\n");

        spec_is_replay = false;

        last_nl_pos    = 0;
        generated_text = "";
        has_new_line   = false;
        truncated      = false;
        stop           = STOP_TYPE_NONE;
        stopping_word  = "";
        n_sent_text    = 0;

        if (can_speculate()) {
            spec_draft.clear();
            spec_i_batch.clear();
            spec_ckpt.clear();
        }
        generated_tokens.clear();
        generated_token_probs.clear();
        json_schema = json();

        task_prev = std::move(task);
        task.reset();

        // note: callback_on_reset() must have run before this, see release()
        stats = {};
        n_accepted_per_pos.clear();

        n_predict_max = -1;

        llama_set_sampler(ctx_tgt, id, nullptr);

        // clear alora start
        alora_invocation_start = -1;

        // clear multimodal state
        mbatch.reset();
    }

    void init_sampler() const {
        common_sampler_reset(smpl.get());

        if (!task->need_sampling()) {
            return;
        }

        const int64_t t_start = ggml_time_us();

        int n_text = 0;

        for (int i = 0; i < (int) prompt.tokens.size(); i++) {
            const llama_token id = prompt.tokens[i];

            if (id != LLAMA_TOKEN_NULL) {
                common_sampler_accept(smpl.get(), id, false);
                n_text++;
            }
        }

        SLT_TRC(*this, "init sampler, took %0.2f ms, tokens: text = %d, total = %d\n",
                (ggml_time_us() - t_start) / 1000.0, n_text, (int) prompt.tokens.size());
    }

    bool need_embd() const {
        GGML_ASSERT(task);
        return task->need_embd();
    }

    // if the context does not have a memory module then all embeddings have to be computed within a single ubatch
    // also we cannot split if the pooling would require any past tokens
    // (MTP supports splitting — uses task->need_embd() not need_embd())
    bool can_split() const {
        GGML_ASSERT(task);

        return
            !task->need_embd() ||
            (llama_get_memory(ctx_tgt) && llama_pooling_type(ctx_tgt) == LLAMA_POOLING_TYPE_LAST);
    }

    bool can_batch_with(server_slot & other_slot) const {
        GGML_ASSERT(task);

        return task->type == other_slot.task->type
            && inp_embd.size() == other_slot.inp_embd.size()
            && are_lora_equal(lora, other_slot.lora);
    }

    // returns -1 if the generation is limitless
    int32_t n_remaining() const {
        return n_predict_max == -1 ? -1 : n_predict_max - (int32_t) stats.n_gen;
    }

    bool has_budget() const {
        return n_predict_max == -1 || n_remaining() > 0;
    }

    bool is_processing() const {
        return state != SLOT_STATE_IDLE;
    }

    bool can_speculate() const {
        return !!spec;
    }

    void add_token(const completion_token_output & token) {
        if (!is_processing()) {
            SLT_WRN(*this, "%s", "slot is not processing\n");
            return;
        }

        generated_token_probs.push_back(token);
    }

    int get_n_draft_max() const {
        GGML_ASSERT(task);

        if (!can_speculate()) {
            return 0;
        }

        // determine the max draft that fits the current slot state
        // note: slot.prompt is not yet expanded with the `id` token sampled above
        //       also, need to leave space for 1 extra token to allow context shifts
        int n_draft_max = n_ctx - prompt.n_tokens() - 2;

        if (n_remaining() > 0) {
            n_draft_max = std::min(n_draft_max, n_remaining() - 1);
        }

        SLT_DBG(*this, "max possible draft: %d\n", n_draft_max);

        return n_draft_max;
    }

    // add sampled token of this slot to the batch, optionally add the speculative draft tokens if any
    void handle_last_sampled_token(server_batch & batch) {
        bool add_ok = true;
        if (spec_draft.empty()) {
            // no speculative decoding
            i_batch = batch.size();

            if (!inp_embd.empty()) {
                add_ok &= batch.add(id, inp_embd, prompt.tokens.pos_next(), true, false);
            } else {
                add_ok &= batch.add(id, sampled, prompt.tokens.pos_next(), true, false);
            }

            SLT_DBG(*this, "slot decode token, id=%d, n_ctx = %d, n_tokens = %d, truncated = %d\n",
                    sampled, n_ctx, prompt.n_tokens(), truncated);
        } else {
            SLT_DBG(*this, "generate_draft: id=%d, #tokens=%zu, #draft=%zu, pos_next=%d\n",
                    sampled, prompt.tokens.size(), spec_draft.size(), prompt.tokens.pos_next());

            GGML_ASSERT(spec_i_batch.empty());

            spec_i_batch.push_back(batch.size());
            for (size_t i = 0; i < spec_draft.size(); i++) {
                spec_i_batch.push_back(batch.size() + i + 1);
            }

            auto pos0 = prompt.tokens.pos_next();

            add_ok &= batch.add(id, sampled, pos0++, true, false);
            for (auto token : spec_draft) {
                add_ok &= batch.add(this->id, token, pos0++, true, false);
            }
        }

        GGML_ASSERT(add_ok && "batch must be large enough to hold the sampled and draft tokens");

        prompt.tokens.push_back(sampled);
        prompt.tokens.insert(spec_draft);
    }

    // the outcome of the finishing task's learn (task_params::learn), for its final response
    std::optional<learn_outcome> learned;
    bool learn_release_requested = false;
    int64_t learn_graph_steps = 0;

    void release() {
        if (state == SLOT_STATE_LEARNING) {
            // Cancellation cannot hand partially restored KV to the next request.
            // Finish the admitted transaction, then release without a final response.
            learn_release_requested = true;
            return;
        }
        if (is_processing()) {
            GGML_ASSERT(task);

            SLT_INF(*this, "stop processing: n_tokens = %d, truncated = %d\n", prompt.n_tokens(), truncated);

            t_last_used = ggml_time_us();

            state = SLOT_STATE_IDLE;

            // do not keep context of the child slots - the parent's context is enough
            if (task->is_child()) {
                prompt_clear();
            }

            callback_on_reset(*this);

            reset();

            callback_on_release(id);
        }
    }

    size_t find_stopping_strings(const std::string & text, const size_t last_token_size, bool is_full_stop) {
        GGML_ASSERT(task);

        size_t stop_pos = std::string::npos;

        for (const std::string & word : task->params.antiprompt) {
            size_t pos;

            if (is_full_stop) {
                const size_t tmp      = word.size() + last_token_size;
                const size_t from_pos = text.size() > tmp ? text.size() - tmp : 0;

                pos = text.find(word, from_pos);
            } else {
                // otherwise, partial stop
                pos = string_find_partial_stop(text, word);
            }

            if (pos != std::string::npos && (stop_pos == std::string::npos || pos < stop_pos)) {
                if (is_full_stop) {
                    stop           = STOP_TYPE_WORD;
                    stopping_word  = word;
                    has_next_token = false;
                }
                stop_pos = pos;
            }
        }

        return stop_pos;
    }

    void print_timings_tg() {
        if (stats.n_gen < 100) {
            return;
        }

        const int64_t t_now = ggml_time_us();

        if (t_now - t_print_last < 3*1000*1000) {
            return;
        }

        const double n_gen_second     = stats.n_gen_tps();
        const double n_gen_second_win = 1e6 / (t_now - t_print_last) * (stats.n_gen - n_gen_last);

        t_print_last = t_now;
        n_gen_last = stats.n_gen;

        SLT_INF(*this, "n_gen = %6d, tg = %6.2f t/s, tg_3s = %6.2f t/s\n", (int) stats.n_gen, n_gen_second, n_gen_second_win);
    }

    void print_timings_pp() const {
        const double t_prompt_total = stats.t_prompt_ms();

        if (t_prompt_total < 3000.0) {
            return;
        }

        const double n_prompt_second = stats.n_prompt_tps();
        const double f_progress = task->n_tokens() > 0 ? (double) prompt.n_tokens() / task->n_tokens() : 0.0;

        SLT_INF(*this, "prompt processing, n_tokens = %6d, progress = %.2f, t = %6.2f s / %.2f tokens per second\n",
                (int) stats.n_prompt_processed, f_progress, t_prompt_total / 1e3, n_prompt_second);
    }

    void print_timings() const {
        const double t_prompt_total = stats.t_prompt_ms();
        const double t_gen_total    = stats.t_gen_ms();

        const double t_prompt        = stats.t_prompt_per_token_ms();
        const double n_prompt_second = stats.n_prompt_tps();

        const double t_gen        = stats.t_gen_per_token_ms();
        const double n_gen_second = stats.n_gen_tps();

        SLT_INF(*this,
                "prompt eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n",
                t_prompt_total, (int) stats.n_prompt_processed, t_prompt, n_prompt_second);

        SLT_INF(*this,
                "       eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n",
                t_gen_total, (int) stats.n_gen, t_gen, n_gen_second);

        SLT_INF(*this,
                "      total time = %10.2f ms / %5d tokens\n",
                t_prompt_total + t_gen_total, (int) (stats.n_prompt_processed + stats.n_gen));

        SLT_INF(*this,
                "   graphs reused = %10d\n",
                llama_perf_context(ctx_tgt).n_reused);

        const int32_t n_draft_total       = stats.n_draft_tokens;
        const int32_t n_draft_accepted    = stats.n_draft_accepted;
        const int32_t n_draft_verif_steps = stats.n_draft_verif_steps;

        if (n_draft_total > 0) {
            const float  draft_ratio  = (float) n_draft_accepted / n_draft_total;
            const double mean_acc_len = n_draft_verif_steps > 0 ? 1.0 + (double) n_draft_accepted / (double) n_draft_verif_steps : 1.0;

            std::string acceptance_rates_per_pos;
            if (n_draft_verif_steps > 0) {
                for (size_t i = 0; i < n_accepted_per_pos.size(); ++i) {
                    if (i > 0) {
                        acceptance_rates_per_pos += ", ";
                    }
                    acceptance_rates_per_pos += string_format("%.3f", (double) n_accepted_per_pos[i] / (double) n_draft_verif_steps);
                }
            }

            SLT_INF(*this,
                    "draft acceptance = %0.5f (%5d accepted / %5d generated), mean len = %5.2f\n",
                    draft_ratio, n_draft_accepted, n_draft_total, mean_acc_len);
            SLT_TRC(*this,
                    "     acc per pos = (%s)\n", acceptance_rates_per_pos.c_str());
        }

        common_speculative_print_stats(spec);
    }

    json to_json(bool only_metrics = false) const {
        json res;

        res = {
            {"id",            id},
            {"n_ctx",         n_ctx},
            {"speculative",   can_speculate()},
            {"is_processing", is_processing()},
            {"is_learning", state == SLOT_STATE_LEARNING},
            {"learn_graph_steps", learn_graph_steps},
        };

        // The engine's own positions for this slot's sequence (Continuum fork): what its cache holds
        // now, in flight or idle. -1 when the sequence is empty.
        {
            llama_memory_t mem = llama_get_memory(ctx_tgt);
            res["kv_pos_min"] = llama_memory_seq_pos_min(mem, id);
            res["kv_pos_max"] = llama_memory_seq_pos_max(mem, id);
        }

        // The slot's context checkpoints as they sit in host memory now (Continuum fork): their count
        // and their bytes, so the core prices the checkpoint store from what it holds, not a guess.
        {
            size_t bytes = 0;
            json at = json::array();
            for (const auto & ckpt : prompt.checkpoints) {
                bytes += ckpt.size();
                at.push_back({ {"n_tokens", ckpt.n_tokens}, {"bytes", ckpt.size()} });
            }
            res["checkpoints"] = { {"count", prompt.checkpoints.size()}, {"bytes", bytes}, {"at", at} };
        }

        const auto & ptask = task ? task : task_prev;

        if (ptask) {
            res["id_task"] = ptask->id;
            res["n_prompt_tokens"]           = (int32_t) prompt.tokens.size();
            res["n_prompt_tokens_processed"] = stats.n_prompt_processed;
            res["n_prompt_tokens_cache"]     = stats.n_prompt_cached;
            res["params"] = ptask->params.to_json(only_metrics);
            res["next_token"] = json::array({
                {
                    {"has_next_token", has_next_token},
                    {"has_new_line",   has_new_line},
                    {"n_remain",       n_remaining()},
                    {"n_decoded",      stats.n_gen},
                }
            });

            if (!only_metrics) {
                res["prompt"] = ptask->tokens.detokenize(ctx_tgt, true);
                res["generated"] = generated_text.empty() ? debug_generated_text : generated_text;
            }
        }

        return res;
    }

    void copy_state_to(server_slot & other) const {
        GGML_ASSERT(state == SLOT_STATE_DONE_PROMPT);

        GGML_ASSERT(mem->remove(other.id,     -1, -1, kv_contexts::both) == edit_outcome::done);
        GGML_ASSERT(mem->copy(id, other.id, -1, -1, kv_contexts::both) == edit_outcome::done);

        other.i_batch = i_batch;

        other.stats = stats;

        other.prompt = prompt.clone();
        other.init_sampler();
    }
};

// returns 0 on success
// caller need to update prompt.tokens after a successful call to keep track of the processing progress
// note: this is not a member of server_slot because we want to run it inside yield_to_queue
//       slot is passed as const to avoid accidental modification of the slot state
//       some pointers are allowed to be used, they are not used by to_json()
inline int process_mtmd_chunk(const server_slot & slot, mtmd::batch_ptr & mbatch, size_t idx, size_t & n_tokens_out) {
    GGML_ASSERT(slot.mctx);
    const auto & mctx = slot.mctx;
    const auto & input_tokens = slot.task->tokens;
    const auto & chunk = input_tokens.find_chunk(idx);
    int32_t res = 0;

    auto try_decode = [&]() -> int32_t {
        if (mbatch) {
            float * embd = mtmd_batch_get_output_embd(mbatch.get(), chunk.get());
            if (embd) {
                void * cb_data = slot.spec;
                static auto cb = [](llama_batch batch, void * user_data) {
                    common_speculative * spec = static_cast<common_speculative *>(user_data);
                    if (!common_speculative_process(spec, batch)) {
                        return 1;
                    }
                    return 0;
                };

                llama_pos new_n_past; // unused for now
                res = mtmd_helper_decode_image_chunk(
                    mctx,
                    slot.ctx_tgt,
                    chunk.get(),
                    embd,
                    slot.prompt.tokens.pos_next(),
                    slot.id,
                    llama_n_batch(slot.ctx_tgt),
                    &new_n_past,
                    cb,
                    cb_data
                );
                if (res != 0) {
                    SLT_ERR(slot, "failed to decode mtmd chunk, idx = %zu, res = %d\n", idx, res);
                    return -1;
                }
                n_tokens_out = mtmd_input_chunk_get_n_tokens(chunk.get());
                return 0; // success
            }
        }
        return 1; // (non-error) need to create & encode batch
    };

    // if the batch is already exist, try searching & encode
    res = try_decode();
    if (res == 0) {
        return 0;
    }
    if (res < 0) {
        // fatal error
        return res;
    }

    // otherwise, the batch is either uninitialized or is used up
    // we need to create & encode a new batch
    mbatch.reset(mtmd_batch_init(mctx));
    res = mtmd_batch_add_chunk(mbatch.get(), chunk.get());
    GGML_ASSERT(res == 0); // we should never have an empty batch

    // try batching as much as possible
    int n_added = 1;
    size_t idx_cur = idx;
    while (res == 0) {
        auto [next_chunk, next_idx] = input_tokens.find_next_media_chunk(idx_cur);
        if (next_chunk == nullptr) {
            break;
        }
        res = mtmd_batch_add_chunk(mbatch.get(), next_chunk->get());
        n_added += (res == 0 ? 1 : 0);
        idx_cur = next_idx;
        SLT_DBG(slot, "try adding media chunk idx = %zu to batch, res = %d\n", next_idx, res);
        // if res != 0, batch is full or chunk is not compatible -> this loop breaks
    }

    // TODO @ngxson : move this log line to debug when it become more stable
    SLT_TRC(slot, "encoding mtmd batch from idx = %zu, n_chunks = %d\n", idx, n_added);

    res = mtmd_batch_encode(mbatch.get());
    if (res != 0) {
        SLT_ERR(slot, "failed to encode mtmd batch for chunk idx = %zu, res = %d\n", idx, res);
        return -1;
    }

    return try_decode();
}

//
// server_context_impl (private implementation)
//

