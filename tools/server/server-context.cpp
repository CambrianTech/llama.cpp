#include <limits>
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
static std::pair<uint64_t, int64_t> lora_file_revision(const std::string & path) {
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

static common_speculative_output_limits server_output_limits(const common_params & params) {
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

    common_memory mem;

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

    // Continuum fork: the mind whose sequence this lane holds (server-context-minds.inc); invalid: anonymous
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

        mem.seq_rm(id, -1, -1);

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

        mem.seq_rm(other.id,     -1, -1);
        mem.seq_cp(id, other.id, -1, -1);

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
static int process_mtmd_chunk(const server_slot & slot, mtmd::batch_ptr & mbatch, size_t idx, size_t & n_tokens_out) {
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
#include "server-context-respond.inc"

#include "server-context-checkpoint.inc"

#include "server-context-tasks.inc"

    void iterate(std::vector<server_slot> & slots, std::function<void(server_slot &)> callback) {
        for (auto & slot : slots) {
            try {
                callback(slot);
            } catch (const std::exception & e) {
                SLT_ERR(slot, "got exception: %s\n", e.what());
                send_error(slot, std::string("got exception: ") + e.what(), ERROR_TYPE_SERVER);
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
                send_error(*slot, std::string("got exception: ") + e.what(), ERROR_TYPE_SERVER);
                slot->release();
            }
        }
    }

    void abort_all_slots(const std::string & reason) {
        for (auto & slot : slots) {
            if (slot.is_processing()) {
                send_error(slot, reason, ERROR_TYPE_SERVER);
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

//
// server_context (public API)
//

server_context::server_context() : impl(new server_context_impl()) {}
server_context::~server_context() = default;

bool server_context::load_model(common_params & params) {
    const bool ok = impl->load_model(params);
    if (ok && impl->model_tgt != nullptr) {
        server_context_impl * p = impl.get();
        impl->trainer = std::make_unique<server_trainer>(impl->model_tgt, impl->params_base,
                                                         server_trainer::serving_view{
                                                             [p]() { return p->n_busy_slots.load(); },
                                                             [p]() { return p->idle_ms(); },
                                                             [p]() { return p->n_tokens_generated.load(std::memory_order_relaxed); }},
                                                         // the SAME templates serving frames turns with
                                                         [p](const json & messages, const json & tools, bool add_generation_prompt) {
                                                             common_chat_templates_inputs in;
                                                             in.messages              = common_chat_msgs_parse_oaicompat(messages);
                                                             if (tools.is_array() && !tools.empty()) {
                                                                 in.tools = common_chat_tools_parse_oaicompat(tools);
                                                             }
                                                             in.add_generation_prompt = add_generation_prompt;
                                                             in.use_jinja             = p->params_base.use_jinja;
                                                             return common_chat_templates_apply(p->chat_params.tmpls.get(), in).prompt;
                                                         });
    }
    return ok;
}

void server_context::start_loop() {
    auto & params = impl->params_base;
    impl->queue_tasks.start_loop(params.sleep_idle_seconds * 1000);
}

void server_context::terminate() {
    impl->queue_tasks.terminate();
}

llama_context * server_context::get_llama_context() const {
    return impl->ctx_tgt;
}

server_response_reader server_context::get_response_reader() {
    return impl->get_response_reader();
}

server_context_meta server_context::get_meta() const {
    auto bos_id = llama_vocab_bos(impl->vocab);
    auto eos_id = llama_vocab_eos(impl->vocab);
    auto bos_token_str = bos_id != LLAMA_TOKEN_NULL ? common_token_to_piece(impl->ctx_tgt, bos_id, true) : "";
    auto eos_token_str = eos_id != LLAMA_TOKEN_NULL ? common_token_to_piece(impl->ctx_tgt, eos_id, true) : "";

    const char * ftype_name = llama_ftype_name(llama_model_ftype(impl->model_tgt));

    return server_context_meta {
        /* build_info             */ std::string(llama_build_info()),
        /* learn_engine           */ impl->learn_engine,
        /* model_name             */ impl->model_name,
        /* model_aliases          */ impl->model_aliases,
        /* model_tags             */ impl->model_tags,
        /* model_path             */ impl->params_base.model.path,
        /* has_mtmd               */ impl->mctx != nullptr,
        /* has_inp_image          */ impl->chat_params.allow_image,
        /* has_inp_audio          */ impl->chat_params.allow_audio,
        /* has_inp_video          */ impl->chat_params.allow_video,
        /* json_ui_settings       */ impl->json_ui_settings,
        /* slot_n_ctx             */ impl->get_slot_n_ctx(),
        /* pooling_type           */ llama_pooling_type(impl->ctx_tgt),

        /* chat_params            */ impl->chat_params,
        /* chat_template_caps     */ common_chat_templates_get_caps(impl->chat_params.tmpls.get()),

        /* bos_token_str          */ bos_token_str,
        /* eos_token_str          */ eos_token_str,
        /* fim_pre_token          */ llama_vocab_fim_pre(impl->vocab),
        /* fim_sub_token          */ llama_vocab_fim_suf(impl->vocab),
        /* fim_mid_token          */ llama_vocab_fim_mid(impl->vocab),
        /* fim_pad_token          */ llama_vocab_fim_pad(impl->vocab),
        /* fim_rep_token          */ llama_vocab_fim_rep(impl->vocab),
        /* fim_sep_token          */ llama_vocab_fim_sep(impl->vocab),

        /* logit_bias_eog         */ impl->params_base.sampling.logit_bias_eog,

        /* model_vocab_type       */ llama_vocab_type(impl->vocab),
        /* model_vocab_n_tokens   */ llama_vocab_n_tokens(impl->vocab),
        /* model_n_ctx_train      */ llama_model_n_ctx_train(impl->model_tgt),
        /* model_n_embd_inp       */ llama_model_n_embd(impl->model_tgt),
        /* model_n_params         */ llama_model_n_params(impl->model_tgt),
        /* model_size             */ llama_model_size(impl->model_tgt),
        /* model_ftype            */ ftype_name,
        /* model_weight_buffers   */ [&] {
            // MEASURED placement: where the weights were actually allocated.
            // Not n_gpu_layers, not the "offloaded N/M" banner — both of those
            // echo the REQUEST back (the banner is min(n_gpu_layers, n_layer+1)),
            // so a lane that asked for the GPU and fell back to the CPU prints
            // numbers identical to one that succeeded.
            json bufs = json::array();
            const size_t n = llama_model_n_weight_backends(impl->model_tgt);
            for (size_t i = 0; i < n; i++) {
                const char * backend = llama_model_weight_backend_name(impl->model_tgt, i);
                bufs.push_back(json {
                    { "backend",    backend ? backend : "" },
                    { "size_bytes", llama_model_weight_backend_bytes(impl->model_tgt, i) },
                });
            }
            return bufs;
        }(),
        /* memory_breakdown       */ [&] {
            json out = json::array();
            for (const auto & [buft, mb] : llama_get_memory_breakdown(impl->ctx_tgt)) {
                out.push_back(json {
                    { "buffer_type",   ggml_backend_buft_name(buft) },
                    { "model_bytes",   mb.model },
                    { "context_bytes", mb.context },
                    { "compute_bytes", mb.compute },
                });
            }
            return out;
        }(),
        /* context_geometry       */ json {
            { "allocated_tokens",        llama_n_ctx(impl->ctx_tgt) },
            { "max_tokens_per_sequence", impl->n_ctx_slot },
            { "sequence_capacity",       llama_n_seq_max(impl->ctx_tgt) },
            // the engine owns mind switches (server-context-minds.inc): a request naming its "page" is switched by
            // sequence moves; the core reads this BEFORE dispatch to skip its own page save/restore (Codex on #139)
            { "resident_minds", json {
                { "enabled",    impl->minds_on() },
                { "lanes",      impl->params_base.n_parallel },
                { "parked_ids", (int64_t) impl->park_free.size() },
                { "incarnation", mind_incarnation() },
            } },
        },
    };
}

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

void server_context::set_state_callback(server_state_callback_t callback) {
    impl->callback_state = std::move(callback);
}

//
// server_routes
//

// The tokens a parsed request is served as: a chat request with a projector loaded goes through
// mtmd (its media markers and files), everything else through tokenize_input_prompts. One place, so
// what /learn checks a slot against is what serving put there by construction.
static std::vector<server_tokens> serving_prompt_tokens(const llama_vocab * vocab, mtmd_context * mctx, const json & prompt,
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

std::unique_ptr<server_res_generator> server_routes::handle_completions_impl(
            const server_http_req & req,
            server_task_type type,
            const json & data,
            const std::vector<raw_buffer> & files,
            task_response_type res_type) {
    GGML_ASSERT(type == SERVER_TASK_TYPE_COMPLETION || type == SERVER_TASK_TYPE_INFILL);

    auto res = create_response();
    auto completion_id = gen_chatcmplid();
    auto & rd = res->rd;
    auto & params = this->params;

    res->set_req(&req); // will also set spipe if needed

    int32_t sse_ping_interval = params.sse_ping_interval;

    try {
        std::vector<server_task> tasks;

        const auto & prompt = data.at("prompt");
        // TODO: this log can become very long, put it behind a flag or think about a more compact format
        //SRV_DBG("Prompt: %s\n", prompt.is_string() ? prompt.get<std::string>().c_str() : prompt.dump(2).c_str());

        if (!params.path_prompts_log_dir.empty()) {
            const auto file_path = std::filesystem::path(params.path_prompts_log_dir) / string_format("%012" PRId64 ".txt", ggml_time_ms());
            std::ofstream f(file_path);
            if (f) {
                f << (prompt.is_string() ? prompt.get<std::string>().c_str() : prompt.dump(2).c_str());
            } else {
                SRV_ERR("failed to create %s\n", file_path.string().c_str());
            }
        }

        // process prompt
        std::vector<server_tokens> inputs = serving_prompt_tokens(ctx_server.vocab, ctx_server.mctx, prompt, files,
                                                                  res_type != TASK_RESPONSE_TYPE_NONE);

        // tasks.reserve(inputs.size()); // TODO: this is inaccurate due to child tasks

        // message delimiters for checkpointing
        json delims = json_value(data, "message_delimiters", json::array());
        auto delimiters = common_chat_msg_delimiters_parse(delims);
        delimiters.tokenize(ctx_server.vocab);

        for (size_t i = 0; i < inputs.size(); i++) {
            server_task task = server_task(type);

            task.id = rd.get_new_id();

            task.tokens = std::move(inputs[i]);
            task.params = server_schema::eval_llama_cmpl_schema(
                    ctx_server.vocab,
                    params,
                    meta->logit_bias_eog,
                    data);

            task.params.message_spans = task.tokens.find_message_spans(delimiters);
            // Continuum fork: where the caller's view ends, so the prompt is checkpointed there.
            if (data.contains("view_messages") && data.contains("messages")) {
                std::string refusal;
                task.params.view_end_pos = view_end_position(data.at("messages"), json_value(data, "view_messages", (int64_t) -1),
                                                             delimiters, task.params.message_spans, task.tokens.size(), refusal);
                if (task.params.view_end_pos < 0) {
                    SRV_WRN("view_messages = %s names no token position (%s): this prompt gets no checkpoint at its view end\n",
                            data.at("view_messages").dump().c_str(), refusal.c_str());
                }
            }

            task.id_slot = json_value(data, "id_slot", -1);
            task.params.page = mind_page_parse(json_value(data, "page", std::string())); // text only at this edge
            task.params.learn_pin = json_value(data, "learn_pin", false);
            sse_ping_interval = task.params.sse_ping_interval;

            // OAI-compat
            task.params.res_type          = res_type;
            task.params.oaicompat_cmpl_id = completion_id;
            task.params.oaicompat_model   = meta->model_name;

            // prepare child tasks
            if (task.params.n_cmpl > 1) {
                int n_children = task.params.n_cmpl - 1;
                for (int j = 0; j < n_children; j++) {
                    task.add_child(task.id, rd.get_new_id());
                }
            }

            tasks.push_back(std::move(task));
        }

        rd.post_tasks(std::move(tasks));
    } catch (const std::exception & e) {
        res->error(format_error_response(e.what(), ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    bool stream = json_value(data, "stream", false);

    if (!stream) {
        // non-stream, wait for the results
        auto all_results = rd.wait_for_all(req.should_stop);
        if (all_results.is_terminated) {
            return res; // connection is closed
        } else if (all_results.error) {
            res->error(all_results.error->to_json());
            return res;
        } else {
            json arr = json::array();
            for (auto & res : all_results.results) {
                GGML_ASSERT(dynamic_cast<server_task_result_cmpl_final*>(res.get()) != nullptr);
                arr.push_back(res->to_json());
            }
            GGML_ASSERT(!arr.empty() && "empty results");
            if (arr.size() == 1) {
                // if single request, return single object instead of array
                res->ok(arr[0]);
            } else if (res_type == TASK_RESPONSE_TYPE_OAI_CHAT || res_type == TASK_RESPONSE_TYPE_OAI_CMPL) {
                // if multiple results in OAI format, we need to re-format them
                json & choices = arr[0]["choices"];
                for (size_t i = 1; i < arr.size(); i++) {
                    choices.push_back(std::move(arr[i]["choices"][0]));
                }
                res->ok(arr[0]);
            } else {
                // multi-results, non-OAI compat
                res->ok(arr);
            }
        }
    } else {
        // in streaming mode, the first error must be treated as non-stream response
        // this is to match the OAI API behavior
        // ref: https://github.com/ggml-org/llama.cpp/pull/16486#discussion_r2419657309
        auto first_result = rd.next(req.should_stop);
        if (first_result == nullptr) {
            GGML_ASSERT(req.should_stop());
            return res; // connection is closed
        }

        if (first_result->is_error()) {
            res->error(first_result->to_json());
            return res;
        }

        GGML_ASSERT(
            dynamic_cast<server_task_result_cmpl_partial*>(first_result.get()) != nullptr ||
            dynamic_cast<server_task_result_cmpl_final*>  (first_result.get()) != nullptr
        );

        // next responses are streamed
        // to be sent immediately
        json first_result_json = first_result->to_json();
        if (first_result_json == nullptr) {
            res->data = ""; // simply send HTTP headers and status code
        } else if (res_type == TASK_RESPONSE_TYPE_ANTHROPIC) {
            res->data = format_anthropic_sse(first_result_json);
        } else if (res_type == TASK_RESPONSE_TYPE_OAI_RESP) {
            res->data = format_oai_resp_sse(first_result_json);
        } else {
            res->data = format_oai_sse(first_result_json);
        }
        res->status = 200;
        res->content_type = "text/event-stream";
        res->set_next([res_this = res.get(), res_type, sse_ping_interval](std::string & output) -> bool {
            static auto format_error = [](task_response_type res_type, const json & res_json) {
                if (res_type == TASK_RESPONSE_TYPE_ANTHROPIC) {
                    return format_anthropic_sse({
                        {"event", "error"},
                        {"data", res_json},
                    });
                } else {
                    return format_oai_sse(json {{ "error", res_json }});
                }
            };

            auto effective_should_stop = [&res_this]() {
                return res_this->should_stop();
            };

            try {
                if (effective_should_stop()) {
                    SRV_DBG("%s", "stopping streaming due to should_stop condition\n");
                    return false; // should_stop condition met
                }

                if (!res_this->data.empty()) {
                    // flush the first chunk
                    output = std::move(res_this->data);
                    res_this->data.clear();
                    return true;
                }

                server_response_reader & rd = res_this->rd;

                // check if there is more data
                if (!rd.has_next()) {
                    switch (res_type) {
                        case TASK_RESPONSE_TYPE_NONE:
                        case TASK_RESPONSE_TYPE_OAI_RESP:
                        case TASK_RESPONSE_TYPE_ANTHROPIC:
                            output = "";
                            break;

                        default:
                            output = "data: [DONE]\n\n";
                            break;
                    }
                    SRV_DBG("%s", "all results received, terminating stream\n");
                    return false; // no more data, terminate
                }

                // receive subsequent results
                bool timeout = false;
                int64_t start_time = ggml_time_ms();
                auto result = rd.next([&timeout, &start_time, sse_ping_interval, &effective_should_stop]() {
                    if (effective_should_stop()) {
                        return true; // should_stop condition met
                    } else if (sse_ping_interval > 0 && ggml_time_ms() - start_time > (int64_t)sse_ping_interval * 1000) {
                        timeout = true;
                        return true; // timeout
                    }
                    return false;
                });

                if (timeout) {
                    // some clients may time out (e.g. undici) will time out if no data is received for a while, so we need to send a ping to keep the connection alive
                    SRV_DBG("%s", "sending SSE ping\n");
                    output = ":\n\n";
                    return true;
                }

                if (result == nullptr) {
                    SRV_DBG("%s", "stopping streaming due to should_stop condition\n");
                    GGML_ASSERT(effective_should_stop());
                    return false; // should_stop condition met
                }

                // send the results
                if (result->is_error()) {
                    json res_json = result->to_json();
                    output = format_error(res_type, res_json);
                    SRV_DBG("%s", "error received during streaming, terminating stream\n");
                    return false; // terminate on error
                } else {
                    GGML_ASSERT(
                        dynamic_cast<server_task_result_cmpl_partial*>(result.get()) != nullptr
                        || dynamic_cast<server_task_result_cmpl_final*>(result.get()) != nullptr
                    );
                    json res_json = result->to_json();
                    if (res_type == TASK_RESPONSE_TYPE_ANTHROPIC) {
                        output = format_anthropic_sse(res_json);
                    } else if (res_type == TASK_RESPONSE_TYPE_OAI_RESP) {
                        output = format_oai_resp_sse(res_json);
                    } else {
                        output = format_oai_sse(res_json);
                    }
                }

                // has next data, continue
                return true;

            } catch (const std::exception & e) {
                json error_json = format_error_response(e.what(), ERROR_TYPE_SERVER);
                output = format_error(res_type, error_json);

                // terminate on exception
                return false;
            }
        });
    }

    return res;
}

std::unique_ptr<server_res_generator> server_routes::create_response(bool bypass_sleep) {
    return std::make_unique<server_res_generator>(queue_tasks, queue_results, params.sleep_idle_seconds, bypass_sleep);
}

server_routes::server_routes(const common_params & params, server_context & ctx_server)
        : params(params),
          ctx_server(*ctx_server.impl),
          queue_tasks(ctx_server.impl->queue_tasks),
          queue_results(ctx_server.impl->queue_results) {
    init_routes();

    // note: this must be registered before load_model()
    //       so that on sleep phase, the callback is called before ctx is destroyed
    queue_tasks.on_sleeping_state([this](bool is_sleeping) {
        update_cached_responses(is_sleeping);
    });
}

static json get_res_model_info(const server_context_meta & meta) {
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

static json get_res_models(const server_context_meta & meta) {
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

static json get_res_props(const server_context_meta & meta, const common_params & params, bool is_sleeping) {
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

json server_routes::get_model_info() const {
    return get_res_model_info(*meta);
}

void server_routes::init_routes() {
    // IMPORTANT: all lambda functions must start with create_response()
    // this is to ensure that the server_res_generator can handle sleeping case correctly

    this->get_health = [this](const server_http_req &) {
        // error and loading states are handled by middleware
        auto res = create_response(true);

        // this endpoint can be accessed during sleeping
        // the next LOC is to avoid someone accidentally use ctx_server
        bool ctx_server; // do NOT delete this line
        GGML_UNUSED(ctx_server);

        res->ok({{"status", "ok"}});
        return res;
    };

    this->get_metrics = [this](const server_http_req & req) {
        auto res = create_response(true);
        if (!params.endpoint_metrics) {
            res->error(format_error_response("This server does not support metrics endpoint. Start it with `--metrics`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        // render response using cached_metrics
        auto use_cached_metrics = [&]() {
            std::unique_lock<std::mutex> lock(mutex_cache);
            res->headers["Process-Start-Time-Unix"] = std::to_string(cached_metrics.t_start);
            server_task_result_metrics tmp;
            tmp.metrics = cached_metrics;
            res->content_type = "text/plain; version=0.0.4";
            res->status = 200;
            res->data = tmp.to_metrics();
            // the gauges are averaged over the window between two scrapes
            cached_metrics.reset_bucket();
            should_reset_buckets = true;
        };

        if (queue_tasks.is_sleeping()) {
            use_cached_metrics();

        } else {
            // request slots data using task queue
            {
                server_task task(SERVER_TASK_TYPE_METRICS);
                task.id = res->rd.get_new_id();
                // the gauges are averaged over the window between two scrapes
                task.metrics_reset_bucket = true;
                res->rd.post_task(std::move(task), true); // high-priority task
            }

            // a task posted right before sleeping is never processed, do not wait for it
            auto result = res->rd.next([&]{
                return req.should_stop() || queue_tasks.is_sleeping();
            });
            if (!result) {
                if (!req.should_stop()) {
                    use_cached_metrics();
                }
                return res;
            }

            if (result->is_error()) {
                res->error(result->to_json());
                return res;
            }

            auto res_task = dynamic_cast<server_task_result_metrics*>(result.get());
            GGML_ASSERT(res_task != nullptr);

            res->headers["Process-Start-Time-Unix"] = std::to_string(res_task->metrics.t_start);
            res->content_type = "text/plain; version=0.0.4";
            res->status = 200;
            res->data = res_task->to_metrics();
        }

        return res;
    };

    this->get_slots = [this](const server_http_req & req) {
        auto res = create_response();
        if (!params.endpoint_slots) {
            res->error(format_error_response("This server does not support slots endpoint. Start it with `--slots`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        const auto operation_id = req.get_param("operation_id");
        if (!operation_id.empty()) {
            std::lock_guard<std::mutex> lock(mutex_restore_progress);
            for (const auto & entry : restore_progress) {
                if (entry.second->operation_id == operation_id) {
                    res->ok(entry.second->snapshot());
                    return res;
                }
            }
            res->error(format_error_response("Restore operation is unknown", ERROR_TYPE_NOT_FOUND));
            return res;
        }

        // request slots data using task queue
        {
            server_task task(SERVER_TASK_TYPE_SLOT_GET);
            task.id = res->rd.get_new_id();
            res->rd.post_task(std::move(task), true); // high-priority task
        }

        // get the result
        auto result = res->rd.next(req.should_stop);
        if (!result) {
            // connection was closed
            GGML_ASSERT(req.should_stop());
            return res;
        }

        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }

        auto * res_task = dynamic_cast<server_task_result_slots*>(result.get());
        GGML_ASSERT(res_task != nullptr);

        // optionally return "fail_on_no_slot" error
        if (!req.get_param("fail_on_no_slot").empty()) {
            if (res_task->n_idle_slots == 0) {
                res->error(format_error_response("no slot available", ERROR_TYPE_UNAVAILABLE));
                return res;
            }
        }

        res->ok(res_task->to_json());
        return res;
    };

    // GET /cells (Continuum fork): every cell holder's occupancy, by the engine's own count, with the
    // slots' positions read in the same pass. Admission reads free cells here, never from arithmetic.
    this->get_cells = [this](const server_http_req & req) {
        auto res = create_response();
        {
            server_task task(SERVER_TASK_TYPE_SLOT_GET);
            task.id = res->rd.get_new_id();
            res->rd.post_task(std::move(task), true); // high-priority task
        }
        auto result = res->rd.next(req.should_stop);
        if (!result) {
            GGML_ASSERT(req.should_stop());
            return res;
        }
        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }
        auto * res_task = dynamic_cast<server_task_result_slots*>(result.get());
        GGML_ASSERT(res_task != nullptr);
        json slots = json::array();
        for (const auto & slot : res_task->slots_data) {
            slots.push_back({ {"id", slot.at("id")}, {"kv_pos_min", slot.at("kv_pos_min")}, {"kv_pos_max", slot.at("kv_pos_max")}, {"is_processing", slot.at("is_processing")} });
        }
        res->ok({ {"holders", res_task->cells}, {"slots", slots} });
        return res;
    };

    this->post_slots = [this](const server_http_req & req) {
        auto res = create_response();
        if (params.slot_save_path.empty()) {
            res->error(format_error_response("This server does not support slots action. Start it with `--slot-save-path`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        std::string id_slot_str = req.get_param("id_slot");

        int id_slot;
        try {
            id_slot = std::stoi(id_slot_str);
        } catch (const std::exception &) {
            res->error(format_error_response("Invalid slot ID", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        std::string action = req.get_param("action");

        if (action == "save" || action == "save-size") {
            return handle_slots_save(req, id_slot, action == "save-size");
        }
        if (action == "restore") {
            return handle_slots_restore(req, id_slot);
        }
        if (action == "erase") {
            return handle_slots_erase(req, id_slot);
        }

        res->error(format_error_response("Invalid action", ERROR_TYPE_INVALID_REQUEST));
        return res;
    };

    this->get_props = [this](const server_http_req &) {
        auto res = create_response(true);
        // note: do NOT use ctx_server here, this endpoint must be accessible during sleep
        if (queue_tasks.is_sleeping()) {
            std::unique_lock<std::mutex> lock(mutex_cache);
            res->ok(cached_props);
        } else {
            res->ok(get_res_props(*meta, params, false));
        }
        return res;
    };

    this->post_props = [this](const server_http_req &) {
        auto res = create_response();
        if (!params.endpoint_props) {
            res->error(format_error_response("This server does not support changing global properties. Start it with `--props`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }
        // update any props here

        res->ok({{ "success", true }});
        return res;
    };

    this->post_infill = [this](const server_http_req & req) {
        auto res = create_response();
        // check model compatibility
        std::string err;
        if (llama_vocab_fim_pre(ctx_server.vocab) == LLAMA_TOKEN_NULL) {
            err += "prefix token is missing. ";
        }
        if (llama_vocab_fim_suf(ctx_server.vocab) == LLAMA_TOKEN_NULL) {
            err += "suffix token is missing. ";
        }
        if (llama_vocab_fim_mid(ctx_server.vocab) == LLAMA_TOKEN_NULL) {
            err += "middle token is missing. ";
        }
        if (!err.empty()) {
            res->error(format_error_response(string_format("Infill is not supported by this model: %s", err.c_str()), ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        // validate input
        json data = json::parse(req.body);
        if (data.contains("prompt") && !data.at("prompt").is_string()) {
            // prompt is optional
            res->error(format_error_response("\"prompt\" must be a string", ERROR_TYPE_INVALID_REQUEST));
        }

        if (!data.contains("input_prefix")) {
            res->error(format_error_response("\"input_prefix\" is required", ERROR_TYPE_INVALID_REQUEST));
        }

        if (!data.contains("input_suffix")) {
            res->error(format_error_response("\"input_suffix\" is required", ERROR_TYPE_INVALID_REQUEST));
        }

        if (data.contains("input_extra") && !data.at("input_extra").is_array()) {
            // input_extra is optional
            res->error(format_error_response("\"input_extra\" must be an array of {\"filename\": string, \"text\": string}", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        json input_extra = json_value(data, "input_extra", json::array());
        for (const auto & chunk : input_extra) {
            // { "text": string, "filename": string }
            if (!chunk.contains("text") || !chunk.at("text").is_string()) {
                res->error(format_error_response("extra_context chunk must contain a \"text\" field with a string value", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
            // filename is optional
            if (chunk.contains("filename") && !chunk.at("filename").is_string()) {
                res->error(format_error_response("extra_context chunk's \"filename\" field must be a string", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
        }
        data["input_extra"] = input_extra; // default to empty array if it's not exist

        std::string prompt = json_value(data, "prompt", std::string());
        std::vector<server_tokens> tokenized_prompts = tokenize_input_prompts(ctx_server.vocab, ctx_server.mctx, prompt, false, true);
        SRV_DBG("creating infill tasks, n_prompts = %d\n", (int) tokenized_prompts.size());
        data["prompt"] = format_prompt_infill(
            ctx_server.vocab,
            data.at("input_prefix"),
            data.at("input_suffix"),
            data.at("input_extra"),
            params.n_batch,
            params.n_predict,
            meta->slot_n_ctx,
            params.spm_infill,
            tokenized_prompts[0].get_tokens() // TODO: this could maybe be multimodal.
        );

        std::vector<raw_buffer> files; // dummy
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_INFILL,
            data,
            files,
            TASK_RESPONSE_TYPE_NONE); // infill is not OAI compatible
    };

    this->post_completions = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files; // dummy
        const json body = json::parse(req.body);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body,
            files,
            TASK_RESPONSE_TYPE_NONE);
    };

    this->post_completions_oai = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files; // dummy
        const json body = json::parse(req.body);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body,
            files,
            TASK_RESPONSE_TYPE_OAI_CMPL);
    };

    this->post_chat_completions = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files;
        json body = json::parse(req.body);
        json body_parsed = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            TASK_RESPONSE_TYPE_OAI_CHAT);
    };

    // POST /v1/chat/completions/cells (Continuum fork): the cells a chat request would add to the KV
    // pool on slot `id_slot` (required), by this server's own template and tokenizer and that slot's
    // cache. Admission sizes a request from this, never from the caller's estimate.
    this->post_chat_completions_cells = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files;
        json body = json::parse(req.body);
        if (!body.contains("id_slot") || !body.at("id_slot").is_number_integer()) {
            res->error(format_error_response("\"id_slot\" (integer) is required", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        const int id_slot = body.at("id_slot").get<int>();
        json body_parsed = oaicompat_chat_params_parse(body, meta->chat_params, files);
        json prompt = body_parsed.at("prompt");
        server_task task(SERVER_TASK_TYPE_CELLS_NEEDED);
        if (ctx_server.mctx != nullptr) {
            if (!prompt.is_string()) {
                res->error(format_error_response("for mtmd, input prompt must be a string", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
            task.tokens = process_mtmd_prompt(ctx_server.mctx, prompt.get<std::string>(), files, true);
        } else {
            task.tokens = server_tokens(tokenize_mixed(ctx_server.vocab, prompt, true, true), false);
        }
        task.id = res->rd.get_new_id();
        task.slot_action.id_slot = id_slot;
        res->rd.post_task(std::move(task), true); // high-priority task
        auto result = res->rd.next(req.should_stop);
        if (!result) {
            GGML_ASSERT(req.should_stop());
            return res;
        }
        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }
        res->ok(result->to_json());
        return res;
    };

    this->post_chat_completions_tok = [this](const server_http_req & req) {
        return handle_count_tokens(ctx_server.vocab, ctx_server.mctx, req, TASK_RESPONSE_TYPE_OAI_CHAT);
    };

    this->post_control = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);

        const std::string cmpl_id = json_value(body, "id", std::string());
        const std::string action  = json_value(body, "action", std::string());
        if (cmpl_id.empty()) {
            res->error(format_error_response("missing completion id", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        if (action != "reasoning_end") {
            res->error(format_error_response("unknown control action", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        auto & rd = res->rd;
        {
            server_task task(SERVER_TASK_TYPE_CONTROL);
            task.id              = rd.get_new_id();
            task.params.control_cmpl_id = cmpl_id;
            task.params.control_action  = action;
            rd.post_task(std::move(task));
        }

        auto result = rd.next(req.should_stop);
        if (!result) {
            GGML_ASSERT(req.should_stop());
            return res;
        }
        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }
        res->ok(result->to_json());
        return res;
    };

    this->post_responses_oai = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files;
        json body = server_chat_convert_responses_to_chatcmpl(json::parse(req.body));
        SRV_DBG("%s\n", "Request converted: OpenAI Responses -> OpenAI Chat Completions");
        SRV_DBG("converted request: %s\n", body.dump().c_str());
        json body_parsed = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            TASK_RESPONSE_TYPE_OAI_RESP);
    };

    this->post_responses_tok_oai = [this](const server_http_req & req) {
        return handle_count_tokens(ctx_server.vocab, ctx_server.mctx, req, TASK_RESPONSE_TYPE_OAI_RESP);
    };

    this->post_transcriptions_oai = [this](const server_http_req & req) {
        auto res = create_response();

        if (!meta->has_mtmd || !meta->chat_params.allow_audio) {
            res->error(format_error_response("The current model does not support audio input.", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        std::vector<raw_buffer> files;
        json body = convert_transcriptions_to_chatcmpl(
            json::parse(req.body),
            meta->chat_params.tmpls.get(),
            req.files,
            files);
        SRV_DBG("%s\n", "Request converted: OpenAI Transcriptions -> OpenAI Chat Completions");
        SRV_DBG("converted request: %s\n", body.dump().c_str());
        json body_parsed = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            TASK_RESPONSE_TYPE_OAI_ASR);
    };

    this->post_anthropic_messages = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files;
        json body = server_chat_convert_anthropic_to_oai(json::parse(req.body));
        SRV_DBG("%s\n", "Request converted: Anthropic -> OpenAI Chat Completions");
        SRV_DBG("converted request: %s\n", body.dump().c_str());
        json body_parsed = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            TASK_RESPONSE_TYPE_ANTHROPIC);
    };

    this->post_anthropic_count_tokens = [this](const server_http_req & req) {
        return handle_count_tokens(ctx_server.vocab, ctx_server.mctx, req, TASK_RESPONSE_TYPE_ANTHROPIC);
    };

    // same with handle_chat_completions, but without inference part
    this->post_apply_template = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files; // dummy, unused
        json body = json::parse(req.body);
        json data = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);
        res->ok({{ "prompt", std::move(data.at("prompt")) }});
        return res;
    };

    this->get_models = [this](const server_http_req &) {
        auto res = create_response(true);
        // note: do NOT use ctx_server here, this endpoint must be accessible during sleep
        if (queue_tasks.is_sleeping()) {
            std::unique_lock<std::mutex> lock(mutex_cache);
            res->ok(cached_models);
        } else {
            res->ok(get_res_models(*meta));
        }
        return res;
    };

    this->post_tokenize = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);
        json tokens_response = json::array();
        if (body.count("content") != 0) {
            const bool add_special = json_value(body, "add_special", false);
            const bool parse_special = json_value(body, "parse_special", true);
            const bool with_pieces = json_value(body, "with_pieces", false);

            llama_tokens tokens = tokenize_mixed(ctx_server.vocab, body.at("content"), add_special, parse_special);

            if (with_pieces) {
                for (const auto& token : tokens) {
                    std::string piece = common_token_to_piece(ctx_server.vocab, token);
                    json piece_json;

                    // Check if the piece is valid UTF-8
                    if (is_valid_utf8(piece)) {
                        piece_json = piece;
                    } else {
                        // If not valid UTF-8, store as array of byte values
                        piece_json = json::array();
                        for (unsigned char c : piece) {
                            piece_json.push_back(static_cast<int>(c));
                        }
                    }

                    tokens_response.push_back({
                        {"id", token},
                        {"piece", piece_json}
                    });
                }
            } else {
                tokens_response = tokens;
            }
        }

        res->ok(json{{"tokens", std::move(tokens_response)}});
        return res;
    };

    this->post_detokenize = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);

        std::string content;
        if (body.count("tokens") != 0) {
            const llama_tokens tokens = body.at("tokens").get<llama_tokens>();
            content = tokens_to_str(ctx_server.vocab, tokens);
        }

        res->ok(json{{"content", std::move(content)}});
        return res;
    };

    this->post_embeddings = [this](const server_http_req & req) {
        return handle_embeddings_impl(req, TASK_RESPONSE_TYPE_NONE);
    };

    this->post_embeddings_oai = [this](const server_http_req & req) {
        return handle_embeddings_impl(req, TASK_RESPONSE_TYPE_OAI_EMBD);
    };

    this->post_rerank = [this](const server_http_req & req) {
        auto res = create_response();
        if (!params.embedding || params.pooling_type != LLAMA_POOLING_TYPE_RANK) {
            res->error(format_error_response("This server does not support reranking. Start it with `--reranking`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        const json body = json::parse(req.body);

        // if true, use TEI API format, otherwise use Jina API format
        // Jina: https://jina.ai/reranker/
        // TEI: https://huggingface.github.io/text-embeddings-inference/#/Text%20Embeddings%20Inference/rerank
        bool is_tei_format = body.contains("texts");

        json query;
        if (body.count("query") == 1) {
            query = body.at("query");
            if (!query.is_string()) {
                res->error(format_error_response("\"query\" must be a string", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
        } else {
            res->error(format_error_response("\"query\" must be provided", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        std::vector<std::string> documents = json_value(body, "documents",
                                             json_value(body, "texts", std::vector<std::string>()));
        if (documents.empty()) {
            res->error(format_error_response("\"documents\" must be a non-empty string array", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        int top_n = json_value(body, "top_n", (int)documents.size());

        // create and queue the task
        json responses = json::array();
        auto & rd = res->rd;
        {
            std::vector<server_task> tasks;
            tasks.reserve(documents.size());
            for (size_t i = 0; i < documents.size(); i++) {
                auto tmp = format_prompt_rerank(ctx_server.model_tgt, ctx_server.vocab, ctx_server.mctx, query, documents[i]);
                server_task task = server_task(SERVER_TASK_TYPE_RERANK);
                task.id     = rd.get_new_id();
                task.tokens = std::move(tmp);
                tasks.push_back(std::move(task));
            }
            rd.post_tasks(std::move(tasks));
        }

        // wait for the results
        auto all_results = rd.wait_for_all(req.should_stop);

        // collect results
        if (all_results.is_terminated) {
            return res; // connection is closed
        } else if (all_results.error) {
            res->error(all_results.error->to_json());
            return res;
        } else {
            for (auto & res : all_results.results) {
                GGML_ASSERT(dynamic_cast<server_task_result_rerank*>(res.get()) != nullptr);
                responses.push_back(res->to_json());
            }
        }

        // write JSON response
        json root = format_response_rerank(
            body,
            meta->model_name,
            responses,
            is_tei_format,
            documents,
            top_n);

        res->ok(root);
        return res;
    };

    // the adapter set the main loop last published: answered at once, never queued behind slots
    this->get_lora_adapters = [this](const server_http_req &) {
        auto res = create_response();
        res->ok(ctx_server.lora_report_snapshot());
        return res;
    };

    this->get_train = [this](const server_http_req &) {
        auto res = create_response();
        res->ok(ctx_server.trainer ? ctx_server.trainer->status() : json::object({{"state", "unavailable"}}));
        return res;
    };

    this->post_train = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);
        const json r = ctx_server.trainer ? ctx_server.trainer->start(body)
                                           : json::object({{"ok", false}, {"error", "no model is loaded"}});
        if (!r.value("ok", false)) {
            res->error(format_error_response(r.value("error", std::string("training refused")), ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        res->ok(r);
        return res;
    };

    this->post_train_measure = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);
        const json r = ctx_server.trainer ? ctx_server.trainer->measure(body)
                                           : json::object({{"ok", false}, {"error", "no model is loaded"}});
        if (!r.value("ok", false)) {
            res->error(format_error_response(r.value("error", std::string("measure refused")), ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        res->ok(r);
        return res;
    };

    this->post_train_cancel = [this](const server_http_req &) {
        auto res = create_response();
        const json r = ctx_server.trainer ? ctx_server.trainer->cancel()
                                           : json::object({{"ok", false}, {"error", "no model is loaded"}});
        if (!r.value("ok", false)) {
            res->error(format_error_response(r.value("error", std::string("nothing to cancel")), ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        res->ok(r);
        return res;
    };

    this->post_lora_adapters_save = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);
        const auto path = confine_out(params.train_dir, json_value(body, "name", std::string()));
        if (params.train_dir.empty() || path.empty() || !body.contains("id") || !body.at("id").is_number_integer() ||
            body.at("id").get<double>() < 0 || body.at("id").get<double>() > INT32_MAX) {
            res->error(format_error_response("save requires an integer id and a bare .gguf name under --train-dir", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        auto & rd = res->rd;
        server_task task(SERVER_TASK_TYPE_SAVE_LORA);
        task.id = rd.get_new_id();
        task.save_lora = body.at("id").get<int32_t>();
        task.load_lora_path = path;
        if (body.contains("checkpoint")) {
            const auto & checkpoint = body.at("checkpoint");
            if (!checkpoint.is_object() || !checkpoint.contains("binding") ||
                !checkpoint.at("binding").is_string() || checkpoint.at("binding").get<std::string>().empty() ||
                !checkpoint.contains("engine") || !checkpoint.at("engine").is_string() ||
                checkpoint.at("engine").get<std::string>().empty() ||
                !checkpoint.contains("provenance") || !checkpoint.at("provenance").is_object()) {
                res->error(format_error_response("checkpoint requires engine, nonempty binding and provenance object", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
            task.save_checkpoint = checkpoint;
        }
        rd.post_task(std::move(task));
        auto result = rd.next(req.should_stop);
        if (!result) {
            return res; // queued snapshot remains owned by the loop after disconnect
        }
        if (result->is_error()) {
            res->error(result->to_json());
        } else {
            res->ok(result->to_json());
        }
        return res;
    };

    auto adapter_memory_row = +[](const char * device, const char * buft, size_t bytes, bool host, void * data) {
        static_cast<json *>(data)->push_back({{"device", device}, {"buffer_type", buft},
            {"bytes", bytes}, {"host", host}});
    };
    struct allocation_guard {
        server_residency_mirror & mirror;
        std::string id;
        bool transferred = false;
        allocation_guard(server_residency_mirror & owner, std::string reference) : mirror(owner), id(std::move(reference)) {}
        ~allocation_guard() { if (!transferred) { mirror.finish(id, false); } }
    };
    auto load_adapter = [this, adapter_memory_row](const server_http_req & req, std::shared_ptr<allocation_guard> custody) {
        auto res = create_response();
        const json body = json::parse(req.body);
        const std::string name = json_value(body, "name", std::string());
        const std::string path = confine_out(params.train_dir, name);
        if (params.train_dir.empty() || path.empty()) {
            res->error(format_error_response("\"name\" must be a bare file name ending in .gguf inside the server's --train-dir "
                                             "(where /train writes its output)", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        if (ctx_server.model_tgt == nullptr) {
            res->error(format_error_response("no model is loaded", ERROR_TYPE_UNAVAILABLE));
            return res;
        }
        if (body.value("dry_run", false)) {
            json receipt = {{"engine", ctx_server.learning_engine_identity()}, {"reserved", false}, {"weights", json::array()}};
            if (!llama_adapter_lora_measure(ctx_server.model_tgt, path.c_str(), adapter_memory_row, &receipt["weights"])) {
                res->error(format_error_response("cannot measure adapter placement", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
            res->ok(receipt);
            return res;
        }
        // Reading the tensors happens here, on the HTTP thread, like the trainer's own
        // adapter init: the server loop only appends the pointer between decode steps.
        const auto residency = body.value("residency", std::string());
        if (!custody) {
            const auto subject = json{{"load", {{"name", name}}}};
            const auto refusal = ctx_server.claim_weight_residency(residency, subject);
            if (refusal != "none") {
                res->error(format_error_response("adapter residency refused: " + refusal, ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
            custody = std::make_shared<allocation_guard>(ctx_server.residencies(), residency);
        }
        auto * adapter = llama_adapter_lora_init_admitted(ctx_server.model_tgt, path.c_str(),
            [](const char * device, const char * buft, size_t bytes, bool host, void * data) {
                auto & c = *static_cast<allocation_guard *>(data);
                return c.mirror.allows(c.id, {device, buft, "weights", bytes, host});
            }, custody.get());
        if (adapter == nullptr) {
            res->error(format_error_response("failed to load LoRA adapter '" + name + "' (missing, or not a LoRA for this model)",
                                             ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        json weights = json::array();
        llama_adapter_lora_visit_memory(adapter, adapter_memory_row, &weights);
        auto & rd = res->rd;
        {
            server_task task(SERVER_TASK_TYPE_LOAD_LORA);
            task.id             = rd.get_new_id();
            task.load_lora      = adapter;
            task.load_lora_path = path;
            task.load_lora_residency = residency;
            rd.post_task(std::move(task));
            custody->transferred = true;
        }
        auto result = rd.next(req.should_stop);
        if (!result) {
            // connection closed after the task was posted: the loop still owns the adapter
            GGML_ASSERT(req.should_stop());
            return res;
        }
        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }
        auto loaded = result->to_json();
        loaded["weights"] = std::move(weights);
        res->ok(loaded);
        return res;
    };

    this->post_lora_adapters_load = [load_adapter](const server_http_req & req) { return load_adapter(req, nullptr); };

    this->post_lora_adapters_fresh = [this, adapter_memory_row, load_adapter](const server_http_req & req) -> server_http_res_ptr {
        auto res = create_response();
        const json body = json::parse(req.body);
        const auto path = confine_out(params.train_dir, json_value(body, "name", std::string()));
        if (params.train_dir.empty() || path.empty() || ctx_server.model_tgt == nullptr) {
            res->error(format_error_response("fresh requires a loaded model and a bare .gguf name under --train-dir", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        const double rank = body.value("rank", 8.0);
        const double alpha = body.value("alpha", 16.0);
        const double top = body.value("top_layers", 0.0);
        const double seed = body.value("seed", 42.0);
        if (!std::isfinite(rank) || rank < 1 || rank > 256 || std::floor(rank) != rank ||
            !std::isfinite(alpha) || alpha <= 0 || alpha > std::numeric_limits<float>::max() ||
            !std::isfinite(top) || top < 0 || top > llama_model_n_layer(ctx_server.model_tgt) || std::floor(top) != top ||
            !std::isfinite(seed) || seed < 0 || seed > UINT32_MAX || std::floor(seed) != seed) {
            res->error(format_error_response("invalid rank, alpha, top_layers or seed", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        std::vector<std::string> targets;
        std::stringstream parts(body.value("targets", std::string("attn_q,attn_v")));
        std::string target;
        while (std::getline(parts, target, ',')) {
            if (!target.empty()) {
                targets.push_back(target);
            }
        }
        if (targets.empty()) {
            res->error(format_error_response("targets must name at least one module", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        if (body.value("dry_run", false)) {
            json receipt = {{"engine", ctx_server.learning_engine_identity()}, {"reserved", false}, {"weights", json::array()}};
            if (!common_lora_write_fresh(ctx_server.model_tgt, path, (int32_t) rank, (float) alpha,
                    targets, (uint32_t) seed, (int32_t) top, adapter_memory_row, &receipt["weights"])) {
                res->error(format_error_response("cannot measure fresh adapter placement", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
            res->ok(receipt);
            return res;
        }
        const auto residency = body.value("residency", std::string());
        const auto refusal = ctx_server.claim_weight_residency(residency, json{{"fresh", body}});
        if (refusal != "none") {
            res->error(format_error_response("fresh residency refused: " + refusal, ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        auto custody = std::make_shared<allocation_guard>(ctx_server.residencies(), residency);
        struct measurement { allocation_guard & custody; bool fits = true; } measured{*custody};
        const bool described = common_lora_write_fresh(ctx_server.model_tgt, path, (int32_t) rank, (float) alpha,
            targets, (uint32_t) seed, (int32_t) top,
            [](const char * device, const char * buft, size_t bytes, bool host, void * data) {
                auto & m = *static_cast<measurement *>(data);
                m.fits &= m.custody.mirror.allows(m.custody.id, {device, buft, "weights", bytes, host});
            }, &measured);
        if (!described || !measured.fits) {
            res->error(format_error_response("fresh residency does not cover measured weights", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        const auto error = server_publish_adapter(path, [&](const std::string & temporary) {
            return common_lora_write_fresh(ctx_server.model_tgt, temporary, (int32_t) rank, (float) alpha,
                                           targets, (uint32_t) seed, (int32_t) top);
        });
        if (!error.empty()) {
            res->error(format_error_response(error, ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        // Reuse load's ownership transfer and dormant scale-zero installation.
        return load_adapter(req, std::move(custody));
    };

    // POST /learn: one reply learned on the slot that served its prompt, between that slot's turns, in
    // this serving context (SERVER_TASK_TYPE_LEARN). Body: id_slot, n_prefix (the served prompt's
    // tokens), reply (text) or reply_tokens, lora_id (a loaded adapter: the shadow), lr.
    // Duty rest returns a deferred outcome: the caller retains replayable input context, reply or
    // labels, and adapter parameters, then retries after retry_after_ms. A slot's prompt may have
    // changed by then. The slot-release queue cannot wake a cooldown on an already idle slot.
    // Both routes use the same served template, reply markers and token owner.
    auto render_learn_example = [this](const json & body) {
        learn_request example;
        // the conversation she was served, its last message her reply (or the correction). The
        // served prompt is every earlier message through the SAME parser and template arguments
        // /v1/chat/completions used (chat_template_kwargs included: a thinking template renders a
        // different prompt without them), with the generation prompt; the reply is what the model
        // would have generated after it, including the template's native turn/tool markers
        const json & messages = body.at("messages");
        if (!messages.is_array() || messages.size() < 2 || !messages.back().is_object()
                || messages.back().value("role", std::string()) != "assistant") {
            throw std::invalid_argument("\"messages\" must end with the assistant reply to learn, after the conversation it answers");
        }
        json served = body;
        served["messages"] = json::array();
        for (size_t i = 0; i + 1 < messages.size(); ++i) {
            served["messages"].push_back(messages[i]);
        }
        std::vector<raw_buffer> files;
        const json parsed = oaicompat_chat_params_parse(served, meta->chat_params, files);
        if (!files.empty()) {
            throw std::invalid_argument("a conversation with media cannot be learned yet: only text prompts");
        }
        example.prompt   = serving_prompt_tokens(ctx_server.vocab, ctx_server.mctx, parsed.at("prompt"), files, true)[0].get_text_tokens();
        example.n_prefix = (int32_t) example.prompt.size();
        // Reuse the trainer's template-derived spans. Only this final assistant turn is
        // labelled; tool syntax, reasoning and end markers come from the served template.
        auto render = [&](const json & turns, const json & tools, bool generation) {
            json framed = body;
            framed["messages"] = turns;
            framed["tools"] = tools;
            framed["add_generation_prompt"] = generation;
            std::vector<raw_buffer> media;
            const json rendered = oaicompat_chat_params_parse(framed, meta->chat_params, media,
                generation ? server_chat_render::serving : server_chat_render::closed_transcript);
            if (!media.empty()) {
                throw std::invalid_argument("learning media is not supported");
            }
            return rendered.at("prompt").get<std::string>();
        };
        std::string why;
        if (!train_reply_rendered(messages, json_value(body, "tools", json()), render, why)) {
            throw std::invalid_argument(why);
        }
        std::vector<bool> trained(messages.size(), false);
        trained.back() = true;
        std::vector<llama_token> tokens;
        std::vector<uint8_t> loss;
        if (!train_render(messages, trained, json_value(body, "tools", json()), render,
                          ctx_server.vocab, "messages", tokens, loss, why)) {
            throw std::invalid_argument(why);
        }
        const auto first = std::find(loss.begin(), loss.end(), uint8_t{1});
        const size_t prefix = (size_t) std::distance(loss.begin(), first);
        if (first == loss.end() || prefix != example.prompt.size()
                || !std::equal(example.prompt.begin(), example.prompt.end(), tokens.begin())
                || !std::all_of(first, loss.end(), [](uint8_t label) { return label == 1; })) {
            throw std::invalid_argument("the template's learned reply is not a contiguous suffix of the exact served prompt");
        }
        example.reply.assign(tokens.begin() + prefix, tokens.end());
        return example;
    };
    auto grant_route = [this](const server_http_req & req, const char * op) {
        auto res = create_response();
        server_task task(SERVER_TASK_TYPE_LEARN_GRANTS);
        task.learn_grants = req.body.empty() ? json::object() : json::parse(req.body);
        if (!task.learn_grants.is_object()) {
            res->error(format_error_response("grant body must be an object", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        for (const char * key : {"id", "engine"}) {
            if (task.learn_grants.contains(key) && !task.learn_grants.at(key).is_string()) {
                res->error(format_error_response("grant identity must be a string", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
        }
        for (const char * key : {"period_ms", "share_ppm", "expires_in_ms", "lora_id"}) {
            if (task.learn_grants.contains(key) && (!task.learn_grants.at(key).is_number_integer() ||
                    task.learn_grants.at(key).get<double>() >= double(INT64_MAX))) {
                res->error(format_error_response("grant terms must be signed integers", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
        }
        task.learn_grants["op"] = op;
        if (task.learn_grants.value("ticket", false)) {
            // a ticketed mutation answers at once; its result arrives on GET /learn/events (card 7fc7e2ce)
            task.id = queue_tasks.get_new_id();
            const int ticket = task.id;
            queue_results.add_ticket(ticket);
            queue_tasks.post(std::move(task));
            res->ok({{"ticket", ticket}});
            return res;
        }
        auto & rd = res->rd;
        task.id = rd.get_new_id();
        rd.post_task(std::move(task));
        auto result = rd.next(req.should_stop);
        if (result) { res->ok(result->to_json()); }
        return res;
    };
    this->post_learn_preflight = [this, grant_route, render_learn_example](const server_http_req & req) {
        auto res = create_response();
        try {
            json body = json::parse(req.body);
            if (!body.is_object() || body.contains("reply_rows")) {
                throw std::invalid_argument("preflight requires the example, not caller-computed reply_rows");
            }
            size_t rows;
            if (body.contains("messages")) {
                rows = render_learn_example(body).reply.size();
            } else if (body.contains("learn")) {
                // Completion-attached corrections share the completion schema, including EOT.
                rows = server_schema::eval_llama_cmpl_schema(ctx_server.vocab, params, meta->logit_bias_eog,
                    json{{"learn", body.at("learn")}}).learn->reply.size();
            } else {
                throw std::invalid_argument("preflight requires messages or a completion learn example");
            }
            body["reply_rows"] = rows;
            server_http_req measured = req;
            measured.body = body.dump();
            return grant_route(measured, "preflight");
        } catch (const std::exception & e) {
            res->error(format_error_response(e.what(), ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
    };
    this->post_learn_residency = [grant_route](const server_http_req & req) { return grant_route(req, "residency_register"); };
    this->release_learn_residency = [grant_route](const server_http_req & req) { return grant_route(req, "residency_release"); };
    this->revoke_learn_residency = [grant_route](const server_http_req & req) { return grant_route(req, "residency_revoke"); };
    // The learning event stream (card 7fc7e2ce): every ticketed mutation's terminal result, pushed to
    // the one live subscriber, then gone. Opened once per engine incarnation by the core; its end is
    // the incarnation gate (the stream drops with the engine). Idle, it sends SSE keepalives.
    this->get_learn_events = [this](const server_http_req & req) {
        struct ticket_stream : server_res_generator {
            server_response & results;
            const int subscriber;
            ticket_stream(server_queue & tasks, server_response & results, int sleep_idle_seconds)
                : server_res_generator(tasks, results, sleep_idle_seconds, true), results(results),
                  subscriber(results.subscribe_tickets()) {}
            void on_complete() override {
                results.unsubscribe_tickets(subscriber);
                server_res_generator::on_complete();
            }
        };
        auto res = std::make_unique<ticket_stream>(queue_tasks, queue_results, params.sleep_idle_seconds);
        res->set_req(&req); // its connection ends the stream (should_stop)
        res->status = 200;
        res->content_type = "text/event-stream";
        res->data = "data: " + safe_json_to_str({{"engine", ctx_server.learning_engine_identity()}, {"subscribed", true}}) + "\n\n";
        const int keepalive = params.sse_ping_interval > 0 ? params.sse_ping_interval : 1;
        res->set_next([res_this = res.get(), keepalive](std::string & output) -> bool {
            if (res_this->should_stop()) {
                return false;
            }
            if (!res_this->data.empty()) {
                output = std::move(res_this->data);
                res_this->data.clear();
                return true;
            }
            output = res_this->results.next_ticket(res_this->subscriber, keepalive);
            if (output.empty()) {
                output = ":\n\n"; // keepalive
            }
            return true;
        });
        return res;
    };

    // a read answers from the loop's last publication, never queued behind the chunk it reports on
    this->get_learn_grants = [this](const server_http_req &) {
        auto res = create_response();
        res->ok(ctx_server.ledger_as_published());
        return res;
    };

    // A cache is never need: the core lowers the host prompt cache's limit to give its bytes to an
    // active consumer (a learner's step), and raises it again after (card 494c99f1).
    this->post_prompt_cache_limit = [this](const server_http_req & req) {
        auto res = create_response();
        // A malformed body throws common_json_error, which this route's ex_wrapper answers as an error.
        const json body = req.body.empty() ? json::object() : json::parse(req.body);
        // 0 would mean "no limit" to the cache itself, so a release to nothing is 1 MiB, never 0; and the
        // limit in bytes must fit a size_t, so a value past it is refused, never wrapped (Cormac on #110).
        constexpr int64_t max_mib = (int64_t) (SIZE_MAX >> 20) < INT64_MAX ? (int64_t) (SIZE_MAX >> 20) : INT64_MAX;
        if (!body.is_object() || !body.contains("limit_mib") || !body.at("limit_mib").is_number_integer()
                || body.at("limit_mib").get<int64_t>() < 1 || body.at("limit_mib").get<int64_t>() > max_mib) {
            res->error(format_error_response("limit_mib must be an integer from 1 to the bytes a size_t holds, in MiB", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        server_task task(SERVER_TASK_TYPE_PROMPT_CACHE_LIMIT);
        task.prompt_cache_limit_mib = body.at("limit_mib").get<int64_t>();
        auto & rd = res->rd;
        task.id = rd.get_new_id();
        rd.post_task(std::move(task));
        auto result = rd.next(req.should_stop);
        if (result) { res->ok(result->to_json()); }
        return res;
    };
    // The host bytes in-flight KV page switches may hold (server-page-io.h): the core sets it from its
    // live host room, one memory budget; raising it admits waiting switches at once. Read and set here,
    // never through the task loop: the budget is thread-safe and a waiting switch must not wait on decode.
    auto page_io_snapshot = [this]() {
        const auto b = ctx_server.page_io_bytes.snapshot();
        return json {{"budget_bytes", b.budget}, {"in_flight_bytes", b.in_flight}, {"waiting", b.waiting}};
    };
    this->get_page_io = [this, page_io_snapshot](const server_http_req &) {
        auto res = create_response(true);
        res->ok(page_io_snapshot());
        return res;
    };
    this->post_page_io_budget = [this, page_io_snapshot](const server_http_req & req) {
        auto res = create_response(true);
        const json body = req.body.empty() ? json::object() : json::parse(req.body);
        if (!body.is_object() || !body.contains("budget_bytes") || !body.at("budget_bytes").is_number_integer() || body.at("budget_bytes").get<int64_t>() < 0) {
            res->error(format_error_response("budget_bytes must be an unsigned integer: the host bytes in-flight page switches may hold", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        ctx_server.page_io_bytes.set_budget(body.at("budget_bytes").get<uint64_t>());
        res->ok(page_io_snapshot());
        return res;
    };
    this->post_minds_prefetch = [this](const server_http_req & req) { return handle_minds_prefetch(req); };
    this->post_learn_release  = [this](const server_http_req & req) { return handle_learn_release(req); };
    this->post_learn_grants = [grant_route](const server_http_req & req) { return grant_route(req, "register"); };
    this->revoke_learn_grant = [grant_route](const server_http_req & req) { return grant_route(req, "revoke"); };

    this->post_learn = [this, render_learn_example](const server_http_req & req) {
        auto res = create_response();
        if (params.train_dir.empty()) {
            res->error(format_error_response("learning is off: start the server with --train-dir", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }
        const json body = json::parse(req.body);
        server_task task(SERVER_TASK_TYPE_LEARN);
        if (body.contains("restore_checkpoint") && !body.contains("messages")) {
            res->error(format_error_response("checkpoint restoration requires the retained messages", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        task.learn.grant = json_value(body, "grant", std::string());
        task.learn.optimizer_residency = json_value(body, "optimizer_residency", std::string());
        task.learn.workspace_residency = json_value(body, "workspace_residency", std::string());
        task.learn.id_slot  = json_value(body, "id_slot",  -1);
        task.id_slot        = task.learn.id_slot; // a deferred learn comes back when her slot frees
        task.learn.n_prefix = json_value(body, "n_prefix", 0);
        task.learn.lora_id  = json_value(body, "lora_id",  -1);
        task.learn.lr       = json_value(body, "lr",       1e-4f);
        task.learn.side     = json_value(body, "side_sequence", false);
        if (body.contains("messages")) {
            try {
                const auto example = render_learn_example(body);
                task.learn.prompt = example.prompt;
                task.learn.n_prefix = example.n_prefix;
                task.learn.reply = example.reply;
            } catch (const std::exception & e) {
                res->error(format_error_response(e.what(), ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
            if (body.contains("restore_checkpoint")) {
                const auto & checkpoint = body.at("restore_checkpoint");
                if (!checkpoint.is_object() || !checkpoint.contains("engine") || !checkpoint.at("engine").is_string() ||
                    !checkpoint.contains("name") || !checkpoint.at("name").is_string() ||
                    confine_out(params.train_dir, checkpoint.at("name").get<std::string>()).empty() ||
                    !checkpoint.contains("binding") || !checkpoint.at("binding").is_string() ||
                    checkpoint.at("binding").get<std::string>().empty() ||
                    !checkpoint.contains("steps") || !checkpoint.at("steps").is_number_integer() ||
                    checkpoint.at("steps").get<int64_t>() < 0) {
                    res->error(format_error_response("restore_checkpoint requires engine, bare .gguf name, binding and nonnegative steps", ERROR_TYPE_INVALID_REQUEST));
                    return res;
                }
                task.restore_checkpoint = checkpoint;
                auto & rd = res->rd;
                task.id = rd.get_new_id();
                rd.post_task(std::move(task));
                auto result = rd.next(req.should_stop);
                if (result) { res->ok(result->to_json()); }
                return res;
            }
            // A side-sequence learn never serves the example on her slot (that would cut her context back to the
            // example's prompt): her slot must already hold the prompt as a prefix of what she has now, and the
            // learn runs on a side sequence (server-context-learn.inc), refused typed otherwise
            if (task.learn.side) {
                auto & rd = res->rd;
                task.id = rd.get_new_id();
                rd.post_task(std::move(task));
                auto result = rd.next(req.should_stop);
                if (!result) {
                    GGML_ASSERT(req.should_stop());
                    return res;
                }
                if (result->is_error()) {
                    res->error(result->to_json());
                    return res;
                }
                res->ok(result->to_json());
                return res;
            }
            // A past example (S4 (f); fork #89): her slot may no longer hold this prompt. Serve it
            // first through the ordinary completion path (prompt cache, checkpoints, the slot's
            // own admission and the grant), one token, and learn as it finishes: when the slot
            // still holds it the cache makes that free, when it does not it is one prefill. One
            // prefill path, one learn path; the slot is the caller's choice (whose cache it replaces).
            server_task serve(SERVER_TASK_TYPE_COMPLETION);
            serve.tokens = server_tokens(task.learn.prompt, false);
            serve.params = server_schema::eval_llama_cmpl_schema(ctx_server.vocab, params, meta->logit_bias_eog,
                json { { "n_predict", 1 }, { "temperature", 0.0 }, { "cache_prompt", true } });
            serve.params.learn      = task.learn;
            serve.params.learn_only = true;
            serve.params.res_type   = TASK_RESPONSE_TYPE_NONE;
            serve.id_slot           = task.learn.id_slot;
            auto & rd = res->rd;
            serve.id = rd.get_new_id();
            rd.post_task(std::move(serve));
            auto result = rd.next(req.should_stop);
            if (!result) {
                GGML_ASSERT(req.should_stop());
                return res;
            }
            if (result->is_error()) {
                res->error(result->to_json());
                return res;
            }
            res->ok(result->to_json());
            return res;
        } else if (body.contains("reply_tokens")) {
            task.learn.reply = body.at("reply_tokens").get<std::vector<llama_token>>();
        } else {
            task.learn.reply = common_tokenize(ctx_server.vocab, json_value(body, "reply", std::string()), false, true);
        }
        auto & rd = res->rd;
        task.id = rd.get_new_id();
        rd.post_task(std::move(task));
        auto result = rd.next(req.should_stop);
        if (!result) {
            GGML_ASSERT(req.should_stop());
            return res;
        }
        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }
        res->ok(result->to_json());
        return res;
    };

    this->post_lora_adapters_unload = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);
        if (!body.contains("engine") || !body.at("engine").is_string() || body.at("engine").get<std::string>().empty()) {
            res->error(format_error_response("engine incarnation is required", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        if (!body.contains("id") || !body.at("id").is_number_integer()) {
            res->error(format_error_response("\"id\" (an integer adapter id) is required", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        if (body.value("ticket", false)) {
            // a ticketed unload answers at once; its result arrives on GET /learn/events (card 7fc7e2ce)
            server_task task(SERVER_TASK_TYPE_UNLOAD_LORA);
            task.id          = queue_tasks.get_new_id();
            task.unload_lora = body.at("id").get<int32_t>();
            task.unload_engine = body.at("engine").get<std::string>();
            const int ticket = task.id;
            queue_results.add_ticket(ticket);
            queue_tasks.post(std::move(task));
            res->ok({{"ticket", ticket}});
            return res;
        }
        auto & rd = res->rd;
        {
            server_task task(SERVER_TASK_TYPE_UNLOAD_LORA);
            task.id          = rd.get_new_id();
            task.unload_lora = body.at("id").get<int32_t>();
            task.unload_engine = body.at("engine").get<std::string>();
            rd.post_task(std::move(task));
        }
        auto result = rd.next(req.should_stop);
        if (!result) {
            GGML_ASSERT(req.should_stop());
            return res;
        }
        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }
        res->ok(result->to_json());
        return res;
    };

    auto train_control = [this](const server_http_req & req, bool pause) {
        auto res = create_response();
        const json body = json::parse(req.body);
        if (!body.is_object() || !body.contains("out") || !body.at("out").is_string() || body.at("out").get<std::string>().empty()) {
            res->error(format_error_response("out must identify the training job", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        const std::string out = body.at("out");
        const json r = ctx_server.trainer ? (pause ? ctx_server.trainer->pause(out) : ctx_server.trainer->resume(out))
                                         : json::object({{"ok", false}, {"error", "no model is loaded"}});
        if (!r.value("ok", false)) {
            res->error(format_error_response(r.value("error", std::string("training control refused")), ERROR_TYPE_INVALID_REQUEST));
        } else {
            res->ok(r);
        }
        return res;
    };
    this->post_train_pause = [train_control](const server_http_req & req) { return train_control(req, true); };
    this->post_train_resume = [train_control](const server_http_req & req) { return train_control(req, false); };

    this->post_lora_adapters = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);
        if (!body.is_array()) {
            res->error(format_error_response("Request body must be an array", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        auto & rd = res->rd;
        {
            server_task task(SERVER_TASK_TYPE_SET_LORA);
            task.id = rd.get_new_id();
            task.set_lora = parse_lora_request(body);
            rd.post_task(std::move(task));
        }

        // get the result
        auto result = rd.next(req.should_stop);
        if (!result) {
            // connection was closed
            GGML_ASSERT(req.should_stop());
            return res;
        }

        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }

        GGML_ASSERT(dynamic_cast<server_task_result_apply_lora*>(result.get()) != nullptr);
        res->ok(result->to_json());
        return res;
    };
}

std::unique_ptr<server_res_generator> server_routes::handle_slots_save(const server_http_req & req, int id_slot, bool size_only) {
    auto res = create_response();
    const json request_data = json::parse(req.body);
    std::string filename = request_data.at("filename");
    if (!fs_validate_filename(filename)) {
        res->error(format_error_response("Invalid filename", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }
    std::string filepath = params.slot_save_path + filename;

    auto & rd = res->rd;
    {
        server_task task(SERVER_TASK_TYPE_SLOT_SAVE);
        task.id = rd.get_new_id();
        // NOTE (Continuum fork): bind the page-op to its slot. pop_deferred_task()
        // (called from callback_on_release when a slot finishes) matches on
        // task.id_slot, NOT slot_action.id_slot — leaving it at the -1 default meant
        // a deferred page-op was invisible to its own slot's release and got popped
        // out of order behind an unrelated deferred task. Save does not defer, so
        // this is a no-op for save today; set for consistency with restore/erase.
        task.id_slot = id_slot;
        task.slot_action.id_slot  = id_slot;
        task.slot_action.filename = filename;
        task.slot_action.filepath = filepath;
        task.slot_action.size_only = size_only;
        if (request_data.contains("admitted_bytes")) {
            if (!request_data.at("admitted_bytes").is_number_integer()
                || request_data.at("admitted_bytes").get<double>() < 0.0) {
                res->error(format_error_response("admitted_bytes must be an unsigned integer", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
            task.slot_action.has_byte_ceiling = true;
            task.slot_action.byte_ceiling = request_data.at("admitted_bytes").get<uint64_t>();
        }
        // Continuum fork: the checkpoint subset this page carries (see slot_action). The quote
        // (`save-size`) and the write must name the same subset, or the admitted bytes lie.
        if (request_data.contains("keep_newest_checkpoints")) {
            const auto & keep = request_data.at("keep_newest_checkpoints");
            if (!keep.is_number_integer() || keep.get<int64_t>() < 0 || keep.get<int64_t>() > 4096) {
                res->error(format_error_response("keep_newest_checkpoints must be an integer in [0, 4096]", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
            task.slot_action.keep_newest_checkpoints = (int32_t) keep.get<int64_t>();
        }
        if (request_data.contains("keep_view_checkpoints")) {
            if (!request_data.at("keep_view_checkpoints").is_boolean()) {
                res->error(format_error_response("keep_view_checkpoints must be a boolean", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
            task.slot_action.keep_view_checkpoints = request_data.at("keep_view_checkpoints").get<bool>();
        }
        rd.post_task(std::move(task));
    }

    auto result = rd.next(req.should_stop);
    if (!result) {
        // connection was closed
        GGML_ASSERT(req.should_stop());
        return res;
    }

    if (result->is_error()) {
        res->error(result->to_json());
        return res;
    }

    res->ok(result->to_json());
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_slots_restore(const server_http_req & req, int id_slot) {
    auto res = create_response();
    const json request_data = json::parse(req.body);
    std::string filename = request_data.at("filename");
    if (!fs_validate_filename(filename)) {
        res->error(format_error_response("Invalid filename", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }
    std::string filepath = params.slot_save_path + filename;

    std::shared_ptr<server_restore_progress> progress;
    if (request_data.contains("operation_id")) {
        const auto id = request_data.at("operation_id").get<std::string>();
        bool valid = id.size() == 36; // canonical UUID, bounded opaque identity
        for (size_t i = 0; valid && i < id.size(); ++i) {
            valid = (i == 8 || i == 13 || i == 18 || i == 23) ? id[i] == '-'
                : ((id[i] >= '0' && id[i] <= '9') || (id[i] >= 'a' && id[i] <= 'f'));
        }
        if (!valid || id_slot < 0 || id_slot >= params.n_parallel) {
            res->error(format_error_response("Invalid restore operation identity or slot", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        std::lock_guard<std::mutex> lock(mutex_restore_progress);
        for (const auto & entry : restore_progress) {
            if (entry.second->operation_id == id) {
                res->error(format_error_response("Restore operation identity is already retained", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
        }
        auto previous = restore_progress.find(id_slot);
        if (previous != restore_progress.end() && !previous->second->terminal()) {
            res->error(format_error_response("A restore operation already owns this slot", ERROR_TYPE_UNAVAILABLE));
            return res;
        }
        progress = std::make_shared<server_restore_progress>(id, id_slot);
        restore_progress[id_slot] = progress;
    }
    auto & rd = res->rd;
    {
        server_task task(SERVER_TASK_TYPE_SLOT_RESTORE);
        task.id = rd.get_new_id();
        // NOTE (Continuum fork): bind the restore to its slot — THIS is the
        // load-bearing case. RESTORE correctly defers while the slot is_processing()
        // (it mutates KV a live decode would read). It re-runs when the slot's
        // generation ends: callback_on_release -> pop_deferred_task(id_slot), which
        // matches task.id_slot. Without this bind (id_slot == -1) the returner's
        // restore was NOT the task popped when ITS slot freed — pop_deferred_task
        // fell through to "first deferred task", so a competition's slot-pinned
        // completion (which DOES set id_slot) was popped ahead of it and prefilled
        // COLD, while the orphaned restore landed later on a future occupant. Binding
        // it makes the returner's restore the task serviced for its slot, in order,
        // the instant that slot releases — correct and clobber-safe. (Measured
        // 2026-09-03: 27/27 restores failed status=0 before this; the Rust client's
        // 10s abandon compounded it — raised to a slot-wait alongside this.)
        task.id_slot = id_slot;
        task.slot_action.id_slot  = id_slot;
        task.slot_action.filename = filename;
        task.slot_action.filepath = filepath;
        task.slot_action.progress = progress;
        rd.post_task(std::move(task));
    }

    auto result = rd.next(req.should_stop);
    if (!result) {
        // Request cancellation is distinct from acknowledgement that IO stopped.
        if (progress) {
            progress->cancel_requested.store(true, std::memory_order_relaxed);
            auto expected = server_restore_progress::queued;
            progress->phase.compare_exchange_strong(expected, server_restore_progress::cancelled);
        }
        GGML_ASSERT(req.should_stop());
        return res;
    }

    if (result->is_error()) {
        res->error(result->to_json());
        return res;
    }

    GGML_ASSERT(dynamic_cast<server_task_result_slot_save_load*>(result.get()) != nullptr);
    res->ok(result->to_json());
    return res;
}

// Continuum fork: POST /minds/prefetch {"page": "<mind>"}: her state comes from flash into a parked sequence now,
// ahead of her turn (server-context-minds.inc). The reply names where she is.
// an abandoned turn's learn pin, released (the core sends the turn receipt's mind.page and mind.pin)
std::unique_ptr<server_res_generator> server_routes::handle_learn_release(const server_http_req & req) {
    auto res = create_response();
    auto & rd = res->rd;
    const json body = req.body.empty() ? json::object() : json::parse(req.body);
    {
        server_task task(SERVER_TASK_TYPE_LEARN_RELEASE);
        task.id = rd.get_new_id();
        task.params.page = mind_page_parse(json_value(body, "page", std::string()));
        task.params.learn_release_pin = json_value(body, "pin", (uint64_t) 0);
        if (json_value(body, "incarnation", (uint64_t) 0) != mind_incarnation()) {
            res->error(format_error_response("incarnation names another engine: this release was for a replaced engine", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        if (!task.params.page.valid() || task.params.learn_release_pin == 0) {
            res->error(format_error_response("page must name a mind and pin the turn receipt's mind.pin", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        rd.post_task(std::move(task));
    }
    auto result = rd.next(req.should_stop);
    if (!result) {
        GGML_ASSERT(req.should_stop());
        return res;
    }
    if (result->is_error()) {
        res->error(result->to_json());
        return res;
    }
    GGML_ASSERT(dynamic_cast<server_task_result_learn_release*>(result.get()) != nullptr);
    res->ok(result->to_json());
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_minds_prefetch(const server_http_req & req) {
    auto res = create_response();
    auto & rd = res->rd;
    const json body = req.body.empty() ? json::object() : json::parse(req.body);
    {
        server_task task(SERVER_TASK_TYPE_MIND_PREFETCH);
        task.id = rd.get_new_id();
        task.params.page = mind_page_parse(json_value(body, "page", std::string()));
        // a stream meant for another incarnation of this engine (a replacement after a restart) is refused, typed
        if (json_value(body, "incarnation", (uint64_t) 0) != mind_incarnation()) {
            res->error(format_error_response("incarnation names another engine: this prefetch was for a replaced engine", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        if (!task.params.page.valid()) {
            res->error(format_error_response("page must name a mind (32 hexadecimal digits, a UUID)", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        rd.post_task(std::move(task));
    }
    auto result = rd.next(req.should_stop);
    if (!result) {
        GGML_ASSERT(req.should_stop());
        return res;
    }
    if (result->is_error()) {
        res->error(result->to_json());
        return res;
    }
    GGML_ASSERT(dynamic_cast<server_task_result_mind_prefetch*>(result.get()) != nullptr);
    res->ok(result->to_json());
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_slots_erase(const server_http_req & req, int id_slot) {
    auto res = create_response();
    auto & rd = res->rd;
    {
        server_task task(SERVER_TASK_TYPE_SLOT_ERASE);
        task.id = rd.get_new_id();
        // NOTE (Continuum fork): bind to its slot — erase also defers while
        // is_processing() and must be serviced for its own slot on release. See the
        // restore arm above for the pop_deferred_task(id_slot) rationale.
        task.id_slot = id_slot;
        task.slot_action.id_slot = id_slot;
        rd.post_task(std::move(task));
    }

    auto result = rd.next(req.should_stop);
    if (!result) {
        // connection was closed
        GGML_ASSERT(req.should_stop());
        return res;
    }

    if (result->is_error()) {
        res->error(result->to_json());
        return res;
    }

    GGML_ASSERT(dynamic_cast<server_task_result_slot_erase*>(result.get()) != nullptr);
    res->ok(result->to_json());
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_embeddings_impl(const server_http_req & req, task_response_type res_type) {
    auto res = create_response();
    if (!params.embedding) {
        res->error(format_error_response("This server does not support embeddings. Start it with `--embeddings`", ERROR_TYPE_NOT_SUPPORTED));
        return res;
    }

    if (res_type != TASK_RESPONSE_TYPE_NONE && meta->pooling_type == LLAMA_POOLING_TYPE_NONE) {
        res->error(format_error_response("Pooling type 'none' is not OAI compatible. Please use a different pooling type", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    const json body = json::parse(req.body);

    // for the shape of input/content, see tokenize_input_prompts()
    json prompt;
    if (body.count("input") != 0) {
        prompt = body.at("input");
    } else if (body.contains("content")) {
        res_type = TASK_RESPONSE_TYPE_NONE; // "content" field is not OAI compatible
        prompt = body.at("content");
    } else {
        res->error(format_error_response("\"input\" or \"content\" must be provided", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    bool use_base64 = false;
    if (body.count("encoding_format") != 0) {
        const std::string & format = body.at("encoding_format");
        if (format == "base64") {
            use_base64 = true;
        } else if (format != "float") {
            res->error(format_error_response("The format to return the embeddings in. Can be either float or base64", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
    }

    auto tokenized_prompts = tokenize_input_prompts(ctx_server.vocab, ctx_server.mctx, prompt, true, true);
    for (const auto & tokens : tokenized_prompts) {
        // this check is necessary for models that do not add BOS token to the input
        if (tokens.empty()) {
            res->error(format_error_response("Input content cannot be empty", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
    }

    int embd_normalize = params.embd_normalize;
    if (body.count("embd_normalize") != 0) {
        embd_normalize = body.at("embd_normalize").get<int>();
        if (meta->pooling_type == LLAMA_POOLING_TYPE_NONE) {
            SRV_DBG("embd_normalize is not supported by pooling type %d, ignoring it\n", meta->pooling_type);
        }
    }

    // create and queue the task
    json responses = json::array();
    auto & rd = res->rd;
    {
        std::vector<server_task> tasks;
        for (size_t i = 0; i < tokenized_prompts.size(); i++) {
            server_task task = server_task(SERVER_TASK_TYPE_EMBEDDING);

            task.id     = rd.get_new_id();
            task.tokens = std::move(tokenized_prompts[i]);

            // OAI-compat
            task.params.res_type = res_type;
            task.params.embd_normalize = embd_normalize;

            tasks.push_back(std::move(task));
        }
        rd.post_tasks(std::move(tasks));
    }

    // wait for the results
    auto all_results = rd.wait_for_all(req.should_stop);

    // collect results
    if (all_results.is_terminated) {
        return res; // connection is closed
    } else if (all_results.error) {
        res->error(all_results.error->to_json());
        return res;
    } else {
        for (auto & res : all_results.results) {
            GGML_ASSERT(dynamic_cast<server_task_result_embd*>(res.get()) != nullptr);
            responses.push_back(res->to_json());
        }
    }

    // write JSON response
    json root = res_type == TASK_RESPONSE_TYPE_OAI_EMBD
        ? format_embeddings_response_oaicompat(body, meta->model_name, responses, use_base64)
        : json(responses);
    res->ok(root);
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_count_tokens(const llama_vocab * vocab, mtmd_context * mctx, const server_http_req & req, task_response_type res_type) {
    auto res = create_response();
    std::vector<raw_buffer> files;
    json body = json::parse(req.body);
    bool is_oai = false;

    switch (res_type) {
        case TASK_RESPONSE_TYPE_OAI_CHAT:
            {
                is_oai = true;
            } break;
        case TASK_RESPONSE_TYPE_OAI_RESP:
            {
                is_oai = true;
                body = server_chat_convert_responses_to_chatcmpl(body);
            } break;
        case TASK_RESPONSE_TYPE_ANTHROPIC:
            {
                body = server_chat_convert_anthropic_to_oai(body);
            } break;
        default:
            res->error(format_error_response("invalid res_type", ERROR_TYPE_INVALID_REQUEST));
            return res;
    }

    json body_parsed = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);
    json prompt = body_parsed.at("prompt");
    // SRV_DBG("prompt = %s\n", prompt.dump().c_str());

    // TODO @ngxson : refactor this code block, move this to server-common and reuse it in other places
    size_t n_tokens;
    if (mctx != nullptr) {
        if (!prompt.is_string()) {
            throw std::runtime_error("for mtmd, input prompt must be a string.");
        }
        n_tokens = process_mtmd_prompt(mctx, prompt.get<std::string>(), files, true).size();
    } else {
        n_tokens = tokenize_mixed(vocab, prompt, true, true).size();
    }

    json response = {{"input_tokens", static_cast<int64_t>(n_tokens)}};
    if (is_oai) {
        response["object"] = "response.input_tokens";
    }
    res->ok(response);
    return res;
}

void server_routes::update_cached_responses(bool is_sleeping) {
    // caller is task_queue, so ctx_server can be accessed without holding locks
    std::unique_lock<std::mutex> lock(mutex_cache);

    if (is_sleeping) {
        cached_models  = get_res_models(*meta);
        cached_props   = get_res_props(*meta, params, true);
        cached_metrics = ctx_server.get_metrics();

        should_reset_buckets = false;

        SRV_DBG("%s\n", "cached responses updated");

    } else if (should_reset_buckets) {
        // a scrape during sleep already reported these buckets
        ctx_server.reset_metrics_bucket();

        should_reset_buckets = false;
    }
}
