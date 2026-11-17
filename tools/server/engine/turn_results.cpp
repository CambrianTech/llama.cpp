// Continuum fork (CONTINUUM-ENGINE 4.12.1): what a turn says, built from its slot; where it goes is a responder's.
// The bodies are the former server_context_impl::send_* methods, which built a result and sent it; these return it.

#include "turn_results.h"

server_task_result_ptr turn_error(int id_task, const std::string & error, error_type type, int32_t n_prompt_tokens, int32_t n_ctx) {
    SRV_ERR("task id = %d, error: %s\n", id_task, error.c_str());

    if (type == ERROR_TYPE_EXCEED_CONTEXT_SIZE) {
        GGML_ASSERT(n_ctx > 0 && n_prompt_tokens > 0);
    }

    auto res = std::make_unique<server_task_result_error>();
    res->id              = id_task;
    res->err_type        = type;
    res->err_msg         = error;
    res->n_prompt_tokens = n_prompt_tokens;
    res->n_ctx           = n_ctx;

    return res;
}

server_task_result_ptr turn_error(const server_task & task, const std::string & error, error_type type) {
    return turn_error(task.id, error, type);
}

server_task_result_ptr turn_error(const server_slot & slot, const std::string & error, error_type type) {
    return turn_error(slot.task->id, error, type, slot.task->n_tokens(), slot.n_ctx);
}

server_task_result_ptr turn_partial(server_slot & slot, const completion_token_output & tkn, bool is_progress, bool is_begin) {
    auto res = std::make_unique<server_task_result_cmpl_partial>();

    res->id    = slot.task->id;
    res->index = slot.task->index;

    if (is_progress) {
        res->is_progress        = true;
        res->progress.total     = slot.task->n_tokens();
        res->progress.cache     = slot.stats.n_prompt_cached;
        res->progress.processed = slot.prompt.tokens.size();
        res->progress.time_ms   = slot.stats.t_elapsed_us() / 1000;
    }
    if (is_begin) {
        res->is_begin = true;
    } else {
        res->content = tkn.text_to_send;
        res->tokens  = { tkn.tok };
    }

    res->n_decoded             = slot.stats.n_gen;
    res->n_prompt_tokens       = slot.task->n_tokens();
    res->n_prompt_tokens_cache = slot.stats.n_prompt_cached;
    res->post_sampling_probs   = slot.task->params.post_sampling_probs;

    res->verbose           = slot.task->params.verbose;
    res->res_type          = slot.task->params.res_type;
    res->oaicompat_model   = slot.task->params.oaicompat_model;
    res->oaicompat_cmpl_id = slot.task->params.oaicompat_cmpl_id;

    // populate res.probs_output
    if (slot.task->params.sampling.n_probs > 0) {
        res->prob_output = tkn; // copy the token probs
    }

    // populate timings if this is final response or timings_per_token is enabled
    if (slot.stop != STOP_TYPE_NONE || slot.task->params.timings_per_token) {
        res->stats = slot.stats;
    }

    return res;
}

server_task_result_ptr turn_final(server_slot & slot, llama_context * ctx, bool keep_debug_text) {
    if (slot.task->params.learn_only) {
        auto res = std::make_unique<server_task_result_learn>();
        res->id = slot.task->id;
        if (slot.learned) {
            res->outcome = *slot.learned;
        } else {
            res->outcome.failure = "the served prompt finished without its learn step";
        }
        return res;
    }
    auto res = std::make_unique<server_task_result_cmpl_final>();

    res->id      = slot.task->id;
    res->id_slot = slot.id;
    res->learned = slot.learned;

    res->index = slot.task->index;

    // keep copy of last generated text for debugging purposes
    if (keep_debug_text) {
        slot.debug_generated_text = slot.generated_text;
    }

    // in stream mode, content and tokens are already in last partial chunk
    if (slot.task->params.stream) {
        res->content     = "";
        res->tokens      = llama_tokens{};
    } else {
        res->content     = std::move(slot.generated_text);
        res->tokens      = std::move(slot.generated_tokens);
    }
    res->stats           = slot.stats;
    res->prompt          = slot.task->tokens.detokenize(ctx, true);
    res->response_fields = std::move(slot.task->params.response_fields);

    res->truncated             = slot.truncated;
    res->n_decoded             = slot.stats.n_gen;
    res->n_prompt_tokens       = slot.task->n_tokens();
    res->n_prompt_tokens_cache = slot.stats.n_prompt_cached;
    res->n_tokens_cached       = slot.prompt.n_tokens();
    res->has_new_line          = slot.has_new_line;
    res->stopping_word         = slot.stopping_word;
    res->stop                  = slot.stop;
    res->post_sampling_probs   = slot.task->params.post_sampling_probs;

    res->verbose           = slot.task->params.verbose;
    res->stream            = slot.task->params.stream;
    res->include_usage     = slot.task->params.include_usage;
    res->res_type          = slot.task->params.res_type;
    res->oaicompat_model   = slot.task->params.oaicompat_model;
    res->oaicompat_cmpl_id = slot.task->params.oaicompat_cmpl_id;

    // populate res.probs_output
    if (slot.task->params.sampling.n_probs > 0) {
        if (!slot.task->params.stream && slot.stop == STOP_TYPE_WORD) {
            const llama_tokens stop_word_toks = common_tokenize(ctx, slot.stopping_word, false);

            size_t safe_offset = std::min(slot.generated_token_probs.size(), stop_word_toks.size());
            res->probs_output = std::vector<completion_token_output>(
                    slot.generated_token_probs.begin(),
                    slot.generated_token_probs.end() - safe_offset);
        } else {
            res->probs_output = std::vector<completion_token_output>(
                    slot.generated_token_probs.begin(),
                    slot.generated_token_probs.end());
        }
    }

    res->generation_params = slot.task->params; // copy the parameters

    return res;
}

server_task_result_ptr turn_embedding(const server_slot & slot, const llama_batch & batch, const llama_model * model) {
    auto res = std::make_unique<server_task_result_embd>();
    res->id        = slot.task->id;
    res->index     = slot.task->index;
    res->n_tokens  = slot.task->n_tokens();
    res->res_type  = slot.task->params.res_type;

    const int n_embd_out = llama_model_n_embd_out(model);

    std::vector<float> embd_res(n_embd_out, 0.0f);

    for (int i = 0; i < batch.n_tokens; ++i) {
        if (!batch.logits[i] || batch.seq_id[i][0] != slot.id) {
            continue;
        }

        const float * embd = nullptr;
        if (llama_pooling_type(slot.ctx_tgt) == LLAMA_POOLING_TYPE_NONE) {
            embd = llama_get_embeddings_ith(slot.ctx_tgt, i);
        } else {
            embd = llama_get_embeddings_seq(slot.ctx_tgt, batch.seq_id[i][0]);
        }

        if (embd == nullptr) {
            SLT_ERR(slot, "failed to get embeddings, token = %d, seq_id = %d\n", batch.token[i], batch.seq_id[i][0]);

            res->embedding.push_back(std::vector<float>(n_embd_out, 0.0f));
            continue;
        }

        // normalize only when there is pooling
        if (llama_pooling_type(slot.ctx_tgt) != LLAMA_POOLING_TYPE_NONE) {
            common_embd_normalize(embd, embd_res.data(), n_embd_out, slot.task->params.embd_normalize);
            res->embedding.push_back(embd_res);
            break;
        }

        res->embedding.emplace_back(embd, embd + n_embd_out);
    }

    SLT_DBG(slot, "%s", "sending embeddings\n");

    return res;
}

server_task_result_ptr turn_rerank(const server_slot & slot, const llama_batch & batch, llama_context * ctx) {
    auto res = std::make_unique<server_task_result_rerank>();
    res->id       = slot.task->id;
    res->index    = slot.task->index;
    res->n_tokens = slot.task->n_tokens();

    for (int i = 0; i < batch.n_tokens; ++i) {
        if (!batch.logits[i] || batch.seq_id[i][0] != slot.id) {
            continue;
        }

        const float * embd = llama_get_embeddings_seq(ctx, batch.seq_id[i][0]);
        if (embd == NULL) {
            embd = llama_get_embeddings_ith(ctx, i);
        }

        if (embd == NULL) {
            SLT_ERR(slot, "failed to get embeddings, token = %d, seq_id = %d\n", batch.token[i], batch.seq_id[i][0]);

            res->score = -1e6;
            continue;
        }

        res->score = embd[0];
    }

    SLT_DBG(slot, "sending rerank result, res.score = %f\n", res->score);

    return res;
}
