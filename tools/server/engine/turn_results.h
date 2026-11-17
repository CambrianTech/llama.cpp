#pragma once

// Continuum fork (CONTINUUM-ENGINE 4.12.1): what a turn says, built from its slot (errors, partial and final results,
// embeddings, rerank). Where it goes is a responder's (responder.h).

#include "../server-slot.h"

server_task_result_ptr turn_error(int id_task, const std::string & error, error_type type = ERROR_TYPE_SERVER, int32_t n_prompt_tokens = 0, int32_t n_ctx = 0);
server_task_result_ptr turn_error(const server_task & task, const std::string & error, error_type type = ERROR_TYPE_SERVER);
server_task_result_ptr turn_error(const server_slot & slot, const std::string & error, error_type type = ERROR_TYPE_SERVER);
server_task_result_ptr turn_partial(server_slot & slot, const completion_token_output & tkn, bool is_progress, bool is_begin = false);
server_task_result_ptr turn_final(server_slot & slot, llama_context * ctx, bool keep_debug_text);
server_task_result_ptr turn_embedding(const server_slot & slot, const llama_batch & batch, const llama_model * model);
server_task_result_ptr turn_rerank(const server_slot & slot, const llama_batch & batch, llama_context * ctx);
