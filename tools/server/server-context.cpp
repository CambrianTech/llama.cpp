// Continuum fork: struct server_context_impl and its helpers live in server-context-impl.h; route families in
// server-routes-*.cpp.
#include "server-context-impl.h"


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


void server_context::set_state_callback(server_state_callback_t callback) {
    impl->callback_state = std::move(callback);
}

//
// server_routes
//


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




json server_routes::get_model_info() const {
    return get_res_model_info(*meta);
}

void server_routes::init_routes() {
    // IMPORTANT: all lambda functions must start with create_response()
    // this is to ensure that the server_res_generator can handle sleeping case correctly
    init_routes_status();
    init_routes_generation();
    init_routes_tokens();
    init_routes_lora();
    init_routes_learn();
    init_routes_control();
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
