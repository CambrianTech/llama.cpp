// Continuum fork: the learn routes (learning, residency, grants, prompt cache limit, page IO, minds prefetch), out of server_routes::init_routes (pure move; the
// 1,000-line rule).

#include "server-context-impl.h"

void server_routes::init_routes_learn() {
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
}
