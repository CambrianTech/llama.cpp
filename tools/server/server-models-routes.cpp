// Continuum fork: the router's HTTP routes; out of server-models.cpp (pure move; the 1,000-line rule).

#include "server-models-internal.h"

//
// server_models_routes
//

// RAII wrapper similar to server_response_reader, but doesn't use server_queue
static std::atomic<int> sse_client_id_counter = 0;
struct server_models_sse_client {
    server_response & queue_results;
    int client_id;
    server_models_sse_client(server_response & q)
            : queue_results(q), client_id(sse_client_id_counter.fetch_add(1, std::memory_order_relaxed)) {
        SRV_DBG("new SSE client connected, assigned client_id=%d\n", client_id);
        queue_results.add_waiting_task_id(client_id);
    }
    ~server_models_sse_client() {
        SRV_DBG("SSE client disconnected, removing client_id=%d\n", client_id);
        queue_results.remove_waiting_task_id(client_id);
    }

    // return nullptr if should_stop() is true before receiving a result
    // note: if one error is received, it will stop further processing and return error result
    server_task_result_ptr next(const std::function<bool()> & should_stop) {
        while (true) {
            static const int http_polling_seconds = 1; // check should_stop every 1 second
            server_task_result_ptr result = queue_results.recv_with_timeout({client_id}, http_polling_seconds);
            if (result == nullptr) {
                // timeout, check stop condition
                if (should_stop()) {
                    return nullptr;
                }
                // continue waiting otherwise
            } else {
                SRV_DBG("recv result for client_id=%d: %s\n", client_id, safe_json_to_str(result->to_json()).c_str());
                return result;
            }
        }
        // should not reach here
    }
};

static void res_ok(std::unique_ptr<server_http_res> & res, const json & response_data) {
    res->status = 200;
    res->data = safe_json_to_str(response_data);
}

static void res_err(std::unique_ptr<server_http_res> & res, const json & error_data) {
    res->status = json_value(error_data, "code", 500);
    res->data = safe_json_to_str({{ "error", error_data }});
}

static bool router_validate_model(std::string & name, server_models & models, bool models_autoload, std::unique_ptr<server_http_res> & res) {
    if (name.empty()) {
        res_err(res, format_error_response("model name is missing from the request", ERROR_TYPE_INVALID_REQUEST));
        return false;
    }
    auto meta = models.get_meta(name);
    if (!meta.has_value()) {
        res_err(res, format_error_response(string_format("model '%s' not found", name.c_str()), ERROR_TYPE_INVALID_REQUEST));
        return false;
    }
    // resolve alias to canonical model name
    name = meta->name;
    if (!models_autoload && !meta->is_running()) {
        res_err(res, format_error_response("model is not loaded", ERROR_TYPE_INVALID_REQUEST));
        return false;
    }
    return true;
}

static bool is_autoload(const common_params & params, const server_http_req & req) {
    std::string autoload = req.get_param("autoload");
    if (autoload.empty()) {
        return params.models_autoload;
    } else {
        return autoload == "true" || autoload == "1";
    }
}

// percent encode one query or path component, covers reserved chars without pulling in
// httplib::detail. used by the stream routes to forward conversation_id to children safely
static std::string encode_qs(const std::string & in) {
    std::string out;
    out.reserve(in.size() * 3);
    for (unsigned char c : in) {
        bool safe = (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9')
                 || c == '-' || c == '_' || c == '.' || c == '~';
        if (safe) {
            out.push_back(char(c));
        } else {
            char buf[4];
            std::snprintf(buf, sizeof(buf), "%%%02X", c);
            out.append(buf, 3);
        }
    }
    return out;
}

// resolve the child that owns a conversation's stream session via the conv_id -> model map
// populated when the POST was routed. single map lookup then a meta lookup, no polling, no
// parsing of the conv id. returns nullopt when nothing maps, the caller answers not found and
// the client recovers
static std::optional<server_model_meta> resolve_child_for_conv(
        server_models & models, const std::string & conversation_id) {
    if (conversation_id.empty()) {
        return std::nullopt;
    }
    auto tracked = models.conv_models.lookup(conversation_id);
    if (!tracked.has_value()) {
        return std::nullopt;
    }
    auto meta = models.get_meta(*tracked);
    if (meta.has_value() && meta->is_ready()) {
        return meta;
    }
    return std::nullopt;
}

void server_models_routes::init_routes() {
    if (!common_subproc::is_supported()) {
        throw std::runtime_error("subprocess is not enabled on this build");
    }

    this->get_router_props = [this](const server_http_req & req) {
        std::string name = req.get_param("model");
        if (name.empty()) {
            // main instance
            auto res = std::make_unique<server_http_res>();
            res_ok(res, {
                // TODO: add support for this on web UI
                {"role",                 "router"},
                {"max_instances",        params.models_max},
                {"models_autoload",      params.models_autoload},
                // this is a dummy response to make sure the UI doesn't break
                {"model_alias", "llama-server"},
                {"model_path",  "none"},
                {"default_generation_settings", {
                    {"params", json{}},
                    {"n_ctx",  0},
                }},
                // New key
                {"ui_settings",          ui_settings},
                {"build_info",           std::string(llama_build_info())},
                {"cors_proxy_enabled",   params.ui_mcp_proxy},
            });
            return res;
        }
        return proxy_get(req);
    };

    this->proxy_get = [this](const server_http_req & req) {
        std::string method = "GET";
        std::string name = req.get_param("model");
        bool autoload = is_autoload(params, req);
        auto error_res = std::make_unique<server_http_res>();
        if (!router_validate_model(name, models, autoload, error_res)) {
            return error_res;
        }
        if (autoload) {
            models.ensure_model_ready(name, req.should_stop);
        }
        return models.proxy_request(req, method, name, false);
    };

    this->proxy_post = [this](const server_http_req & req) {
        std::string method = "POST";
        json body = json::parse(req.body);
        std::string name = json_value(body, "model", std::string());
        bool autoload = is_autoload(params, req);
        auto error_res = std::make_unique<server_http_res>();
        if (!router_validate_model(name, models, autoload, error_res)) {
            return error_res;
        }
        // remember which child serves this conversation so the stream routes can route straight
        // to it without polling, keyed on the exact conv id from the header. registered before
        // the load wait so a stop issued while the model loads can erase the entry and cancel
        // this request instead of leaving an orphan generation
        std::string conv_id = server_stream_conv_id_from_headers(req.headers);
        uint64_t ticket = models.conv_models.remember(conv_id, name);
        // a dead socket must not cancel a session request, only a stop does (checked right below)
        auto should_stop = ticket == 0 ? req.should_stop : nullptr;
        bool waited = autoload && models.ensure_model_ready(name, should_stop);
        if (ticket != 0 && !models.conv_models.alive(conv_id, ticket)) {
            SRV_INF("request for conv_id=%s cancelled while model name=%s was loading\n",
                    conv_id.c_str(), name.c_str());
            res_err(error_res, format_error_response(
                    "request cancelled by a stop while the model was loading", ERROR_TYPE_INVALID_REQUEST));
            return error_res;
        }
        // a session request that waited for a load detaches from the client socket: the
        // client may have dropped during the wait (page reload) and the session buffer must
        // still receive the generation for a later resume
        return models.proxy_request(req, method, name, true, waited && ticket != 0); // update last usage for POST request only
    };

    this->post_router_models_load = [this](const server_http_req & req) {
        auto res = std::make_unique<server_http_res>();
        json body = json::parse(req.body);
        std::string name = json_value(body, "model", std::string());
        auto meta = models.get_meta(name);
        if (!meta.has_value()) {
            res_err(res, format_error_response("model is not found", ERROR_TYPE_NOT_FOUND));
            return res;
        }
        if (meta->is_running()) {
            res_err(res, format_error_response("model is already running", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        models.load(meta->name);
        res_ok(res, {{"success", true}});
        return res;
    };

    this->get_router_models = [this](const server_http_req & req) {
        bool reload = !req.get_param("reload", "").empty();
        if (reload) {
            models.load_models();
        }
        auto res = std::make_unique<server_http_res>();
        json models_json = json::array();
        auto all_models = models.get_all_meta();
        std::time_t t = std::time(0);
        for (const auto & meta : all_models) {
            if (meta.hidden) {
                continue; // cache model deduplicated by a preset
            }
            json status {
                {"value",  server_model_status_to_string(meta.status)},
                {"args",   meta.args},
            };
            if (!meta.preset.name.empty()) {
                common_preset preset_copy = meta.preset;
                unset_reserved_args(preset_copy, false);
                preset_copy.unset_option("LLAMA_ARG_HOST");
                preset_copy.unset_option("LLAMA_ARG_PORT");
                preset_copy.unset_option("LLAMA_ARG_ALIAS");
                preset_copy.unset_option("LLAMA_ARG_TAGS");
                status["preset"] = preset_copy.to_ini();
            }
            if (meta.is_failed()) {
                status["exit_code"] = meta.exit_code;
                status["failed"]    = true;
            }

            // pi coding agent multimodal compatibility
            json input_modalities = json::array({"text"});
            if (meta.multimodal.inp_vision) {
                input_modalities.push_back("image");
            }
            if (meta.multimodal.inp_audio) {
                input_modalities.push_back("audio");
            }
            json architecture {
                {"input_modalities",  input_modalities},
                {"output_modalities", json::array({"text"})},
            };

            json model_info = json {
                {"id",            meta.name},
                {"aliases",       meta.aliases},
                {"tags",          meta.tags},
                {"object",        "model"},    // for OAI-compat
                {"owned_by",      "llamacpp"}, // for OAI-compat
                {"created",       t},          // for OAI-compat
                {"status",        status},
                {"architecture",  architecture},
                {"source",        server_model_source_to_string(meta.source)},
                {"can_remove",    meta.source == SERVER_MODEL_SOURCE_CACHE},
                // {"need_download", meta.need_download},
                // TODO: add other fields, may require reading GGUF metadata
            };

            // merge with loaded_info from the child process if available
            if (meta.is_running()) {
                for (auto it = meta.loaded_info.begin(); it != meta.loaded_info.end(); ++it) {
                    if (!model_info.contains(it.key())) {
                        model_info[it.key()] = it.value();
                    }
                }
            }
            models_json.push_back(model_info);
        }
        res_ok(res, {
            {"data", models_json},
            {"object", "list"},
        });
        return res;
    };

    this->post_router_models_unload = [this](const server_http_req & req) {
        auto res = std::make_unique<server_http_res>();
        json body = json::parse(req.body);
        std::string name = json_value(body, "model", std::string());
        auto model = models.get_meta(name);
        if (!model.has_value()) {
            res_err(res, format_error_response("model is not found", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        if (!model->is_running() && model->status != SERVER_MODEL_STATUS_DOWNLOADING) {
            res_err(res, format_error_response("model is not running", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        models.unload(model->name);
        res_ok(res, {{"success", true}});
        return res;
    };

    this->get_router_models_sse = [this](const server_http_req & req) {
        auto res = std::make_unique<server_http_res>();
        res->status = 200;
        res->content_type = "text/event-stream";
        auto sse_client = std::make_shared<server_models_sse_client>(models.sse);
        res->next = [this, sse_client, &req](std::string & output) -> bool {
            auto result = sse_client->next([&]() {
                return stopping.load(std::memory_order_relaxed) || req.should_stop();
            });
            if (result == nullptr) {
                return false; // client disconnected or should_stop
            }
            output = "data: " + safe_json_to_str(result->to_json()) + "\n\n";
            return true; // listen for the next event
        };
        return res;
    };

    this->post_router_models = [this](const server_http_req & req) {
        auto res = std::make_unique<server_http_res>();

        json body = json::parse(req.body);
        std::string name = json_value(body, "model", std::string());
        if (name.empty()) {
            throw std::invalid_argument("model must be a non-empty string");
        }

        common_params p;
        p.model.hf_repo  = name;
        p.hf_token       = params.hf_token;

        // validate by fetching metadata
        bool ok = false;
        try {
            common_models_handler_init(p, LLAMA_EXAMPLE_SERVER);
            ok = true;
        } catch (...) {
            SRV_ERR("unknown error while validating model '%s'\n", name.c_str());
            // other exceptions will be handled by the outer ex_wrapper()
            throw;
        }

        if (!ok) {
            throw std::invalid_argument("model validation failed, unable to download");
        }

        // reject if model already exists
        if (models.has_model(name)) {
            throw std::invalid_argument("model '" + name + "' already exists");
        }

        // then, proceed with the actual download
        SRV_INF("starting download for model '%s'\n", name.c_str());
        {
            server_models::load_options load_opts;
            load_opts.mode = SERVER_CHILD_MODE_DOWNLOAD;
            load_opts.custom_meta = server_model_meta{};
            load_opts.custom_meta->source = SERVER_MODEL_SOURCE_CACHE;
            load_opts.custom_meta->name   = name;
            models.load(name, load_opts);
        }

        res_ok(res, {{"success", true}});
        return res;
    };

    this->del_router_models = [this](const server_http_req & req) {
        auto res = std::make_unique<server_http_res>();

        std::string name = req.get_param("model");
        if (name.empty()) {
            throw std::invalid_argument("model must be a non-empty string");
        }

        models.remove(name); // throws on error

        res_ok(res, {{"success", true}});
        return res;
    };

    this->router_stream_get = [this](const server_http_req & req) {
        // GET /v1/stream?conv_id=<id>&from=N. resolve the owning child from the conv_id -> model
        // map, 404 when nothing maps
        auto res = std::make_unique<server_http_res>();
        std::string conv_id = req.get_param("conv_id");
        if (conv_id.empty()) {
            res_err(res, format_error_response("Missing conversation id in path", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        std::optional<server_model_meta> owner = resolve_child_for_conv(models, conv_id);
        if (!owner.has_value()) {
            // a registered conv whose model is still loading earns a retry: the session appears
            // once the load ends and the pending request reaches the child
            auto tracked = models.conv_models.lookup(conv_id);
            auto meta = tracked.has_value() ? models.get_meta(*tracked) : std::nullopt;
            bool transient = meta.has_value() && (meta->status == SERVER_MODEL_STATUS_LOADING ||
                                                  meta->status == SERVER_MODEL_STATUS_DOWNLOADING ||
                                                  meta->status == SERVER_MODEL_STATUS_DOWNLOADED);
            if (transient) {
                res_err(res, format_error_response("Stream owner model is loading, retry later", ERROR_TYPE_UNAVAILABLE));
            } else {
                res_err(res, format_error_response("Stream not found or expired", ERROR_TYPE_NOT_FOUND));
            }
            return res;
        }
        std::string from = req.get_param("from");
        std::string child_path = "/v1/stream?conv_id=" + encode_qs(conv_id);
        if (!from.empty()) {
            child_path += "&from=" + from;
        }
        SRV_TRC("proxying stream resume to model %s on port %d, path=%s\n",
                owner->name.c_str(), owner->port, child_path.c_str());
        auto proxy = std::make_unique<server_http_proxy>(
                "GET",
                "http",
                CHILD_ADDR,
                owner->port,
                child_path,
                req.headers,
                req.body,
                req.files,
                req.should_stop,
                params.timeout_read,
                params.timeout_write);
        return std::unique_ptr<server_http_res>(std::move(proxy));
    };

    this->router_streams_lookup = [this](const server_http_req & req) {
        // POST /v1/streams/lookup. resolve each requested conv id to its owning child via the
        // map, group the ids per child, and query only the children that actually own some of
        // them instead of fanning out to every ready child. a child only answers for the ids
        // it owns, never lists anything else
        auto res = std::make_unique<server_http_res>();
        std::vector<std::string> requested;
        try {
            json body = json::parse(req.body);
            if (body.contains("conversation_ids") && body["conversation_ids"].is_array()) {
                for (const auto & v : body["conversation_ids"]) {
                    if (v.is_string() && !v.get<std::string>().empty()) {
                        requested.push_back(v.get<std::string>());
                    }
                }
            }
        } catch (const std::exception &) {
            res_ok(res, json::array());
            return res;
        }

        // group requested ids by the child port that owns them, drop ids that map to nothing
        std::unordered_map<int, json> per_child;
        for (const auto & cid : requested) {
            auto owner = resolve_child_for_conv(models, cid);
            if (!owner.has_value()) {
                continue;
            }
            per_child[owner->port].push_back(cid);
        }

        json aggregated = json::array();
        for (auto & [port, ids] : per_child) {
            json child_body = {{"conversation_ids", ids}};
            httplib::Client cli(CHILD_ADDR, port);
            cli.set_connection_timeout(0, STREAM_LOOKUP_TIMEOUT_MS * 1000);
            cli.set_read_timeout(0, STREAM_LOOKUP_TIMEOUT_MS * 1000);
            cli.set_write_timeout(0, STREAM_LOOKUP_TIMEOUT_MS * 1000);
            auto resp = cli.Post("/v1/streams/lookup", child_body.dump(), "application/json");
            if (!resp || resp->status != 200) {
                continue;
            }
            try {
                json child_arr = json::parse(resp->body);
                if (!child_arr.is_array()) {
                    continue;
                }
                for (auto & entry : child_arr) {
                    if (entry.is_object()) {
                        aggregated.push_back(entry);
                    }
                }
            } catch (const std::exception &) {
                continue;
            }
        }
        res_ok(res, aggregated);
        return res;
    };

    this->router_stream_delete = [this](const server_http_req & req) {
        // DELETE /v1/stream?conv_id=<id>. resolve the owning child via the map and forward only to
        // it, evict_and_cancel is idempotent on the child
        auto res = std::make_unique<server_http_res>();
        std::string conv_id = req.get_param("conv_id");
        if (conv_id.empty()) {
            res_err(res, format_error_response("Missing conversation id in path", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        std::string child_path = "/v1/stream?conv_id=" + encode_qs(conv_id);
        auto owner = resolve_child_for_conv(models, conv_id);
        if (owner.has_value()) {
            httplib::Client cli(CHILD_ADDR, owner->port);
            cli.set_connection_timeout(0, STREAM_LOOKUP_TIMEOUT_MS * 1000);
            cli.set_read_timeout(0, STREAM_LOOKUP_TIMEOUT_MS * 1000);
            cli.set_write_timeout(0, STREAM_LOOKUP_TIMEOUT_MS * 1000);
            auto resp = cli.Delete(child_path.c_str());
            (void) resp; // the child logs its own miss when the session is unknown there
        } else if (auto tracked = models.conv_models.lookup(conv_id); tracked.has_value()) {
            // the entry exists but its model is still loading: the forget below erases it,
            // which cancels the request parked in proxy_post before the generation starts
            SRV_INF("router stop for conv_id=%s while model name=%s is loading, cancelling the pending request\n",
                    conv_id.c_str(), tracked->c_str());
        } else {
            SRV_WRN("router stop for unknown conv_id=%s, no owning child in the conv map\n",
                    conv_id.c_str());
        }
        // drop the tracking entry, the session is being torn down
        models.conv_models.forget(conv_id);
        res->status = 204;
        res->content_type = "application/json";
        return res;
    };
}
