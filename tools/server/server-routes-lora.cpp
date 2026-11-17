// Continuum fork: the lora routes (LoRA adapters and training: list, save, load, fresh, train, measure, cancel), out of server_routes::init_routes (pure move; the
// 1,000-line rule).

#include "server-context-impl.h"

void server_routes::init_routes_lora() {
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
}
