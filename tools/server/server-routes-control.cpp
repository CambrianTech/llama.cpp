// Continuum fork: the control routes (adapter unload, training pause and resume, adapter set), out of server_routes::init_routes (pure move; the
// 1,000-line rule).

#include "server-context-impl.h"

void server_routes::init_routes_control() {
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
