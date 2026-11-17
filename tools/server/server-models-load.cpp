// Continuum fork: server_models: load, unload, status, waiting, readiness, proxying, child state; out of server-models.cpp (pure move; the 1,000-line rule).

#include "server-models-internal.h"

void server_models::load(const std::string & name) {
    load(name, load_options{});
}

void server_models::load(const std::string & name, const load_options & opts) {
    if (debug_fake_timing) {
        // do not hold the mutex here, other requests must keep making progress
        std::this_thread::sleep_for(std::chrono::seconds(2));
    }

    if (!opts.custom_meta.has_value()) {
        if (!has_model(name)) {
            throw std::runtime_error("model name=" + name + " is not found");
        }
        unload_lru();
    }

    std::unique_lock<std::mutex> lk(mutex);
    // edge case: block until any in-progress reload has finished so we always load
    // against the freshest preset and a consistent mapping state
    cv.wait(lk, [this]() { return !is_reloading; });

    auto meta = opts.custom_meta.has_value() ? *opts.custom_meta : mapping[name].meta;
    if (meta.status != SERVER_MODEL_STATUS_UNLOADED) {
        SRV_INF("model %s is not ready\n", name.c_str());
        return;
    }

    // Re-check capacity under the lock to prevent concurrent loads from
    // exceeding models_max. Without this, the window between unload_lru()
    // releasing its lock and this lock_guard acquiring allows multiple
    // threads to each observe capacity and all proceed to load.
    if (base_params.models_max > 0) {
        size_t count_active = 0;
        for (const auto & m : mapping) {
            if (m.second.meta.is_running()) {
                count_active++;
            }
        }
        if (count_active >= (size_t)base_params.models_max) {
            throw std::runtime_error("model limit reached, try again later");
        }
    }

    // prepare new instance info
    instance_t inst;
    inst.meta             = meta;
    inst.meta.port        = common_http_get_free_port();
    inst.meta.status      = SERVER_MODEL_STATUS_LOADING;
    inst.meta.loaded_info = json{};
    inst.meta.last_used   = ggml_time_ms();

    if (inst.meta.port <= 0) {
        throw std::runtime_error("failed to get a port number");
    }

    inst.subproc = std::make_shared<server_subproc>();
    {
        SRV_INF("spawning server instance with name=%s on port %d\n", inst.meta.name.c_str(), inst.meta.port);

        inst.meta.update_args(ctx_preset, bin_path); // render args

        std::vector<std::string> child_args = inst.meta.args; // copy
        std::vector<std::string> child_env  = base_env; // copy
        child_env.push_back("LLAMA_SERVER_ROUTER_PORT=" + std::to_string(base_params.port));

        if (opts.mode == SERVER_CHILD_MODE_DOWNLOAD) {
            inst.meta.status = SERVER_MODEL_STATUS_DOWNLOADING;
            child_env.push_back("LLAMA_SERVER_CHILD_MODE=download");
            child_env.push_back("LLAMA_ARG_HF_REPO=" + name);
        }

        SRV_INF("%s", "spawning server instance with args:\n");
        for (const auto & arg : child_args) {
            SRV_INF("  %s\n", arg.c_str());
        }
        inst.meta.args = child_args; // save for debugging

        // TODO @ngxson : maybe separate stdout and stderr in the future
        //                so that we can use stdout for commands and stderr for logging
        int options = subprocess_option_no_window | subprocess_option_combined_stdout_stderr;
        if (!inst.subproc->sproc.create(child_args, options, child_env)) {
            throw std::runtime_error("failed to spawn server instance");
        }
    }

    // start a thread to manage the child process
    // captured variables are guaranteed to be destroyed only after the thread is joined
    inst.th = std::thread([
        this, name,
        child_proc = inst.subproc,
        port = inst.meta.port,
        stop_timeout = inst.meta.stop_timeout,
        child_mode = opts.mode
    ]() {
        FILE * stdin_file = child_proc->sproc.stdin_file();
        FILE * stdout_file = child_proc->sproc.stdout_file(); // combined stdout/stderr

        std::thread log_thread([&]() {
            // read stdout/stderr and forward to main server log
            // also handle status report from child process
            std::vector<char> vec_buf(128 * 1024); // large buffer for storing info
            char * buffer = vec_buf.data();
            if (stdout_file) {
                while (fgets(buffer, vec_buf.size(), stdout_file) != nullptr) {
                    std::string str(buffer);
                    if (string_starts_with(buffer, CMD_CHILD_TO_ROUTER_STATE)) {
                        LOG_DBG("[%5d] %s", port, buffer); // prevent spamming the log
                        this->handle_child_state(name, str);
                    } else {
                        // forward log
                        LOG("[%5d] %s", port, buffer);
                    }
                }
            } else {
                SRV_ERR("failed to get stdout/stderr of child process for name=%s\n", name.c_str());
            }
        });

        std::thread stopping_thread([&]() {
            // thread to monitor explicit stop requests; child crash is signalled via child_proc->stopped
            auto is_stopping = [this, &name]() {
                return this->stopping_models.find(name) != this->stopping_models.end();
            };
            {
                std::unique_lock<std::mutex> lk(this->mutex);
                this->cv_stop.wait(lk, [&]() {
                    return is_stopping() || child_proc->stopped.load(std::memory_order_acquire);
                });
            }
            // child crashed or finished on its own, skip graceful shutdown sequence
            if (child_proc->stopped.load(std::memory_order_acquire)) {
                return;
            }
            SRV_INF("stopping model instance name=%s\n", name.c_str());
            fprintf(stdin_file, "%s\n", CMD_ROUTER_TO_CHILD_EXIT);
            fflush(stdin_file);
            int64_t start_time = ggml_time_ms();
            while (true) {
                std::unique_lock<std::mutex> lk(this->mutex);
                if (!is_stopping() || child_proc->stopped.load(std::memory_order_acquire)) {
                    return;
                }
                int64_t elapsed = ggml_time_ms() - start_time;
                if (elapsed >= stop_timeout * 1000) {
                    lk.unlock();
                    SRV_WRN("force-killing model instance name=%s after %d seconds timeout\n", name.c_str(), stop_timeout);
                    child_proc->terminate();
                    return;
                }
                this->cv_stop.wait_for(lk, std::chrono::seconds(1), [&]() {
                    return !is_stopping() || child_proc->stopped.load(std::memory_order_acquire);
                });
            }
        });

        // we reach here when the child process exits (stdout EOF)
        // note: we cannot join() prior to this point because it will close stdin_file
        if (log_thread.joinable()) {
            log_thread.join();
        }

        child_proc->stopped.store(true, std::memory_order_release);
        {
            std::lock_guard<std::mutex> lk(this->mutex);
            stopping_models.erase(name);
            cv_stop.notify_all();
        }
        if (stopping_thread.joinable()) {
            stopping_thread.join();
        }

        // get the exit code
        int exit_code = child_proc->sproc.join();

        // update status and exit code
        if (child_mode == SERVER_CHILD_MODE_DOWNLOAD) {
            // instance will be cleaned up on next load_models() call
        } else {
            this->update_status(name, {
                SERVER_MODEL_STATUS_UNLOADED,
                exit_code
            });
        }
        SRV_INF("instance name=%s exited with status %d\n", name.c_str(), exit_code);
    });

    // clean up old process/thread if exists
    {
        auto & old_instance = mapping[name];
        // old process should have exited already, but just in case, we clean it up here
        if (old_instance.subproc && old_instance.subproc->is_alive()) {
            SRV_WRN("old process for model name=%s is still alive, this is unexpected\n", name.c_str());
            old_instance.subproc->terminate(); // force kill
        }
        if (old_instance.th.joinable()) {
            old_instance.th.join();
        }
    }

    notify_sse("model_status", name, {
        {"status", server_model_status_to_string(inst.meta.status)},
    });

    mapping[name] = std::move(inst);
    cv.notify_all();
}

void server_models::unload(const std::string & name) {
    std::unique_lock<std::mutex> lk(mutex);
    auto it = mapping.find(name);
    if (it != mapping.end()) {
        if (it->second.meta.status == SERVER_MODEL_STATUS_DOWNLOADING) {
            SRV_INF("cancelling download for model name=%s\n", name.c_str());
            it->second.subproc->request_exit();
            // for convenience, we wait the status change here
            wait(lk, name, [](const server_model_meta & new_meta) {
                return new_meta.status != SERVER_MODEL_STATUS_DOWNLOADING;
            });
        } else if (it->second.meta.is_running()) {
            SRV_INF("stopping model instance name=%s\n", name.c_str());
            stopping_models.insert(name);
            if (it->second.meta.status == SERVER_MODEL_STATUS_LOADING) {
                // special case: if model is in loading state, unloading means force-killing it
                SRV_WRN("model name=%s is still loading, force-killing\n", name.c_str());
                it->second.subproc->terminate();
            }
            cv_stop.notify_all();
            // status change will be handled by the managing thread
        } else {
            SRV_WRN("model instance name=%s is not running\n", name.c_str());
        }
    }
}

void server_models::unload_all() {
    std::vector<std::thread> to_join;
    {
        std::lock_guard<std::mutex> lk(mutex);
        for (auto & [name, inst] : mapping) {
            if (inst.meta.status == SERVER_MODEL_STATUS_DOWNLOADING) {
                SRV_INF("cancelling download for model name=%s\n", name.c_str());
                inst.subproc->stopped.store(true, std::memory_order_relaxed);
            } else if (inst.meta.is_running()) {
                SRV_INF("stopping model instance name=%s\n", name.c_str());
                stopping_models.insert(name);
                cv_stop.notify_all();
                // status change will be handled by the managing thread
            }
            // moving the thread to join list to avoid deadlock
            to_join.push_back(std::move(inst.th));
        }
    }
    for (auto & th : to_join) {
        if (th.joinable()) {
            th.join();
        }
    }
}

void server_models::update_status(const std::string & name, const update_status_args & args) {
    std::unique_lock<std::mutex> lk(mutex);
    auto it = mapping.find(name);
    if (it != mapping.end()) {
        auto & meta = it->second.meta;
        meta.status      = args.status;
        meta.exit_code   = args.exit_code;
        if (!args.loaded_info.is_null()) {
            meta.loaded_info = args.loaded_info;
        }
        if (!args.progress.is_null()) {
            meta.progress = args.progress;
        }
    }
    // broadcast status change to SSE
    {
        json data = {
            {"status", server_model_status_to_string(args.status)},
        };
        if (args.status == SERVER_MODEL_STATUS_UNLOADED) {
            data["exit_code"] = args.exit_code;
        }
        if (!args.loaded_info.is_null()) {
            data["info"] = args.loaded_info;
        }
        if (!args.progress.is_null()) {
            data["progress"] = args.progress;
        }
        // note: notify_sse doesn't acquire the lock, so no deadlock here
        notify_sse("status_change", name, data);
    }
    cv.notify_all();
}

void server_models::update_download_progress(const std::string & name, const common_download_progress & progress, bool done, bool ok) {
    json curr;
    {
        std::lock_guard<std::mutex> lk(mutex);
        auto it = mapping.find(name);
        if (it != mapping.end()) {
            if (done) {
                // mark the instance to be erased on next load_models() call
                it->second.meta.status = SERVER_MODEL_STATUS_DOWNLOADED;
                need_reload = true;
            } else {
                json & info = it->second.meta.loaded_info;
                if (!info.contains("progress")) {
                    info["progress"] = json{};
                }
                info["progress"][progress.url] = {
                    {"done",  progress.downloaded},
                    {"total", progress.total},
                };
                curr = it->second.meta.loaded_info; // copy
            }
        }
    }
    if (done) {
        cv.notify_all(); // notify in case unload() is waiting for download to be cancelled
        notify_sse(ok ? "download_finished" : "download_failed", name, {});
    } else {
        notify_sse("download_progress", name, curr);
    }
}

bool server_models::remove(const std::string & name) {
    // do everything under one lock acquisition; avoid get_meta() /
    // unload() because they can trigger load_models() which erases
    // transient DOWNLOADING / DOWNLOADED entries as a side-effect
    std::unique_lock<std::mutex> lk(mutex);

    auto it = mapping.find(name);
    if (it == mapping.end()) {
        throw std::runtime_error("model name=" + name + " is not found");
    }
    if (it->second.meta.source != SERVER_MODEL_SOURCE_CACHE) {
        throw std::runtime_error("model name=" + name + " is not removable (not from cache)");
    }

    if (it->second.meta.status == SERVER_MODEL_STATUS_DOWNLOADING) {
        // cancel in-flight download
        SRV_INF("cancelling download for model name=%s\n", name.c_str());
        it->second.subproc->request_exit();
    } else if (it->second.meta.is_running()) {
        // stop running instance
        SRV_INF("stopping model instance name=%s\n", name.c_str());
        stopping_models.insert(name);
        if (it->second.meta.status == SERVER_MODEL_STATUS_LOADING) {
            it->second.subproc->terminate();
        }
        cv_stop.notify_all();
    }

    // wait until the monitoring thread finishes
    wait(lk, name, [](const server_model_meta & meta) {
        return meta.status == SERVER_MODEL_STATUS_UNLOADED
            || meta.status == SERVER_MODEL_STATUS_DOWNLOADED;
    });

    // re-find after wait - load_models() may have erased the entry during the wait
    it = mapping.find(name);
    if (it == mapping.end()) {
        // load_models() already joined the thread and erased the entry;
        // we just need to clean up the cached files on disk
        lk.unlock();
        bool ok = common_download_remove(name);
        SRV_INF("removing model name=%s from cache (%s)\n", name.c_str(), ok ? "succeeded" : "partial");
        notify_sse("model_remove", name, {});
        return true;
    }

    // join before erasing - thread no longer acquires this mutex
    if (it->second.th.joinable()) {
        it->second.th.join();
    }

    // remove from disk (best-effort: cancelled downloads may have no cached files)
    bool ok = common_download_remove(name);
    mapping.erase(name);
    if (!ok) {
        SRV_WRN("removing model name=%s from disk returned false (no cached files?)\n", name.c_str());
    }
    SRV_INF("removing model name=%s from cache (%s)\n", name.c_str(), ok ? "succeeded" : "partial");
    notify_sse("model_remove", name, {});
    return true;
}

void server_models::wait(const std::string & name, std::function<bool(const server_model_meta &)> predicate) {
    std::unique_lock<std::mutex> lk(mutex);
    wait(lk, name, predicate);
}

void server_models::wait(std::unique_lock<std::mutex> & lk, const std::string & name, std::function<bool(const server_model_meta &)> predicate) {
    cv.wait(lk, [this, &name, &predicate]() {
        auto it = mapping.find(name);
        if (it != mapping.end()) {
            return predicate(it->second.meta);

        }
        // model was removed from mapping by another code path (e.g. load_models()).
        // nothing left to wait for - tell the caller to proceed.
        return true;
    });
}

bool server_models::ensure_model_ready(const std::string & name, const std::function<bool()> & should_stop) {
    auto meta = get_meta(name);
    if (!meta.has_value()) {
        throw std::runtime_error("model name=" + name + " is not found");
    }
    if (meta->is_ready()) {
        return false; // ready for taking requests
    }
    if (meta->status == SERVER_MODEL_STATUS_SLEEPING) {
        return false; // child is sleeping but still running; new request will wake it up
    }

    bool queued   = false;
    bool did_load = false;
    std::string victim;
    {
        std::unique_lock<std::mutex> lk(mutex);
        auto it = mapping.find(name);
        if (it != mapping.end() && it->second.meta.status == SERVER_MODEL_STATUS_UNLOADED) {
            bool has_capacity = sched->has_capacity(lk);
            if (has_capacity && sched->queue_empty(lk)) {
                lk.unlock();
                SRV_INF("model name=%s is not loaded, loading...\n", name.c_str());
                load(name);
                did_load = true;
            } else {
                // also queue when a slot looks free but others wait already, else they starve
                sched->join(lk, name);
                queued = true;
                if (!has_capacity) {
                    // an idle model may sit here right now, do not wait for a request to end
                    victim = sched->pick_victim(lk, name);
                    if (!victim.empty()) {
                        sched->mark_slot_pending(lk, name);
                    }
                }
            }
        }
    }
    if (!victim.empty()) {
        SRV_INF("evicting idle LRU name=%s to make room for name=%s\n", victim.c_str(), name.c_str());
        unload(victim);
    }

    // while queued, this is also where the load happens: the head of the queue does it
    SRV_INF("waiting until model name=%s is fully loaded...\n", name.c_str());
    std::unique_lock<std::mutex> lk(mutex);
    auto leave_queue = [this, &queued, &lk, &name]() {
        if (queued) {
            sched->leave(lk, name);
            queued = false;
        }
    };

    try {
        bool saw_loading = false;
        while (true) {
            auto it = mapping.find(name);
            if (it == mapping.end()) {
                break; // removed by another code path, nothing to wait for
            }
            const server_model_status status = it->second.meta.status;

            if (status == SERVER_MODEL_STATUS_LOADED || status == SERVER_MODEL_STATUS_SLEEPING) {
                break;
            }
            if (status == SERVER_MODEL_STATUS_DOWNLOADING || status == SERVER_MODEL_STATUS_DOWNLOADED) {
                break; // do not wait on a download child
            }
            if (status == SERVER_MODEL_STATUS_LOADING) {
                saw_loading = true;
            } else if (status == SERVER_MODEL_STATUS_UNLOADED) {
                if (did_load || saw_loading) {
                    // a spawn happened and the instance came back down
                    if (it->second.meta.is_failed()) {
                        throw std::runtime_error("model name=" + name + " failed to load");
                    }
                    break; // unloaded by another code path, caller reports "not running"
                }
                if (!queued) {
                    break; // not queued, and the load someone else started fell over
                }
            }

            if (should_stop && should_stop()) {
                // if a model was evicted for us, the free slot goes to the next waiter
                throw std::runtime_error("request cancelled while waiting for model name=" + name);
            }

            // our turn: our model is at the head, and a slot really did free up
            if (status == SERVER_MODEL_STATUS_UNLOADED && sched->try_claim(lk, name)) {
                lk.unlock();
                bool ok = true;
                try {
                    SRV_INF("slot available, loading queued model name=%s\n", name.c_str());
                    load(name);
                    did_load = true;
                } catch (const std::exception & e) {
                    // lost a race for the slot, stay in line and retry
                    SRV_WRN("queued load of name=%s did not go through: %s\n", name.c_str(), e.what());
                    ok = false;
                }
                lk.lock();
                sched->claim_done(lk, name, ok);
                if (ok) {
                    queued = false; // entry is gone, the other waiters watch the status now
                }
                continue;
            }

            cv.wait_for(lk, std::chrono::milliseconds(200));
        }
    } catch (...) {
        leave_queue();
        throw;
    }
    leave_queue();

    return true;
}

server_http_res_ptr server_models::proxy_request(const server_http_req & req, const std::string & method, const std::string & name, bool update_last_used, bool detached) {
    auto meta = get_meta(name);
    if (!meta.has_value()) {
        throw std::runtime_error("model name=" + name + " is not found");
    }
    if (!meta->is_running()) {
        throw std::invalid_argument("model name=" + name + " is not running");
    }
    {
        std::unique_lock<std::mutex> lk(mutex);
        if (update_last_used) {
            mapping[name].meta.last_used = ggml_time_ms();
        }
        mapping[name].req_count++;
    }
    if (debug_fake_timing) {
        // sleep after req_count++, so the model counts as busy while we wait here
        std::this_thread::sleep_for(std::chrono::seconds(2));
    }
    SRV_INF("proxying request to model %s on port %d\n", name.c_str(), meta->port);
    std::string proxy_path = req.path;
    if (!req.query_string.empty()) {
        proxy_path += '?' + req.query_string;
    }
    auto proxy = std::make_unique<server_http_proxy>(
            method,
            "http",
            CHILD_ADDR,
            meta->port,
            proxy_path,
            req.headers,
            req.body,
            req.files,
            // a detached request belongs to a replay session
            detached
                ? std::function<bool()>([]() { return false; })
                : req.should_stop,
            base_params.timeout_read,
            base_params.timeout_write
            );

    proxy->cleanup = [this, name]() {
        bool went_idle = false;
        {
            std::unique_lock<std::mutex> lk(mutex);
            auto it = mapping.find(name);
            if (it != mapping.end() && it->second.req_count > 0) {
                it->second.req_count--;
                went_idle = it->second.req_count == 0;
            }
        }
        if (went_idle) {
            sched->on_model_idle(name);
        }
    };

    return proxy;
}

void server_models::handle_child_state(const std::string & name, const std::string & raw_input) {
    server_state state;
    json payload;

    try {
        json data = json::parse(raw_input.substr(strlen(CMD_CHILD_TO_ROUTER_STATE)));
        state = server_state_from_str(json_value(data, "state", std::string()));
        payload = json_value(data, "payload", json{});
    } catch (const std::exception & e) {
        SRV_ERR("failed to parse child state update for name=%s: %s\n", name.c_str(), e.what());
        return;
    }

    switch (state) {
        case SERVER_STATE_DOWNLOADING:
            {
                std::string result = json_value(payload, "result", std::string());
                std::string url    = json_value(payload, "url",    std::string());
                auto request_exit = [&]() {
                    std::lock_guard<std::mutex> lk(mutex);
                    auto it = mapping.find(name);
                    if (it != mapping.end()) {
                        return it->second.subproc->request_exit();
                    }
                };
                if (result == "download_finished") {
                    update_download_progress(name, {}, true, true);
                    request_exit();
                } else if (result == "download_failed") {
                    update_download_progress(name, {}, true, false);
                    request_exit();
                } else if (!url.empty()) {
                    common_download_progress p;
                    p.url        = url;
                    p.downloaded = json_value(payload, "downloaded", (size_t)0);
                    p.total      = json_value(payload, "total", (size_t)0);
                    update_download_progress(name, p, false);
                }
            } break;
        case SERVER_STATE_LOADING:
            {
                update_status(name, {
                    SERVER_MODEL_STATUS_LOADING,
                    0,
                    nullptr, // no loaded_info yet
                    payload,
                });
            } break;
        case SERVER_STATE_READY:
            {
                update_status(name, {
                    SERVER_MODEL_STATUS_LOADED,
                    0,
                    // note: payload can be empty if this is a wakeup from sleep
                    payload.size() > 0 ? payload : nullptr,
                    {}, // reset progress info
                });
            } break;
        case SERVER_STATE_SLEEPING:
            {
                update_status(name, { SERVER_MODEL_STATUS_SLEEPING });
            } break;
        default:
            // should never happen, but just in case
            GGML_ASSERT(false && "unexpected state from child server");
    }
}
