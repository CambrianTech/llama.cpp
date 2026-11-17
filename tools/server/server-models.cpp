// Continuum fork: model metadata and server_models: construction, discovery, lookup, LRU; out of server-models.cpp (pure move; the 1,000-line rule).

#include "server-models-internal.h"

void server_model_meta::update_args(common_preset_context & ctx_preset, std::string bin_path) {
    // update params
    unset_reserved_args(preset, false);
    preset.set_option(ctx_preset, "LLAMA_ARG_HOST",  CHILD_ADDR);
    preset.set_option(ctx_preset, "LLAMA_ARG_PORT",  std::to_string(port));
    preset.set_option(ctx_preset, "LLAMA_ARG_ALIAS", name);
    // TODO: maybe validate preset before rendering ?
    // render args
    args = preset.to_args(bin_path);

    // unified binary dispatches by subcommand, re-inject it right after the
    // binary path so the child starts as 'llama serve ...' not 'llama ...'
    const char * app_cmd = std::getenv("LLAMA_APP_CMD");
    if (app_cmd != nullptr && app_cmd[0] != '\0' && !bin_path.empty()) {
        args.insert(args.begin() + 1, app_cmd);
    }
}

void server_model_meta::update_caps() {
    try {
        common_params params;
        preset.apply_to_params(params, {
            "LLAMA_ARG_MODEL",
            "LLAMA_ARG_MODEL_URL",
            "LLAMA_ARG_MMPROJ",
            "LLAMA_ARG_MMPROJ_URL",
            "LLAMA_ARG_MMPROJ_AUTO",
            "LLAMA_ARG_HF_REPO",
            "LLAMA_ARG_HF_REPO_FILE",
        });
        params.offline = true;
        common_models_handler handler = common_models_handler_init(params, LLAMA_EXAMPLE_SERVER);
        common_models_handler_apply(handler, params); // note: this won't download the model because offline=true
        if (params.no_mmproj || params.mmproj.path.empty()) {
            multimodal = { false, false };
        } else {
            multimodal = mtmd_get_cap_from_file(params.mmproj.path.c_str());
        }
    } catch (const std::exception & e) {
        LOG_WRN("failed to initialize common_params for multimodal capability detection: %s\n", e.what());
        multimodal = { false, false };
    }
}

//
// server_models
//

server_models::server_models(
        const common_params & params,
        int argc,
        char ** argv)
            : ctx_preset(LLAMA_EXAMPLE_SERVER),
              base_params(params),
              base_env(get_environment()),
              base_preset(ctx_preset.load_from_args(argc, argv)),
              sched(std::make_unique<server_lru_sched>(*this)) {
    // clean up base preset
    unset_reserved_args(base_preset, true);
    // set binary path
    try {
        bin_path = get_server_exec_path().string();
    } catch (const std::exception & e) {
        bin_path = argv[0];
        LOG_WRN("failed to get server executable path: %s\n", e.what());
        LOG_WRN("using original argv[0] as fallback: %s\n", argv[0]);
    }
    load_models();
    debug_fake_timing = !common_get_env("LLAMA_SERVER_DEBUG_FAKE_TIMING").empty();
}

server_models::~server_models() = default;

void server_models::add_model(server_model_meta && meta) {
    if (mapping.find(meta.name) != mapping.end()) {
        throw std::runtime_error(string_format("model '%s' appears multiple times", meta.name.c_str()));
    }

    // check model name does not conflict with existing aliases
    for (const auto & [key, inst] : mapping) {
        if (inst.meta.aliases.count(meta.name)) {
            throw std::runtime_error(string_format("model name '%s' conflicts with alias of model '%s'",
                meta.name.c_str(), key.c_str()));
        }
    }

    // parse aliases from preset's --alias option (comma-separated)
    std::string alias_str;
    if (meta.preset.get_option("LLAMA_ARG_ALIAS", alias_str) && !alias_str.empty()) {
        for (auto & alias : string_split<std::string>(alias_str, ',')) {
            alias = string_strip(alias);
            if (!alias.empty()) {
                meta.aliases.insert(alias);
            }
        }
    }

    // parse tags from preset's --tags option (comma-separated)
    std::string tags_str;
    if (meta.preset.get_option("LLAMA_ARG_TAGS", tags_str) && !tags_str.empty()) {
        for (auto & tag : string_split<std::string>(tags_str, ',')) {
            tag = string_strip(tag);
            if (!tag.empty()) {
                meta.tags.insert(tag);
            }
        }
    }

    // validate aliases do not conflict with existing names or aliases
    for (const auto & alias : meta.aliases) {
        if (mapping.find(alias) != mapping.end()) {
            throw std::runtime_error(string_format("alias '%s' for model '%s' conflicts with existing model name",
                alias.c_str(), meta.name.c_str()));
        }
        for (const auto & [key, inst] : mapping) {
            if (inst.meta.aliases.count(alias)) {
                throw std::runtime_error(string_format("alias '%s' for model '%s' conflicts with alias of model '%s'",
                    alias.c_str(), meta.name.c_str(), key.c_str()));
            }
        }
    }

    meta.update_args(ctx_preset, bin_path); // render args
    meta.update_caps();
    std::string name = meta.name;
    mapping[name] = instance_t{
        /* subproc */ std::make_shared<server_subproc>(),
        /* th      */ std::thread(),
        /* meta    */ std::move(meta)
    };
}

void server_models::notify_sse(const std::string & event, const std::string & model_id, const json & data) {
    std::unique_ptr<server_task_result_router> result = std::make_unique<server_task_result_router>();
    result->data = {
        {"model", model_id},
        {"event", event},
    };
    if (!data.is_null()) {
        result->data["data"] = data;
    }
    SRV_DBG("notifying SSE clients about event '%s' for model '%s': %s\n", event.c_str(), model_id.c_str(), safe_json_to_str(result->data).c_str());
    sse.broadcast(std::move(result));
}

void server_models::load_models() {
    // Phase 1: load presets from all sources - pure I/O, no lock needed
    // 1. cached models
    common_presets cached_models = ctx_preset.load_from_cache();
    SRV_INF("Loaded %zu cached model presets\n", cached_models.size());
    // 2. local models from --models-dir
    common_presets local_models;
    if (!base_params.models_dir.empty()) {
        local_models = ctx_preset.load_from_models_dir(base_params.models_dir);
        SRV_INF("Loaded %zu local model presets from %s\n", local_models.size(), base_params.models_dir.c_str());
    }
    // 3. custom-path models from presets
    common_preset global = {};
    common_presets custom_presets = {};
    if (!base_params.models_preset.empty()) {
        custom_presets = ctx_preset.load_from_ini(base_params.models_preset, global);
        SRV_INF("Loaded %zu custom model presets from %s\n", custom_presets.size(), base_params.models_preset.c_str());
    }

    // cascade, apply global preset first
    cached_models  = ctx_preset.cascade(global, cached_models);
    local_models   = ctx_preset.cascade(global, local_models);
    custom_presets = ctx_preset.cascade(global, custom_presets);

    // note: if a model exists in both cached and local, local takes precedence
    common_presets final_presets;
    std::unordered_map<std::string, server_model_source> source_map;
    for (const auto & [name, preset] : cached_models) {
        final_presets[name] = preset;
        source_map[name] = SERVER_MODEL_SOURCE_CACHE;
    }
    for (const auto & [name, preset] : local_models)  {
        final_presets[name] = preset;
        source_map[name] = SERVER_MODEL_SOURCE_MODELS_DIR;
    }
    for (const auto & [name, custom] : custom_presets) {
        if (final_presets.find(name) != final_presets.end()) {
            final_presets[name].merge(custom);
        } else {
            final_presets[name] = custom;
        }
        source_map[name] = SERVER_MODEL_SOURCE_PRESET;
    }

    // overlay router's own CLI args on top of every model preset so that
    // e.g. `llama-server --temp 0` is honoured by all child processes
    for (auto & [name, preset] : final_presets) {
        preset.merge(base_preset);
    }

    auto get_source = [&](const std::string & name) {
        return source_map.count(name) ? source_map.at(name) : SERVER_MODEL_SOURCE_PRESET;
    };

    // hide cache models whose resolved file is already used by a preset with dedup-cache-models enabled
    std::set<std::string> hidden_models;
    {
        std::set<std::string> preset_paths;
        for (const auto & [name, preset] : custom_presets) {
            std::string val;
            if (!preset.get_option(COMMON_ARG_PRESET_DEDUP_CACHE_MODELS, val) || !common_arg_utils::is_truthy(val)) {
                continue;
            }
            std::string hf_repo;
            if (!preset.get_option("LLAMA_ARG_HF_REPO", hf_repo) || hf_repo.empty()) {
                continue;
            }
            std::string hf_file;
            preset.get_option("LLAMA_ARG_HF_FILE", hf_file);
            std::string path = common_download_resolve_path(hf_repo, hf_file);
            if (!path.empty()) {
                preset_paths.insert(path);
            }
        }
        if (!preset_paths.empty()) {
            for (const auto & [name, preset] : cached_models) {
                if (get_source(name) != SERVER_MODEL_SOURCE_CACHE) {
                    continue; // merged with another source, not a pure cache entry
                }
                std::string path = common_download_resolve_path(name);
                if (!path.empty() && preset_paths.count(path)) {
                    SRV_INF("hiding cache model name=%s (deduplicated by a preset)\n", name.c_str());
                    hidden_models.insert(name);
                }
            }
        }
    }

    // Helpers that read `mapping` - must be called while holding the lock.
    std::unordered_set<std::string> custom_names;
    for (const auto & [name, preset] : custom_presets) custom_names.insert(name);
    auto join_set = [](const std::set<std::string> & s) {
        std::string result;
        for (const auto & v : s) {
            if (!result.empty()) result += ", ";
            result += v;
        }
        return result;
    };
    auto log_available_models = [&]() {
        SRV_INF("Available models (%zu) (*: custom preset)\n", mapping.size());
        for (const auto & [name, inst] : mapping) {
            bool has_custom = custom_names.find(name) != custom_names.end();
            std::string info;
            if (!inst.meta.aliases.empty()) info += " (aliases: " + join_set(inst.meta.aliases) + ")";
            if (!inst.meta.tags.empty())    info += " [tags: "    + join_set(inst.meta.tags)    + "]";
            SRV_INF("  %c %s%s\n", has_custom ? '*' : ' ', name.c_str(), info.c_str());
        }
    };
    auto apply_stop_timeout = [&]() {
        for (auto & [name, inst] : mapping) {
            std::string val;
            if (inst.meta.preset.get_option(COMMON_ARG_PRESET_STOP_TIMEOUT, val)) {
                try {
                    inst.meta.stop_timeout = std::stoi(val);
                } catch (...) {
                    SRV_WRN("invalid stop-timeout value '%s' for model '%s', using default %d seconds\n",
                        val.c_str(), name.c_str(), DEFAULT_STOP_TIMEOUT);
                    inst.meta.stop_timeout = DEFAULT_STOP_TIMEOUT;
                }
            }
        }
    };
    auto apply_hidden = [&]() {
        for (auto & [name, inst] : mapping) {
            inst.meta.hidden = hidden_models.count(name) > 0;
        }
    };
    // update_args() injects HOST/PORT/ALIAS, so strip them before comparing presets
    auto preset_options_for_compare = [](common_preset p) {
        p.unset_option("LLAMA_ARG_HOST");
        p.unset_option("LLAMA_ARG_PORT");
        p.unset_option("LLAMA_ARG_ALIAS");
        return p.options;
    };

    // Phase 2: acquire the lock once for all mapping mutations.
    // We temporarily release it only when calling functions that acquire it internally
    // (unload, load) or when joining threads (the monitoring thread calls update_status
    // which locks the mutex, so joining while holding it would deadlock).
    std::unique_lock<std::mutex> lk(mutex);

    need_reload = false;
    bool is_first_load = mapping.empty();

    if (is_first_load) {
        // FIRST LOAD: add all models, then unlock for autoloading
        for (const auto & [name, preset] : final_presets) {
            server_model_meta meta{
                /* source        */ get_source(name),
                /* preset        */ preset,
                /* name          */ name,
                /* aliases       */ {},
                /* tags          */ {},
                /* port          */ 0,
                /* status        */ SERVER_MODEL_STATUS_UNLOADED,
                /* last_used     */ 0,
                /* args          */ std::vector<std::string>(),
                /* loaded_info   */ {},
                /* progress      */ {},
                /* exit_code     */ 0,
                /* stop_timeout  */ DEFAULT_STOP_TIMEOUT,
                /* multimodal    */ mtmd_caps{false, false},
                // /* need_download */ false,
            };
            add_model(std::move(meta));
        }
        apply_stop_timeout();
        apply_hidden();
        log_available_models();

        // skipped on reload, see startup_models
        if (startup_models.has_value()) {
            std::vector<std::string> models_to_load;
            for (const auto & [name, inst] : mapping) {
                std::string val;
                if (inst.meta.preset.get_option(COMMON_ARG_PRESET_LOAD_ON_STARTUP, val) && common_arg_utils::is_truthy(val)) {
                    models_to_load.push_back(name);
                }
            }
            if ((int)models_to_load.size() > base_params.models_max) {
                throw std::runtime_error(string_format(
                    "number of models to load on startup (%zu) exceeds models_max (%d)",
                    models_to_load.size(), base_params.models_max));
            }

            // to be lazy-loaded after main() setup phase is completed
            startup_models = std::move(models_to_load);
        }

        lk.unlock();
    } else {
        // RELOAD: diff the new preset list against the current mapping and reconcile
        is_reloading = true;

        // find running models whose source was removed or whose preset changed
        std::vector<std::string> to_unload;
        for (const auto & [name, inst] : mapping) {
            if (!inst.meta.is_running()) continue;
            auto it = final_presets.find(name);
            if (it == final_presets.end()) {
                to_unload.push_back(name); // removed from source
            } else if (preset_options_for_compare(inst.meta.preset) != preset_options_for_compare(it->second)) {
                to_unload.push_back(name); // preset changed
            }
        }

        // unload() acquires the lock internally, so release before each call
        for (const auto & name : to_unload) {
            SRV_INF("(reload) unloading model name=%s (source updated or removed)\n", name.c_str());
            lk.unlock();
            unload(name);
            lk.lock();
        }

        // wait for all targeted models to reach UNLOADED; cv.wait handles unlock/relock
        cv.wait(lk, [&]() {
            for (const auto & name : to_unload) {
                auto it = mapping.find(name);
                if (it != mapping.end() && it->second.meta.is_running()) return false;
            }
            return true;
        });

        // collect all threads to join in one pass while the lock is held:
        // - monitoring threads from just-unloaded models (to_unload)
        // - threads of finished downloads (DOWNLOADED), they acquire the mutex on exit
        // - threads of already-UNLOADED models that are being removed from source
        std::vector<std::thread> threads_to_join;
        for (const auto & name : to_unload) {
            auto it = mapping.find(name);
            if (it != mapping.end() && it->second.th.joinable()) {
                threads_to_join.push_back(std::move(it->second.th));
            }
        }
        for (auto & [name, inst] : mapping) {
            if (inst.meta.status == SERVER_MODEL_STATUS_DOWNLOADING) {
                continue; // downloading models are not from config sources, leave them alone
            }
            if (inst.meta.status == SERVER_MODEL_STATUS_DOWNLOADED) {
                // joining this thread under the lock deadlocks: it locks the mutex on its way out
                if (inst.th.joinable()) {
                    threads_to_join.push_back(std::move(inst.th));
                }
                continue;
            }
            if (final_presets.find(name) == final_presets.end() && !inst.meta.is_running() && inst.th.joinable()) {
                threads_to_join.push_back(std::move(inst.th));
            }
        }

        // join outside the lock - monitoring thread calls update_status (needs lock)
        lk.unlock();
        for (auto & th : threads_to_join) th.join();
        lk.lock();

        // erase models no longer in any source
        for (auto it = mapping.begin(); it != mapping.end(); ) {
            if (it->second.meta.status == SERVER_MODEL_STATUS_DOWNLOADING) {
                ++it; // download thread is still busy, skip
            } else if (it->second.meta.status == SERVER_MODEL_STATUS_DOWNLOADED) {
                // download finished, thread is joined above, safe to erase
                GGML_ASSERT(!it->second.th.joinable());
                it = mapping.erase(it);
            } else if (final_presets.find(it->first) == final_presets.end()) {
                SRV_INF("(reload) removing model name=%s (no longer in source)\n", it->first.c_str());
                GGML_ASSERT(!it->second.th.joinable()); // must have been joined above
                it = mapping.erase(it);
            } else {
                ++it;
            }
        }

        // update presets for non-running models still in source
        for (auto & [name, inst] : mapping) {
            if (inst.meta.is_running()) continue;
            auto it = final_presets.find(name);
            if (it == final_presets.end()) continue; // erased above

            inst.meta.preset = it->second;

            // re-parse aliases, then validate against other models
            std::set<std::string> new_aliases;
            std::string alias_str;
            if (inst.meta.preset.get_option("LLAMA_ARG_ALIAS", alias_str) && !alias_str.empty()) {
                for (auto & alias : string_split<std::string>(alias_str, ',')) {
                    alias = string_strip(alias);
                    if (!alias.empty()) new_aliases.insert(alias);
                }
            }
            inst.meta.aliases.clear();
            for (const auto & alias : new_aliases) {
                bool conflict = false;
                for (const auto & [other_name, other_inst] : mapping) {
                    if (other_name == name) continue;
                    if (other_name == alias || other_inst.meta.aliases.count(alias)) {
                        SRV_WRN("(reload) alias '%s' for model '%s' conflicts with model '%s', skipping\n",
                            alias.c_str(), name.c_str(), other_name.c_str());
                        conflict = true;
                        break;
                    }
                }
                if (!conflict) inst.meta.aliases.insert(alias);
            }

            // re-parse tags
            inst.meta.tags.clear();
            std::string tags_str;
            if (inst.meta.preset.get_option("LLAMA_ARG_TAGS", tags_str) && !tags_str.empty()) {
                for (auto & tag : string_split<std::string>(tags_str, ',')) {
                    tag = string_strip(tag);
                    if (!tag.empty()) inst.meta.tags.insert(tag);
                }
            }

            inst.meta.exit_code = 0; // clear failed state so the model can be reloaded
            inst.meta.update_args(ctx_preset, bin_path);
            inst.meta.update_caps();
        }

        // add models that are new in this reload, load-on-startup is not honored here since a
        // reload never spawns an instance
        for (const auto & [name, preset] : final_presets) {
            if (mapping.find(name) == mapping.end()) {
                server_model_meta meta{
                    /* source        */ get_source(name),
                    /* preset        */ preset,
                    /* name          */ name,
                    /* aliases       */ {},
                    /* tags          */ {},
                    /* port          */ 0,
                    /* status        */ SERVER_MODEL_STATUS_UNLOADED,
                    /* last_used     */ 0,
                    /* args          */ std::vector<std::string>(),
                    /* loaded_info   */ {},
                    /* progress      */ {},
                    /* exit_code     */ 0,
                    /* stop_timeout  */ DEFAULT_STOP_TIMEOUT,
                    /* multimodal    */ mtmd_caps{false, false},
                    // /* need_download */ false,
                };
                add_model(std::move(meta));
            }
        }

        apply_stop_timeout();
        apply_hidden();

        // clear reload flag under the lock, this releases the load() calls waiting on !is_reloading
        is_reloading = false;
        cv.notify_all();

        log_available_models();

        lk.unlock();

        notify_sse("models_reload", "*");
    }
}

void server_models::load_startup_models() {
    std::vector<std::string> to_load;
    {
        std::lock_guard<std::mutex> lk(mutex);
        if (!startup_models.has_value()) {
            return; // already drained
        }
        to_load = std::move(*startup_models);
        startup_models.reset();
    }
    for (const auto & name : to_load) {
        SRV_INF("(startup) loading model %s\n", name.c_str());
        load(name);
    }
}

void server_models::update_meta(const std::string & name, const server_model_meta & meta) {
    std::lock_guard<std::mutex> lk(mutex);
    auto it = mapping.find(name);
    if (it != mapping.end()) {
        it->second.meta = meta;
    }
    cv.notify_all(); // notify wait_until_loading_finished
}

bool server_models::has_model(const std::string & name) {
    std::lock_guard<std::mutex> lk(mutex);
    if (mapping.find(name) != mapping.end()) {
        return true;
    }
    for (const auto & [key, inst] : mapping) {
        if (inst.meta.aliases.count(name)) {
            return true;
        }
    }
    return false;
}

std::optional<server_model_meta> server_models::get_meta(const std::string & name) {
    std::unique_lock<std::mutex> lk(mutex);
    if (need_reload) {
        lk.unlock();
        load_models();
        lk.lock();
    }

    auto it = mapping.find(name);
    if (it != mapping.end()) {
        return it->second.meta;
    }
    for (const auto & [key, inst] : mapping) {
        if (inst.meta.aliases.count(name)) {
            return inst.meta;
        }
    }
    return std::nullopt;
}

std::vector<server_model_meta> server_models::get_all_meta() {
    std::unique_lock<std::mutex> lk(mutex);
    if (need_reload) {
        lk.unlock();
        load_models();
        lk.lock();
    }

    std::vector<server_model_meta> result;
    result.reserve(mapping.size());
    for (const auto & [name, inst] : mapping) {
        result.push_back(inst.meta);
    }
    return result;
}

void server_models::unload_lru() {
    if (base_params.models_max <= 0) {
        return; // no limit
    }
    // remove one of the servers if we passed the models_max (least recently used - LRU)
    std::string lru_model_name;
    {
        std::unique_lock<std::mutex> lk(mutex);
        if (sched->has_capacity(lk)) {
            return;
        }
        lru_model_name = sched->pick_victim(lk, "");
    }
    if (!lru_model_name.empty()) {
        SRV_INF("models_max limit reached, removing LRU name=%s\n", lru_model_name.c_str());
        unload(lru_model_name);
        // wait for unload to complete
        {
            std::unique_lock<std::mutex> lk(mutex);
            cv.wait(lk, [this, &lru_model_name]() {
                return mapping[lru_model_name].meta.status == SERVER_MODEL_STATUS_UNLOADED;
            });
        }
    }
}
