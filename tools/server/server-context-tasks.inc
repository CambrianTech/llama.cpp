// Methods of server_context_impl: Task dispatch: every task type the queue delivers.
// Included inside the struct by server-context.cpp; a pure move out of it (the 1,000-line rule).

    // returns false to decline the task, it is offered again after the decode is done
    bool process_single_task(server_task && task, bool is_yielding) {
        // while yielding, an encode / decode is running and only reading the server state is safe
        if (is_yielding && task.type != SERVER_TASK_TYPE_METRICS && task.type != SERVER_TASK_TYPE_SLOT_GET) {
            SRV_DBG("decoding, decline task, id_task = %d\n", task.id);
            return false;
        }

        switch (task.type) {
            case SERVER_TASK_TYPE_COMPLETION:
            case SERVER_TASK_TYPE_INFILL:
            case SERVER_TASK_TYPE_EMBEDDING:
            case SERVER_TASK_TYPE_RERANK:
                {
                    // special case: if input is provided via CLI, tokenize it first
                    // otherwise, no need to tokenize as it's already done inside the HTTP thread
                    if (task.cli) {
                        if (!tokenize_cli_input(task)) {
                            break;
                        }
                    }

                    const int id_task = task.id;

                    server_slot * slot = get_available_slot(task);

                    //
                    // slot scheduling logic
                    //

                    if (slot == nullptr) {
                        // if no slot is available, we defer this task for processing later
                        SRV_DBG("no slot is available, defer task, id_task = %d\n", id_task);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", id_task);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    if (task.is_parent()) {
                        // try getting free slots for all child tasks
                        size_t n_child_tasks = task.child_tasks.size();
                        std::vector<server_slot *> child_slots = get_free_slots(n_child_tasks, slot->id);
                        if (child_slots.size() < n_child_tasks) {
                            SRV_DBG("not enough free slots for child tasks, n_free = %zu, n_children = %zu, defer task, id_task = %d\n", child_slots.size(), n_child_tasks, id_task);
                            queue_tasks.defer(std::move(task));
                            break;
                        }
                        if (!launch_slots_with_parent_task(*slot, child_slots, std::move(task))) {
                            SRV_ERR("failed to launch slot with parent task, id_task = %d\n", id_task);
                            break; // drop the task
                        }
                    } else if (!launch_slot_with_task(*slot, std::move(task))) {
                        SRV_ERR("failed to launch slot with task, id_task = %d\n", id_task);
                        break; // drop the task
                    }

                    if (params_base.cache_idle_slots) {
                        for (auto & slot : slots) {
                            if (!slot.is_processing()) {
                                SLT_TRC(slot, "%s", "saving idle slot to prompt cache\n");

                                if (slot.prompt_save(*prompt_cache)) {
                                    SLT_DBG(slot, "%s", "__TEST_TAG_CACHE_IDLE_SLOT__\n");
                                    prompt_cache->update();
                                }

                                if (params_base.kv_unified) {
                                    // [TAG_IDLE_SLOT_CLEAR]
                                    slot.prompt_clear();
                                }
                            }
                        }
                    }
                } break;
            case SERVER_TASK_TYPE_CANCEL:
                {
                    // release slot linked with the task id
                    for (auto & slot : slots) {
                        if (slot.task && slot.task->id == task.id_target) {
                            slot.release();
                            break;
                        }
                    }
                } break;
            case SERVER_TASK_TYPE_CONTROL:
                {
                    auto res = std::make_unique<server_task_result_control>();
                    res->id = task.id;

                    server_slot * slot = get_slot_by_cmpl_id(task.params.control_cmpl_id);
                    if (slot == nullptr) {
                        SRV_WRN("control %s on unknown completion id=%s, no live slot\n",
                                task.params.control_action.c_str(), task.params.control_cmpl_id.c_str());
                        res->success = false;
                        res->message = "no active completion for this id";
                        turn_sink->deliver(std::move(res));
                        break;
                    }

                    if (task.params.control_action == "reasoning_end") {
                        // the budget sampler only exists when reasoning control was armed
                        if (!slot->task->params.sampling.reasoning_control) {
                            res->success = false;
                            res->message = "reasoning control not enabled for this completion";
                            turn_sink->deliver(std::move(res));
                            break;
                        }
                        // act on the live slot mid generation, never defer
                        common_sampler_reasoning_budget_force(slot->smpl.get());
                        res->success = true;
                    } else {
                        res->success = false;
                        res->message = "unknown control action";
                    }

                    turn_sink->deliver(std::move(res));
                } break;
            case SERVER_TASK_TYPE_NEXT_RESPONSE:
                {
                    // do nothing
                } break;
            case SERVER_TASK_TYPE_METRICS:
                {
                    int n_processing_slots = 0;

                    for (server_slot & slot : slots) {
                        if (slot.is_processing()) {
                            n_processing_slots++;
                        }
                    }
                    SRV_DBG("n_processing_slots = %d\n", n_processing_slots);

                    auto res = std::make_unique<server_task_result_metrics>();
                    res->id                  = task.id;
                    res->n_processing_slots  = n_processing_slots;
                    res->n_tasks_deferred    = queue_tasks.queue_tasks_deferred_size();
                    res->metrics             = metrics;

                    if (task.metrics_reset_bucket) {
                        metrics.reset_bucket();
                    }
                    turn_sink->deliver(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SLOT_GET:
                {
                    json slots_data = json::array();

                    int n_idle_slots = 0;

                    for (server_slot & slot : slots) {
                        if (!slot.is_processing()) {
                            n_idle_slots++;
                        }

                        slots_data.push_back(slot.to_json(slots_debug == 0));
                    }
                    SRV_DBG("n_idle_slots = %d\n", n_idle_slots);

                    auto res = std::make_unique<server_task_result_slots>();
                    res->id           = task.id;
                    res->slots_data   = std::move(slots_data);
                    res->n_idle_slots = n_idle_slots;
                    // Every cell holder's occupancy, read in the same main-loop pass as the slots so
                    // the two agree (Continuum fork, GET /cells).
                    {
                        llama_memory_t mem = llama_get_memory(ctx_tgt);
                        std::vector<llama_memory_cells_report> holders(8);
                        int32_t n = llama_memory_cell_usage(mem, holders.data(), (int32_t) holders.size());
                        if (n > (int32_t) holders.size()) {
                            holders.resize(n);
                            n = llama_memory_cell_usage(mem, holders.data(), n);
                        }
                        json cells = json::array();
                        for (int32_t i = 0; i < n; ++i) {
                            cells.push_back({ {"holder", holders[i].holder}, {"used", holders[i].used}, {"size", holders[i].size} });
                        }
                        res->cells = std::move(cells);
                    }

                    turn_sink->deliver(std::move(res));
                } break;
            case SERVER_TASK_TYPE_CELLS_NEEDED:
                {
                    // read-only: the slot's cached prompt against the request's tokens
                    server_slot * slot = get_slot_by_id(task.slot_action.id_slot);
                    if (slot == nullptr) {
                        turn_sink->deliver(turn_error(task, "Invalid slot ID", ERROR_TYPE_INVALID_REQUEST));
                        break;
                    }
                    auto res = std::make_unique<server_task_result_cells_needed>();
                    res->id       = task.id;
                    res->id_slot  = slot->id;
                    res->n_tokens = task.tokens.size();
                    // The prefix the sequence's CELLS still hold, never more than its remembered prompt:
                    // a page-out can empty the sequence and keep prompt.tokens (Cormac on #92).
                    const llama_pos held = llama_memory_seq_pos_max(llama_get_memory(ctx_tgt), slot->id) + 1;
                    res->n_cached = std::min(slot->prompt.tokens.get_common_prefix(task.tokens), (size_t) std::max<llama_pos>(held, 0));
                    turn_sink->deliver(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SLOT_SAVE:
                {
                    page_save(std::move(task)); // server-context-page.inc
                } break;
            case SERVER_TASK_TYPE_SLOT_RESTORE:
                {
                    page_restore(std::move(task));
                } break;
            case SERVER_TASK_TYPE_SLOT_PAGE_IO_DONE:
                {
                    page_io_done(std::move(task));
                } break;
            case SERVER_TASK_TYPE_LEARN_RELEASE:
                {
                    auto res = std::make_unique<server_task_result_learn_release>();
                    res->id      = task.id;
                    res->outcome = residents->release_pin(task.params.page, task.params.learn_release_pin);
                    turn_sink->deliver(std::move(res));
                } break;
            case SERVER_TASK_TYPE_MIND_PREFETCH:
                {
                    const int64_t t0 = ggml_time_us();
                    auto res = std::make_unique<server_task_result_mind_prefetch>();
                    res->id         = task.id;
                    res->place      = residents->enabled() ? residents->prefetch(task.params.page) : mind_place::minds_off;
                    res->elapsed_ms = (ggml_time_us() - t0) / 1000.0;
                    turn_sink->deliver(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SLOT_ERASE:
                {
                    const int id_slot = task.slot_action.id_slot;
                    server_slot * slot = get_slot_by_id(id_slot);
                    if (slot == nullptr) {
                        turn_sink->deliver(turn_error(task, "Invalid slot ID", ERROR_TYPE_INVALID_REQUEST));
                        break;
                    }
                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", task.id);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    // Erase token cache
                    const size_t n_erased = slot->prompt.tokens.size();

                    slot->prompt_clear();
                    slot->page = mind_page{}; // erased: no mind's sequence is here now
                    page_slot_reassigned(*slot); // a restore buffered for this slot is dropped, never applied over it

                    auto res = std::make_unique<server_task_result_slot_erase>();
                    res->id       = task.id;
                    res->id_slot  = id_slot;
                    res->n_erased = n_erased;
                    turn_sink->deliver(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SET_LORA:
                {
                    auto new_loras = construct_lora_list(task.set_lora);
                    // logging
                    for (size_t i = 0; i < new_loras.size(); ++i) {
                        SRV_TRC("set lora adapter idx=%zu scale=%f\n", i, new_loras[i].scale);
                    }
                    // TODO @ngxson : make lora_adapters a dedicated member of server_context
                    params_base.lora_adapters = new_loras;
                    publish_lora_report();
                    auto res = std::make_unique<server_task_result_apply_lora>();
                    res->id = task.id;
                    turn_sink->deliver(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SAVE_LORA:
                {
                    if (!task.save_checkpoint.is_null() && task.save_checkpoint.at("engine").get<std::string>() != learn_engine) {
                        turn_sink->deliver(turn_error(task, "stale_engine", ERROR_TYPE_INVALID_REQUEST));
                        break;
                    }
                    const auto & loras = params_base.lora_adapters;
                    const int32_t id = task.save_lora;
                    if (id < 0 || (size_t) id >= loras.size() || loras[id].ptr == nullptr) {
                        turn_sink->deliver(turn_error(task, "no loaded LoRA adapter with id " + std::to_string(id), ERROR_TYPE_INVALID_REQUEST));
                        break;
                    }
                    // Queue ownership excludes concurrent writes, but a resumable reply
                    // retains this shadow between chunks through stable-tail restoration.
                    if (learning && (learning->shadow == loras[id].ptr || !task.save_checkpoint.is_null())) {
                        turn_sink->deliver(turn_error(task, "shadow snapshot awaits learning restoration", ERROR_TYPE_UNAVAILABLE));
                        break;
                    }
                    const int64_t steps = llama_opt_shadow_steps(ctx_tgt, loras[id].ptr);
                    if (!task.save_checkpoint.is_null()) {
                        bool allocated = false;
                        const auto memory = learn_memory_receipt();
                        for (const auto & row : memory.at("optimizer_owned_buffers")) {
                            allocated |= row.at("lora_id") == id && row.at("bytes").get<uint64_t>() != 0;
                        }
                        if (!allocated) {
                            auto refused = std::make_unique<server_task_result_learn_grants>();
                            refused->id = task.id;
                            refused->receipt = {{"success", false}, {"refusal", "no_optimizer_state"},
                                {"engine", learn_engine}, {"id", id}};
                            turn_sink->deliver(std::move(refused));
                            break;
                        }
                    }
                    json checkpoint;
                    std::string saved_path = task.load_lora_path;
                    std::string error;
                    if (task.save_checkpoint.is_null()) {
                        error = server_publish_adapter(saved_path, [&](const std::string & temporary) {
                            return llama_adapter_lora_save(loras[id].ptr, temporary.c_str()) == 0;
                        });
                    } else {
                        const std::string generation = saved_path + ".checkpoint";
                        const std::string binding = task.save_checkpoint.at("binding").get<std::string>();
                        checkpoint = {{"version", 1}, {"binding", binding}, {"steps", steps},
                            {"adapter", "adapter.gguf"}, {"optimizer", "optimizer.gguf"},
                            {"provenance", task.save_checkpoint.at("provenance")}};
                        error = server_publish_adapter_generation(generation, [&](const std::string & temporary) {
                            if (llama_adapter_lora_save(loras[id].ptr, (temporary + "/adapter.gguf").c_str()) != 0) { return false; }
                            FILE * state = fopen((temporary + "/optimizer.gguf").c_str(), "wb");
                            if (!state) { return false; }
                            const bool written = llama_opt_shadow_state_write(ctx_tgt, loras[id].ptr, state, binding.c_str());
                            const bool closed = fclose(state) == 0;
                            if (!written || !closed) { return false; }
                            const std::string manifest = checkpoint.dump();
                            FILE * file = fopen((temporary + "/manifest.json").c_str(), "wb");
                            if (!file) { return false; }
                            const bool complete = fwrite(manifest.data(), 1, manifest.size(), file) == manifest.size();
                            const bool manifest_closed = fclose(file) == 0;
                            return complete && manifest_closed;
                        });
                        saved_path = generation + "/adapter.gguf";
                        checkpoint["path"] = generation;
                    }
                    if (!error.empty()) {
                        turn_sink->deliver(turn_error(task, error, ERROR_TYPE_INVALID_REQUEST));
                        break;
                    }
                    auto res = std::make_unique<server_task_result_lora_change>();
                    res->id = task.id;
                    res->lora_id = id;
                    res->path = saved_path;
                    // read on the loop between steps, beside the write: the file holds exactly these
                    res->steps = steps;
                    res->checkpoint = std::move(checkpoint);
                    turn_sink->deliver(std::move(res));
                } break;
            case SERVER_TASK_TYPE_LOAD_LORA:
                {
                    // Read from disk off this loop by the HTTP handler; appending it here,
                    // between decode steps, is what makes the list change atomic for slots.
                    // It joins dormant (scale 0), so no slot's output or KV changes until a
                    // request or POST /lora-adapters gives it a scale. Ids are never reused:
                    // a stale id must never address a different adapter.
                    // Idempotent: a client that dropped after posting and retries gets the id
                    // it never learned, not a second copy under a new one.
                    int32_t existing = -1;
                    for (size_t i = 0; i < params_base.lora_adapters.size(); ++i) {
                        if (params_base.lora_adapters[i].ptr != nullptr && params_base.lora_adapters[i].path == task.load_lora_path) {
                            existing = (int32_t) i;
                            break;
                        }
                    }
                    if (existing >= 0) {
                        llama_adapter_lora_free(task.load_lora);
                        residency_mirror.finish(task.load_lora_residency, false);
                        auto res = std::make_unique<server_task_result_lora_change>();
                        res->id      = task.id;
                        res->lora_id = existing;
                        res->path    = task.load_lora_path;
                        turn_sink->deliver(std::move(res));
                        break;
                    }
                    char buf[1024];
                    common_adapter_lora_info la;
                    la.path  = task.load_lora_path;
                    la.scale = 0.0f;
                    la.ptr   = task.load_lora;
                    llama_adapter_meta_val_str(la.ptr, "adapter.lora.task_name", buf, sizeof(buf));
                    la.task_name = buf;
                    llama_adapter_meta_val_str(la.ptr, "adapter.lora.prompt_prefix", buf, sizeof(buf));
                    la.prompt_prefix = buf;
                    params_base.lora_adapters.push_back(la);
                    residency_mirror.finish(task.load_lora_residency, true, (int32_t) params_base.lora_adapters.size() - 1);
                    publish_lora_report();
                    SRV_INF("loaded lora adapter id=%zu path=%s (dormant)\n", params_base.lora_adapters.size() - 1, la.path.c_str());
                    auto res = std::make_unique<server_task_result_lora_change>();
                    res->id      = task.id;
                    res->lora_id = (int32_t) params_base.lora_adapters.size() - 1;
                    res->path    = la.path;
                    turn_sink->deliver(std::move(res));
                } break;
            case SERVER_TASK_TYPE_LEARN_GRANTS:
                process_learn_grants(task);
                break;
            case SERVER_TASK_TYPE_PROMPT_CACHE_LIMIT:
                {
                    // The host prompt cache gives its bytes back: a cache is never need (card 494c99f1).
                    // The limit is lowered or raised, and update() evicts the oldest states until the
                    // cache fits it. The reply is what the cache holds now, so the core frees exactly
                    // what it measured, never what it hoped.
                    auto result = std::make_unique<server_task_result_learn_grants>();
                    result->id = task.id;
                    if (!prompt_cache) {
                        result->receipt = {{"refusal", "prompt_cache_disabled"}};
                    } else {
                        const size_t before = prompt_cache->size();
                        prompt_cache->limit_size = 1024ull * 1024ull * (size_t) task.prompt_cache_limit_mib;
                        prompt_cache->update();
                        result->receipt = {{"refusal", "none"}, {"limit_mib", task.prompt_cache_limit_mib},
                            {"held_bytes", prompt_cache->size()}, {"freed_bytes", before - std::min(before, prompt_cache->size())},
                            {"states", prompt_cache->states.size()}};
                    }
                    turn_sink->deliver(std::move(result));
                } break;
            case SERVER_TASK_TYPE_LEARN:
                {
                    if (!task.restore_checkpoint.is_null()) {
                        auto result = std::make_unique<server_task_result_learn_grants>();
                        result->id = task.id;
                        result->receipt = restore_learning_checkpoint(task.learn, task.restore_checkpoint);
                        turn_sink->deliver(std::move(result));
                        break;
                    }
                    auto res = std::make_unique<server_task_result_learn>();
                    res->id = task.id;
                    const auto & L = task.learn;
                    const int64_t t0 = ggml_time_us();
                    server_slot * slot = get_slot_by_id(L.id_slot);
                    auto refuse = [&](const std::string & why) {
                        res->outcome.ok      = false;
                        res->outcome.failure = why;
                        SRV_WRN("learn refused: %s\n", why.c_str());
                        turn_sink->deliver(std::move(res));
                    };
                    if (slot == nullptr) {
                        refuse("no slot " + std::to_string(L.id_slot));
                        break;
                    }
                    if (slot->is_processing()) {
                        // her slot is serving: learn between its turns, never during one
                        queue_tasks.defer(std::move(task));
                        break;
                    }
                    res->outcome = learn_on_slot(*slot, L, t0);
                    if (slot->state == SLOT_STATE_LEARNING) {
                        learning->separate_request = true;
                        slot->task = std::make_unique<const server_task>(std::move(task));
                    } else {
                        turn_sink->deliver(std::move(res));
                    }
                } break;
            case SERVER_TASK_TYPE_UNLOAD_LORA:
                {
                    if (task.unload_engine != learn_engine) {
                        turn_sink->deliver(turn_error(task, "stale_engine: unload must name this engine incarnation", ERROR_TYPE_INVALID_REQUEST));
                        break;
                    }
                    auto & loras = params_base.lora_adapters;
                    const int32_t id = task.unload_lora;
                    if (id < 0 || (size_t) id >= loras.size() || loras[id].ptr == nullptr) {
                        turn_sink->deliver(turn_error(task, "no loaded LoRA adapter with id " + std::to_string(id), ERROR_TYPE_INVALID_REQUEST));
                        break;
                    }
                    llama_adapter_lora * ptr = loras[id].ptr;
                    if (loras[id].scale != 0.0f) {
                        turn_sink->deliver(turn_error(task, "LoRA adapter " + std::to_string(id) + " is active by default (scale " +
                                   std::to_string(loras[id].scale) + "); set its scale to 0 first", ERROR_TYPE_INVALID_REQUEST));
                        break;
                    }
                    for (const auto & slot : slots) {
                        if (slot.is_processing() && (size_t) id < slot.lora.size() && slot.lora[id].ptr == ptr &&
                            slot.lora[id].scale != 0.0f) {
                            turn_sink->deliver(turn_error(task, "LoRA adapter " + std::to_string(id) + " is in use by slot " +
                                       std::to_string(slot.id) + "; retry when it finishes", ERROR_TYPE_UNAVAILABLE));
                            ptr = nullptr;
                            break;
                        }
                    }
                    if (ptr == nullptr) {
                        break;
                    }
                    // the shadow a learn is training stays bound from begin through restoration (its
                    // optimizer and accumulators reference its tensors between chunks), whether or not
                    // any slot's stable set names it
                    if (learning && learning->shadow == ptr) {
                        turn_sink->deliver(turn_error(task, "LoRA adapter " + std::to_string(id) + " is the shadow a learn is training on slot " +
                                   std::to_string(learning->slot->id) + "; retry when its restoration finishes", ERROR_TYPE_UNAVAILABLE));
                        break;
                    }
                    // a shadow that learned holds an optimizer referencing its tensors: release it
                    // first (its moments go; a snapshot is the durable form of what it learned)
                    if (!llama_opt_release(ctx_tgt, ptr)) {
                        turn_sink->deliver(turn_error(task, "LoRA adapter " + std::to_string(id) + " cannot be unloaded yet: " + llama_opt_failure(ctx_tgt),
                                   ERROR_TYPE_UNAVAILABLE));
                        break;
                    }
                    // An idle slot that last ran with it keeps KV computed WITH the adapter;
                    // once the adapter is gone its list would read as the base's, so that
                    // cache must not be reused.
                    for (auto & slot : slots) {
                        if ((size_t) id < slot.lora.size() && slot.lora[id].ptr == ptr) {
                            if (slot.lora[id].scale != 0.0f) {
                                slot.prompt.clear();
                            }
                            slot.lora[id].scale = 0.0f;
                            slot.lora[id].ptr   = nullptr;
                        }
                    }
                    // --cache-ram states computed with it would restore its KV into a slot
                    // that no longer runs it
                    if (prompt_cache) {
                        prompt_cache->evict_lora(ptr);
                    }
                    // the context may still hold it from the last batch; clear before freeing.
                    // Only ctx_tgt ever receives adapters (common_set_adapter_lora in
                    // update_slots); the draft/MTP context never does.
                    llama_set_adapters_lora(ctx_tgt, nullptr, 0, nullptr);
                    llama_adapter_lora_free(ptr);
                    residency_mirror.freed_lora(id);
                    learning_residencies.erase(id);
                    // Failure retains the reference; the existing release operation
                    // retries this same owner after the transient reserve failure clears.
                    reclaim_learning_workspace();
                    // An empty path is the retired mark: a model reload after sleep skips it
                    // (common_init_from_params) and the id stays taken.
                    auto res = std::make_unique<server_task_result_lora_change>();
                    res->id      = task.id;
                    res->lora_id = id;
                    res->path    = loras[id].path;
                    res->loaded  = false;
                    loras[id].ptr = nullptr;
                    loras[id].path.clear();
                    publish_lora_report();
                    SRV_INF("retired lora adapter id=%d path=%s\n", id, res->path.c_str());
                    turn_sink->deliver(std::move(res));
                } break;
        }

        return true;
    }
