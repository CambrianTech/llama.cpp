// Serialized checkpoint restoration uses the same example renderer and allocation authority as learning.
    json restore_learning_checkpoint(const learn_request & request, const json & checkpoint) {
        json out = {{"engine", learn_engine}, {"restored", false}, {"committed_steps", 0}, {"refusal", "none"}};
        auto refuse = [&](const std::string & why) { out["refusal"] = why; return out; };
        if (checkpoint.at("engine").get<std::string>() != learn_engine) { return refuse("stale_engine"); }
        if (learning) { return refuse("busy"); }
        const int32_t id = request.lora_id;
        const auto & adapters = params_base.lora_adapters;
        if (id < 0 || (size_t) id >= adapters.size() || !adapters[id].ptr || adapters[id].scale != 0.0f) {
            return refuse("restore requires a dormant loaded shadow");
        }
        for (const auto & slot : slots) {
            if ((size_t) id < slot.lora.size() && slot.lora[id].scale != 0.0f) {
                return refuse("restore shadow is referenced by a serving slot");
            }
        }
        if (learning_residencies.count(id)) { return refuse("restore requires a fresh optimizer; unload and reseat"); }
        if (request.prompt.empty() || request.reply.empty() || request.n_prefix <= 0 ||
            request.n_prefix + request.reply.size() > llama_n_ctx_seq(ctx_tgt) || get_slot_by_id(request.id_slot) == nullptr) {
            return refuse("restore requires a whole rendered example within the served window and a valid slot");
        }
        const auto path = confine_out(params_base.train_dir, checkpoint.at("name").get<std::string>()) + ".checkpoint";
        const std::string binding = checkpoint.at("binding").get<std::string>();
        const int64_t steps = checkpoint.at("steps").get<int64_t>();
        json manifest;
        try {
            std::ifstream file(path + "/manifest.json");
            std::ostringstream contents;
            contents << file.rdbuf();
            manifest = json::parse(contents.str());
            if (manifest.at("version") != 1 || manifest.at("binding") != binding || manifest.at("steps") != steps ||
                manifest.at("adapter") != "adapter.gguf" || manifest.at("optimizer") != "optimizer.gguf" ||
                !manifest.at("provenance").is_object()) { return refuse("checkpoint manifest mismatch"); }
        } catch (const std::exception &) { return refuse("checkpoint manifest unreadable"); }
        std::unique_ptr<gguf_context, decltype(&gguf_free)> metadata(
            gguf_init_from_file((path + "/optimizer.gguf").c_str(), {true, nullptr}), gguf_free);
        if (!metadata) { return refuse("optimizer metadata unreadable"); }
        const int64_t iteration = gguf_find_key(metadata.get(), "optimizer.iteration");
        const int64_t bound = gguf_find_key(metadata.get(), "optimizer.state.binding");
        if (iteration < 0 || bound < 0 || gguf_get_kv_type(metadata.get(), iteration) != GGUF_TYPE_INT64 ||
            gguf_get_kv_type(metadata.get(), bound) != GGUF_TYPE_STRING ||
            gguf_get_val_i64(metadata.get(), iteration) <= 0 ||
            gguf_get_val_i64(metadata.get(), iteration) - 1 != steps ||
            std::string(gguf_get_val_str(metadata.get(), bound)) != binding) { return refuse("optimizer generation mismatch"); }
        // a deleter type, not decltype(&fclose): fclose's attributes are dropped from a template argument (-Wignored-attributes)
        struct file_closer { void operator()(FILE * f) const { fclose(f); } };
        std::unique_ptr<FILE, file_closer> weights(fopen((path + "/adapter.gguf").c_str(), "rb"));
        std::unique_ptr<FILE, file_closer> optimizer(fopen((path + "/optimizer.gguf").c_str(), "rb"));
        if (!weights || !optimizer) { return refuse("checkpoint components unreadable"); }
        const auto started = ggml_time_us();
        auto held = learn_ledger.hold_restoration(request.grant, started);
        if (held.refusal != server_learn_refusal::none) { return refuse(learn_refusal_name(held.refusal)); }
        if (const auto why = learn_ledger.start(held.reservation, started); why != server_learn_refusal::none) {
            learn_ledger.cancel(held.reservation, started);
            return refuse(learn_refusal_name(why));
        }
        auto stable = construct_lora_list({});
        struct release {
            server_context_impl * self;
            int32_t id;
            uint64_t reservation;
            int64_t started;
            std::vector<common_adapter_lora_info> & stable;
            ~release() {
                common_set_adapter_lora(self->ctx_tgt, stable);
                self->settle_learning_residency(id);
                const auto now = ggml_time_us();
                self->learn_ledger.finish(reservation, now - started, 0, now);
            }
        } cleanup{this, id, held.reservation, started, stable};
        if (const auto why = admit_learning_residency(request); !why.empty()) { return refuse(why); }
        auto attached = stable;
        attached[id].scale = 1.0f;
        common_set_adapter_lora(ctx_tgt, attached);
        learn_lr = request.lr;
        if (!select_learning_shadow(id)) { return refuse(llama_opt_failure(ctx_tgt)); }
        llama_batch retained = llama_batch_init((int32_t) request.reply.size(), 0, 1);
        struct free_batch { llama_batch & batch; ~free_batch() { llama_batch_free(batch); } } batch_cleanup{retained};
        retained.n_tokens = (int32_t) request.reply.size();
        for (int32_t i = 0; i < retained.n_tokens; ++i) {
            retained.token[i] = i == 0 ? request.prompt.back() : request.reply[i - 1];
            retained.pos[i] = request.n_prefix - 1 + i;
            retained.n_seq_id[i] = 1;
            retained.seq_id[i][0] = request.id_slot;
            retained.logits[i] = 1;
            retained.label[i] = request.reply[i];
        }
        if (!llama_opt_prepare_restore(ctx_tgt, retained)) { return refuse(llama_opt_failure(ctx_tgt)); }
        common_set_adapter_lora(ctx_tgt, stable);
        if (!llama_opt_state_restore(ctx_tgt, adapters[id].ptr, weights.get(), optimizer.get(), binding.c_str())) {
            out["effects_uncertain"] = llama_opt_shadow_quarantined(adapters[id].ptr);
            return refuse("checkpoint restore refused; retain custody until unload");
        }
        out["restored"] = true;
        out["steps"] = llama_opt_shadow_steps(ctx_tgt, adapters[id].ptr);
        out["provenance"] = manifest.at("provenance");
        out["effects_uncertain"] = false;
        return out;
    }
