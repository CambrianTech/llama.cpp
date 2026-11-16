// Methods of server_context_impl; one owner for completion and /learn custody.
    static const char * learn_refusal_name(server_learn_refusal r) {
        switch (r) {
#define LEARN_REFUSAL(x) case server_learn_refusal::x: return #x
            LEARN_REFUSAL(none); LEARN_REFUSAL(invalid); LEARN_REFUSAL(stale_engine);
            LEARN_REFUSAL(grant_mismatch); LEARN_REFUSAL(unknown_grant); LEARN_REFUSAL(never_fits); LEARN_REFUSAL(uncontended_only);
            LEARN_REFUSAL(budget_exhausted); LEARN_REFUSAL(started); LEARN_REFUSAL(not_started);
            LEARN_REFUSAL(pending_calibration); LEARN_REFUSAL(chunk_exceeds_latency);
            LEARN_REFUSAL(already_started); LEARN_REFUSAL(unknown_reservation);
#undef LEARN_REFUSAL
        }
        return "invalid";
    }

    json learn_memory_receipt() const {
        json out = {{"optimizer_owned_buffers", json::array()}, {"shared_compute_buffers", json::array()}};
        struct visitor { const server_context_impl * self; json * rows; } v{this, &out["optimizer_owned_buffers"]};
        llama_opt_visit_memory(ctx_tgt, [](const llama_adapter_lora * shadow, const char * device,
                const char * buft, const char * kind, size_t bytes, bool host, void * data) {
            const auto & v = *static_cast<visitor *>(data);
            int32_t id = -1;
            const auto & adapters = v.self->params_base.lora_adapters;
            for (size_t i = 0; i < adapters.size(); ++i) {
                if (adapters[i].ptr == shadow) { id = (int32_t) i; break; }
            }
            v.rows->push_back({{"lora_id", id}, {"device", device}, {"buffer_type", buft},
                {"kind", kind}, {"bytes", bytes}, {"host", host}});
        }, &v);
        // One serving scheduler, reported once. This is CURRENT TOTAL compute, not a guessed
        // training delta. Core compares measured snapshots under the same context geometry.
        for (const auto & entry : llama_get_memory_breakdown(ctx_tgt)) {
            const auto buft = entry.first;
            const auto device = ggml_backend_buft_get_device(buft);
            if (entry.second.compute) {
                out["shared_compute_buffers"].push_back({{"device", device ? ggml_backend_dev_name(device) : "unknown"},
                    {"buffer_type", ggml_backend_buft_name(buft)}, {"bytes", entry.second.compute},
                    {"host", ggml_backend_buft_is_host(buft)}});
            }
        }
        out["optimizer_unallocated_lora_ids"] = json::array();
        for (size_t i = 0; i < params_base.lora_adapters.size(); ++i) {
            if (!params_base.lora_adapters[i].ptr) { continue; }
            bool observed = false;
            for (const auto & row : out["optimizer_owned_buffers"]) {
                observed |= row.at("lora_id").get<int32_t>() == (int32_t) i;
            }
            if (!observed) { out["optimizer_unallocated_lora_ids"].push_back(i); }
        }
        out["allocation_estimate"] = false;
        return out;
    }

    llama_opt_params learning_shadow_params(int32_t id) {
        llama_opt_params lopt{};
        lopt.param_filter = llama_opt_param_filter_all;
        lopt.get_opt_pars = learn_opt_pars;
        lopt.get_opt_pars_ud = &learn_lr;
        lopt.optimizer_type = GGML_OPT_OPTIMIZER_TYPE_ADAMW;
        lopt.adapter = params_base.lora_adapters[id].ptr;
        const auto binding = learning_residencies.find(id);
        if (binding != learning_residencies.end()) {
            lopt.static_admission_data = lopt.workspace_admission_data = binding->second.get();
            lopt.static_admission = [](ggml_backend_buffer_type_t buft, const char * kind, size_t bytes, void * data) {
                const auto required = std::string(kind) + "_required";
                return static_cast<learning_residency *>(data)->allows(buft, required.c_str(), bytes);
            };
            lopt.workspace_admission = [](ggml_backend_buffer_type_t buft, size_t required, size_t, void * data) {
                return static_cast<learning_residency *>(data)->allows(buft, "shared_compute_required", required);
            };
        }
        return lopt;
    }

    bool select_learning_shadow(int32_t id) {
        return llama_opt_init(ctx_tgt, model_tgt, learning_shadow_params(id));
    }

    json learn_preflight(const json & body) {
        json out = {{"engine", learn_engine}, {"refusal", "none"},
            {"reserved", false}, {"requirements", json::array()}};
        auto refuse = [&](const char * reason, const std::string & detail) {
            out["refusal"] = reason;
            out["detail"] = detail;
            return out;
        };
        if (body.value("engine", std::string()) != learn_engine) {
            return refuse("stale_engine", "preflight must name this engine incarnation");
        }
        if (learning) { return refuse("busy", "learning retains the context through restoration"); }
        if (params_base.train_dir.empty()) { return refuse("unsupported", "learning is disabled"); }
        const int64_t id = body.value("lora_id", int64_t(-1));
        if (id < 0 || (size_t) id >= params_base.lora_adapters.size() || !params_base.lora_adapters[id].ptr) {
            return refuse("invalid", "preflight requires a loaded shadow");
        }
        if (const auto why = train_repack_refusal(model_tgt, params_base); !why.empty()) {
            return refuse("unsupported", why);
        }
        const int64_t reply_rows = body.value("reply_rows", int64_t(0));
        if (reply_rows <= 0 || reply_rows > llama_n_ctx_seq(ctx_tgt)) {
            return refuse("invalid", "reply_rows must describe a nonempty example within the served window");
        }
        // The loop serializes this with decoding and adapter unload. Restore the active
        // default composition on exit; update_slots installs each batched slot before decode.
        auto saved = construct_lora_list({});
        struct restore_adapters {
            llama_context * ctx;
            std::vector<common_adapter_lora_info> & set;
            ~restore_adapters() { common_set_adapter_lora(ctx, set); }
        } restore{ctx_tgt, saved};
        auto set = construct_lora_list({});
        set[id].scale = 1.0f;
        common_set_adapter_lora(ctx_tgt, set);
        const bool ok = llama_opt_measure_shadow_graph(ctx_tgt, learning_shadow_params((int32_t) id), (uint32_t) reply_rows, [](const llama_adapter_lora *, const char * device,
                const char * buft, const char * kind, size_t bytes, bool host, void * data) {
            static_cast<json *>(data)->push_back({{"device", device}, {"buffer_type", buft},
                {"kind", kind}, {"bytes", bytes}, {"host", host}});
        }, &out["requirements"]);
        if (!ok) { return refuse("unsupported", llama_opt_failure(ctx_tgt)); }
        out["lora_id"] = id;
        out["reply_rows"] = reply_rows;
        out["served_window"] = llama_n_ctx_seq(ctx_tgt);
        return out;
    }

    static std::string residency_subject(const json & subject) {
        if (!subject.is_object() || subject.size() != 1) { return {}; }
        if (subject.contains("workspace") && subject.at("workspace") == true) { return "workspace"; }
        json normalized;
        if (subject.contains("fresh")) {
            const auto & f = subject.at("fresh");
            normalized = {{"fresh", {{"name", f.at("name").get<std::string>()},
                {"rank", f.value("rank", 8)}, {"alpha", f.value("alpha", 16.0)},
                {"targets", f.value("targets", std::string("attn_q,attn_v"))},
                {"top_layers", f.value("top_layers", 0)}, {"seed", f.value("seed", uint32_t(42))}}}};
        } else if (subject.contains("load")) {
            normalized = {{"load", {{"name", subject.at("load").at("name").get<std::string>()}}}};
        } else if (subject.contains("loaded")) {
            normalized = {{"loaded", {{"lora_id", subject.at("loaded").at("lora_id").get<int32_t>()},
                {"path", subject.at("loaded").at("path").get<std::string>()}}}};
        } else { return {}; }
        return normalized.dump();
    }

    std::string register_residency(const json & body) {
        try {
            server_residency r;
            r.id = body.at("id").get<std::string>();
            r.engine = body.at("engine").get<std::string>();
            r.owner = body.at("owner").get<std::string>();
            r.subject = residency_subject(body.at("subject"));
            auto bytes = [](const json & j) -> uint64_t {
                if (!j.is_number_integer() || j.get<double>() < 0 || j.get<double>() >= double(INT64_MAX)) {
                    throw std::runtime_error("invalid residency bytes");
                }
                return j.get<uint64_t>();
            };
            if (!body.at("rows").is_array() || !body.at("leases").is_array()) { return "invalid"; }
            for (const auto & row : body.at("rows")) {
                r.rows.push_back({row.at("device").get<std::string>(), row.at("buffer_type").get<std::string>(),
                    row.at("kind").get<std::string>(), bytes(row.at("bytes")), row.at("host").get<bool>()});
            }
            for (const auto & lease : body.at("leases")) {
                r.leases.push_back({lease.at("kind").get<std::string>(), lease.at("lease_id").get<std::string>(), bytes(lease.at("bytes"))});
            }
            return residency_mirror.register_residency(r);
        } catch (const std::exception &) { return "invalid"; }
    }

    json residency_snapshot() const {
        json out = json::array();
        for (const auto & r : residency_mirror.snapshot()) {
            json rows = json::array(), leases = json::array();
            for (const auto & row : r.rows) {
                rows.push_back({{"device", row.device}, {"buffer_type", row.buffer_type}, {"kind", row.kind},
                    {"bytes", row.bytes}, {"host", row.host}});
            }
            for (const auto & lease : r.leases) {
                leases.push_back({{"kind", lease.kind}, {"lease_id", lease.lease_id}, {"bytes", lease.bytes}});
            }
            out.push_back({{"id", r.id}, {"engine", r.engine}, {"owner", r.owner}, {"subject", r.subject == "workspace" ? json{{"workspace", true}} : json::parse(r.subject)},
                {"rows", rows}, {"leases", leases}, {"pending", r.pending}, {"live", r.live},
                {"revoked", r.revoked}, {"lora_id", r.lora_id}});
        }
        return out;
    }

    void process_learn_grants(const server_task & task) {
        const auto now = ggml_time_us();
        auto result = std::make_unique<server_task_result_learn_grants>();
        result->id = task.id;
        json out = {{"engine", learn_engine}, {"clock_us", now}, {"refusal", "none"}};
        const auto & body = task.learn_grants;
        const auto op = body.value("op", std::string());
        if (op == "preflight") {
            result->receipt = learn_preflight(body);
            queue_results.send(std::move(result));
            return;
        }
        if (op == "residency_register") {
            out["refusal"] = register_residency(body);
        } else if ((op == "residency_release" || op == "residency_revoke") &&
                   body.value("engine", std::string()) != learn_engine) {
            out["refusal"] = "stale_engine";
        } else if (op == "residency_release") {
            const auto id = body.value("id", std::string());
            if (id == learning_workspace_residency) {
                // Serialized with learning and unload. A failed serving reserve
                // remains busy and may be retried without another shadow unload.
                reclaim_learning_workspace();
            }
            out["refusal"] = residency_mirror.release(id);
        } else if (op == "residency_revoke") {
            out["refusal"] = residency_mirror.revoke(body.value("id", std::string()));
        } else if (op == "register") {
            const auto duration = body.value("expires_in_ms", int64_t(0));
            server_learn_grant g{body.value("id", std::string()), body.value("engine", std::string()),
                body.value("period_ms", int64_t(0)), body.value("share_ppm", int64_t(0)), 0, body.value("max_chunk_us", int64_t(0))};
            auto refusal = server_learn_refusal::invalid;
            if (body.contains("share_ppm") && duration > 0 && duration <= server_learn_ledger::kMaxPeriodMs &&
                    now <= server_learn_ledger::kMaxClockUs - duration * 1000) {
                g.expires_at_us = now + duration * 1000;
                refusal = learn_ledger.register_grant(g, now);
            }
            out["refusal"] = learn_refusal_name(refusal);
        } else if (op == "revoke") {
            learn_ledger.revoke(body.value("id", std::string()), now);
            out["revoked"] = true; // idempotent: this id can admit no new work
        }
        const json receipt = ledger_receipt(now);
        for (const auto & entry : receipt.items()) { out[entry.key()] = entry.value(); }
        out["residencies"] = residency_snapshot();
        result->receipt = std::move(out);
        queue_results.send(std::move(result));
    }

    // The loop-owned half of the ledger (grants, their use, the memory the optimizer holds), read
    // ON THE LOOP: the reply to a mutation, and the snapshot every boundary publishes.
    json ledger_receipt(int64_t now) const {
        json out = {{"max_period_ms", server_learn_ledger::kMaxPeriodMs}, {"max_expiry_ms", server_learn_ledger::kMaxPeriodMs},
            {"grants", json::array()}};
        for (const auto & a : learn_ledger.snapshot()) {
            out["grants"].push_back({{"id", a.grant.id}, {"period_ms", a.grant.period_ms},
                {"max_chunk_us", a.grant.max_chunk_us}, {"share_ppm", a.grant.share_ppm}, {"expires_at_us", a.grant.expires_at_us},
                {"revoked", a.revoked}, {"outstanding", a.outstanding},
                {"committed_steps", learn_ledger.committed_steps(a.grant.id)}});
            int64_t used = 0;
            if (learn_ledger.used_us(a.grant.id, now, used) == server_learn_refusal::none) {
                out["grants"].back()["used_us"] = used;
            }
        }
        out["memory"] = learn_memory_receipt();
        return out;
    }

    // A control read never queues behind the work it reports on (card 7fc7e2ce, Fable's design):
    // the loop publishes its half of the ledger at every task and slot-update boundary, stamped,
    // and a reader on any thread takes the last publication plus the residency mirror (its own
    // mutex). At most one boundary stale, and the stamp says when.
    struct ledger_publication {
        std::mutex mutex;
        std::condition_variable published; // signalled at every publication: the wait-on-ticket wake
        json receipt;
        uint64_t sequence = 0;
    };
    mutable ledger_publication ledger_published;
    uint64_t ledger_published_revision = 0; // loop-owned: the ledger revision the last publication holds

    // ON THE LOOP, at a slot-update boundary: publish only when the ledger changed since the last
    // publication (a task boundary, below, always publishes).
    void publish_ledger_if_changed() {
        if (learn_ledger.revision() != ledger_published_revision) {
            publish_ledger();
        }
    }

    // ON THE LOOP, at a boundary.
    void publish_ledger() {
        ledger_published_revision = learn_ledger.revision();
        const auto now = ggml_time_us();
        json receipt = ledger_receipt(now);
        receipt["clock_us"] = now;
        {
            std::lock_guard<std::mutex> lock(ledger_published.mutex);
            receipt["sequence"] = ++ledger_published.sequence;
            ledger_published.receipt = std::move(receipt);
        }
        ledger_published.published.notify_all();
    }

    // ANY THREAD: the last publication, the live residency mirror and the engine identity (const).
    json published_ledger() const {
        json out;
        {
            std::lock_guard<std::mutex> lock(ledger_published.mutex);
            out = ledger_published.receipt;
        }
        out["engine"] = learn_engine;
        out["refusal"] = "none";
        out["residencies"] = residency_snapshot();
        return out;
    }


    // One reply learned on `slot`, which holds the served prompt (POST /learn between its turns, or a
    // completion's own learn before its slot is released): the slot's stable set plus the shadow on
    // the reply's labelled rows, then the slot back to its whole served prompt.
    learn_outcome learn_on_slot(server_slot & slot, const learn_request & L, int64_t t0) {
        if (learning) {
            learn_outcome out;
            out.deferred = true;
            out.failure = "the lane optimizer is restoring an admitted reply; retain this experience for retry";
            return out;
        }
        const learn_geometry geometry{L.lora_id, L.n_prefix, (int32_t) L.reply.size(), slot.prompt.tokens.size()};
        const bool warm = learning_residencies.count(L.lora_id) != 0;
        const auto prior = learn_chunk_us.find(geometry);
        int64_t boundary_us = 0;
        if (prior != learn_chunk_us.end()) {
            const auto & cost = prior->second;
            const auto entry = warm ? cost.entry_warm : cost.entry_cold;
            if (entry > 0 && cost.exit > 0) { boundary_us = std::max(entry, cost.exit); }
        }
        bool contended = false;
        for (const auto & other : slots) { contended |= other.id != slot.id && other.is_processing(); }
        auto boundary = learn_ledger.admit_chunk(L.grant, boundary_us, t0, contended);
        if (boundary.refusal != server_learn_refusal::none) {
            learn_outcome out;
            out.refusal = learn_refusal_name(boundary.refusal);
            out.failure = "entry/restoration geometry has no admitted measured latency budget";
            out.deferred = true;
            return out;
        }
        // No work ran under this reservation. The serialized loop transfers to the
        // existing restoration obligation below, which debits setup and exit exactly once.
        learn_ledger.cancel(boundary.reservation, t0);
        // Validate authority before touching adapters, optimizer, or sequence state.
        // This zero-estimate obligation remains live until the served tail is restored.
        // Its measured setup/restoration time is charged even if authority ends meanwhile.
        auto probe = learn_ledger.hold_restoration(L.grant, t0);
        if (probe.refusal != server_learn_refusal::none) {
            learn_outcome out;
            out.refusal = learn_refusal_name(probe.refusal);
            out.failure = "learning requires a live registered authority grant";
            return out;
        }
        const auto started = learn_ledger.start(probe.reservation, t0);
        if (started != server_learn_refusal::none) {
            learn_ledger.cancel(probe.reservation, t0);
            learn_outcome out;
            out.refusal = learn_refusal_name(started);
            out.failure = "learning authority ended before sequence custody began";
            return out;
        }
        auto out = learn_on_slot_admitted(slot, L, t0);
        const auto setup_us = ggml_time_us() - t0;
        auto & entry = warm ? learn_chunk_us[geometry].entry_warm : learn_chunk_us[geometry].entry_cold;
        entry = std::max(entry, setup_us);
        if (learning) {
            learning->served_tokens = std::get<3>(geometry);
            learning->restoration_reservation = probe.reservation;
            learning->restoration_us = setup_us;
            learning->entry_us = setup_us;
            learning->compute_us += setup_us;
            learning->max_chunk_us = setup_us;
        } else {
            const auto settled = learn_ledger.finish(probe.reservation, setup_us, 0, ggml_time_us());
            if (settled != server_learn_refusal::none) { out.refusal = learn_refusal_name(settled); }
        }
        return out;
    }

    learn_outcome learn_on_slot_admitted(server_slot & slot, const learn_request & L, int64_t t0) {
        auto refused = [&](const std::string & why) {
            learn_outcome out;
            out.failure = why;
            SRV_WRN("learn refused: %s\n", why.c_str());
            return out;
        };
        // the same refusal /train gives: repacked CPU weights fail the backward (x86) or crash
        // the copy to Metal (an M5 at -ngl 0), card 5b497852
        if (const std::string why = train_repack_refusal(model_tgt, params_base); !why.empty()) {
            return refused(why);
        }
        auto & loras = params_base.lora_adapters;
        if (L.lora_id < 0 || (size_t) L.lora_id >= loras.size() || loras[L.lora_id].ptr == nullptr) {
            return refused("no loaded adapter " + std::to_string(L.lora_id) + " to train (POST /lora-adapters/load it first)");
        }
        const int32_t n0 = L.n_prefix - 1; // the last prompt token is re-decoded with the reply's first token as its label
        if (n0 < 0 || L.n_prefix > slot.prompt.n_tokens() || L.reply.empty()) {
            return refused("slot " + std::to_string(slot.id) + " holds " + std::to_string(slot.prompt.n_tokens())
                + " prompt tokens; the served prefix is " + std::to_string(L.n_prefix) + " and the reply must not be empty");
        }
        // the served prompt, when the request carries it: her slot must still hold exactly it (a
        // slot that has served other turns since, or compacted, would teach the wrong context)
        if (!L.prompt.empty()) {
            int32_t i = 0;
            while (i < L.n_prefix && slot.prompt.tokens[i] == L.prompt[i]) {
                ++i;
            }
            if (i < L.n_prefix) {
                return refused("slot " + std::to_string(slot.id) + " no longer holds this turn's prompt (it differs at token "
                    + std::to_string(i) + " of " + std::to_string(L.n_prefix) + "): serve the prompt again, then learn");
            }
        }
        // the recurrent part cannot be cut back: her state one token before the prompt's end
        const common_prompt_checkpoint * cp = nullptr;
        const bool needs_cp = ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL || ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS;
        if (needs_cp) {
            for (const auto & c : slot.prompt.checkpoints) {
                if (c.n_tokens == n0) {
                    cp = &c;
                }
            }
            if (cp == nullptr) {
                return refused("slot " + std::to_string(slot.id) + " has no checkpoint one token before its prompt's end ("
                    + std::to_string(n0) + "): the lane learns only with --train-dir, which keeps one");
            }
        }
        // the tokens from the last prompt token on, each labelled with the next
        std::vector<llama_token> toks;
        toks.push_back(slot.prompt.tokens[n0]);
        toks.insert(toks.end(), L.reply.begin(), L.reply.end());
        std::vector<llama_token> labels(toks.size(), -1);
        for (size_t i = 0; i + 1 < toks.size(); ++i) {
            labels[i] = toks[i + 1];
        }
        if (const auto why = admit_learning_residency(L); !why.empty()) { return refused(why); }
        // her stable set plus the shadow: the shadow applies to labelled rows only
        std::vector<common_adapter_lora_info> set = slot.lora.empty() ? construct_lora_list({}) : slot.lora;
        if ((size_t) L.lora_id >= set.size()) {
            set = construct_lora_list({});
        }
        set[L.lora_id].scale = 1.0f;
        common_set_adapter_lora(ctx_tgt, set);
        learn_lr = L.lr;
        if (!select_learning_shadow(L.lora_id)) {
            common_set_adapter_lora(ctx_tgt, slot.lora);
            settle_learning_residency(L.lora_id);
            return refused("adapter " + std::to_string(L.lora_id) + " cannot be trained: " + llama_opt_failure(ctx_tgt));
        }
        // drop what follows n0 (attention) and put her recurrent state back at n0
        if (cp != nullptr) {
            cp->load_tgt(ctx_tgt, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        }
        const auto & served_tokens = slot.prompt.tokens.get_tokens();
        ggml_opt_result_t result = ggml_opt_result_init();
        const bool begun = llama_opt_learn_begin(ctx_tgt, slot.id, n0, toks.data(), labels.data(), (int32_t) toks.size(),
            served_tokens.data() + n0, (int32_t) served_tokens.size() - n0, result);
        common_set_adapter_lora(ctx_tgt, slot.lora);
        learning = std::make_unique<pending_learn>(&slot, cp, std::move(set), result,
            toks[0], L.n_prefix, (int32_t) toks.size() - 1, loras[L.lora_id].ptr, t0);
        learning->began = begun;
        learning->grant = L.grant;
        learning->lora_id = L.lora_id;
        if (!begun) {
            learning->finished = false;
            learning->failure = llama_opt_failure(ctx_tgt);
        }
        slot.state = SLOT_STATE_LEARNING;
        slot.learn_release_requested = false;
        slot.learn_graph_steps = 0;
        return {}; // no outcome is published until advance_learning restores the slot
    }

    void advance_learning() {
        if (!learning) { return; }
        auto & pending = *learning;
        auto & slot = *pending.slot;
        if (!pending.finished) {
            const auto chunk_started = ggml_time_us();
            uint64_t reservation = 0;
            if (llama_opt_learn_needs_gradient(ctx_tgt)) {
                auto & binding = *learning_residencies.at(pending.lora_id);
                if (!binding.authorized()) {
                    pending.refusal = "residency_revoked";
                    pending.failure = "residency revoked; restoring the stable tail before releasing custody";
                    pending.drained = true;
                    llama_opt_learn_drain(ctx_tgt);
                }
            }
            if (llama_opt_learn_needs_gradient(ctx_tgt) && !llama_opt_learn_draining_graph(ctx_tgt)) {
                bool contended = false;
                for (const auto & other : slots) {
                    contended |= other.id != slot.id && other.is_processing();
                }
                const learn_geometry geometry{pending.lora_id, pending.prefix, pending.n_labels, pending.served_tokens};
                const auto measured = learn_chunk_us.find(geometry);
                auto admission = learn_ledger.admit_chunk(pending.grant,
                    measured == learn_chunk_us.end() ? 0 : measured->second.gradient, chunk_started, contended);
                if (admission.refusal == server_learn_refusal::budget_exhausted) { return; }
                if (admission.refusal == server_learn_refusal::none) {
                    admission.refusal = learn_ledger.start(admission.reservation, chunk_started);
                    if (admission.refusal == server_learn_refusal::none) { reservation = admission.reservation; }
                    else { learn_ledger.cancel(admission.reservation, chunk_started); }
                }
                if (admission.refusal != server_learn_refusal::none) {
                    pending.refusal = learn_refusal_name(admission.refusal);
                    pending.failure = "authority refused the next gradient chunk; stable tail restored";
                    pending.drained = true;
                    llama_opt_learn_drain(ctx_tgt);
                }
            }
            learning_residencies.at(pending.lora_id)->restoring = !llama_opt_learn_needs_gradient(ctx_tgt) || llama_opt_learn_draining_graph(ctx_tgt);
            common_set_adapter_lora(ctx_tgt, pending.adapters);
            const auto steps_before = llama_opt_shadow_steps(ctx_tgt, pending.shadow);
            const bool graph_step = llama_opt_learn_needs_gradient(ctx_tgt);
            const auto status = llama_opt_learn_graph_step(ctx_tgt);
            if (graph_step) { ++slot.learn_graph_steps; }
            const auto committed = llama_opt_shadow_steps(ctx_tgt, pending.shadow) - steps_before;
            GGML_ASSERT(committed >= 0); // optimizer custody prevents generation replacement here
            pending.committed_steps += committed;
            if (!reservation) { pending.restoration_steps += committed; }
            common_set_adapter_lora(ctx_tgt, slot.lora);
            const auto chunk_finished = ggml_time_us();
            const auto elapsed = chunk_finished - chunk_started;
            pending.compute_us += elapsed;
            pending.max_chunk_us = std::max(pending.max_chunk_us, elapsed);
            // Classify the work, not its charging reservation: a revoked graph
            // still runs graph nodes under restoration custody until it commits.
            auto & phase_max = graph_step ? pending.max_graph_us : pending.max_restore_us;
            phase_max = std::max(phase_max, elapsed);
            if (!reservation) {
                pending.restoration_us += elapsed;
                pending.max_restoration_chunk_us = std::max(pending.max_restoration_chunk_us, elapsed);
            }
            if (reservation) {
                const learn_geometry geometry{pending.lora_id, pending.prefix, pending.n_labels, pending.served_tokens};
                auto & cost = learn_chunk_us[geometry].gradient;
                cost = std::max(cost, elapsed);
                const auto settled = learn_ledger.finish(reservation, elapsed,
                    committed, chunk_finished);
                if (settled != server_learn_refusal::none) {
                    pending.refusal = learn_refusal_name(settled);
                    pending.failure = "measured chunk settlement refused";
                    pending.drained = true;
                    llama_opt_learn_drain(ctx_tgt);
                }
            }
            if (status != LLAMA_OPT_LEARN_MORE) {
                pending.finished = status == LLAMA_OPT_LEARN_DONE;
                if (status != LLAMA_OPT_LEARN_DONE) { pending.failure = llama_opt_failure(ctx_tgt); }
            }
            // Even the final chunk yields before the last prompt token is restored below.
            return;
        }
        const auto restore_started = ggml_time_us();
        const auto * cp = pending.checkpoint;
        const int32_t n0 = pending.prefix - 1;
        const bool ok = *pending.finished;
        double loss = 0.0, unc = 0.0;
        if (ok) { ggml_opt_result_loss(pending.result, &loss, &unc); }
        // her own adapter set again before anything else is decoded for her (the shadow is
        // already off for a batch without labels: llama-graph.cpp lora_on, n_shadow_rows =
        // the ubatch's label count; this does not rely on it)
        common_set_adapter_lora(ctx_tgt, slot.lora);
        // The native owner restored the exact served tail once (not the supplied correction).
        // Target KV, token metadata and the untouched draft therefore still name the same tail.
        // If begin refused after loading the recurrent prefix, retain only that proven prefix.
        if (!pending.began || !ok) {
            llama_memory_seq_rm(llama_get_memory(ctx_tgt), slot.id, n0, -1);
            if (cp) { cp->load_tgt(ctx_tgt, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY); }
            if (ctx_dft) {
                llama_memory_seq_rm(llama_get_memory(ctx_dft), slot.id, n0, -1);
                if (cp) { cp->load_dft(ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY); }
            }
            slot.prompt.tokens.keep_first(n0);
            slot.prompt.checkpoints.remove_if([&](const common_prompt_checkpoint & c) { return c.n_tokens > n0; });
        }
        learn_outcome out;
        out.ok       = ok && !pending.drained;
        out.refusal = pending.refusal;
        out.deferred = pending.drained;
        out.failure  = pending.failure;
        out.committed_steps = pending.committed_steps;
        out.effects_uncertain = llama_opt_shadow_quarantined(pending.shadow);
        if (out.effects_uncertain) {
            out.ok = false;
            out.refusal = "effects_uncertain";
        }
        out.loss     = pending.committed_steps > 0 ? loss : 0.0;
        out.n_labels = pending.committed_steps > 0 ? pending.n_labels : 0;
        // Reconcile retained buffers and release result storage before settling
        // execution custody: both are synchronous parts of leaving learning.
        settle_learning_residency(pending.lora_id);
        ggml_opt_result_free(pending.result);
        pending.result = nullptr;
        const auto cleanup_us = ggml_time_us() - restore_started;
        pending.compute_us += cleanup_us;
        pending.restoration_us += cleanup_us;
        pending.max_chunk_us = std::max(pending.max_chunk_us, cleanup_us);
        auto & exit_cost = learn_chunk_us[learn_geometry{pending.lora_id, pending.prefix, pending.n_labels, pending.served_tokens}].exit;
        exit_cost = std::max(exit_cost, std::max(cleanup_us, pending.max_restoration_chunk_us));
        const auto settled = learn_ledger.finish(pending.restoration_reservation, pending.restoration_us, pending.restoration_steps, ggml_time_us());
        if (settled != server_learn_refusal::none) {
            out.ok = false;
            out.refusal = learn_refusal_name(settled);
            out.failure = "restoration custody settlement refused";
        }
        out.max_chunk_ms = pending.max_chunk_us / 1000.0;
        out.restoration_ms = pending.restoration_us / 1000.0;
        out.entry_ms = pending.entry_us / 1000.0;
        out.max_graph_ms = pending.max_graph_us / 1000.0;
        out.max_restore_ms = pending.max_restore_us / 1000.0;
        out.cleanup_ms = cleanup_us / 1000.0;
        out.t_ms     = pending.compute_us / 1000.0;
        SRV_INF("learned on slot %d: %d labels from position %d, loss %.4f, %.1f ms%s\n", slot.id, out.n_labels, n0, loss,
                out.t_ms, ok ? "" : (" (failed: " + out.failure + ")").c_str());
        const bool separate = pending.separate_request;
        const bool cancelled = slot.learn_release_requested;
        learning.reset();
        slot.state = SLOT_STATE_DONE_PROMPT; // release now owns fully restored KV
        if (!cancelled) {
            if (separate) {
                auto res = std::make_unique<server_task_result_learn>();
                res->id = slot.task->id;
                res->outcome = std::move(out);
                queue_results.send(std::move(res));
            } else {
                slot.learned = std::move(out);
                send_final_response(slot);
            }
        }
        slot.release();
    }

    // A completion that carries a learn learns its reply on its own slot as it finishes, before the
    // slot is released: no other request can take the slot between the turn and its learning (on
    // a one-slot lane that is busy almost always, a separate POST /learn lost that race 12 times in
    // 12 on the M5's 27B, 2026-10-15).
    void learn_if_asked(server_slot & slot) {
        if (!slot.task->params.learn) {
            return;
        }
        learn_request L = *slot.task->params.learn;
        L.id_slot  = slot.id;
        L.n_prefix = slot.task->n_tokens();
        slot.learned = learn_on_slot(slot, L, ggml_time_us());
    }


    // Continuum fork: what of learning cannot cross a live reconfigure (server-reconfigure.h), owned here
    // beside the learn duties. A learn in progress holds its graph and accumulators in the serving context
    // until its restoration ends. Each shadow's optimizer (its moments) lives in the context as well, and
    // the fork moves it only through llama_opt_shadow_state_write and, on the other side,
    // llama_opt_prepare_restore + llama_opt_state_restore, whose preparation needs the shadow's retained
    // labelled experience on a fresh context: a reconfigure holds no such experience, so the moments
    // cannot be paged with the slots' state and the rebuild is refused while any exist.
    std::optional<server_reconfigure_refused> learning_blocks_reconfigure() const {
        if (learning) {
            const int32_t id = learning->slot->id;
            return server_reconfigure_refused { SERVER_RECONFIGURE_LEARN_IN_PROGRESS,
                "a learn is in progress on slot " + std::to_string(id) + "; retry when its restoration finishes", id };
        }
        struct held_optimizers {
            size_t bytes = 0;
            std::set<const llama_adapter_lora *> shadows;
        } held;
        llama_opt_visit_memory(ctx_tgt, [](const llama_adapter_lora * shadow, const char *, const char *, const char *,
                                           size_t bytes, bool, void * user_data) {
            auto & h = *static_cast<held_optimizers *>(user_data);
            h.bytes += bytes;
            if (bytes > 0) {
                h.shadows.insert(shadow);
            }
        }, &held);
        if (held.bytes > 0) {
            return server_reconfigure_refused { SERVER_RECONFIGURE_LEARNING_BOUND,
                "the serving context holds the optimizer of " + std::to_string(held.shadows.size()) + " shadow(s), " +
                std::to_string(held.bytes >> 20) + " MiB; their moments do not cross a rebuild (snapshot and unload them first)", -1 };
        }
        if (!learning_workspace_residency.empty()) {
            return server_reconfigure_refused { SERVER_RECONFIGURE_LEARNING_BOUND,
                "the training workspace still replaces the serving scheduler (residency " + learning_workspace_residency +
                "); it is reclaimed when the last shadow is released", -1 };
        }
        return std::nullopt;
    }
