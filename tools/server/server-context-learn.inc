// Methods of server_context_impl; one owner for completion and /learn custody.
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
        for (const auto & other : slots) {
            if (other.id != slot.id && other.is_processing() && t0 < learn_next_us) {
                learn_outcome out;
                out.deferred = true;
                out.retry_after_ms = (learn_next_us - t0 + 999) / 1000;
                out.failure = "learning deferred while another slot is busy; retain replayable input context and learning inputs for retry";
                return out;
            }
        }
        return learn_on_slot_admitted(slot, L, t0);
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
        if (learn_lora_id >= 0 && learn_lora_id != L.lora_id) {
            return refused("this server trains adapter " + std::to_string(learn_lora_id) + "; one shadow per server run");
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
        // her stable set plus the shadow: the shadow applies to labelled rows only
        std::vector<common_adapter_lora_info> set = slot.lora.empty() ? construct_lora_list({}) : slot.lora;
        if ((size_t) L.lora_id >= set.size()) {
            set = construct_lora_list({});
        }
        set[L.lora_id].scale = 1.0f;
        common_set_adapter_lora(ctx_tgt, set);
        learn_lr = L.lr;
        if (learn_lora_id < 0) {
            llama_opt_params lopt{};
            lopt.param_filter    = llama_opt_param_filter_all;
            lopt.get_opt_pars    = learn_opt_pars;
            lopt.get_opt_pars_ud = &learn_lr;
            lopt.optimizer_type  = GGML_OPT_OPTIMIZER_TYPE_ADAMW;
            lopt.adapter         = loras[L.lora_id].ptr;
            llama_opt_init(ctx_tgt, model_tgt, lopt);
            learn_lora_id = L.lora_id;
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
        learning = std::unique_ptr<pending_learn>(new pending_learn{ &slot, cp, std::move(set), result,
            toks[0], L.n_prefix, (int32_t) toks.size() - 1, t0 });
        learning->began = begun;
        if (!begun) {
            learning->finished = false;
            learning->failure = llama_opt_failure(ctx_tgt);
        }
        slot.state = SLOT_STATE_LEARNING;
        slot.learn_release_requested = false;
        return {}; // no outcome is published until advance_learning restores the slot
    }

    void advance_learning() {
        if (!learning) { return; }
        auto & pending = *learning;
        auto & slot = *pending.slot;
        if (!pending.finished) {
            const auto chunk_started = ggml_time_us();
            common_set_adapter_lora(ctx_tgt, pending.adapters);
            const auto status = llama_opt_learn_step(ctx_tgt);
            common_set_adapter_lora(ctx_tgt, slot.lora);
            pending.compute_us += ggml_time_us() - chunk_started;
            if (status != LLAMA_OPT_LEARN_MORE) {
                pending.finished = status == LLAMA_OPT_LEARN_DONE;
                pending.failure = status == LLAMA_OPT_LEARN_DONE ? "" : llama_opt_failure(ctx_tgt);
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
        out.ok       = ok;
        out.failure  = pending.failure;
        out.loss     = loss;
        out.n_labels = pending.n_labels;
        pending.compute_us += ggml_time_us() - restore_started;
        out.t_ms     = pending.compute_us / 1000.0;
        SRV_INF("learned on slot %d: %d labels from position %d, loss %.4f, %.1f ms%s\n", slot.id, out.n_labels, n0, loss,
                out.t_ms, ok ? "" : (" (failed: " + out.failure + ")").c_str());
        const bool separate = pending.separate_request;
        const bool cancelled = slot.learn_release_requested;
        const auto finished = ggml_time_us();
        learn_next_us = finished + (int64_t) (pending.compute_us * (1.0/LEARN_DUTY - 1.0));
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

