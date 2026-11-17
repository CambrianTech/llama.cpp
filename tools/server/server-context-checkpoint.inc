// Methods of server_context_impl: Prompt checkpoints: when, how big, room for them, creation.
// Included inside the struct by server-context.cpp; a pure move out of it (the 1,000-line rule).

    // Continuum fork: the minimum spacing between this sequence's checkpoints. A fixed
    // `--checkpoint-min-step` is used as given; `auto` spaces them so `--ctx-checkpoints` of them
    // cover this sequence's own prompt, never a slot window: under `--kv-unified` a slot's window
    // is the whole pool, and pool / 32 (8,192 on a 262,144 pool) skipped every divergence point
    // of a ~5k-token turn (continuum, measured cache_n 0 on every turn, 2026-09-13).
    int64_t checkpoint_step(const server_slot & slot) const {
        if (params_base.checkpoint_min_step >= 0) {
            return params_base.checkpoint_min_step;
        }
        if (params_base.n_ctx_checkpoints <= 0) {
            return 0;
        }
        return (slot.task->n_tokens() + params_base.n_ctx_checkpoints - 1) / params_base.n_ctx_checkpoints;
    }

    enum class checkpoint_kind { prompt, speculative };

    // which parts of a sequence's state a checkpoint copies
    struct checkpoint_parts {
        bool tgt;
        bool dft;
        bool spec; // the speculative implementation's own state
    };

    // Continuum fork: the one place a checkpoint's state copy is allocated. A copy that cannot be
    // allocated is refused, not a slot failure: the checkpoint is cleared, false is returned and
    // the caller goes on without it (a restore from a prompt checkpoint is cold; a draft that a
    // speculative checkpoint would protect is dropped, and the step decodes without it).
    bool copy_checkpoint(const server_slot & slot, common_prompt_checkpoint & ckpt, checkpoint_kind kind, checkpoint_parts parts) {
        const bool is_prompt = kind == checkpoint_kind::prompt;
        try {
            if constexpr (server_test_faults::enabled) {
                if (server_test_faults::take(is_prompt ? test_faults.prompt_checkpoints : test_faults.spec_checkpoints)) {
                    throw std::bad_alloc(); // where the copies below allocate
                }
            }
            if (parts.tgt) {
                ckpt.update_tgt(ctx_tgt, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            }
            // Continuum fork (#79): a draft whose memory rolls back by trimming (plain attention KV, the MTP
            // head) has nothing a checkpoint must hold: on restore, slot.mem->remove trims its live KV to the
            // checkpoint. PARTIAL_ONLY returned that whole KV (~2 KB/token, 246 MB at 118k tokens) in EVERY
            // checkpoint, 79% of each, up to n_ctx_checkpoints copies per slot in host RAM. A saved page
            // carries the draft KV once instead (the `.dft` sidecar, server-context-page.inc).
            if (parts.dft && ctx_dft_seq_rm_type != COMMON_CONTEXT_SEQ_RM_TYPE_PART) {
                ckpt.update_dft(ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            }
            if (parts.spec) {
                common_speculative_get_state(spec.get(), slot.id, ckpt.data_spec);
            }
            return true;
        } catch (const std::bad_alloc &) {
            SLT_WRN(slot, "%s checkpoint refused: its state copy could not be allocated (n_tokens = %" PRId64 "), %s\n",
                    is_prompt ? "context" : "speculative", ckpt.n_tokens,
                    is_prompt ? "a restore from here will be cold" : "the draft is dropped");
            ckpt.clear();
            return false;
        }
    }

    // ---- the one host budget (CONTINUUM-ENGINE 4.7.6, step 1) ------------------------------------------
    // 0: no budget (--cache-ram -1 or 0). Checkpoints are charged first (they serve the minds in the slots);
    // the prompt cache holds what is left and evicts down to it.
    size_t host_budget = 0;

    // every host byte the engine holds beside the prompt cache: the lanes' checkpoints, the parked minds'
    // checkpoints, and a flashed mind's RAM copy until her file has landed (server-context-minds.inc)
    // `in_flight`: also a flashed mind's RAM copy (freed within ms of its write landing). The make-room decision
    // counts only what stays resident, so sending one mind down is seen to free her checkpoints.
    size_t checkpoint_bytes(bool in_flight = true) const {
        size_t b = 0;
        for (const auto & s : slots) {
            for (const auto & c : s.prompt.checkpoints) {
                b += c.size();
            }
        }
        for (const auto & m : minds) {
            for (const auto & c : m.prompt.checkpoints) {
                b += c.size();
            }
            if (in_flight && m.ram) {
                b += m.ram->size();
            }
        }
        return b;
    }

    // The prompt cache's share: the budget less every checkpoint (1 byte, not 0, when none is left: the
    // cache reads 0 as "no limit").
    void fit_prompt_cache_to_budget() {
        if (!prompt_cache || host_budget == 0) {
            return;
        }
        const size_t ck = checkpoint_bytes();
        prompt_cache->limit_size = host_budget > ck ? host_budget - ck : 1;
        prompt_cache->update();
    }

    // Room for a checkpoint of `bytes` under the budget: the slot holding the most checkpoint bytes gives up its
    // least valuable one (checkpoint_to_evict keeps her head while anything else is left), until it fits. False:
    // it cannot fit even with every checkpoint gone (larger than the budget); the caller makes none.
    bool make_room_for_checkpoint(size_t bytes) {
        if (host_budget == 0) {
            return true;
        }
        if (bytes > host_budget) {
            return false;
        }
        // the coldest parked mind goes down to flash first (her checkpoints with her): the minds acting keep theirs
        while (checkpoint_bytes(false) + bytes > host_budget && minds_on() && flash_coldest()) {
        }
        size_t held = checkpoint_bytes(false); // what stays resident: copies on their way to flash are leaving
        while (held + bytes > host_budget) {
            server_slot * from = nullptr;
            size_t        most = 0;
            for (auto & s : slots) {
                size_t b = 0;
                for (const auto & c : s.prompt.checkpoints) {
                    b += c.size();
                }
                if (b > most) {
                    most = b;
                    from = &s;
                }
            }
            if (from == nullptr) {
                return false;
            }
            // her newest checkpoint is where her next act resumes (her context only appends): it goes last.
            // Otherwise the store's own order (her head kept while anything else is left), never a pinned one.
            auto & cks = from->prompt.checkpoints;
            auto newest = cks.end();
            for (auto it = cks.begin(); it != cks.end(); ++it) {
                if (newest == cks.end() || it->n_tokens > newest->n_tokens) {
                    newest = it;
                }
            }
            auto victim = cks.end();
            for (auto it = cks.begin(); it != cks.end(); ++it) {
                if (it != newest && it->pin == 0) {
                    victim = it;
                    break;
                }
            }
            if (victim == cks.end() && newest != cks.end() && newest->pin == 0) {
                victim = newest; // all that is left
            }
            if (victim == cks.end()) {
                return false; // every checkpoint left is pinned for learning
            }
            SLT_WRN(*from, "host budget: erasing a checkpoint (n_tokens = %" PRId64 ", %.3f MiB) for a new one of %.3f MiB\n",
                    victim->n_tokens, (float) victim->size() / 1024 / 1024, (float) bytes / 1024 / 1024);
            held -= victim->size();
            from->prompt.checkpoints.erase(victim);
        }
        return true;
    }

    // n_tokens_cur: the number of tokens added to the batch for the current slot
    void create_checkpoint(server_slot & slot, const int64_t n_tokens_cur, llama_pos pos_min, llama_pos pos_max) {
        const int id_task = slot.task->id;

        // evict checkpoints within min-step of a previous checkpoint, unless they were
        // created by the current task
        const int64_t step = checkpoint_step(slot);
        int64_t last = -1;
        for (auto it = slot.prompt.checkpoints.begin(); it != slot.prompt.checkpoints.end(); ) {
            // a checkpoint pinned for learning stays until learned or released, however close (/learn/release)
            if (it->id_task != id_task && it->pin == 0 && last >= 0 && it->n_tokens <= last + step) {
                SLT_TRC(slot, "erasing context checkpoint too close to an earlier one (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", size = %.3f MiB)\n",
                        it->pos_min, it->pos_max, it->n_tokens, (float) it->size() / 1024 / 1024);

                it = slot.prompt.checkpoints.erase(it);
                continue;
            }

            last = it->n_tokens;
            ++it;
        }

        while (slot.prompt.checkpoints.size() >= (size_t) params_base.n_ctx_checkpoints) {
            // make room for the new checkpoint, if needed: her head stays while there is anything else
            // to evict (server-checkpoint-policy.h)
            auto victim = checkpoint_to_evict(slot.prompt.checkpoints, slot.prompt.view_end);
            if (victim == slot.prompt.checkpoints.end()) {
                break; // every checkpoint is pinned for learning: the store holds one more until they are learned
            }

            SLT_WRN(slot, "erasing old context checkpoint (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", size = %.3f MiB)\n",
                    victim->pos_min, victim->pos_max, victim->n_tokens, (float) victim->size() / 1024 / 1024);

            slot.prompt.checkpoints.erase(victim);
        }

        // under the one host budget: quote the copy before making it (its peak), and make room first
        {
            size_t quote = llama_state_seq_get_size_ext(ctx_tgt, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            if (ctx_dft && ctx_dft_seq_rm_type != COMMON_CONTEXT_SEQ_RM_TYPE_PART) {
                quote += llama_state_seq_get_size_ext(ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            }
            if (!make_room_for_checkpoint(quote)) {
                SLT_WRN(slot, "host budget: a checkpoint of %.3f MiB does not fit beside what stays resident (budget %.3f MiB); none is made\n",
                        (float) quote / 1024 / 1024, (float) host_budget / 1024 / 1024);
                return;
            }
        }

        // the copy is made aside and kept only once complete
        common_prompt_checkpoint cur;

        cur.id_task = id_task;

        // [TAG_CHECKPOINTS_FIX_POS_MIN]
        // TODO: here we incorrectly deterimne that the saved checkpoint data covers the [pos_min, pos_max] range
        //       this is not true for SWA models: https://github.com/ggml-org/llama.cpp/pull/24411#issuecomment-4677983225
        cur.update_pos(slot.prompt.n_tokens() - n_tokens_cur, pos_min, pos_max);

        // with the draft's speculative state stashed beside it
        if (!copy_checkpoint(slot, cur, checkpoint_kind::prompt, { /* tgt */ true, /* dft */ true, /* spec */ true })) {
            return;
        }

        auto & kept = slot.prompt.checkpoints.emplace_back(std::move(cur));
        if (slot.task->params.learn_pin && kept.n_tokens == (int64_t) slot.task->n_tokens() - 1) {
            kept.pin = ++learn_pin_clock; // the core will learn this turn: kept until learned or released (/learn/release)
            slot.stats.mind_pin       = kept.pin;
            slot.stats.mind_pinned_at = kept.n_tokens;
        }

        // Continuum fork (CONTINUUM-ENGINE 4.7.3, placed checkpoints): her new view-end checkpoint
        // supersedes every checkpoint an earlier act left on this prefix except her head's (the oldest,
        // the restore point when her next prompt diverges before her view end). What stays per slot is
        // her head, her newest view end, and this act's end checkpoints.
        erase_superseded_checkpoints(slot.prompt.checkpoints, kept, slot.prompt.view_end, id_task,
            [&](const common_prompt_checkpoint & it) {
                SLT_TRC(slot, "erasing a checkpoint her view end supersedes (n_tokens = %" PRId64 ", size = %.3f MiB)\n",
                        it.n_tokens, (float) it.size() / 1024 / 1024);
            });

        fit_prompt_cache_to_budget(); // the cache yields what the checkpoints now hold

        SLT_TRC(slot,
                "created context checkpoint %d of %d (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", size = %.3f MiB)\n",
                (int) slot.prompt.checkpoints.size(), params_base.n_ctx_checkpoints, kept.pos_min,
                kept.pos_max, kept.n_tokens, (float) kept.size() / 1024 / 1024);
    }
