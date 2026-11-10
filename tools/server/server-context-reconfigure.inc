// Members of server_context_impl (included in its body): what a live reconfigure needs of the server
// (server-reconfigure.h). The logic is server-reconfigure.cpp's; this binds it to the server's context
// lifetimes (destroy_contexts / build_contexts) and its slots.
    // LIVE RECONFIGURE (server-reconfigure.h): the weights loaded since start (a reconfigure never adds
    // one), the parked slots' host state (their pages are reconfigure_state's), and the geometry as
    // /props reports it, republished by the main loop after every load and reconfigure
    int64_t n_weight_loads = 0;
    server_reconfigure_state reconfigure_state;
    std::map<int, server_slot> parked_slots;
    mutable std::mutex geometry_report_mutex;
    json geometry_report = json::object();
    std::atomic<int32_t> n_seq_live{0};

    bool kv_pool_unified() const {
        // measured from the context, not the flag: one pool when a sequence may hold all of it
        return llama_n_seq_max(ctx_tgt) <= 1 || llama_n_ctx_seq(ctx_tgt) >= llama_n_ctx(ctx_tgt);
    }

    void publish_geometry() {
        json parked = json::array();
        for (const auto & [id, p] : reconfigure_state.parked) {
            parked.push_back({ { "slot", id }, { "tokens_held", p.tokens_held }, { "busy", p.busy } });
        }
        json report = {
            { "n_ctx",          llama_n_ctx(ctx_tgt) },
            { "n_seq",          (int32_t) slots.size() },
            { "n_ctx_seq",      get_slot_n_ctx() },
            { "kv_unified",     kv_pool_unified() },
            { "n_weight_loads", n_weight_loads },
            { "n_applied",      reconfigure_state.n_applied },
            { "n_refused",      reconfigure_state.n_refused },
            { "parked",         parked },
            // the same keys /props' load-time context_geometry has, read from the live context
            { "context_geometry", {
                { "allocated_tokens",        llama_n_ctx(ctx_tgt) },
                { "max_tokens_per_sequence", llama_n_ctx_seq(ctx_tgt) },
                { "sequence_capacity",       llama_n_seq_max(ctx_tgt) },
            } },
            { "last",           reconfigure_state.last ? server_reconfigure_to_json(*reconfigure_state.last) : json(nullptr) },
        };
        n_seq_live.store((int32_t) slots.size());
        std::lock_guard<std::mutex> lock(geometry_report_mutex);
        geometry_report = std::move(report);
    }

    // a slot leaves service whole and a fresh, empty slot takes its id
    void park_slot(int id) {
        GGML_ASSERT(parked_slots.find(id) == parked_slots.end());
        parked_slots.emplace(id, std::move(slots[id]));
        slots[id] = server_slot();
        bind_slot(slots[id]);
        init_slot(slots[id], id);
    }

    // the parked slot takes its id back (the reconfigure pages its KV in)
    void unpark_slot(int id) {
        auto it = parked_slots.find(id);
        GGML_ASSERT(it != parked_slots.end());
        slots[id] = std::move(it->second);
        parked_slots.erase(it);
        bind_slot(slots[id]);
    }

    // a parked slot's work ends without its state ever returning: its client is told why. Its sampler
    // was attached to a context since freed, so it is dropped, never released (release resets it there).
    void drop_parked_slot(int id, const std::string & why) {
        auto it = parked_slots.find(id);
        if (it == parked_slots.end()) {
            return;
        }
        if (it->second.task && !why.empty()) {
            send_error(it->second, why, ERROR_TYPE_UNAVAILABLE);
        }
        parked_slots.erase(it);
        reconfigure_state.parked.erase(id);
    }

    void drop_parked(const std::string & why) {
        while (!parked_slots.empty()) {
            drop_parked_slot(parked_slots.begin()->first, why);
        }
        reconfigure_state.parked.clear();
    }

    // after a reconfigure placed the states: each live sampler on the new context, drafting begun again on
    // the rebuilt speculative state, and the queued work offered the free slots (a grow adds some)
    void resume_after_reconfigure() {
        for (auto & slot : slots) {
            if (slot.task && slot.smpl) {
                if (server_slot::samples_on_backend(slot.task->params)) {
                    // a backend chain is initialized once, for the context it was first attached to
                    // (llama_sampler_chain_backend_init): on the rebuilt context the slot samples with a
                    // fresh chain that carries the same state (common_sampler_copy, as a rollback does)
                    try {
                        common_params_sampling sampling = slot.task->params.sampling; // init takes them mutable; the task's are const
                        common_sampler_ptr fresh(common_sampler_init(model_tgt, sampling));
                        common_sampler_copy(slot.smpl.get(), fresh.get());
                        slot.smpl = std::move(fresh);
                    } catch (const std::exception & e) {
                        send_error(slot, std::string("the sampler could not be made again after a reconfigure: ") + e.what());
                        slot.release();
                        continue;
                    }
                }
                slot.attach_sampler(slot.task->params);
            }
            if (slot.state == SLOT_STATE_GENERATING && slot.can_speculate()) {
                common_speculative_begin(spec.get(), slot.id, slot.prompt.tokens.get_text_tokens());
            }
        }
        for (auto & slot : slots) {
            if (!slot.is_processing()) {
                queue_tasks.pop_deferred_task(slot.id);
            }
        }
        publish_busy_slots();
    }

    // what a live reconfigure needs of this server: its contexts' lifetime and its slots
    struct reconfigure_engine final : server_reconfigure_engine {
        server_context_impl & s;
        explicit reconfigure_engine(server_context_impl & owner) : s(owner) {}

        server_geometry live() const override {
            // the geometry as asked for (fit's choice when the launch named none), so asking again is a no-op
            const int32_t n_ctx = s.params_base.n_ctx > 0 ? s.params_base.n_ctx : (int32_t) llama_n_ctx(s.ctx_tgt);
            return { n_ctx, (int32_t) s.slots.size() };
        }
        int32_t n_ctx_seq()  const override { return s.get_slot_n_ctx(); }
        int32_t n_cells()    const override { return (int32_t) llama_n_ctx(s.ctx_tgt); }
        bool    kv_unified() const override { return s.kv_pool_unified(); }

        std::optional<server_reconfigure_refused> blocked() const override {
            if (auto learn = s.learning_blocks_reconfigure()) {
                return learn;
            }
            for (const auto & slot : s.slots) {
                if (slot.state == SLOT_STATE_WAIT_OTHER) {
                    return server_reconfigure_refused { SERVER_RECONFIGURE_FAN_OUT,
                        "a parent prompt is being copied into its child slots (n_cmpl > 1)", slot.id };
                }
            }
            return std::nullopt;
        }

        int32_t tokens_held(int32_t id) const override {
            llama_memory_t mem = llama_get_memory(s.ctx_tgt);
            if (mem == nullptr) {
                return 0; // a context with no memory module (an embedding model) holds no state between tasks
            }
            const llama_pos pos_max = llama_memory_seq_pos_max(mem, id);
            if (pos_max < 0) {
                return 0;
            }
            return std::max<int32_t>(s.slots[id].prompt.n_tokens(), pos_max + 1);
        }
        bool busy(int32_t id) const override { return s.slots[id].is_processing(); }

        bool page_out(int32_t id, server_slot_page & page) override { return s.slots[id].page_out(page); }
        bool page_in (int32_t id, const server_slot_page & page) override { return s.slots[id].page_in(page); }

        void park  (int32_t id) override { s.park_slot(id); }
        void unpark(int32_t id) override { s.unpark_slot(id); }

        void destroy_contexts() override { s.destroy_contexts(); }
        bool build_contexts(const server_geometry & g) override {
            s.params_base.n_ctx      = g.n_ctx;
            s.params_base.n_parallel = g.n_seq;
            // no warmup: the weights are already resident, and the warmup would only clear the new memory
            return s.build_contexts(false);
        }

        void resume() override { s.resume_after_reconfigure(); }
    };
