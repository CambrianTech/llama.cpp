// Members of server_context_impl. Core owns physical leases; this binds their
// immutable references to the existing context allocators and actual buffer lifetime.
    struct learning_residency {
        server_residency_mirror * mirror;
        std::string optimizer, workspace;
        std::vector<server_residency_row> requirements;
        bool restoring = false;

        bool allows(ggml_backend_buffer_type_t buft, const char * kind, size_t bytes) const {
            const auto device = ggml_backend_buft_get_device(buft);
            const bool shared = std::string(kind) == "shared_compute_required";
            const bool admitted = mirror->allows(shared ? workspace : optimizer,
                {device ? ggml_backend_dev_name(device) : "host", ggml_backend_buft_name(buft),
                 kind, bytes, ggml_backend_buft_is_host(buft)}, restoring);
            if (!admitted) {
                SRV_WRN("learning residency refused actual allocation: %s %s %s %zu bytes\n",
                    device ? ggml_backend_dev_name(device) : "host", ggml_backend_buft_name(buft), kind, bytes);
            }
            return admitted;
        }
        bool authorized() const {
            for (const auto & row : requirements) {
                if (!mirror->allows(row.kind == "shared_compute_required" ? workspace : optimizer, row)) { return false; }
            }
            return true;
        }
    };
    std::map<int32_t, std::unique_ptr<learning_residency>> learning_residencies;
    std::string learning_workspace_residency;

    std::string admit_learning_residency(const learn_request & request) {
        if (request.optimizer_residency.empty() || request.workspace_residency.empty()) {
            return "learning requires optimizer_residency and workspace_residency references";
        }
        const auto old = learning_residencies.find(request.lora_id);
        if (old != learning_residencies.end() && old->second->optimizer != request.optimizer_residency) {
            return "optimizer residency cannot change while its buffers are retained";
        }
        if (!learning_workspace_residency.empty() && learning_workspace_residency != request.workspace_residency) {
            return "workspace residency cannot change while the engine retains its scheduler";
        }
        const auto measured = learn_preflight({{"engine", learn_engine}, {"lora_id", request.lora_id}});
        if (measured.at("refusal") != "none") { return measured.value("detail", std::string("preflight refused")); }
        server_residency_need optimizer{request.optimizer_residency,
            residency_subject({{"loaded", {{"lora_id", request.lora_id},
                {"path", params_base.lora_adapters[request.lora_id].path}}}}), {}};
        server_residency_need workspace{request.workspace_residency, "workspace", {}};
        auto binding = std::make_unique<learning_residency>();
        binding->mirror = &residency_mirror;
        binding->optimizer = request.optimizer_residency;
        binding->workspace = request.workspace_residency;
        for (const auto & row : measured.at("requirements")) {
            const auto kind = row.at("kind").get<std::string>();
            if (kind == "shared_compute_held") { continue; }
            server_residency_row need{row.at("device"), row.at("buffer_type"), kind,
                row.at("bytes").get<uint64_t>(), row.at("host").get<bool>()};
            (kind == "shared_compute_required" ? workspace.rows : optimizer.rows).push_back(need);
            binding->requirements.push_back(need);
        }
        // First opt_init replaces the scheduler: held bytes are NOT retained credit.
        // Admit the complete required workspace before that transition, then recheck
        // every actual allocator plan. The workspace remains an engine-wide binding.
        std::vector<std::string> acquired;
        const auto refusal = residency_mirror.claim_plan({optimizer, workspace}, acquired);
        if (refusal != "none") { return "learning residency plan refused: " + refusal; }
        if (old == learning_residencies.end()) {
            learning_residencies[request.lora_id] = std::move(binding);
        } else {
            *old->second = std::move(*binding); // retain the allocator callback's address
        }
        learning_workspace_residency = request.workspace_residency;
        return {};
    }

    void settle_learning_residency(int32_t id) {
        const auto it = learning_residencies.find(id);
        if (it == learning_residencies.end()) { return; }
        auto & binding = *it->second;
        const auto memory = learn_memory_receipt();
        bool optimizer_live = false, workspace_live = false;
        for (const auto & row : memory.at("optimizer_owned_buffers")) {
            optimizer_live |= row.at("lora_id") == id && row.at("bytes").get<uint64_t>() != 0;
        }
        for (const auto & row : memory.at("shared_compute_buffers")) {
            workspace_live |= row.at("bytes").get<uint64_t>() != 0;
        }
        residency_mirror.finish(binding.optimizer, optimizer_live, id);
        residency_mirror.finish(binding.workspace, workspace_live);
        binding.restoring = false;
    }
