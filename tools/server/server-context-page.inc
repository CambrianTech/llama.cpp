// Continuum fork: KV page switches (POST /slots/{id}?action=save|restore). The task loop only copies
// a sequence's state out of or into the context (bounded, proportional to its bytes); the page's
// files are written or read on page_io (server-page-io.h) while every other slot keeps decoding,
// and the IO's completion is a SERVER_TASK_TYPE_SLOT_PAGE_IO_DONE task back on the loop, which alone
// touches the slot. The HTTP contract and the files are unchanged; the receipt adds copy_ms (the
// loop's share) and io_ms (the worker's).

    // The checkpoints a saved page carries: the oldest (the head of the first user turn) and the
    // newest `keep_newest_checkpoints`, in list order, or all of them when the caller asked for no
    // subset. With `keep_view_checkpoints`, also the newest at or before the last prompt's view end
    // (where its next prompt diverges) and those made for that prompt's end (a next prompt that
    // resends it whole); a slot that has run no prompt since a restore keeps them all. The quote and
    // the write share this list.
    std::vector<const common_prompt_checkpoint *> page_kept_checkpoints(const server_slot & slot, const server_task & task) const {
        std::vector<const common_prompt_checkpoint *> kept_checkpoints;
        const auto & all  = slot.prompt.checkpoints;
        const int32_t keep = task.slot_action.keep_newest_checkpoints;
        const bool by_view = task.slot_action.keep_view_checkpoints;
        const int32_t view_end = slot.prompt.view_end;
        const int32_t end_from = slot.prompt.end_from;
        const bool keep_all = by_view ? end_from < 0 : keep < 0;
        const common_prompt_checkpoint * at_view_end = nullptr;
        for (const auto & c : all) {
            if (by_view && view_end >= 0 && c.n_tokens <= view_end) {
                at_view_end = &c;
            }
        }
        const size_t n = all.size();
        size_t i = 0;
        std::string kept_at;
        for (const auto & c : all) {
            const bool newest = keep > 0 && i + (size_t) keep >= n;
            const bool viewed = by_view && (&c == at_view_end || c.n_tokens >= end_from);
            if (keep_all || i == 0 || newest || viewed) {
                kept_checkpoints.push_back(&c);
                kept_at += (kept_at.empty() ? "" : ", ") + std::to_string(c.n_tokens);
            }
            ++i;
        }
        if (by_view && !task.slot_action.size_only) {
            SLT_INF(slot, "page keeps %zu of %zu checkpoints at n_tokens = [%s] (view end = %d, prompt end from = %d)\n",
                    kept_checkpoints.size(), n, kept_at.c_str(), view_end, end_from);
        }
        return kept_checkpoints;
    }

    // The `.ckpt` sidecar: the kept checkpoints, each with its target, draft and speculative state.
    template <typename Put>
    static void page_write_checkpoints(const std::vector<const common_prompt_checkpoint *> & kept, Put && put) {
        const uint32_t magic   = 0x434B5054u; // 'CKPT'
        const uint32_t version = 1;
        const uint32_t n_ckpt  = (uint32_t) kept.size();
        put(&magic, sizeof(magic)); put(&version, sizeof(version)); put(&n_ckpt, sizeof(n_ckpt));
        for (const auto * k : kept) {
            const auto & c = *k;
            const int64_t  n_tokens = c.n_tokens;
            const int32_t  pos_min  = c.pos_min;
            const int32_t  pos_max  = c.pos_max;
            const uint64_t sz_tgt   = c.data_tgt.size();
            const uint64_t sz_dft   = c.data_dft.size();
            const uint64_t sz_spec  = c.data_spec.size();
            put(&n_tokens, sizeof(n_tokens)); put(&pos_min, sizeof(pos_min)); put(&pos_max, sizeof(pos_max));
            put(&sz_tgt, sizeof(sz_tgt)); put(&sz_dft, sizeof(sz_dft)); put(&sz_spec, sizeof(sz_spec));
            put(c.data_tgt.data(), c.data_tgt.size());
            put(c.data_dft.data(), c.data_dft.size());
            put(c.data_spec.data(), c.data_spec.size());
        }
    }

    // The `.lora` sidecar: the adapters the state was computed under, each with its file revision.
    template <typename Put>
    static void page_write_adapters(const std::vector<common_adapter_lora_info> & lora, Put && put) {
        const uint32_t magic   = 0x4C4F5241u; // 'LORA'
        const uint32_t version = 2;
        const uint32_t n_lora  = (uint32_t) lora.size();
        put(&magic, sizeof(magic)); put(&version, sizeof(version)); put(&n_lora, sizeof(n_lora));
        for (const auto & l : lora) {
            const uint32_t n_path = (uint32_t) l.path.size();
            const auto     rev    = lora_file_revision(l.path);
            put(&n_path, sizeof(n_path));
            put(l.path.data(), n_path);
            put(&l.scale, sizeof(l.scale));
            put(&rev.first, sizeof(rev.first)); put(&rev.second, sizeof(rev.second));
        }
    }

    // NOTE (Continuum fork): SLOT_SAVE does NOT defer on a live slot. Reaching this arm already
    // guarantees is_yielding == false (process_single_task declines every task but METRICS/SLOT_GET
    // while a decode is yielding), so no llama_decode is in flight and the sequence's KV is at a
    // consistent inter-batch boundary; is_processing() means only "this slot has an assigned turn".
    // A save is READ-ONLY against that KV, so snapshotting a mid-turn slot is safe, and it is the
    // warm restore we want (resume mid-thought). Deferring bounced the save for the whole turn and a
    // ~30k-token turn tripped the client's 10 s timeout. The snapshot is taken here, on the loop; the
    // files are written on a worker.
    void page_save(server_task && task) {
        auto job = std::make_shared<server_page_job>();
        if (task.slot_action.page_reserved > 0) {
            // admitted after it waited: it holds its bytes, returned on any refusal below
            job->reservation.hold(&page_io_bytes, task.slot_action.page_reserved);
        }
        const int id_slot = task.slot_action.id_slot;
        server_slot * slot = get_slot_by_id(id_slot);
        if (slot == nullptr) {
            send_error(task, "Invalid slot ID", ERROR_TYPE_INVALID_REQUEST);
            return;
        }

        std::vector<char> packed;
        try {
            packed = slot->prompt.tokens.serialize();
        } catch (const std::exception & err) {
            send_error(task, err.what(), ERROR_TYPE_NOT_SUPPORTED);
            return;
        }
        GGML_ASSERT(packed.size() % sizeof(llama_token) == 0);
        const auto kept = page_kept_checkpoints(*slot, task);

        if (task.slot_action.size_only || task.slot_action.has_byte_ceiling) {
            size_t projected_bytes = llama_state_seq_file_size(
                ctx_tgt, slot->id, reinterpret_cast<const llama_token *>(packed.data()),
                packed.size() / sizeof(llama_token));
            if (projected_bytes == 0) {
                send_error(task, "Unable to size slot state", ERROR_TYPE_SERVER);
                return;
            }
            bool size_overflow = false;
            auto count = [&](const void *, size_t n) {
                if (n > std::numeric_limits<size_t>::max() - projected_bytes) {
                    size_overflow = true;
                } else {
                    projected_bytes += n;
                }
            };
            page_write_checkpoints(kept, count);
            page_write_adapters(slot->lora, count);
            if (size_overflow) {
                send_error(task, "Slot file size overflow", ERROR_TYPE_SERVER);
                return;
            }
            if (task.slot_action.size_only) {
                auto res = std::make_unique<server_task_result_slot_save_load>();
                res->id = task.id;
                res->id_slot = id_slot;
                res->filename = task.slot_action.filename;
                res->is_save = true;
                res->size_only = true;
                res->n_bytes = projected_bytes;
                queue_results.send(std::move(res));
                return;
            }
            // Recheck at this task boundary: decode may have advanced after the quote.
            if (task.slot_action.has_byte_ceiling && projected_bytes > task.slot_action.byte_ceiling) {
                send_error(task, "Slot state exceeds admitted file bytes", ERROR_TYPE_INVALID_REQUEST);
                return;
            }
        }

        // The bytes this switch will hold, admitted against the page IO budget before any is allocated.
        const size_t n_body = llama_state_seq_file_body_size(ctx_tgt, slot->id);
        if (n_body == 0) {
            send_error(task, "Unable to save slot", ERROR_TYPE_SERVER);
            return;
        }
        size_t n_ckpt = 0, n_lora = 0;
        page_write_checkpoints(kept, [&](const void *, size_t n) { n_ckpt += n; });
        page_write_adapters(slot->lora, [&](const void *, size_t n) { n_lora += n; });
        const uint64_t bytes = (uint64_t) n_body + packed.size() + n_ckpt + n_lora;
        if (task.slot_action.page_reserved == 0) {
            task.slot_action.page_asked_us = ggml_time_us();
            auto parked = std::make_shared<server_task>(std::move(task));
            parked->slot_action.page_reserved = bytes;
            // admitted later: the release that made room posts it back to this loop, holding its bytes
            if (!page_io_bytes.admit_or_park(bytes, [this, parked]() { queue_tasks.post(std::move(*parked)); })) {
                SRV_DBG("slot %d: save of %" PRIu64 " bytes waits for the page IO budget, id_task = %d\n", id_slot, bytes, parked->id);
                return;
            }
            task = std::move(*parked);
            job->reservation.hold(&page_io_bytes, bytes);
        } else {
            job->reservation.resize(bytes); // decode may have grown the slot since it was admitted
        }
        job->bytes    = bytes;
        job->admit_ms = (ggml_time_us() - task.slot_action.page_asked_us) / 1000.0;

        // The loop's share: the slot's state, tokens and sidecars copied into the switch. Each buffer
        // is sized once, so the copy is one pass over the bytes.
        const int64_t t_copy = ggml_time_us();
        job->is_save  = true;
        job->id_task  = task.id;
        job->id_slot  = id_slot;
        job->filename = task.slot_action.filename;
        job->filepath = task.slot_action.filepath;
        job->n_tokens = slot->prompt.tokens.size();
        job->body.resize(n_body);
        if (llama_state_seq_get_file_body(ctx_tgt, slot->id, job->body.data(), n_body) != n_body) {
            send_error(task, "Unable to save slot", ERROR_TYPE_SERVER);
            return;
        }
        job->tokens.resize(packed.size() / sizeof(llama_token));
        std::memcpy(job->tokens.data(), packed.data(), packed.size());
        job->ckpt_sidecar.reserve(n_ckpt);
        page_write_checkpoints(kept, [&](const void * p, size_t n) { server_page_put(job->ckpt_sidecar, p, n); });
        page_write_adapters(slot->lora, [&](const void * p, size_t n) { server_page_put(job->lora_sidecar, p, n); });
        job->copy_ms = (ggml_time_us() - t_copy) / 1000.0;

        page_io_submit(task.id, std::move(job), page_write);
    }

    // The worker's share of a save: the page and its sidecars on disk.
    static void page_write(server_page_job & job) {
        const server_page_file file { LLAMA_STATE_SEQ_MAGIC, LLAMA_STATE_SEQ_VERSION };
        job.n_bytes = file.write(job.filepath, job.tokens.data(), job.tokens.size(), job.body, job.error);
        if (job.n_bytes == 0) {
            return;
        }
        if (!server_page_write_bytes(job.filepath + ".ckpt", job.ckpt_sidecar)) {
            SRV_WRN("slot %d: checkpoint sidecar could not be written: %s (the restore will be cold)\n", job.id_slot, job.filepath.c_str());
        }
        if (!server_page_write_bytes(job.filepath + ".lora", job.lora_sidecar)) {
            SRV_WRN("slot %d: adapter sidecar could not be written: %s (this page will not restore)\n", job.id_slot, job.filepath.c_str());
        }
        // on disk now: the copy is freed here, and its bytes returned to the budget, not when the loop
        // takes the completion
        std::vector<uint8_t>().swap(job.body);
        std::vector<uint8_t>().swap(job.ckpt_sidecar);
        job.reservation.release();
    }

    // A restore reads its page on a worker at once, even while its slot finishes another turn (the
    // read overlaps that turn's decode); it is applied on the loop when the read completes and the
    // slot is free (the completion defers while the slot is_processing(): it mutates KV a live
    // decode reads, and pop_deferred_task(id_slot) services it the instant that slot releases).
    void page_restore(server_task && task) {
        const int id_slot = task.slot_action.id_slot;
        server_slot * slot = get_slot_by_id(id_slot);
        if (slot == nullptr) {
            send_error(task, "Invalid slot ID", ERROR_TYPE_INVALID_REQUEST);
            return;
        }
        auto progress = task.slot_action.progress;
        if (progress) {
            auto expected = server_restore_progress::queued;
            if (!progress->phase.compare_exchange_strong(expected, server_restore_progress::state)) {
                send_error(task, "Restore operation cancelled", ERROR_TYPE_INVALID_REQUEST);
                return;
            }
        }
        auto job = std::make_shared<server_page_job>();
        job->is_save   = false;
        job->id_task   = task.id;
        job->id_slot   = id_slot;
        job->filename  = task.slot_action.filename;
        job->filepath  = task.slot_action.filepath;
        job->occupancy = slot->n_occupancy;
        job->progress  = std::move(progress);
        // the bytes it will hold: the page and its sidecars as they are on disk (a missing file holds
        // none and fails its read)
        for (const char * suffix : { "", ".ckpt", ".lora" }) {
            std::error_code ec;
            const auto n = std::filesystem::file_size(job->filepath + suffix, ec);
            job->bytes += ec ? 0 : (uint64_t) n;
        }
        // a later task or erase on this slot drops it (page_slot_reassigned)
        auto & pending = pending_restores[id_slot];
        pending.erase(std::remove_if(pending.begin(), pending.end(), [](const auto & w) { return w.expired(); }), pending.end());
        pending.push_back(job);

        const int64_t asked = ggml_time_us();
        auto start = [this, job, asked]() {
            job->reservation.hold(&page_io_bytes, job->bytes);
            job->admit_ms = (ggml_time_us() - asked) / 1000.0;
            page_io_submit(job->id_task, job, page_read);
        };
        if (page_io_bytes.admit_or_park(job->bytes, start)) {
            start();
        } else {
            SRV_DBG("slot %d: restore of %" PRIu64 " bytes waits for the page IO budget, id_task = %d\n", id_slot, job->bytes, task.id);
        }
    }

    // A slot was given another task or erased: count it, and drop every restore buffered for it.
    void page_slot_reassigned(server_slot & slot) {
        ++slot.n_occupancy;
        auto it = pending_restores.find(slot.id);
        if (it == pending_restores.end()) {
            return;
        }
        for (const auto & w : it->second) {
            if (auto job = w.lock()) {
                page_drop_restore(*job);
            }
        }
        pending_restores.erase(it);
    }

    // The restore's bytes return to the budget at once. A read in flight stops at its next part and
    // frees its own buffers; a read already done is freed here and answered now, so a completion
    // deferred behind the slot's new task never holds it.
    void page_drop_restore(server_page_job & job) {
        if (job.dropped.exchange(true)) {
            return;
        }
        job.reservation.release();
        if (!job.io_done.load()) {
            return;
        }
        page_free_buffers(job);
        if (job.progress) {
            job.progress->phase.store(server_restore_progress::failed, std::memory_order_relaxed);
        }
        send_page_refusal(job.id_task, page_reassigned_message, "slot_reassigned");
        job.answered = true;
    }

    static constexpr const char * page_reassigned_message =
        "Unable to restore slot: it was given another task or erased while its page was read; "
        "its occupant is kept, the page is not applied";

    static void page_free_buffers(server_page_job & job) {
        std::vector<uint8_t>().swap(job.body);
        std::vector<int32_t>().swap(job.tokens);
        job.checkpoints.clear();
    }

    // The worker's share of a restore: the adapter identity, the page and its checkpoints read. The
    // identity is read first so a page whose identity is unknown is refused without reading the rest.
    static void page_read(server_page_job & job) {
        auto * progress = job.progress.get();
        {
            const std::string lora_path = job.filepath + ".lora";
            server_page_reader in(lora_path);
            uint32_t magic = 0, version = 0, n_lora = 0;
            in.get(&magic, sizeof(magic)); in.get(&version, sizeof(version)); in.get(&n_lora, sizeof(n_lora));
            bool ok = in.ok && magic == 0x4C4F5241u && version == 2 && n_lora <= 4096;
            for (uint32_t i = 0; ok && i < n_lora; ++i) {
                uint32_t n_path = 0;
                in.get(&n_path, sizeof(n_path));
                ok = in.ok && n_path <= 4096;
                if (!ok) {
                    break;
                }
                server_page_adapter a { std::string(n_path, '\0'), 0.0f, 0, 0 };
                in.get(a.path.data(), n_path);
                in.get(&a.scale, sizeof(a.scale));
                in.get(&a.size, sizeof(a.size)); in.get(&a.mtime, sizeof(a.mtime));
                ok = in.ok;
                job.adapters.push_back(std::move(a));
            }
            if (!ok) {
                job.error = "the page's adapter identity is missing or unreadable: " + lora_path;
                return;
            }
        }

        const server_page_file file { LLAMA_STATE_SEQ_MAGIC, LLAMA_STATE_SEQ_VERSION };
        auto page = file.read(job.filepath, [&job, progress](uint64_t done, uint64_t total) {
            return !job.dropped.load(std::memory_order_relaxed)
                && (progress == nullptr || server_restore_progress::on_state_io(done, total, progress));
        }, job.error);
        if (!page) {
            return;
        }
        job.tokens  = std::move(page->tokens);
        job.body    = std::move(page->body);
        job.n_bytes = page->n_bytes;

        // The checkpoints come back with the state. Any defect in the sidecar leaves the slot as a
        // pre-fork restore left it: valid tokens + state, no checkpoints.
        job.ckpt_path = job.filepath + ".ckpt";
        server_page_reader in(job.ckpt_path);
        if (progress) {
            progress->phase.store(server_restore_progress::sidecar, std::memory_order_relaxed);
            if (in.ok) {
                progress->sidecar_total.store(in.size, std::memory_order_relaxed);
            }
        }
        bool cancelled = false;
        auto get = [&](void * ptr, size_t n) {
            if (job.dropped.load(std::memory_order_relaxed)
                    || (progress && progress->cancel_requested.load(std::memory_order_relaxed))) {
                cancelled = true;
                in.ok = false;
            }
            in.get(ptr, n);
            if (progress && in.ok) {
                progress->sidecar_bytes.fetch_add(n, std::memory_order_relaxed);
            }
        };
        uint32_t magic = 0, version = 0, n_ckpt = 0;
        get(&magic, sizeof(magic)); get(&version, sizeof(version)); get(&n_ckpt, sizeof(n_ckpt));
        bool ok = in.ok && magic == 0x434B5054u && version == 1 && n_ckpt <= 4096;
        for (uint32_t i = 0; ok && i < n_ckpt; ++i) {
            int64_t n_tokens = 0; int32_t pos_min = 0, pos_max = 0;
            uint64_t sz_tgt = 0, sz_dft = 0, sz_spec = 0;
            get(&n_tokens, sizeof(n_tokens)); get(&pos_min, sizeof(pos_min)); get(&pos_max, sizeof(pos_max));
            get(&sz_tgt, sizeof(sz_tgt)); get(&sz_dft, sizeof(sz_dft)); get(&sz_spec, sizeof(sz_spec));
            // a checkpoint is at most one sequence state; refuse absurd sizes
            ok = in.ok && sz_tgt < (1ull << 36) && sz_dft < (1ull << 36) && sz_spec < (1ull << 30);
            if (!ok) {
                break;
            }
            auto & c = job.checkpoints.emplace_back();
            c.update_pos(n_tokens, pos_min, pos_max);
            c.id_task = -1;
            c.data_tgt.resize(sz_tgt);   get(c.data_tgt.data(),  sz_tgt);
            c.data_dft.resize(sz_dft);   get(c.data_dft.data(),  sz_dft);
            c.data_spec.resize(sz_spec); get(c.data_spec.data(), sz_spec);
            ok = in.ok;
            if (progress && ok) {
                progress->checkpoints.fetch_add(1, std::memory_order_relaxed);
            }
        }
        if (cancelled) {
            job.error = "Restore operation cancelled";
            return;
        }
        job.checkpoints_ok = ok;
        if (!ok) {
            job.checkpoints.clear();
        }
    }

    // Hands a switch's file IO to page_io; its completion comes back to the loop as a task with the
    // switch's own id, so its HTTP waiter receives the receipt.
    void page_io_submit(int id_task, std::shared_ptr<server_page_job> job, void (*io)(server_page_job &)) {
        std::string gate;
        if constexpr (server_test_faults::enabled) {
            gate = test_faults.page_io_gate;
        }
        const std::string file = job->filepath;
        page_io->submit(file, [this, job = std::move(job), id_task, io, gate]() {
            if constexpr (server_test_faults::enabled) {
                server_test_faults::hold(gate);
            }
            const int64_t t_io = ggml_time_us();
            try {
                if (!job->dropped.load()) {
                    io(*job);
                }
            } catch (const std::exception & err) {
                job->error = err.what();
            }
            job->io_ms = (ggml_time_us() - t_io) / 1000.0;
            if (job->dropped.load()) {
                page_free_buffers(*job); // dropped while it read: nothing it read is applied
                job->reservation.release();
            }
            job->io_done.store(true);
            server_task done(SERVER_TASK_TYPE_SLOT_PAGE_IO_DONE);
            done.id                  = id_task;
            done.id_slot             = job->id_slot; // a deferred restore is serviced when ITS slot releases
            done.slot_action.id_slot = job->id_slot;
            done.slot_action.page    = job;
            queue_tasks.post(std::move(done));
        });
    }

    // A refusal the caller matches on by its typed `refusal`, never by its message.
    void send_page_refusal(int id_task, const std::string & message, const char * refusal) {
        SRV_WRN("task id = %d, refused (%s): %s\n", id_task, refusal, message.c_str());
        auto res = std::make_unique<server_task_result_error>();
        res->id       = id_task;
        res->err_type = ERROR_TYPE_INVALID_REQUEST;
        res->err_msg  = message;
        res->refusal  = refusal;
        queue_results.send(std::move(res));
    }

    void page_io_done(server_task && task) {
        const auto job = task.slot_action.page;
        if (job->is_save) {
            if (!job->error.empty()) {
                send_error(task, "Unable to save slot: " + job->error, ERROR_TYPE_SERVER);
                return;
            }
            auto res = std::make_unique<server_task_result_slot_save_load>();
            res->id       = task.id;
            res->id_slot  = job->id_slot;
            res->filename = job->filename;
            res->is_save  = true;
            res->n_tokens = job->n_tokens;
            res->n_bytes  = job->n_bytes;
            res->copy_ms  = job->copy_ms;
            res->io_ms    = job->io_ms;
            res->t_ms     = job->copy_ms + job->io_ms;
            page_io_receipt(*res, *job);
            queue_results.send(std::move(res));
            return;
        }
        if (job->answered) {
            return; // dropped after its read and answered then (page_drop_restore)
        }
        page_apply_restore(std::move(task), *job);
    }

    // The loop's share of a restore: the page read, checked, set into its slot.
    void page_apply_restore(server_task && task, server_page_job & job) {
        auto * progress = job.progress.get();
        auto fail = [&](const std::string & why) {
            if (progress) {
                progress->phase.store(progress->cancel_requested.load(std::memory_order_relaxed)
                    ? server_restore_progress::cancelled : server_restore_progress::failed, std::memory_order_relaxed);
            }
            send_error(task, "Unable to restore slot: " + why, ERROR_TYPE_INVALID_REQUEST);
        };
        server_slot * slot = get_slot_by_id(job.id_slot);
        if (slot == nullptr) {
            fail("Invalid slot ID");
            return;
        }
        if (!job.error.empty()) {
            fail(job.error); // the read failed: the slot was never touched
            return;
        }
        if (slot->n_occupancy != job.occupancy || job.dropped.load()) {
            if (progress) {
                progress->phase.store(server_restore_progress::failed, std::memory_order_relaxed);
            }
            send_page_refusal(task.id, page_reassigned_message, "slot_reassigned");
            return;
        }
        if (slot->is_processing()) {
            SRV_DBG("page read for slot %d, which is processing: defer its apply, id_task = %d\n", slot->id, task.id);
            queue_tasks.defer(std::move(task));
            return;
        }
        if (progress && progress->cancel_requested.load(std::memory_order_relaxed)) {
            fail("Restore operation cancelled");
            return;
        }

        // The adapters the state was computed under, against those loaded now: a page whose identity
        // is foreign never touches the KV (its identity is its ACTIVE adapters, server-page-adapters.h).
        std::vector<common_adapter_lora_info> page_lora = params_base.lora_adapters; // copy: ptr, task_name, prompt_prefix
        {
            std::vector<server_loaded_adapter> loaded;
            loaded.reserve(page_lora.size());
            for (const auto & l : page_lora) {
                const auto rev = lora_file_revision(l.path);
                loaded.push_back({ l.path, rev.first, rev.second });
            }
            std::string why;
            const auto scales = server_page_adapter_scales(job.adapters, loaded, why);
            if (!scales) {
                fail("the page's adapter identity does not match the loaded adapters (" + why + "): " + job.filepath + ".lora");
                return;
            }
            for (size_t i = 0; i < page_lora.size(); ++i) {
                page_lora[i].scale = (*scales)[i];
            }
        }

        const int64_t t_copy = ggml_time_us();
        // the tokens are checked before the KV is touched
        std::optional<server_tokens> restored;
        try {
            restored = server_tokens::deserialize(job.tokens, mctx != nullptr);
        } catch (const std::exception & err) {
            fail(err.what());
            return;
        }
        if (restored->size() > (size_t) slot->n_ctx) {
            fail("Restored prompt does not fit in the slot context");
            return;
        }
        if (!restored->validate(ctx_tgt)) {
            fail("Invalid tokens in slot save file");
            return;
        }
        if (llama_state_seq_set_file_body(ctx_tgt, slot->id, job.body.data(), job.body.size()) == 0) {
            slot->prompt_clear();
            fail("No available space in KV cache or invalid slot save file");
            return;
        }
        std::vector<uint8_t>().swap(job.body);

        slot->prompt.clear();
        slot->prompt.tokens = std::move(*restored);
        slot->lora = std::move(page_lora);
        if (job.checkpoints_ok) {
            slot->prompt.checkpoints = std::move(job.checkpoints);
            SLT_INF(*slot, "restored %zu context checkpoints from the sidecar: %s\n", slot->prompt.checkpoints.size(), job.ckpt_path.c_str());
        } else {
            slot->prompt.checkpoints.clear();
            SLT_WRN(*slot, "no usable checkpoint sidecar (%s): the restore is cold — the next divergence re-prefills from the head\n", job.ckpt_path.c_str());
        }
        job.copy_ms = (ggml_time_us() - t_copy) / 1000.0;
        job.reservation.release(); // its state is the slot's now: nothing of it is in flight

        if (progress) {
            progress->phase.store(server_restore_progress::completed, std::memory_order_relaxed);
        }
        auto res = std::make_unique<server_task_result_slot_save_load>();
        res->id       = task.id;
        res->id_slot  = job.id_slot;
        res->filename = job.filename;
        res->is_save  = false;
        res->n_tokens = slot->prompt.tokens.size();
        res->n_bytes  = job.n_bytes;
        res->copy_ms  = job.copy_ms;
        res->io_ms    = job.io_ms;
        res->t_ms     = job.copy_ms + job.io_ms;
        page_io_receipt(*res, job);
        queue_results.send(std::move(res));
    }

    // The page IO budget as this switch's receipt reports it.
    void page_io_receipt(server_task_result_slot_save_load & res, const server_page_job & job) const {
        const auto budget = page_io_bytes.snapshot();
        res.admit_ms           = job.admit_ms;
        res.io_bytes           = job.bytes;
        res.io_bytes_in_flight = budget.in_flight;
        res.io_budget_bytes    = budget.budget;
    }
