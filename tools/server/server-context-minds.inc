// Methods of server_context_impl: resident minds and the switch between them (server-mind.h).
//
// Every mind the core names has exactly one home:
//   lane     her sequence is a slot's (she decodes);
//   parked   her sequence is one of the pool's parked ids (unified memory: on an M5 the same physical memory the
//            GPU reads; nothing copied, her attention cells re-tagged and her recurrent state moved by id);
//   flash    her sequence was copied out of the pool (a memcpy inside unified memory), its cells freed at once,
//            and the bytes written to her page file by the page-IO worker, off the decode thread.
// A switch moves her between homes; it never serializes her into a cache, never drops her, never truncates her.
// The pool runs out of CELLS, not of minds: when a decode finds no room, the coldest parked mind goes to flash
// (was: the idle slot's mind was purged). A mind on flash comes back by mapping her file and inserting it.

    struct resident_mind {
        mind_page                             page;
        server_prompt                         prompt;  // her tokens, checkpoints and view end (host bookkeeping)
        std::vector<common_adapter_lora_info> lora;
        uint64_t                              last_use = 0;
        uint64_t                              generation = 0; // advances each time her state moves
        int32_t                               seq      = -1; // parked: the pool sequence id; flash: -1
        std::string                           file;          // flash: her page file
        // her bytes in RAM, held by this record (never by the worker alone) until her file has landed: while the
        // write is in flight, or after it failed, she lives here and is inserted from here, never lost
        std::shared_ptr<std::vector<uint8_t>> ram;
        std::shared_ptr<std::atomic<int>>     written;         // 0 pending, 1 landed, -1 failed
        std::vector<int64_t>                  released;        // pins released while she was on flash (positions)
    };

    // unpin her checkpoint at `at` in `cks`; true when one was pinned there
    static bool unpin_at(std::list<common_prompt_checkpoint> & cks, int64_t at) {
        for (auto & c : cks) {
            if (c.pinned && c.n_tokens == at) {
                c.pinned = false;
                return true;
            }
        }
        return false;
    }

    // An abandoned turn's learn pin, released wherever she is: on a lane, parked, or on flash (applied when her
    // checkpoints come back). The checkpoint stays as an ordinary one: the budget may now evict it.
    pin_release learn_release(mind_page page, int64_t at) {
        for (auto & s : slots) {
            if (s.page == page) {
                return unpin_at(s.prompt.checkpoints, at) ? pin_release::released : pin_release::not_pinned;
            }
        }
        resident_mind * m = find_mind(page);
        if (m == nullptr) {
            return pin_release::unknown_mind;
        }
        if (!m->file.empty() || m->ram) {
            m->released.push_back(at); // her checkpoints are in her flash bytes
            return pin_release::released_on_flash;
        }
        return unpin_at(m->prompt.checkpoints, at) ? pin_release::released : pin_release::not_pinned;
    }

    // her checkpoints as bytes beside her state: count, then per checkpoint its position fields, three sizes and
    // the three blobs (target, draft, speculative state)
    static constexpr size_t ckpt_header_bytes = sizeof(int64_t) + 3 * sizeof(int32_t) + 1 + 3 * sizeof(uint64_t);

    static void ckpt_write(const std::list<common_prompt_checkpoint> & cks, uint8_t * w) {
        auto put = [&](const void * p, size_t n) { std::memcpy(w, p, n); w += n; };
        const uint64_t count = cks.size();
        put(&count, sizeof(count));
        for (const auto & c : cks) {
            const int32_t  id = c.id_task, mn = c.pos_min, mx = c.pos_max;
            const uint8_t  pin = c.pinned ? 1 : 0;
            const uint64_t a = c.data_tgt.size(), b = c.data_dft.size(), d = c.data_spec.size();
            put(&c.n_tokens, sizeof(c.n_tokens)); put(&id, 4); put(&mn, 4); put(&mx, 4); put(&pin, 1);
            put(&a, 8); put(&b, 8); put(&d, 8);
            put(c.data_tgt.data(), a); put(c.data_dft.data(), b); put(c.data_spec.data(), d);
        }
    }

    static bool ckpt_read(const uint8_t * r, uint64_t n, std::list<common_prompt_checkpoint> & out) {
        const uint8_t * end = r + n;
        auto get = [&](void * p, size_t k) { if ((size_t) (end - r) < k) { return false; } std::memcpy(p, r, k); r += k; return true; };
        uint64_t count = 0;
        if (!get(&count, sizeof(count))) {
            return false;
        }
        for (uint64_t i = 0; i < count; ++i) {
            common_prompt_checkpoint c;
            int32_t id, mn, mx; uint8_t pin; uint64_t a, b, d;
            if (!get(&c.n_tokens, sizeof(c.n_tokens)) || !get(&id, 4) || !get(&mn, 4) || !get(&mx, 4) || !get(&pin, 1) ||
                !get(&a, 8) || !get(&b, 8) || !get(&d, 8) || (uint64_t) (end - r) < a + b + d) {
                return false;
            }
            c.id_task = id; c.pos_min = mn; c.pos_max = mx; c.pinned = pin != 0;
            c.data_tgt.assign(r, r + a);  r += a;
            c.data_dft.assign(r, r + b);  r += b;
            c.data_spec.assign(r, r + d); r += d;
            out.push_back(std::move(c));
        }
        return r == end;
    }

    std::vector<int32_t>       park_free;  // free parked sequence ids (sized once at load)
    std::vector<resident_mind> minds;      // parked and flashed minds (reserved once at load)
    uint64_t                   mind_clock = 0;

    // At load: the parked ids are what the context reports beyond the lanes (llama_n_seq_max), never assumed.
    bool minds_init() {
        const int32_t n_seq  = (int32_t) llama_n_seq_max(ctx_tgt);
        const int32_t n_lane = params_base.n_parallel;
        if (params_base.n_minds <= 0) {
            return true;
        }
        if (!params_base.kv_unified) {
            SRV_ERR("%s", "--minds needs --kv-unified: a parked mind shares the pool's cells by sequence id\n");
            return false;
        }
        if (params_base.slot_save_path.empty()) {
            SRV_ERR("%s", "--minds needs --slot-save-path: a mind the pool cannot hold goes to flash, never dropped\n");
            return false;
        }
        for (int32_t s = n_seq - 1; s >= n_lane; --s) {
            park_free.push_back(s);
        }
        minds.reserve((size_t) params_base.n_minds * 2);
        SRV_INF("resident minds: %d lanes, %zu parked sequence ids (context reports %d), flash at %s\n",
                n_lane, park_free.size(), n_seq, params_base.slot_save_path.c_str());
        return true;
    }

    bool minds_on() const { return !park_free.empty() || !minds.empty(); }

    resident_mind * find_mind(mind_page page) {
        for (auto & m : minds) {
            if (m.page == page) {
                return &m;
            }
        }
        return nullptr;
    }

    void seq_move(llama_seq_id from, llama_seq_id to) {
        llama_memory_seq_rm(llama_get_memory(ctx_tgt), to, -1, -1);
        llama_memory_seq_cp(llama_get_memory(ctx_tgt), from, to, -1, -1);
        llama_memory_seq_rm(llama_get_memory(ctx_tgt), from, -1, -1);
        if (ctx_dft) {
            llama_memory_seq_rm(llama_get_memory(ctx_dft), to, -1, -1);
            llama_memory_seq_cp(llama_get_memory(ctx_dft), from, to, -1, -1);
            llama_memory_seq_rm(llama_get_memory(ctx_dft), from, -1, -1);
        }
    }

    std::string mind_file(mind_page page) const {
        char name[64];
        std::snprintf(name, sizeof(name), "mind-%016llx%016llx.bin", (unsigned long long) page.hi, (unsigned long long) page.lo);
        return params_base.slot_save_path + name;
    }

    // The coldest parked mind leaves the pool for flash. False: nothing is parked.
    bool flash_coldest() {
        resident_mind * cold = nullptr;
        for (auto & m : minds) {
            if (m.seq >= 0 && (cold == nullptr || m.last_use < cold->last_use)) {
                cold = &m;
            }
        }
        if (cold == nullptr) {
            return false;
        }
        const llama_seq_id seq = cold->seq;
        if (!flash_out(*cold, seq)) {
            return false;
        }
        park_free.push_back(seq);
        return true;
    }

    // Mind `m`'s state in sequence `seq` leaves the pool for flash: copied out (unified memory, a memcpy), the
    // sequence's cells freed now, the bytes written by the page-IO worker off the decode thread.
    bool flash_out(resident_mind & m, llama_seq_id seq) {
        resident_mind * cold = &m;
        const int64_t t0 = ggml_time_us();
        auto bytes = std::make_shared<std::vector<uint8_t>>();
        const size_t n_tgt = llama_state_seq_get_size_ext(ctx_tgt, seq, LLAMA_STATE_SEQ_FLAGS_NONE);
        const size_t n_dft = ctx_dft ? llama_state_seq_get_size_ext(ctx_dft, seq, LLAMA_STATE_SEQ_FLAGS_NONE) : 0;
        // her checkpoints go down with her (they are host bytes too: ~100 MB each on a 35B hybrid)
        size_t n_ck = sizeof(uint64_t);
        for (const auto & c : cold->prompt.checkpoints) {
            n_ck += ckpt_header_bytes + c.data_tgt.size() + c.data_dft.size() + c.data_spec.size();
        }
        bytes->resize(3 * sizeof(uint64_t) + n_tgt + n_dft + n_ck);
        uint8_t * w = bytes->data();
        const uint64_t h[3] = { n_tgt, n_dft, n_ck };
        std::memcpy(w, h, sizeof(h));
        w += sizeof(h);
        ckpt_write(cold->prompt.checkpoints, w + n_tgt + n_dft);
        if (llama_state_seq_get_data_ext(ctx_tgt, w, n_tgt, seq, LLAMA_STATE_SEQ_FLAGS_NONE) != n_tgt ||
            (ctx_dft && llama_state_seq_get_data_ext(ctx_dft, w + n_tgt, n_dft, seq, LLAMA_STATE_SEQ_FLAGS_NONE) != n_dft)) {
            SRV_ERR("mind %016llx: her state could not be copied out of the pool; she stays where she is\n", (unsigned long long) cold->page.hi);
            return false;
        }
        const int64_t t_copy = ggml_time_us() - t0;
        std::list<common_prompt_checkpoint>().swap(cold->prompt.checkpoints); // their bytes are in hers now
        llama_memory_seq_rm(llama_get_memory(ctx_tgt), seq, -1, -1);
        if (ctx_dft) {
            llama_memory_seq_rm(llama_get_memory(ctx_dft), seq, -1, -1);
        }
        cold->seq     = -1;
        cold->file    = mind_file(cold->page);
        cold->ram     = bytes; // the record holds her bytes until her file has landed
        cold->written = std::make_shared<std::atomic<int>>(0);
        flash_write(*cold);
        SRV_INF("mind %016llx%016llx out of the pool: %.1f MiB copied out in %.2f ms, cells freed; held in RAM until written\n",
                (unsigned long long) cold->page.hi, (unsigned long long) cold->page.lo,
                bytes->size() / (1024.0 * 1024.0), t_copy / 1000.0);
        return true;
    }

    // Her bytes to her file on the page-IO worker; the worker shares them, the record keeps them.
    void flash_write(resident_mind & m) {
        const std::string path = m.file;
        auto bytes = m.ram;
        auto done  = m.written;
        done->store(0, std::memory_order_release);
        page_io->submit(path, [path, bytes, done] {
            FILE * f = std::fopen(path.c_str(), "wb");
            bool ok = f != nullptr && std::fwrite(bytes->data(), 1, bytes->size(), f) == bytes->size();
            ok = (f != nullptr && std::fclose(f) == 0) && ok;
            done->store(ok ? 1 : -1, std::memory_order_release);
        });
    }

    // Between steps: a landed file releases her RAM copy; a failed write keeps it (she lives in RAM) and is
    // written again. Never waits.
    void minds_reap() {
        for (auto & m : minds) {
            if (!m.ram || !m.written) {
                continue;
            }
            const int st = m.written->load(std::memory_order_acquire);
            if (st == 1) {
                m.ram.reset();
            } else if (st == -1) {
                // she lives in RAM (never lost); the write is not retried every step against a failing disk: her
                // next trip out of a lane writes her again
                SRV_WRN("mind %016llx%016llx: her flash write failed; she stays in RAM\n",
                        (unsigned long long) m.page.hi, (unsigned long long) m.page.lo);
                m.written.reset();
            }
        }
    }

    // Her bytes back into `seq`, from where they are: RAM while her write is in flight or failed (never a wait on
    // the disk), else her file mapped and inserted directly (no read into a buffer first).
    bool flash_insert(resident_mind & m, llama_seq_id seq) {
        const int64_t t0 = ggml_time_us();
        std::unique_ptr<server_mapped_file> map;
        auto ram = m.ram; // held for the insert even if the write lands meanwhile
        const uint8_t * base = nullptr;
        size_t          size = 0;
        if (ram) {
            base = ram->data();
            size = ram->size();
        } else {
            map = std::make_unique<server_mapped_file>(m.file);
            if (!map->ok()) {
                return false;
            }
            base = map->data();
            size = map->size();
        }
        if (size < 3 * sizeof(uint64_t)) {
            return false;
        }
        uint64_t h[3];
        std::memcpy(h, base, sizeof(h));
        if (size != 3 * sizeof(uint64_t) + h[0] + h[1] + h[2]) {
            return false;
        }
        const uint8_t * r = base + sizeof(h);
        std::list<common_prompt_checkpoint> checkpoints;
        if (!ckpt_read(r + h[0] + h[1], h[2], checkpoints)) {
            return false;
        }
        // her cells come from the pool: when it is full, the coldest parked mind goes down to flash first, and the
        // insert is tried again (a level above makes room for the level below, never a drop)
        bool ok = false;
        while (true) {
            llama_memory_seq_rm(llama_get_memory(ctx_tgt), seq, -1, -1);
            ok = llama_state_seq_set_data_ext(ctx_tgt, r, h[0], seq, LLAMA_STATE_SEQ_FLAGS_NONE) == h[0];
            if (ok || !flash_coldest()) {
                break;
            }
        }
        while (ok && ctx_dft && h[1] > 0) {
            llama_memory_seq_rm(llama_get_memory(ctx_dft), seq, -1, -1);
            ok = llama_state_seq_set_data_ext(ctx_dft, r + h[0], h[1], seq, LLAMA_STATE_SEQ_FLAGS_NONE) == h[1];
            if (ok || !flash_coldest()) {
                break;
            }
        }
        if (!ok) {
            llama_memory_seq_rm(llama_get_memory(ctx_tgt), seq, -1, -1);
            if (ctx_dft) {
                llama_memory_seq_rm(llama_get_memory(ctx_dft), seq, -1, -1);
            }
            SRV_WRN("mind %016llx%016llx: the pool cannot hold her even with every other mind on flash; she stays on flash\n",
                    (unsigned long long) m.page.hi, (unsigned long long) m.page.lo);
            return false; // her file and record are kept: she is still on flash, never lost
        }
        SRV_INF("mind %016llx%016llx back from %s: %.1f MiB inserted in %.2f ms\n", (unsigned long long) m.page.hi,
                (unsigned long long) m.page.lo, ram ? "RAM" : "flash", size / (1024.0 * 1024.0), (ggml_time_us() - t0) / 1000.0);
        m.prompt.checkpoints = std::move(checkpoints);
        for (const int64_t at : m.released) {
            unpin_at(m.prompt.checkpoints, at); // released while she was on flash
        }
        m.released.clear();
        // her file is removed on the page-IO worker, ordered after its own write (it may still be landing)
        const std::string path = m.file;
        page_io->submit(path, [path] { std::error_code ec; std::filesystem::remove(path, ec); });
        m.file.clear();
        m.ram.reset();
        return true;
    }

    // Ahead of her turn (the core names who is next): a mind on flash is inserted into a free parked sequence
    // now, while the others decode, so her switch is the sequence move. Returns where she is afterwards.
    mind_place mind_prefetch(mind_page page) {
        for (const auto & s : slots) {
            if (s.page == page) {
                return mind_place::lane;
            }
        }
        resident_mind * m = find_mind(page);
        if (m == nullptr) {
            return mind_place::unknown;
        }
        if (m->seq >= 0) {
            m->last_use = ++mind_clock; // she is next: the last to be sent down
            return mind_place::parked;
        }
        if (park_free.empty()) {
            flash_coldest();
        }
        if (park_free.empty()) {
            return mind_place::deferred_no_parked_id;
        }
        const llama_seq_id seq = park_free.back();
        park_free.pop_back();
        // flash_coldest() may run inside the insert and move records: hold her by page, not by pointer
        if (!flash_insert(*find_mind(page), seq)) {
            park_free.push_back(seq);
            return mind_place::deferred_pool_full;
        }
        resident_mind * back = find_mind(page);
        back->seq      = seq;
        back->last_use = ++mind_clock;
        return mind_place::parked;
    }

    // `slot` serves `page` next. The mind it holds (if any) is parked; hers comes back from where she is.
    // false: the switch could not complete (the leaving mind could not leave intact, or the arriving mind's state
    // could not be inserted): the request waits for a lane, never runs cold beside a stored version of her.
    bool mind_switch(server_slot & slot, mind_page page) {
        slot.stats.mind_hi = page.hi;
        slot.stats.mind_lo = page.lo;
        if (slot.page == page) {
            slot.stats.mind_came_from  = mind_from::lane;
            slot.stats.mind_switch_us  = 0;
            slot.stats.mind_generation = slot.page_generation;
            return true;
        }
        const int64_t t0 = ggml_time_us();
        // the lane's mind leaves, whole
        if (slot.page.valid() && slot.prompt.n_tokens() > 0) {
            resident_mind m;
            m.page       = slot.page;
            m.lora       = slot.lora;
            m.last_use   = ++mind_clock;
            m.generation = slot.page_generation + 1;
            if (park_free.empty()) {
                flash_coldest(); // the pool's coldest parked mind makes room, if one is parked
            }
            m.prompt = std::move(slot.prompt);
            slot.prompt.clear();
            if (!park_free.empty()) {
                m.seq = park_free.back();
                park_free.pop_back();
                seq_move(slot.id, m.seq);
            } else if (!flash_out(m, slot.id)) {
                slot.prompt = std::move(m.prompt); // she stays whole on her lane
                SLT_ERR(slot, "%s", "the leaving mind could be neither parked nor copied out; she is kept on this lane\n");
                return false; // her state is intact on the lane; the arriving request waits for another lane
            }
            minds.push_back(std::move(m));
        } else if (!slot.page.valid()) {
            // content no named mind owns (a core page restore, or scratch): adopted, never cleared; the prompt's
            // prefix match keeps what is hers and re-prefills only what diverges
        }
        const int64_t t_out = ggml_time_us() - t0;
        // hers comes in from where she is
        const char * from = "new";
        mind_from    kind = slot.page.valid() || slot.prompt.n_tokens() == 0 ? mind_from::fresh : mind_from::adopted;
        uint64_t     generation = 1;
        for (size_t i = 0; i < minds.size(); ++i) {
            if (minds[i].page != page) {
                continue;
            }
            resident_mind & m = minds[i];
            bool ok;
            if (m.seq >= 0) {
                seq_move(m.seq, slot.id);
                park_free.push_back(m.seq);
                from = "parked";
                kind = mind_from::parked;
                ok   = true;
            } else {
                const bool in_ram = (bool) m.ram;
                from = in_ram ? "RAM" : "flash";
                kind = in_ram ? mind_from::ram : mind_from::flash;
                ok   = flash_insert(m, slot.id);
            }
            generation = m.generation + 1;
            if (!ok) {
                // she stays where she is (record and bytes intact: one version of her); the request waits for
                // cells, never runs cold beside her stored state
                slot.prompt_clear();
                slot.page = mind_page{};
                SLT_WRN(slot, "%s", "the arriving mind could not be inserted (the pool is full even with every other mind out); her request waits\n");
                return false;
            }
            slot.prompt = std::move(m.prompt);
            slot.lora   = std::move(m.lora);
            // the record may move under flash_coldest() above: find her again by page before erasing
            for (size_t j = 0; j < minds.size(); ++j) {
                if (minds[j].page == page) {
                    minds.erase(minds.begin() + (long) j);
                    break;
                }
            }
            break;
        }
        slot.page            = page;
        slot.page_generation = generation;
        slot.stats.mind_came_from  = kind;
        slot.stats.mind_switch_us  = ggml_time_us() - t0;
        slot.stats.mind_generation = generation;
        SLT_INF(slot, "mind switch: out %.3f ms, in (%s) %.3f ms, she holds %d tokens\n", t_out / 1000.0, from,
                (ggml_time_us() - t0 - t_out) / 1000.0, slot.prompt.n_tokens());
        return true;
    }
