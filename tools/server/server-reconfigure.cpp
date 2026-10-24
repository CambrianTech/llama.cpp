#include "server-reconfigure.h"

#include "server-checkpoints.h"

#include "arg.h"

#include <algorithm>
#include <stdexcept>

json server_geometry::to_json() const {
    return json {
        { "n_ctx",        n_ctx },
        { "n_parallel",   n_parallel },
        { "cache_type_k", ggml_type_name(type_k) },
        { "cache_type_v", ggml_type_name(type_v) },
    };
}

int32_t server_geometry_n_ctx_seq(const server_geometry & geo, bool kv_unified, int32_t n_ctx_train) {
    const int32_t n_ctx = GGML_PAD(geo.n_ctx, 256);
    int32_t n_ctx_seq = kv_unified ? n_ctx : GGML_PAD(n_ctx / std::max(1, geo.n_parallel), 256);
    if (n_ctx_train > 0) {
        n_ctx_seq = std::min(n_ctx_seq, n_ctx_train);
    }
    return n_ctx_seq;
}

static int32_t parse_positive(const json & body, const char * key, int32_t cur, int32_t max) {
    if (!body.contains(key)) {
        return cur;
    }
    const json & v = body.at(key);
    if (!v.is_number_integer() || v.get<int64_t>() < 1 || v.get<int64_t>() > max) {
        throw server_page_error(ERROR_TYPE_INVALID_REQUEST, string_format("%s must be an integer in [1, %d]", key, max));
    }
    return v.get<int32_t>();
}

static ggml_type parse_kv_type(const json & body, const char * key, ggml_type cur) {
    if (!body.contains(key)) {
        return cur;
    }
    if (!body.at(key).is_string()) {
        throw server_page_error(ERROR_TYPE_INVALID_REQUEST, string_format("%s must be a KV cache type name", key));
    }
    try {
        return common_kv_cache_type_from_str(body.at(key).get<std::string>());
    } catch (const std::runtime_error & e) {
        throw server_page_error(ERROR_TYPE_INVALID_REQUEST, e.what());
    }
}

server_reconfigure_request server_reconfigure_request::parse(const json & body, const server_geometry & cur) {
    if (!body.is_object()) {
        throw server_page_error(ERROR_TYPE_INVALID_REQUEST, "the body must be a JSON object");
    }
    static const char * known[] = { "n_ctx", "n_parallel", "cache_type_k", "cache_type_v", "force" };
    for (const auto & item : body.items()) {
        const std::string & key = item.key();
        if (std::find_if(std::begin(known), std::end(known), [&](const char * k) { return key == k; }) == std::end(known)) {
            // a misspelled field silently keeping the old geometry is the bug this refuses
            throw server_page_error(ERROR_TYPE_INVALID_REQUEST, "unknown field: " + key);
        }
    }
    server_reconfigure_request req;
    req.geo.n_ctx      = parse_positive(body, "n_ctx",      cur.n_ctx,      INT32_MAX / 2);
    req.geo.n_parallel = parse_positive(body, "n_parallel", cur.n_parallel, (int32_t) llama_max_parallel_sequences());
    req.geo.type_k     = parse_kv_type (body, "cache_type_k", cur.type_k);
    req.geo.type_v     = parse_kv_type (body, "cache_type_v", cur.type_v);
    if (body.contains("force")) {
        if (!body.at("force").is_boolean()) {
            throw server_page_error(ERROR_TYPE_INVALID_REQUEST, "force must be a boolean");
        }
        req.force = body.at("force").get<bool>();
    }
    return req;
}

server_reconfigure_plan server_reconfigure_decide(
        const server_reconfigure_request & req,
        const server_geometry & cur,
        const std::vector<server_reconfigure_slot> & slots,
        bool kv_unified,
        int32_t n_ctx_train,
        const std::string & name_prefix) {
    server_reconfigure_plan plan;
    plan.n_ctx_seq  = server_geometry_n_ctx_seq(req.geo, kv_unified, n_ctx_train);
    plan.cold_start = !req.geo.same_kv_format(cur);

    if (plan.cold_start && !req.force) {
        throw server_page_error(ERROR_TYPE_INVALID_REQUEST, string_format(
            "a KV type change (%s/%s -> %s/%s) cannot carry the residents' states over: pass \"force\": true to cold start",
            ggml_type_name(cur.type_k), ggml_type_name(cur.type_v), ggml_type_name(req.geo.type_k), ggml_type_name(req.geo.type_v)));
    }

    for (const auto & s : slots) {
        if (!s.processing) {
            continue;
        }
        if (s.n_tokens + 1 > plan.n_ctx_seq) {
            throw server_page_error(ERROR_TYPE_INVALID_REQUEST, string_format(
                "slot %d is serving a request holding %d tokens and the new geometry gives a sequence %d: "
                "shrinking under a live request would truncate it",
                s.id, s.n_tokens, plan.n_ctx_seq));
        }
        if (!s.at_boundary) {
            plan.wait = true;
            plan.wait_reason = string_format("slot %d is between a decode and its sampling", s.id);
        } else if (s.id >= req.geo.n_parallel) {
            plan.wait = true;
            plan.wait_reason = string_format("slot %d serves a request and has no place among %d slots", s.id, req.geo.n_parallel);
        } else if (plan.cold_start) {
            plan.wait = true;
            plan.wait_reason = string_format("slot %d serves a request and a cold start has no state to resume it from", s.id);
        }
    }
    if (plan.wait) {
        return plan;
    }

    // slots of the new geometry that nobody keeps: an occupied idle slot past n_parallel moves into one
    std::vector<bool> taken(req.geo.n_parallel, false);
    for (const auto & s : slots) {
        if (s.n_tokens > 0 && s.id < req.geo.n_parallel && s.n_tokens <= plan.n_ctx_seq) {
            taken[s.id] = true;
        }
    }

    for (const auto & s : slots) {
        if (s.n_tokens == 0) {
            continue;
        }
        server_reconfigure_move m;
        m.id_from    = s.id;
        m.processing = s.processing;
        m.n_tokens   = s.n_tokens;
        if (plan.cold_start) {
            m.fate   = SERVER_RECONFIGURE_DROPPED;
            m.reason = "KV type change (cold start)";
        } else if (s.n_tokens > plan.n_ctx_seq) {
            m.fate   = SERVER_RECONFIGURE_KEPT_IN_STORE;
            m.reason = string_format("%d tokens do not fit a sequence of %d", s.n_tokens, plan.n_ctx_seq);
        } else if (s.id < req.geo.n_parallel) {
            m.id_to = s.id;
        } else {
            const auto free = std::find(taken.begin(), taken.end(), false);
            if (free == taken.end()) {
                m.fate   = SERVER_RECONFIGURE_KEPT_IN_STORE;
                m.reason = string_format("no free slot among %d", req.geo.n_parallel);
            } else {
                *free  = true;
                m.id_to = (int) (free - taken.begin());
            }
        }
        if (m.fate == SERVER_RECONFIGURE_KEPT_IN_STORE) {
            m.page = string_format("%s.slot%d", name_prefix.c_str(), s.id);
        }
        plan.moves.push_back(std::move(m));
    }
    return plan;
}

static const char * fate_name(server_reconfigure_fate f) {
    switch (f) {
        case SERVER_RECONFIGURE_RESTORED:      return "restored";
        case SERVER_RECONFIGURE_KEPT_IN_STORE: return "kept_in_store";
        case SERVER_RECONFIGURE_DROPPED:       return "dropped";
    }
    return "unknown";
}

json server_reconfigure_move::to_json() const {
    json j = {
        { "id_slot_before", id_from },
        { "id_slot_after",  id_to },
        { "in_flight",      processing },
        { "n_tokens",       n_tokens },
        { "n_bytes",        n_bytes },
        { "fate",           fate_name(fate) },
    };
    if (!page.empty()) {
        j["page"] = page;
    }
    if (!reason.empty()) {
        j["reason"] = reason;
    }
    return j;
}

json server_reconfigure_receipt::to_json() const {
    json moved = json::array();
    int n_restored = 0, n_kept = 0, n_dropped = 0;
    for (const auto & m : moves) {
        moved.push_back(m.to_json());
        n_restored += m.fate == SERVER_RECONFIGURE_RESTORED;
        n_kept     += m.fate == SERVER_RECONFIGURE_KEPT_IN_STORE;
        n_dropped  += m.fate == SERVER_RECONFIGURE_DROPPED;
    }
    json j = {
        { "before",           before.to_json() },
        { "after",            after.to_json() },
        { "n_ctx_seq_before", n_ctx_seq_before },
        { "n_ctx_seq_after",  n_ctx_seq_after },
        { "cold_start",       cold_start },
        { "rolled_back",      rolled_back },
        { "t_drain_ms",       t_drain_ms },
        { "t_page_out_ms",    t_page_out_ms },
        { "t_free_ms",        t_free_ms },
        { "t_init_ms",        t_init_ms },
        { "t_page_in_ms",     t_page_in_ms },
        { "t_total_ms",       t_total_ms },
        { "n_bytes",          n_bytes },
        { "n_bytes_dropped",  n_bytes_dropped },
        { "n_tasks_deferred", n_tasks_deferred },
        { "n_restored",       n_restored },
        { "n_kept_in_store",  n_kept },
        { "n_dropped",        n_dropped },
        { "pages",            moved },
    };
    if (rolled_back) {
        j["rollback_reason"] = rollback_reason;
    }
    return j;
}

void server_reconfigure_run(
        server_reconfigure_host & host,
        const server_reconfigure_request & req,
        server_reconfigure_plan & plan,
        server_reconfigure_receipt & rc,
        const std::string & page_prefix) {
    rc.before           = host.geometry();
    rc.n_ctx_seq_before = host.n_ctx_seq();
    server_page_store & store = host.pages();

    // the store must take every page that cannot be restored before anything moves, counted as it charges them
    std::vector<server_page_shape> shapes;
    shapes.reserve(plan.moves.size());
    {
        size_t n_kept = 0;
        for (const auto & m : plan.moves) {
            shapes.push_back(server_page_measure(host.target(), host.draft(), host.speculative(), m.id_from));
            if (m.fate == SERVER_RECONFIGURE_KEPT_IN_STORE) {
                n_kept += shapes.back().n_bytes(server_checkpoints_bytes(host.slot_prompt(m.id_from).checkpoints));
            }
        }
        if (store.limit_bytes > 0 && store.n_bytes() + n_kept > store.limit_bytes) {
            throw server_page_error(ERROR_TYPE_UNAVAILABLE, string_format(
                "%zu bytes of slot state cannot be restored under the new geometry and the page store (%zu of %llu bytes) cannot hold them",
                n_kept, store.n_bytes(), (unsigned long long) store.limit_bytes));
        }
    }

    // 1. page out: every occupied slot's state into host RAM. A cold start's states are captured
    //    too and dropped only once the new geometry stands, so a failed build rolls back whole.
    const int64_t t0 = ggml_time_us();
    std::vector<server_page_ptr> pages(plan.moves.size());
    for (size_t i = 0; i < plan.moves.size(); ++i) {
        auto & m = plan.moves[i];
        pages[i] = server_page_capture(host.target(), host.draft(), m.id_from, std::move(shapes[i]));
        server_prompt & prompt = host.slot_prompt(m.id_from);
        const bool in_place = m.fate == SERVER_RECONFIGURE_RESTORED && m.id_to == m.id_from;
        if (!in_place) {
            // it leaves its slot: the page carries the tokens, checkpoints and adapters
            pages[i]->prompt = std::move(prompt);
            pages[i]->lora   = host.slot_lora(m.id_from);
            prompt.clear();
        }
        m.n_bytes = pages[i]->n_bytes() + (in_place ? server_checkpoints_bytes(prompt.checkpoints) : 0);
        rc.n_bytes += m.fate == SERVER_RECONFIGURE_DROPPED ? 0 : m.n_bytes;
    }
    const int64_t t1 = ggml_time_us();

    // 2. free both contexts; the model, its mmap and the adapters stay
    host.free_contexts();
    const int64_t t2 = ggml_time_us();

    // 3. build the new geometry, or the old one again
    if (!host.build_contexts(req.geo)) {
        rc.rolled_back     = true;
        rc.rollback_reason = "the new geometry could not be built (see the log); the old one was rebuilt";
        SRV_ERR("reconfigure to n_ctx = %d, n_parallel = %d failed; rebuilding the old geometry\n", req.geo.n_ctx, req.geo.n_parallel);
        if (!host.build_contexts(rc.before)) {
            GGML_ABORT("reconfigure: neither the new nor the old context geometry could be built");
        }
    }
    const int64_t t3 = ggml_time_us();

    // 4. the slots of the geometry that stands
    host.set_slots(rc.rolled_back ? rc.before.n_parallel : req.geo.n_parallel);

    // 5. page in: every state back, or into the store with a typed reason; never truncated
    for (size_t i = 0; i < plan.moves.size(); ++i) {
        auto & m = plan.moves[i];
        if (rc.rolled_back) {
            m.fate  = SERVER_RECONFIGURE_RESTORED; // the old geometry holds every state it held
            m.id_to = m.id_from;
            m.reason.clear();
            m.page.clear();
        }
        if (m.fate == SERVER_RECONFIGURE_DROPPED) {
            rc.n_bytes_dropped += m.n_bytes;
            pages[i].reset();
            continue;
        }
        // a page that left its slot carries the slot's tokens; one restored in place does not
        const bool carries_prompt = pages[i]->prompt.tokens.size() > 0;
        if (m.fate == SERVER_RECONFIGURE_RESTORED) {
            try {
                server_page_restore(*pages[i], host.target(), host.draft(), host.speculative(), m.id_to);
                if (carries_prompt) {
                    host.slot_prompt(m.id_to) = std::move(pages[i]->prompt);
                    host.slot_lora(m.id_to)   = pages[i]->lora;
                }
                continue;
            } catch (const server_page_error & err) {
                m.reason = err.what();
                if (host.slot_processing(m.id_to)) {
                    // its KV is gone and its request cannot continue: it ends with a typed error
                    // instead of going on with a context it no longer has
                    host.slot_fail(m.id_to, std::string("the reconfigure could not restore this request's state: ") + err.what());
                    m.fate = SERVER_RECONFIGURE_DROPPED;
                    continue;
                }
                if (!carries_prompt) {
                    pages[i]->prompt = std::move(host.slot_prompt(m.id_to));
                    pages[i]->lora   = host.slot_lora(m.id_to);
                }
                host.slot_prompt(m.id_to).clear();
                m.fate = SERVER_RECONFIGURE_KEPT_IN_STORE;
                m.page = string_format("%s.slot%d", page_prefix.c_str(), m.id_from);
            }
        }
        m.id_to = -1;
        try {
            store.put(m.page, pages[i]);
            SRV_WRN("reconfigure: slot %d's %d tokens stay in the page store as '%s': %s\n",
                    m.id_from, m.n_tokens, m.page.c_str(), m.reason.c_str());
        } catch (const server_page_error & err) {
            // admitted above unless its restore failed here: a state the store cannot take is dropped, typed
            m.reason = (m.reason.empty() ? "" : m.reason + "; ") + "the page store refused it: " + err.what();
            SRV_ERR("reconfigure: slot %d's %d tokens are dropped: %s\n", m.id_from, m.n_tokens, m.reason.c_str());
            m.fate = SERVER_RECONFIGURE_DROPPED;
            m.page.clear();
            rc.n_bytes_dropped += m.n_bytes;
        }
    }

    rc.cold_start = plan.cold_start && !rc.rolled_back;
    if (rc.cold_start) {
        // every state in the old KV format: the store's pages and the prompt cache go with them
        const size_t n_resident = rc.n_bytes_dropped;
        const size_t n_store    = store.n_bytes();
        const size_t n_pages    = store.n_pages();
        const size_t n_cache    = host.drop_prompt_cache();
        store.clear();
        rc.n_bytes_dropped += n_store + n_cache;
        SRV_WRN("reconfigure: KV type change %s/%s -> %s/%s is a cold start: dropped %zu resident bytes, %zu page-store bytes (%zu pages), %zu prompt-cache bytes\n",
                ggml_type_name(rc.before.type_k), ggml_type_name(rc.before.type_v),
                ggml_type_name(req.geo.type_k), ggml_type_name(req.geo.type_v),
                n_resident, n_store, n_pages, n_cache);
    }
    const int64_t t4 = ggml_time_us();

    rc.after           = host.geometry();
    rc.n_ctx_seq_after = host.n_ctx_seq();
    rc.moves           = plan.moves;
    rc.t_page_out_ms   = (t1 - t0) / 1000.0;
    rc.t_free_ms       = (t2 - t1) / 1000.0;
    rc.t_init_ms       = (t3 - t2) / 1000.0;
    rc.t_page_in_ms    = (t4 - t3) / 1000.0;
    rc.t_total_ms      = (t4 - t0) / 1000.0;

    SRV_INF("reconfigure: n_ctx %d -> %d, n_parallel %d -> %d, %s/%s; pause %.1f ms (page out %.1f, free %.1f, init %.1f, page in %.1f) for %zu bytes; drain %.1f ms%s\n",
            rc.before.n_ctx, rc.after.n_ctx, rc.before.n_parallel, rc.after.n_parallel,
            ggml_type_name(rc.after.type_k), ggml_type_name(rc.after.type_v),
            rc.t_total_ms, rc.t_page_out_ms, rc.t_free_ms, rc.t_init_ms, rc.t_page_in_ms, rc.n_bytes, rc.t_drain_ms,
            rc.rolled_back ? " (ROLLED BACK)" : "");
}
