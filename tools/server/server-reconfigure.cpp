#include "server-reconfigure.h"

#include "server-common.h"

#include "ggml.h"

#include <algorithm>
#include <limits>

const char * server_reconfigure_refusal_name(server_reconfigure_refusal_reason reason) {
    switch (reason) {
        case SERVER_RECONFIGURE_INVALID_GEOMETRY:     return "invalid_geometry";
        case SERVER_RECONFIGURE_CACHE_TYPE_CHANGE:    return "cache_type_change";
        case SERVER_RECONFIGURE_LEARN_IN_PROGRESS:    return "learn_in_progress";
        case SERVER_RECONFIGURE_LEARNING_BOUND:       return "learning_bound";
        case SERVER_RECONFIGURE_FAN_OUT:              return "fan_out_in_progress";
        case SERVER_RECONFIGURE_PAGE_OUT_FAILED:      return "page_out_failed";
        case SERVER_RECONFIGURE_CONTEXT_BUILD_FAILED: return "context_build_failed";
    }
    GGML_ABORT("unknown reconfigure refusal");
}

namespace {

// a slot's state while the contexts are rebuilt
struct paged_slot {
    server_slot_page page;
    int32_t tokens_held = 0;
    bool    busy        = false;
};

server_reconfigure_refused refuse(server_reconfigure_refusal_reason reason, std::string detail, int32_t slot = -1) {
    return server_reconfigure_refused { reason, std::move(detail), slot };
}

} // namespace

server_reconfigure_outcome server_reconfigure(server_reconfigure_engine & e, server_reconfigure_state & st, const server_geometry & target) {
    auto refused = [&](server_reconfigure_refused r) -> server_reconfigure_outcome {
        SRV_WRN("reconfigure to n_ctx = %d, n_seq = %d refused (%s): %s\n",
                target.n_ctx, target.n_seq, server_reconfigure_refusal_name(r.reason), r.detail.c_str());
        st.n_refused++;
        st.last = r;
        return r;
    };

    // every sequence needs at least one cell of its own
    if (target.n_seq < 1 || target.n_ctx < target.n_seq) {
        return refused(refuse(SERVER_RECONFIGURE_INVALID_GEOMETRY, "n_seq must be at least 1 and n_ctx at least n_seq"));
    }
    if (auto blocked = e.blocked()) {
        return refused(*blocked);
    }

    const int64_t t_start = ggml_time_us();
    const server_geometry before = e.live();

    server_reconfigure_applied out;
    out.rebuilt = target != before;

    std::map<int32_t, paged_slot> pages;   // slots in service whose state crosses the rebuild
    std::vector<int32_t>          parked_now;

    // the slot leaves service whole: its host state to the engine's park, its page here
    auto park = [&](int32_t id, paged_slot && p) {
        e.park(id);
        st.parked[id] = server_parked_slot { std::move(p.page), p.tokens_held, p.busy };
        parked_now.push_back(id);
    };

    if (out.rebuilt) {
        // 1. page out every slot holding state, unfinished generations included. Reading never changes
        //    a context, so a failure here leaves everything as it was.
        for (int32_t id = 0; id < before.n_seq; ++id) {
            const int32_t held = e.tokens_held(id);
            if (held == 0) {
                continue;
            }
            paged_slot p;
            p.tokens_held = held;
            p.busy        = e.busy(id);
            if (!e.page_out(id, p.page)) {
                return refused(refuse(SERVER_RECONFIGURE_PAGE_OUT_FAILED, "the slot's state could not be read", id));
            }
            out.tokens_held += held;
            out.bytes_paged += p.page.data.size();
            pages.emplace(id, std::move(p));
        }

        // 2. the ids a smaller n_seq removes leave service with their state, busy or not
        for (int32_t id = target.n_seq; id < before.n_seq; ++id) {
            auto it = pages.find(id);
            if (it == pages.end() && !e.busy(id)) {
                continue; // nothing there: the id just goes
            }
            paged_slot p;
            if (it != pages.end()) {
                p = std::move(it->second);
                pages.erase(it);
            } else {
                p.busy = true; // a task that has not filled its prompt yet
            }
            out.paged_out.push_back({ id, p.tokens_held, 0 });
            park(id, std::move(p));
        }

        // 3. the contexts, rebuilt on the same weights
        e.destroy_contexts();
        if (!e.build_contexts(target)) {
            // put back exactly what was: the previous geometry, the slots this call parked, every state
            if (!e.build_contexts(before)) {
                GGML_ABORT("reconfigure: the previous geometry (n_ctx = %d, n_seq = %d) could not be rebuilt either; "
                           "the server has no context to serve with", before.n_ctx, before.n_seq);
            }
            for (int32_t id : parked_now) {
                auto it = st.parked.find(id);
                e.unpark(id);
                pages[id] = paged_slot { std::move(it->second.page), it->second.tokens_held, it->second.busy };
                st.parked.erase(it);
            }
            for (auto & [id, p] : pages) {
                if (p.tokens_held > 0 && !e.page_in(id, p.page)) {
                    // the geometry that held it a moment ago does not take it back: keep it paged, by name
                    SRV_ERR("reconfigure: slot %d's state did not return to the previous geometry; it stays parked\n", id);
                    e.park(id);
                    st.parked[id] = server_parked_slot { std::move(p.page), p.tokens_held, p.busy };
                }
            }
            e.resume();
            return refused(refuse(SERVER_RECONFIGURE_CONTEXT_BUILD_FAILED,
                "the contexts for the target geometry could not be created; the previous geometry was rebuilt"));
        }
    }

    out.live      = e.live();
    out.n_ctx_seq = e.n_ctx_seq();

    const bool    unified = e.kv_unified();
    const int64_t n_cells = e.n_cells();
    // with one pool every sequence draws on, the states placed so far use part of it
    int64_t used = 0;
    if (!out.rebuilt) {
        for (int32_t id = 0; id < out.live.n_seq; ++id) {
            used += e.tokens_held(id);
        }
    }
    // a state fits when its sequence keeps room for the next token
    auto room = [&]() -> int32_t {
        const int64_t free = unified ? n_cells - used : std::numeric_limits<int64_t>::max();
        return (int32_t) std::min<int64_t>(out.n_ctx_seq, free);
    };

    // 4. page in: busy slots first (work in progress), then idle ones, each in id order. A state the
    //    target cannot hold is not cut: it is parked whole and named.
    std::vector<int32_t> order;
    for (const auto & [id, p] : pages) {
        if (p.busy) {
            order.push_back(id);
        }
    }
    for (const auto & [id, p] : pages) {
        if (!p.busy) {
            order.push_back(id);
        }
    }
    for (int32_t id : order) {
        paged_slot & p = pages.at(id);
        const int32_t fit = room();
        if (p.tokens_held < fit && e.page_in(id, p.page)) {
            used += p.tokens_held;
            continue;
        }
        out.paged_out.push_back({ id, p.tokens_held, std::max(fit, 0) });
        park(id, std::move(p));
    }

    // 5. parked slots come back where the live geometry holds them and their id is free
    for (auto it = st.parked.begin(); it != st.parked.end();) {
        const int32_t id = it->first;
        const server_parked_slot & p = it->second;
        const bool this_call = std::find(parked_now.begin(), parked_now.end(), id) != parked_now.end();
        if (this_call || id >= out.live.n_seq || p.tokens_held >= room() || e.busy(id) || e.tokens_held(id) > 0) {
            ++it;
            continue;
        }
        e.unpark(id);
        if (p.tokens_held > 0 && !e.page_in(id, p.page)) {
            e.park(id); // the page stays where it is, parked
            ++it;
            continue;
        }
        used += p.tokens_held;
        out.paged_in.push_back(id);
        it = st.parked.erase(it);
    }

    e.resume();

    for (const auto & [id, p] : st.parked) {
        out.parked.push_back(id);
    }
    out.pause_ms = (ggml_time_us() - t_start) / 1000.0;

    SRV_INF("reconfigured n_ctx = %d -> %d, n_seq = %d -> %d (%d tokens per sequence): paused %.2f ms for %lld tokens "
            "(%.3f MiB) of state, %zu slots paged out, %zu placed back, %zu parked\n",
            before.n_ctx, out.live.n_ctx, before.n_seq, out.live.n_seq, out.n_ctx_seq, out.pause_ms,
            (long long) out.tokens_held, out.bytes_paged / (1024.0 * 1024.0),
            out.paged_out.size(), out.paged_in.size(), out.parked.size());

    st.n_applied++;
    if (out.rebuilt) {
        st.n_rebuilt++;
        st.pause_ms_total    += out.pause_ms;
        st.pause_ms_max       = std::max(st.pause_ms_max, out.pause_ms);
        st.tokens_held_total += out.tokens_held;
        st.bytes_paged_total += out.bytes_paged;
    }
    st.last = out;
    return out;
}

json server_reconfigure_cost_json(const server_reconfigure_state & st) {
    json last = nullptr;
    if (st.last) {
        if (const auto * a = std::get_if<server_reconfigure_applied>(&*st.last)) {
            last = { { "pause_ms", a->pause_ms }, { "tokens_held", a->tokens_held }, { "bytes_paged", a->bytes_paged }, { "rebuilt", a->rebuilt } };
        }
    }
    return json {
        { "n_rebuilt",         st.n_rebuilt },
        { "pause_ms_total",    st.pause_ms_total },
        { "pause_ms_max",      st.pause_ms_max },
        { "tokens_held_total", st.tokens_held_total },
        { "bytes_paged_total", st.bytes_paged_total },
        // pause per thousand tokens held, over every rebuild that held any (null until one did)
        { "pause_ms_per_1k_tokens", st.tokens_held_total > 0 ? json(st.pause_ms_total * 1000.0 / st.tokens_held_total) : json(nullptr) },
        { "last_applied",      last },
    };
}

std::variant<server_geometry, server_reconfigure_refused> server_reconfigure_from_json(
        const json & body, const char * cache_type_k, const char * cache_type_v) {
    if (!body.is_object()
            || !body.contains("n_ctx") || !body.at("n_ctx").is_number_integer()
            || !body.contains("n_seq") || !body.at("n_seq").is_number_integer()) {
        return refuse(SERVER_RECONFIGURE_INVALID_GEOMETRY, "\"n_ctx\" and \"n_seq\" (integers) are required");
    }
    const int64_t n_ctx = body.at("n_ctx").get<int64_t>();
    const int64_t n_seq = body.at("n_seq").get<int64_t>();
    if (n_ctx < 1 || n_seq < 1 || n_ctx > std::numeric_limits<int32_t>::max() || n_seq > std::numeric_limits<int32_t>::max()) {
        return refuse(SERVER_RECONFIGURE_INVALID_GEOMETRY, "\"n_ctx\" and \"n_seq\" must be positive 32-bit integers");
    }
    // the KV type is the cache's, never the geometry's: a change of it is the intentional cold start
    for (const auto & [key, live] : { std::make_pair("cache_type_k", cache_type_k), std::make_pair("cache_type_v", cache_type_v) }) {
        if (!body.contains(key)) {
            continue;
        }
        if (!body.at(key).is_string() || body.at(key).get<std::string>() != live) {
            return refuse(SERVER_RECONFIGURE_CACHE_TYPE_CHANGE,
                std::string("\"") + key + "\" is " + live + "; a KV cache type change is a cold start, not a reconfigure");
        }
    }
    return server_geometry { (int32_t) n_ctx, (int32_t) n_seq };
}

json server_reconfigure_to_json(const server_reconfigure_outcome & outcome) {
    if (const auto * r = std::get_if<server_reconfigure_refused>(&outcome)) {
        // an error body the HTTP edge sends as is ({"error": ...}), with the refusal typed beside the message
        error_type type = ERROR_TYPE_SERVER; // the server failed at something it should do
        switch (r->reason) {
            case SERVER_RECONFIGURE_INVALID_GEOMETRY:
            case SERVER_RECONFIGURE_CACHE_TYPE_CHANGE:
                type = ERROR_TYPE_INVALID_REQUEST; // the request asks for what no reconfigure is
                break;
            case SERVER_RECONFIGURE_LEARN_IN_PROGRESS:
            case SERVER_RECONFIGURE_LEARNING_BOUND:
            case SERVER_RECONFIGURE_FAN_OUT:
                type = ERROR_TYPE_UNAVAILABLE; // the server's state refuses it now; asked again later it may not
                break;
            case SERVER_RECONFIGURE_PAGE_OUT_FAILED:
            case SERVER_RECONFIGURE_CONTEXT_BUILD_FAILED:
                break;
        }
        json j = format_error_response(std::string("reconfigure refused (") + server_reconfigure_refusal_name(r->reason) + "): " + r->detail, type);
        j["applied"] = false;
        j["refused"] = server_reconfigure_refusal_name(r->reason);
        j["detail"]  = r->detail;
        if (r->slot >= 0) {
            j["slot"] = r->slot;
        }
        return j;
    }
    const auto & a = std::get<server_reconfigure_applied>(outcome);
    json paged_out = json::array();
    for (const auto & s : a.paged_out) {
        paged_out.push_back({ { "slot", s.slot }, { "tokens_held", s.tokens_held }, { "tokens_fit", s.tokens_fit } });
    }
    return json {
        { "applied",     true },
        { "rebuilt",     a.rebuilt },
        { "live",        { { "n_ctx", a.live.n_ctx }, { "n_seq", a.live.n_seq }, { "n_ctx_seq", a.n_ctx_seq } } },
        { "paged_out",   paged_out },
        { "paged_in",    a.paged_in },
        { "parked",      a.parked },
        { "pause_ms",    a.pause_ms },
        { "tokens_held", a.tokens_held },
        { "bytes_paged", a.bytes_paged },
    };
}
