#pragma once

// ONE learning budget for the whole server. Every learning step (a POST /learn, a completion's own
// learn, a /train window) runs under a grant the core's scheduler issued, and debits that grant's
// account. A grant is { id, engine incarnation, period, share }: the share of engine time, per
// period, its learning may take while another slot is serving. An idle engine learns free.
//
// What it guarantees (each one a check in test-server-learn-ledger):
//  - a grant from another engine incarnation (issued before a restart) is refused, never replenished;
//  - a grant is immutable: the same id with another period or share is refused;
//  - a retry or a duplicate never adds budget: every request naming the id debits the same account;
//  - admission reserves the step's estimated cost, finish debits what it measured and releases
//    the rest, cancel releases all of it;
//  - a step that could never fit its grant's whole period is refused as such, never retried forever.

#include <algorithm>
#include <cstdint>
#include <map>
#include <string>

struct server_learn_grant {
    std::string id;
    std::string engine;
    int64_t     period_ms = 0;
    int64_t     share_ppm = 0;
};

enum class server_learn_refusal {
    none,
    invalid,          // a field out of range: period_ms <= 0, share_ppm outside [1, 1e6], an empty id
    stale_engine,     // issued to another engine incarnation
    grant_mismatch,   // this id was registered with another period or share
    never_fits,       // the step costs more than the grant's whole period allows
    budget_exhausted, // the period's share is spent: retry_after_ms says when it renews
};

struct server_learn_admission {
    server_learn_refusal refusal        = server_learn_refusal::none;
    int64_t              retry_after_ms = 0;
    uint64_t             reservation    = 0; // nonzero when admitted
};

class server_learn_ledger {
public:
    explicit server_learn_ledger(std::string engine) : engine(std::move(engine)) {}

    // est_us: the step's expected cost (its last measured time). contended: another slot is
    // serving at admission, so the step's time is debited; an uncontended step is admitted
    // without a debit (a slot that starts during it waits at most that one step).
    server_learn_admission admit(const server_learn_grant & g, int64_t est_us, int64_t now_us, bool contended) {
        server_learn_admission out;
        if (g.id.empty() || g.period_ms <= 0 || g.share_ppm < 1 || g.share_ppm > 1000000 || est_us < 0) {
            out.refusal = server_learn_refusal::invalid;
            return out;
        }
        if (g.engine != engine) {
            out.refusal = server_learn_refusal::stale_engine;
            return out;
        }
        auto it = accounts.find(g.id);
        if (it == accounts.end()) {
            it = accounts.emplace(g.id, account{ g, now_us, 0, 0 }).first;
        } else if (it->second.grant.period_ms != g.period_ms || it->second.grant.share_ppm != g.share_ppm) {
            out.refusal = server_learn_refusal::grant_mismatch;
            return out;
        }
        account & a = it->second;
        roll(a, now_us);
        const int64_t cost = contended ? est_us : 0;
        if (cost > budget_us(a.grant)) {
            out.refusal = server_learn_refusal::never_fits;
            return out;
        }
        if (a.used_us + a.reserved_us + cost > budget_us(a.grant)) {
            out.refusal        = server_learn_refusal::budget_exhausted;
            out.retry_after_ms = (a.period_start_us + a.grant.period_ms * 1000 - now_us + 999) / 1000;
            return out;
        }
        a.reserved_us += cost;
        out.reservation = ++next_reservation;
        reservations.emplace(out.reservation, reservation{ g.id, cost, contended });
        return out;
    }

    // The step ran for measured_us: debit it (when it was admitted contended) and release the
    // rest of its reservation. Unknown or already settled reservations are ignored.
    void finish(uint64_t id, int64_t measured_us, int64_t now_us) {
        settle(id, std::max<int64_t>(measured_us, 0), now_us);
    }

    // The step never ran: its whole reservation returns to the account.
    void cancel(uint64_t id, int64_t now_us) {
        settle(id, 0, now_us);
    }

    // Debited time of the grant's current period (0 for an unknown grant).
    int64_t used_us(const std::string & grant, int64_t now_us) {
        auto it = accounts.find(grant);
        if (it == accounts.end()) {
            return 0;
        }
        roll(it->second, now_us);
        return it->second.used_us;
    }

    size_t n_accounts() const {
        return accounts.size();
    }

private:
    struct account {
        server_learn_grant grant;
        int64_t            period_start_us;
        int64_t            used_us;
        int64_t            reserved_us;
    };
    struct reservation {
        std::string grant;
        int64_t     cost_us;
        bool        contended;
    };

    // share_ppm of period_ms, in us: period_ms * 1000 * share_ppm / 1e6
    static int64_t budget_us(const server_learn_grant & g) {
        return g.period_ms * g.share_ppm / 1000;
    }

    // A new period starts where the old one ended; whole idle periods are skipped, not banked.
    static void roll(account & a, int64_t now_us) {
        const int64_t period_us = a.grant.period_ms * 1000;
        if (now_us >= a.period_start_us + period_us) {
            a.period_start_us += (now_us - a.period_start_us) / period_us * period_us;
            a.used_us = 0;
        }
    }

    void settle(uint64_t id, int64_t measured_us, int64_t now_us) {
        auto r = reservations.find(id);
        if (r == reservations.end()) {
            return;
        }
        auto a = accounts.find(r->second.grant);
        if (a != accounts.end()) {
            a->second.reserved_us -= r->second.cost_us;
            roll(a->second, now_us);
            if (r->second.contended) {
                a->second.used_us += measured_us;
            }
        }
        reservations.erase(r);
        forget_idle(now_us);
    }

    // Accounts are bounded: one with no reservation and no debit two periods back is forgotten
    // (a later request under its id registers it afresh; its spent budget was a past period's).
    void forget_idle(int64_t now_us) {
        for (auto it = accounts.begin(); it != accounts.end();) {
            const account & a = it->second;
            const bool idle = a.reserved_us == 0 && now_us >= a.period_start_us + 2 * a.grant.period_ms * 1000;
            it = idle ? accounts.erase(it) : std::next(it);
        }
    }

    std::string                         engine;
    std::map<std::string, account>      accounts;
    std::map<uint64_t, reservation>     reservations;
    uint64_t                            next_reservation = 0;
};
