#pragma once

// ONE learning budget for the whole server. The core's resource authority REGISTERS a grant
// { id, engine incarnation, period, share, expiry }: the share of engine time, per period, the
// learning it covers may take while another slot is serving. Every learning step (a POST /learn,
// a completion's own learn, a /train window) references a registered grant and debits its
// account. Receiving grant fields on a step authorizes nothing: an unregistered id is refused.
//
// What it guarantees (each one a check in test-server-learn-ledger):
//  - registration is the authority's: an unknown grant id is refused, never minted from a step;
//  - a grant from another engine incarnation (issued before a restart) is refused at registration;
//  - a grant is immutable while it lives: re-registering its id with other terms is refused, with
//    the same terms is a no-op; it ends only by revocation or expiry, never by idleness, and a
//    changed budget is a new grant the authority issues;
//  - a retry or a duplicate never adds budget: every step naming the id debits the same account;
//  - every reservation, a free (uncontended) one included, is live custody until settled;
//  - a reservation is cancelled (its whole cost released) only before its step started; a
//    started step settles with its measured time and committed steps, after quiescence;
//  - a step that could never fit its grant's whole period is refused as such;
//  - a step whose grant was revoked or expired after admission does not start (typed refusal);
//  - arithmetic is checked: clocks live in an explicit domain, periods and costs are bounded, and
//    accumulated time and steps are checked before they change: a typed refusal, never a wrap.

#include <climits>
#include <cstdint>
#include <iterator>
#include <map>
#include <string>

struct server_learn_grant {
    std::string id;
    std::string engine;
    int64_t     period_ms     = 0;
    int64_t     share_ppm     = 0;
    int64_t     expires_at_us = 0; // the authority's expiry, on the engine's clock
};

enum class server_learn_refusal {
    none,
    invalid,          // a field out of range (see kMaxPeriodMs, share in [1, 1e6], an empty id, a negative cost)
    stale_engine,     // registered for another engine incarnation
    grant_mismatch,   // this live id is registered with other terms
    unknown_grant,    // no live grant has this id (never registered, revoked or expired)
    never_fits,       // the step costs more than the grant's whole period allows
    budget_exhausted, // the period's share is spent: retry_after_ms says when it renews
    started,          // cancel after the step started: it must settle with its measured time
    unknown_reservation,
};

struct server_learn_admission {
    server_learn_refusal refusal        = server_learn_refusal::none;
    int64_t              retry_after_ms = 0;
    uint64_t             reservation    = 0; // nonzero when admitted
};

class server_learn_ledger {
public:
    // The longest period a grant may name, and the largest step cost: the bounds under which every
    // product and sum below stays in range (1 day of ms x 1e6 ppm, and 1 day of us).
    static constexpr int64_t kMaxPeriodMs = 86400000;
    static constexpr int64_t kMaxCostUs   = 86400000000;
    // The accepted engine clock domain (us since the engine's epoch): with periods <= 1 day, every
    // sum of a clock value and a period stays below INT64_MAX. The core translates its own expiry
    // into this clock; a wall-clock time interpreted as engine time is its bug, not this ledger's.
    static constexpr int64_t kMaxClockUs = int64_t(1) << 62;
    // One settlement commits at most this many optimizer steps.
    static constexpr int64_t kMaxStepsPerSettle = int64_t(1) << 20;

    explicit server_learn_ledger(std::string engine) : engine(std::move(engine)) {}

    // The authority registers (or re-confirms) a grant.
    server_learn_refusal register_grant(const server_learn_grant & g, int64_t now_us) {
        if (!in_clock(now_us) || g.id.empty() || g.period_ms <= 0 || g.period_ms > kMaxPeriodMs || g.share_ppm < 1 ||
                g.share_ppm > 1000000 || g.expires_at_us <= now_us || !in_clock(g.expires_at_us)) {
            return server_learn_refusal::invalid;
        }
        if (g.engine != engine) {
            return server_learn_refusal::stale_engine;
        }
        // bounded: grants the authority let expire, whose last reservation settled, go here
        for (auto a = accounts.begin(); a != accounts.end();) {
            const bool done = now_us >= a->second.grant.expires_at_us && a->second.outstanding == 0;
            a = done ? accounts.erase(a) : std::next(a);
        }
        const auto it = accounts.find(g.id);
        if (it != accounts.end()) {
            const auto & r = it->second.grant;
            const bool same = r.period_ms == g.period_ms && r.share_ppm == g.share_ppm && r.expires_at_us == g.expires_at_us;
            return same ? server_learn_refusal::none : server_learn_refusal::grant_mismatch;
        }
        accounts.emplace(g.id, account{ g, now_us, 0, 0, 0, false });
        return server_learn_refusal::none;
    }

    // The authority ends a grant. Its account goes once its last reservation settles.
    void revoke(const std::string & id, int64_t now_us) {
        if (!in_clock(now_us)) {
            return;
        }
        const auto it = accounts.find(id);
        if (it != accounts.end()) {
            it->second.revoked = true;
            retire_if_done(it, now_us);
        }
    }

    // est_us: the step's expected cost (its last measured time). contended: another slot is
    // serving at admission, so the step's time is debited; an uncontended step is admitted
    // without a debit but is still a live reservation.
    server_learn_admission admit(const std::string & grant, int64_t est_us, int64_t now_us, bool contended) {
        server_learn_admission out;
        if (est_us < 0 || est_us > kMaxCostUs || !in_clock(now_us)) {
            out.refusal = server_learn_refusal::invalid;
            return out;
        }
        const auto it = accounts.find(grant);
        if (it == accounts.end() || it->second.revoked || now_us >= it->second.grant.expires_at_us) {
            out.refusal = server_learn_refusal::unknown_grant;
            return out;
        }
        account & a = it->second;
        roll(a, now_us);
        const int64_t cost   = contended ? est_us : 0;
        const int64_t budget = budget_us(a.grant);
        if (cost > budget) {
            out.refusal = server_learn_refusal::never_fits;
            return out;
        }
        if (a.used_us + a.reserved_us + cost > budget) {
            out.refusal        = server_learn_refusal::budget_exhausted;
            out.retry_after_ms = (a.period_start_us + a.grant.period_ms * 1000 - now_us + 999) / 1000;
            return out;
        }
        a.reserved_us += cost;
        a.outstanding += 1;
        out.reservation = ++next_reservation;
        reservations.emplace(out.reservation, reservation{ grant, cost, contended, false });
        return out;
    }

    // The step begins executing: from here it settles only with its measured time. Refused when its
    // grant was revoked or expired after admission (unknown_grant): the reservation stays held, so
    // the caller cancels it (it never ran) and nothing is erased silently.
    server_learn_refusal start(uint64_t id, int64_t now_us) {
        if (!in_clock(now_us)) {
            return server_learn_refusal::invalid;
        }
        const auto r = reservations.find(id);
        if (r == reservations.end()) {
            return server_learn_refusal::unknown_reservation;
        }
        const auto a = accounts.find(r->second.grant);
        if (a == accounts.end() || a->second.revoked || now_us >= a->second.grant.expires_at_us) {
            return server_learn_refusal::unknown_grant;
        }
        r->second.started = true;
        return server_learn_refusal::none;
    }

    // The step quiesced after measured_us with committed_steps optimizer steps: debit the measured
    // time (when it was admitted contended) and release the rest of its reservation.
    server_learn_refusal finish(uint64_t id, int64_t measured_us, int64_t committed_steps, int64_t now_us) {
        if (measured_us < 0 || measured_us > kMaxCostUs || committed_steps < 0 || committed_steps > kMaxStepsPerSettle ||
                !in_clock(now_us)) {
            return server_learn_refusal::invalid;
        }
        return settle(id, measured_us, committed_steps, now_us, /*cancel =*/ false);
    }

    // The step never ran: its whole reservation returns. Refused once it started.
    server_learn_refusal cancel(uint64_t id, int64_t now_us) {
        if (!in_clock(now_us)) {
            return server_learn_refusal::invalid;
        }
        return settle(id, 0, 0, now_us, /*cancel =*/ true);
    }

    // Debited time and committed steps of a live grant's current period (0 for an unknown grant).
    int64_t used_us(const std::string & grant, int64_t now_us) {
        const auto it = accounts.find(grant);
        if (it == accounts.end()) {
            return 0;
        }
        roll(it->second, now_us);
        return it->second.used_us;
    }

    int64_t committed_steps(const std::string & grant) const {
        const auto it = accounts.find(grant);
        return it == accounts.end() ? 0 : it->second.steps;
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
        int64_t            outstanding; // live reservations, free ones included
        bool               revoked;
        int64_t            steps = 0;
    };
    struct reservation {
        std::string grant;
        int64_t     cost_us;
        bool        contended;
        bool        started;
    };

    static bool in_clock(int64_t t) {
        return t >= 0 && t <= kMaxClockUs;
    }

    // share_ppm of period_ms, in us: in range by kMaxPeriodMs x 1e6 (8.64e13)
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

    server_learn_refusal settle(uint64_t id, int64_t measured_us, int64_t committed, int64_t now_us, bool cancel) {
        const auto r = reservations.find(id);
        if (r == reservations.end()) {
            return server_learn_refusal::unknown_reservation;
        }
        if (cancel && r->second.started) {
            return server_learn_refusal::started;
        }
        const auto a = accounts.find(r->second.grant);
        if (a != accounts.end()) {
            roll(a->second, now_us);
            const int64_t debit = r->second.contended ? measured_us : 0;
            // checked before anything changes: a sum past INT64_MAX is refused, never wrapped
            if (a->second.used_us > INT64_MAX - debit || a->second.steps > INT64_MAX - committed) {
                return server_learn_refusal::invalid;
            }
            a->second.reserved_us -= r->second.cost_us;
            a->second.outstanding -= 1;
            a->second.used_us += debit;
            a->second.steps += committed;
        }
        reservations.erase(r);
        if (a != accounts.end()) {
            retire_if_done(a, now_us);
        }
        return server_learn_refusal::none;
    }

    // An account ends only when the authority ended its grant (revoked or expired) and its last
    // reservation settled: never by idleness, so a live grant's terms cannot be reset.
    void retire_if_done(std::map<std::string, account>::iterator it, int64_t now_us) {
        const account & a = it->second;
        if ((a.revoked || now_us >= a.grant.expires_at_us) && a.outstanding == 0) {
            accounts.erase(it);
        }
    }

    std::string                     engine;
    std::map<std::string, account>  accounts;
    std::map<uint64_t, reservation> reservations;
    uint64_t                        next_reservation = 0;
};
