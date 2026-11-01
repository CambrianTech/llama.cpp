// what this catches: the learning ledger's contract (server-learn-ledger.h) without a model, in
// milliseconds: authority-only registration, immutable live grants, no replenishment by retry,
// free reservations as live custody, cancel only before start, checked ranges, bounded accounts.
#include "../tools/server/server-learn-ledger.h"

#include <cstdlib>
#include <iostream>

static void require(bool condition, const char * message) {
    if (!condition) {
        std::cerr << "FAILED: " << message << '\n';
        std::exit(1);
    }
}

using R = server_learn_refusal;

int main() {
    server_learn_ledger ledger("engine-a");
    const int64_t day_us = 86400000000;
    // 10% of a 1000 ms period: 100 ms of contended learning per period, for a day
    const server_learn_grant g{ "grant-1", "engine-a", 1000, 100000, day_us };

    // registration is the authority's, with checked ranges
    require(ledger.register_grant({ "", "engine-a", 1000, 100000, day_us }, 0) == R::invalid, "empty id");
    require(ledger.register_grant({ "x", "engine-a", 0, 100000, day_us }, 0) == R::invalid, "zero period");
    require(ledger.register_grant({ "x", "engine-a", server_learn_ledger::kMaxPeriodMs + 1, 100000, day_us }, 0) == R::invalid,
            "a period past the checked range");
    require(ledger.register_grant({ "x", "engine-a", 1000, 0, day_us }, 0) == R::invalid, "zero share is no learning");
    require(ledger.register_grant({ "x", "engine-a", 1000, 1000001, day_us }, 0) == R::invalid, "share over 1e6");
    require(ledger.register_grant({ "x", "engine-a", 1000, 100000, 0 }, 0) == R::invalid, "already expired");
    require(ledger.register_grant({ "x", "engine-b", 1000, 100000, day_us }, 0) == R::stale_engine, "another incarnation");
    require(ledger.n_accounts() == 0, "refused grants register nothing");
    require(ledger.admit("never-registered", 10, 0, true).refusal == R::unknown_grant, "a step cannot mint a grant");
    require(ledger.admit("grant-1", server_learn_ledger::kMaxCostUs + 1, 0, true).refusal == R::invalid, "a cost past the range");

    require(ledger.register_grant(g, 0) == R::none, "the authority registers");
    require(ledger.register_grant(g, 0) == R::none, "re-confirming the same terms is a no-op");
    require(ledger.register_grant({ "grant-1", "engine-a", 1000, 900000, day_us }, 0) == R::grant_mismatch, "a live grant is immutable");

    // reserve 60 ms, then a second 60 ms does not fit beside it
    const auto a1 = ledger.admit("grant-1", 60000, 0, true);
    require(a1.refusal == R::none && a1.reservation != 0, "first step admitted");
    const auto a2 = ledger.admit("grant-1", 60000, 1000, true);
    require(a2.refusal == R::budget_exhausted && a2.retry_after_ms == 999, "reserved time counts against the share");

    // started: it settles with what it measured, never cancels
    require(ledger.start(a1.reservation), "start");
    require(ledger.cancel(a1.reservation, 30000) == R::started, "a started step cannot be cancelled");
    require(ledger.finish(a1.reservation, 30000, 1, 40000) == R::none, "it settles after quiescence");
    require(ledger.used_us("grant-1", 40000) == 30000 && ledger.committed_steps("grant-1") == 1, "debit and steps measured");
    require(ledger.finish(a1.reservation, 30000, 1, 40000) == R::unknown_reservation, "never settled twice");

    // 70 ms remain: a retry of the same request spends the same account
    const auto a3 = ledger.admit("grant-1", 70000, 41000, true);
    require(a3.refusal == R::none, "the remaining 70 ms fit");
    require(ledger.admit("grant-1", 1, 42000, true).refusal == R::budget_exhausted, "a duplicate never replenishes");
    require(ledger.cancel(a3.reservation, 43000) == R::none, "never started: cancelled");
    require(ledger.used_us("grant-1", 43000) == 30000, "cancel releases the whole reservation");

    // a step larger than the whole period's share can never run: said so, not retried
    require(ledger.admit("grant-1", 100001, 44000, true).refusal == R::never_fits, "never fits");

    // a free (uncontended) reservation is live custody: revoking the grant does not erase its
    // account while it is outstanding, and its later settlement still lands
    const auto free_step = ledger.admit("grant-1", 500000, 45000, false);
    require(free_step.refusal == R::none, "uncontended step admitted");
    ledger.revoke("grant-1", 46000);
    require(ledger.n_accounts() == 1, "a live free reservation keeps its account through revocation");
    require(ledger.admit("grant-1", 1, 47000, true).refusal == R::unknown_grant, "a revoked grant admits nothing new");
    require(ledger.start(free_step.reservation) && ledger.finish(free_step.reservation, 500000, 1, 3 * 1000000) == R::none,
            "the outstanding step settles long after");
    require(ledger.n_accounts() == 0, "then the revoked account goes");

    // bounded without idleness eviction: an expired grant with nothing outstanding goes at the
    // next registration, and a live idle one stays (its terms cannot be reset)
    require(ledger.register_grant({ "short", "engine-a", 1000, 100000, 2000000 }, 1000000) == R::none, "short grant");
    require(ledger.register_grant({ "idle", "engine-a", 1000, 100000, day_us }, 1000000) == R::none, "idle grant");
    require(ledger.register_grant({ "next", "engine-a", 1000, 100000, day_us }, 5000000) == R::none, "a later registration");
    require(ledger.n_accounts() == 2, "the expired grant is gone, the idle live one stays");
    require(ledger.register_grant({ "idle", "engine-a", 1000, 500000, day_us }, 9000000) == R::grant_mismatch,
            "an idle live grant keeps its terms");

    std::cout << "test-server-learn-ledger: OK\n";
    return 0;
}
