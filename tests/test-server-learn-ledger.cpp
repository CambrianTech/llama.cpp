// what this catches: the learning ledger's contract (server-learn-ledger.h) without a model, in
// milliseconds: stale and mismatched grants refused, no replenishment by retry, reserve/settle
// conservation, never-fitting steps refused as such, and bounded accounts.
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
    // 10% of a 1000 ms period: 100 ms of contended learning per period
    const server_learn_grant g{ "grant-1", "engine-a", 1000, 100000 };

    require(ledger.admit({ "", "engine-a", 1000, 100000 }, 10, 0, true).refusal == R::invalid, "empty id");
    require(ledger.admit({ "x", "engine-a", 0, 100000 }, 10, 0, true).refusal == R::invalid, "zero period");
    require(ledger.admit({ "x", "engine-a", 1000, 0 }, 10, 0, true).refusal == R::invalid, "zero share is no learning");
    require(ledger.admit({ "x", "engine-a", 1000, 1000001 }, 10, 0, true).refusal == R::invalid, "share over 1e6");
    require(ledger.admit({ "x", "engine-b", 1000, 100000 }, 10, 0, true).refusal == R::stale_engine, "another incarnation");
    require(ledger.n_accounts() == 0, "refused grants register nothing");

    // reserve 60 ms, then a second 60 ms does not fit beside it
    const auto a1 = ledger.admit(g, 60000, 0, true);
    require(a1.refusal == R::none && a1.reservation != 0, "first step admitted");
    const auto a2 = ledger.admit(g, 60000, 1000, true);
    require(a2.refusal == R::budget_exhausted && a2.retry_after_ms == 999, "reserved time counts against the share");

    // the same id with another share is refused; the account is untouched
    require(ledger.admit({ "grant-1", "engine-a", 1000, 900000 }, 1, 1000, true).refusal == R::grant_mismatch, "grant immutable");

    // the step ran 30 ms: the rest of its reservation returns
    ledger.finish(a1.reservation, 30000, 40000);
    require(ledger.used_us("grant-1", 40000) == 30000, "debit is the measured time");
    ledger.finish(a1.reservation, 30000, 40000);
    require(ledger.used_us("grant-1", 40000) == 30000, "a settled reservation never debits twice");

    // 70 ms remain: a retry of the same request (same grant) spends the same account
    const auto a3 = ledger.admit(g, 70000, 41000, true);
    require(a3.refusal == R::none, "the remaining 70 ms fit");
    require(ledger.admit(g, 1, 42000, true).refusal == R::budget_exhausted, "a duplicate never replenishes");
    ledger.cancel(a3.reservation, 43000);
    require(ledger.used_us("grant-1", 43000) == 30000, "cancel releases the whole reservation");

    // an idle engine learns free, even past the share
    const auto free_step = ledger.admit(g, 500000, 44000, false);
    require(free_step.refusal == R::none, "uncontended step admitted");
    ledger.finish(free_step.reservation, 500000, 544000);
    require(ledger.used_us("grant-1", 544000) == 30000, "uncontended time is not debited");

    // a step larger than the whole period's share can never run: said so, not retried
    require(ledger.admit(g, 100001, 545000, true).refusal == R::never_fits, "never fits");

    // the next period renews the share; whole idle periods are not banked
    require(ledger.used_us("grant-1", 1000000) == 0, "new period, nothing spent");
    const auto a4 = ledger.admit(g, 100000, 1000000, true);
    require(a4.refusal == R::none, "a full period's share fits a fresh period");
    ledger.finish(a4.reservation, 100000, 1100000);

    // bounded: an account idle two periods with no reservation is forgotten at the next settle
    const auto other = ledger.admit({ "grant-2", "engine-a", 1000, 100000 }, 1000, 3500000, true);
    ledger.finish(other.reservation, 1000, 3501000);
    require(ledger.n_accounts() == 1, "the idle grant-1 account is forgotten");

    std::cout << "test-server-learn-ledger: OK\n";
    return 0;
}
