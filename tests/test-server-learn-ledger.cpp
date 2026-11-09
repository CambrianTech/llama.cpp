// what this catches: the learning ledger's contract (server-learn-ledger.h) without a model, in
// milliseconds: authority-only registration, immutable live grants, no replenishment by retry,
// free reservations as live custody, cancel only before start, checked ranges, bounded accounts.
#include "../tools/server/server-learn-ledger.h"
#include "../tools/server/server-learn-residency.h"

#include <climits>
#include <cstdlib>
#include <iostream>

static void require(bool condition, const char * message) {
    if (!condition) {
        std::cerr << "FAILED: " << message << '\n';
        std::exit(1);
    }
}

using R = server_learn_refusal;

static int64_t used(const server_learn_ledger & l, const char * grant, int64_t now) {
    int64_t out = -1;
    require(l.used_us(grant, now, out) == R::none, "a valid observation");
    return out;
}

static void residency_custody() {
    server_residency_mirror mirror("engine-a");
    server_residency r{"r1", "engine-a", "core-a", "shadow-a",
        {{"CUDA0", "CUDA0", "weights", 128, false}, {"host", "CPU", "optimizer_host_required", 32, true}},
        {{"vram", "vram-1", 128}, {"ram", "ram-1", 32}}};
    auto bad = r; bad.engine = "old";
    require(mirror.register_residency(bad) == "stale_engine", "residency binds engine");
    bad = r; bad.leases[0].bytes = 127;
    require(mirror.register_residency(bad) == "uncovered", "no bytes without physical lease");
    bad = r; bad.rows.push_back(r.rows[0]);
    require(mirror.register_residency(bad) == "invalid", "duplicate rows cannot mint coverage");
    require(mirror.register_residency(r) == "none", "register residency");
    require(mirror.register_residency(r) == "none", "idempotent registration");
    bad = r; bad.owner = "core-b";
    require(mirror.register_residency(bad) == "residency_mismatch", "new core cannot adopt silently");
    bad = r; bad.id = "r2";
    require(mirror.register_residency(bad) == "lease_already_bound", "one lease cannot fund two residencies");
    require(mirror.claim("r1", "shadow-b") == "subject_mismatch", "subject checked before allocation");
    require(mirror.claim("r1", "shadow-a") == "none", "claim preallocation custody");
    require(mirror.claim("r1", "shadow-a") == "busy", "concurrent duplicate cannot allocate twice");
    require(mirror.release("r1") == "busy", "cannot release while loading");
    require(mirror.allows("r1", r.rows[0]), "exact aligned bytes admitted");
    auto larger = r.rows[0]; ++larger.bytes;
    require(!mirror.allows("r1", larger), "larger allocation refused");
    mirror.finish("r1", true, 3);
    require(mirror.revoke("r1") == "none", "revoke allocation authority");
    require(!mirror.allows("r1", r.rows[0]), "revocation stops new allocation");
    require(mirror.release("r1") == "busy", "revocation does not free resident bytes");
    mirror.freed("r1");
    require(mirror.release("r1") == "none", "release after actual free");
    require(mirror.register_residency(r) == "retired", "retired reference cannot be reused");
}

static void residency_plan() {
    server_residency_mirror mirror("engine");
    server_residency optimizer;
    optimizer.id = "optimizer"; optimizer.engine = "engine"; optimizer.owner = "core";
    optimizer.subject = "mind-a";
    optimizer.rows = {{"GPU", "CUDA", "optimizer_static_required", 64, false}};
    optimizer.leases = {{"vram", "lease-opt", 64}};
    auto workspace = optimizer;
    workspace.id = "workspace"; workspace.subject = "workspace";
    workspace.rows[0].kind = "shared_compute_required";
    workspace.leases[0].lease_id = "lease-workspace";
    require(mirror.register_residency(optimizer) == "none" &&
            mirror.register_residency(workspace) == "none", "register separate allocation owners");
    std::vector<server_residency_need> plan = {
        {optimizer.id, optimizer.subject, optimizer.rows},
        {workspace.id, workspace.subject, workspace.rows}};
    std::vector<std::string> acquired;
    ++plan[1].rows[0].bytes;
    require(mirror.claim_plan(plan, acquired) == "uncovered" && acquired.empty(), "second refusal rolls back whole plan");
    for (const auto & row : mirror.snapshot()) { require(!row.pending && !row.live, "failed plan changes no custody"); }
    --plan[1].rows[0].bytes;
    plan[1].rows.push_back(plan[1].rows[0]);
    require(mirror.claim_plan(plan, acquired) == "invalid", "duplicate requirement cannot multiply one allowance");
    plan[1].rows.pop_back();
    require(mirror.claim_plan(plan, acquired) == "none" && acquired.size() == 2, "whole plan acquires custody");
    require(mirror.claim_plan(plan, acquired) == "busy", "pending plan cannot be reused concurrently");
    mirror.finish(optimizer.id, true, 7);
    mirror.finish(workspace.id, true);
    require(mirror.claim_plan(plan, acquired) == "none" && acquired.empty(), "reuse live owners without another lease");
    mirror.freed_lora(7);
    require(mirror.release(optimizer.id) == "none", "mind unload frees only optimizer");
    require(mirror.release(workspace.id) == "busy", "engine workspace survives mind unload");
    require(mirror.revoke(workspace.id) == "none", "revoke workspace authority");
    require(mirror.claim_plan({plan[1]}, acquired) == "unknown_residency", "revoked live workspace refuses new work");
    require(mirror.release(workspace.id) == "busy", "revocation does not free workspace");
    // What this catches: a larger lease must take custody of retained bytes as
    // well as the next graph, atomically, even when the former lease is revoked.
    auto replacement = workspace;
    replacement.id = "replacement";
    replacement.rows[0].bytes = 63;
    replacement.leases[0] = {"vram", "lease-replacement", 128};
    require(mirror.register_residency(replacement) == "none", "register short replacement");
    server_residency_need next{replacement.id, "workspace", replacement.rows};
    require(mirror.claim_plan({next}, acquired, workspace.id) == "uncovered", "old retained bytes must be covered");
    require(mirror.release(workspace.id) == "busy", "failed replacement retains old custody");
    require(mirror.release(replacement.id) == "none", "failed replacement claims nothing");
    replacement.id = "grown";
    replacement.rows[0].bytes = 128;
    require(mirror.register_residency(replacement) == "none", "register full replacement");
    next = {replacement.id, "workspace", replacement.rows};
    require(mirror.claim_plan({next}, acquired, workspace.id) == "none" && acquired.empty(), "transfer existing allocation without new pending allocation");
    require(mirror.release(workspace.id) == "none", "old reference releasable after custody transfer");
    require(mirror.allows(replacement.id, workspace.rows[0]), "retained bytes belong to replacement");
    require(mirror.release(replacement.id) == "busy", "new owner retains workspace");
    mirror.freed(replacement.id);
    require(mirror.release(replacement.id) == "none", "actual workspace free permits release");
}

int main() {
    residency_plan();
    residency_custody();
    server_learn_ledger ledger("engine-a");
    const int64_t day_us = 86400000000;
    // 10% of a 1000 ms period: 100 ms of contended learning per period, for a day
    const server_learn_grant g{ "grant-1", "engine-a", 1000, 100000, day_us };

    // registration is the authority's, with checked ranges
    require(ledger.register_grant({ "", "engine-a", 1000, 100000, day_us }, 0) == R::invalid, "empty id");
    require(ledger.register_grant({ "x", "engine-a", 0, 100000, day_us }, 0) == R::invalid, "zero period");
    require(ledger.register_grant({ "x", "engine-a", server_learn_ledger::kMaxPeriodMs + 1, 100000, day_us }, 0) == R::invalid,
            "a period past the checked range");
    require(ledger.register_grant({ "x", "engine-a", 1000, -1, day_us }, 0) == R::invalid, "negative share invalid");
    {
    server_learn_ledger calibration("engine-a");
    // What this catches: calibration authority must never become a tiny guessed
    // contended quota, or admit an unmeasured zero-cost contended step.
    require(calibration.register_grant({ "bootstrap", "engine-a", 0, 0, day_us }, 0) == R::none, "explicit idle calibration grant");
    require(calibration.admit("bootstrap", 0, 0, true).refusal == R::uncontended_only, "zero estimate cannot bypass idle-only authority");
    require(calibration.admit("bootstrap", 1, 0, true).refusal == R::uncontended_only, "measured contention also refused");
    auto bootstrap = calibration.admit("bootstrap", 0, 0, false);
    require(bootstrap.refusal == R::none && bootstrap.reservation != 0, "idle calibration retains custody");
    require(calibration.start(bootstrap.reservation, 0) == R::none, "start idle measurement");
    require(calibration.finish(bootstrap.reservation, 100, 0, 100) == R::none, "settle idle measurement");
    auto restoration = calibration.hold_restoration("bootstrap", 100);
    require(restoration.refusal == R::none && calibration.start(restoration.reservation, 100) == R::none, "restoration custody with zero contended authority");
    require(calibration.finish(restoration.reservation, 50, 0, 150) == R::none, "restoration remains measured");
    require(used(calibration, "bootstrap", 150) == 50, "do not erase restoration debit");
    auto idle_again = calibration.admit("bootstrap", 100, 150, false);
    require(idle_again.refusal == R::none, "measured restoration must not starve subsequent idle learning");
    require(calibration.cancel(idle_again.reservation, 150) == R::none, "release unused idle custody");
    require(calibration.register_grant({ "bootstrap", "engine-a", 1000, 100, day_us }, 100) == R::grant_mismatch, "calibrated authority needs a new immutable id");
    }
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
    require(ledger.finish(a1.reservation, 30000, 1, 900) == R::not_started, "finish before start is refused");
    require(ledger.start(a1.reservation, 1000) == R::none, "start");
    require(ledger.start(a1.reservation, 1001) == R::already_started, "one reservation authorizes one execution");
    require(ledger.cancel(a1.reservation, 30000) == R::started, "a started step cannot be cancelled");
    require(ledger.finish(a1.reservation, 30000, 1, 40000) == R::none, "it settles after quiescence");
    require(used(ledger, "grant-1", 40000) == 30000 && ledger.committed_steps("grant-1") == 1, "debit and steps measured");
    require(ledger.finish(a1.reservation, 30000, 1, 40000) == R::unknown_reservation, "never settled twice");

    // 70 ms remain: a retry of the same request spends the same account
    const auto a3 = ledger.admit("grant-1", 70000, 41000, true);
    require(a3.refusal == R::none, "the remaining 70 ms fit");
    require(ledger.admit("grant-1", 1, 42000, true).refusal == R::budget_exhausted, "a duplicate never replenishes");
    require(ledger.finish(a3.reservation, 70000, 1, 42500) == R::not_started, "a never-started step cannot settle committed work");
    require(ledger.cancel(a3.reservation, 43000) == R::none, "never started: cancelled (custody kept for it)");
    require(used(ledger, "grant-1", 43000) == 30000, "cancel releases the whole reservation");

    // a step larger than the whole period's share can never run: said so, not retried
    require(ledger.admit("grant-1", 100001, 44000, true).refusal == R::never_fits, "never fits");

    // a free (uncontended) reservation is live custody: revoking the grant does not erase its
    // account while it is outstanding, and its later settlement still lands
    const auto free_step = ledger.admit("grant-1", 500000, 45000, false);
    require(free_step.refusal == R::none, "uncontended step admitted");
    ledger.revoke("grant-1", 46000);
    require(ledger.n_accounts() == 1, "a live free reservation keeps its account through revocation");
    require(ledger.admit("grant-1", 1, 47000, true).refusal == R::unknown_grant, "a revoked grant admits nothing new");
    require(ledger.start(free_step.reservation, 46500) == R::unknown_grant, "a revoked grant's reservation does not start");
    require(ledger.cancel(free_step.reservation, 3 * 1000000) == R::none, "it never ran: cancelled, custody released");
    require(ledger.n_accounts() == 0, "then the revoked account goes");

    // admitted, then the grant expires before the step starts: refused, cancellable
    require(ledger.register_grant({ "brief", "engine-a", 1000, 100000, 3500000 }, 3000000) == R::none, "a brief grant");
    const auto late = ledger.admit("brief", 1000, 3100000, true);
    require(late.refusal == R::none, "admitted before expiry");
    require(ledger.start(late.reservation, 3600000) == R::unknown_grant, "expired before start: refused");
    require(ledger.cancel(late.reservation, 3600000) == R::none, "and cancelled, never erased silently");

    // a started step still settles after its grant ends
    require(ledger.register_grant({ "ending", "engine-a", 1000, 100000, 3900000 }, 3700000) == R::none, "an ending grant");
    const auto running = ledger.admit("ending", 1000, 3700000, true);
    require(ledger.start(running.reservation, 3700000) == R::none, "started while live");
    require(ledger.finish(running.reservation, 1000, 1, 4000000) == R::none, "settles after the grant expired");

    // the clock domain and the per-settle step bound are explicit, checked refusals
    require(ledger.admit("grant-x", 1, -1, true).refusal == R::invalid, "a negative clock");
    require(ledger.admit("grant-x", 1, server_learn_ledger::kMaxClockUs + 1, true).refusal == R::invalid, "a clock past the domain");
    require(ledger.register_grant({ "far", "engine-a", 1000, 100000, server_learn_ledger::kMaxClockUs + 1 }, 4000000) == R::invalid,
            "an expiry past the domain");
    require(ledger.register_grant({ "steps", "engine-a", 1000, 100000, day_us }, 4000000) == R::none, "steps grant");
    const auto big = ledger.admit("steps", 1, 4000000, true);
    require(ledger.start(big.reservation, 4000000) == R::none, "start");
    require(ledger.finish(big.reservation, 1, INT64_MAX, 4000001) == R::invalid, "INT64_MAX steps in one settle is refused, not wrapped");
    require(ledger.finish(big.reservation, 1, 1, 4000001) == R::none, "a valid settle still lands");

    // bounded without idleness eviction: an expired grant with nothing outstanding goes at the
    // next registration, and a live idle one stays (its terms cannot be reset)
    server_learn_ledger fresh("engine-a");
    require(fresh.register_grant({ "short", "engine-a", 1000, 100000, 2000000 }, 1000000) == R::none, "short grant");
    require(fresh.register_grant({ "idle", "engine-a", 1000, 100000, day_us }, 1000000) == R::none, "idle grant");
    require(fresh.register_grant({ "next", "engine-a", 1000, 100000, day_us }, 5000000) == R::none, "a later registration");
    require(fresh.n_accounts() == 2, "the expired grant is gone, the idle live one stays");
    require(fresh.register_grant({ "idle", "engine-a", 1000, 500000, day_us }, 9000000) == R::grant_mismatch,
            "an idle live grant keeps its terms");

    // an observation outside the clock domain is refused and changes nothing
    {
        server_learn_ledger obs("engine-a");
        require(obs.register_grant({ "o", "engine-a", 1000, 100000, day_us }, 0) == R::none, "obs grant");
        const auto o1 = obs.admit("o", 50000, 0, true);
        require(obs.start(o1.reservation, 0) == R::none && obs.finish(o1.reservation, 50000, 1, 10) == R::none, "debit 50 ms");
        int64_t out = 0;
        require(obs.used_us("o", INT64_MAX, out) == R::invalid, "a clock past the domain is refused");
        require(used(obs, "o", 20) == 50000, "and the period it observed is unchanged");
        require(obs.admit("o", 60000, 30, true).refusal == R::budget_exhausted, "admission still sees the spent period");
    }

    std::cout << "test-server-learn-ledger: OK\n";
    return 0;
}
