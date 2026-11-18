// A persona's token history appends only a request's new ids, in one validated transaction
// (src/llama-persona-tokens.h, CONTINUUM-ENGINE 4.14).

#include "../src/llama-persona-tokens.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <atomic>
#include <new>

// Every allocation in this test binary is counted, so "an accepted append never allocates" is measured, not stated.
static std::atomic<uint64_t> g_allocations { 0 };
void * operator new(std::size_t n) {
    g_allocations.fetch_add(1, std::memory_order_relaxed);
    if (void * p = std::malloc(n ? n : 1)) {
        return p;
    }
    throw std::bad_alloc();
}
void operator delete(void * p) noexcept { std::free(p); }
void operator delete(void * p, std::size_t) noexcept { std::free(p); }

#define CHECK(cond) do { if (!(cond)) { fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #cond); exit(1); } } while (0)

using status = llama_persona_append_status;

static llama_persona_request_id id(uint64_t n) { return { 1, n }; }

static llama_persona_append_request header(uint64_t persona, uint64_t epoch, uint64_t seq, uint64_t frontier, llama_persona_request_id request) {
    llama_persona_append_request h;
    h.persona  = persona;
    h.epoch    = epoch;
    h.seq      = seq;
    h.frontier = frontier;
    h.request  = request;
    return h;
}

int main() {
    // what this catches: her history rebuilt per act. Three acts: each request appends only its new ids
    // after what she generated; her history is exactly served + generated + new, in order.
    {
        llama_persona_tokens her(7, 1, 64, 16);
        const llama_token prompt[] = { 1, 40, 9, 2, 7 };
        CHECK(her.append(header(7, 1, 1, 0, id(1)), prompt, 5).status == status::appended);
        std::vector<llama_token> expect(prompt, prompt + 5);
        for (uint64_t a = 0; a < 3; a++) {
            for (llama_token g : { (llama_token) (50 + a), (llama_token) (60 + a) }) { // her reasoning and call
                CHECK(her.generated(g));
                expect.push_back(g);
            }
            const llama_token delta[] = { 9, 1, (llama_token) (70 + a), 9, 2, 7 };
            const auto r = her.append(header(7, 1, a + 2, her.frontier(), id(a + 2)), delta, 6);
            CHECK(r.status == status::appended && r.from + 6 == r.to && r.to == her.frontier());
            expect.insert(expect.end(), delta, delta + 6);
        }
        CHECK(her.frontier() == expect.size() && her.committed() == 4);
        CHECK(std::equal(expect.begin(), expect.end(), her.data()));
    }
    // what this catches: a lost response read as permission to append again, however late the retry comes.
    // A then B commit; a delayed retry of A (after B) gets A's original receipt and writes nothing; so does a
    // duplicate of B; A's sequence with another request or delta is refused.
    {
        llama_persona_tokens her(7, 1, 16, 16);
        const llama_token a[] = { 1, 2, 3 }, b[] = { 4, 5 }, other[] = { 1, 2, 4 };
        const auto ra = her.append(header(7, 1, 1, 0, id(1)), a, 3);
        const auto rb = her.append(header(7, 1, 2, 3, id(2)), b, 2);
        CHECK(ra.status == status::appended && rb.status == status::appended && her.frontier() == 5);
        const auto late_a = her.append(header(7, 1, 1, 0, id(1)), a, 3);
        CHECK(late_a.status == status::replayed && late_a.from == ra.from && late_a.to == ra.to);
        CHECK(her.append(header(7, 1, 2, 3, id(2)), b, 2).status == status::replayed);
        CHECK(her.append(header(7, 1, 1, 0, id(1)), other, 3).status == status::changed);
        CHECK(her.append(header(7, 1, 1, 0, id(9)), a, 3).status == status::changed);
        CHECK(her.frontier() == 5 && her.committed() == 2);
    }
    // what this catches: empty appends at the same frontier letting a retried request be accepted twice.
    // Each empty append commits its own sequence; a retry of the first is its receipt, never a new append.
    {
        llama_persona_tokens her(7, 1, 16, 16);
        CHECK(her.append(header(7, 1, 1, 0, id(1)), nullptr, 0).status == status::appended);
        CHECK(her.append(header(7, 1, 2, 0, id(2)), nullptr, 0).status == status::appended);
        CHECK(her.append(header(7, 1, 1, 0, id(1)), nullptr, 0).status == status::replayed);
        CHECK(her.committed() == 2 && her.frontier() == 0);
    }
    // what this catches: a reordered request (a later sequence arriving first) appended out of order.
    {
        llama_persona_tokens her(7, 1, 16, 16);
        const llama_token d[] = { 1 };
        CHECK(her.append(header(7, 1, 2, 0, id(2)), d, 1).status == status::out_of_order);
        CHECK(her.append(header(7, 1, 1, 0, id(1)), d, 1).status == status::appended);
        CHECK(her.append(header(7, 1, 2, 1, id(2)), d, 1).status == status::appended);
        CHECK(her.frontier() == 2);
    }
    // what this catches: a stale, foreign or malformed request appended to her history.
    {
        llama_persona_tokens her(7, 2, 16, 16);
        const llama_token d[] = { 1 };
        CHECK(her.append(header(8, 2, 1, 0, id(1)), d, 1).status == status::wrong_persona);
        CHECK(her.append(header(7, 1, 1, 0, id(1)), d, 1).status == status::wrong_epoch);
        CHECK(her.append(header(7, 2, 1, 5, id(1)), d, 1).status == status::moved);
        CHECK(her.append(header(7, 2, 1, 0, {}), d, 1).status == status::invalid);
        CHECK(her.append(header(7, 2, 0, 0, id(1)), d, 1).status == status::invalid);
        CHECK(her.append(header(7, 2, 1, 0, id(1)), nullptr, 1).status == status::invalid);
        CHECK(her.frontier() == 0 && her.committed() == 0);
    }
    // what this catches: an append past her reserved capacity growing it in place (an allocation and a copy
    // of what she holds on the hot path): refused, her owner grows first; a length that would overflow too.
    {
        llama_persona_tokens her(7, 1, 4, 16);
        const llama_token d[] = { 1, 2, 3, 4, 5 };
        CHECK(her.append(header(7, 1, 1, 0, id(1)), d, 5).status == status::no_room && her.frontier() == 0);
        CHECK(her.append(header(7, 1, 1, 0, id(1)), d, SIZE_MAX).status == status::no_room);
    }
    // what this catches: an accepted append allocating its receipt on the hot path, or a refused one leaving
    // a half-written append. Receipts are reserved; when they are used up the append is refused before any
    // write, an empty append included, and her tokens, frontier and commits are unchanged.
    {
        llama_persona_tokens her(7, 1, 16, 2);
        const llama_token d[] = { 1, 2 };
        CHECK(her.append(header(7, 1, 1, 0, id(1)), d, 2).status == status::appended);
        CHECK(her.append(header(7, 1, 2, 2, id(2)), nullptr, 0).status == status::appended);
        CHECK(her.append(header(7, 1, 3, 2, id(3)), d, 2).status == status::no_room);
        CHECK(her.append(header(7, 1, 3, 2, id(3)), nullptr, 0).status == status::no_room);
        CHECK(her.frontier() == 2 && her.committed() == 2 && her.data()[0] == 1 && her.data()[1] == 2);
        CHECK(her.append(header(7, 1, 1, 0, id(1)), d, 2).status == status::replayed);
    }
    // what this catches (Codex on #147 at f3eb: fixed-capacity evidence): any allocation on the append path. At her
    // real size, after the one reserve, accepted, refused (no room, wrong epoch, stale frontier) and replayed
    // appends together allocate nothing: her capacity and receipts are fixed at reserve and the hot path only writes.
    {
        llama_persona_tokens her(7, 1, 1 << 17, 64);
        std::vector<llama_token> prompt(100000, 5);
        std::vector<llama_token> delta(2048, 6);
        std::vector<llama_token> too_much(1 << 17, 7);
        const uint64_t before = g_allocations.load();
        CHECK(her.append(header(7, 1, 1, 0, id(1)), prompt.data(), prompt.size()).status == status::appended);
        for (uint64_t i = 0; i < 8; i++) {
            CHECK(her.append(header(7, 1, i + 2, her.frontier(), id(i + 2)), delta.data(), delta.size()).status == status::appended);
        }
        CHECK(her.append(header(7, 1, 10, her.frontier(), id(10)), too_much.data(), too_much.size()).status == status::no_room);
        CHECK(her.append(header(7, 2, 10, her.frontier(), id(10)), delta.data(), delta.size()).status == status::wrong_epoch);
        CHECK(her.append(header(7, 1, 10, 0, id(10)), delta.data(), delta.size()).status != status::appended);
        CHECK(her.append(header(7, 1, 1, 0, id(1)), prompt.data(), prompt.size()).status == status::replayed);
        CHECK(g_allocations.load() == before);
        fprintf(stderr, "fixed capacity: %llu appends and refusals at a %llu-token history, 0 allocations\n",
                (unsigned long long) 13, (unsigned long long) her.frontier());
    }
    // The cost of an act's append at her real size: a 100k-token history, a 2k-token delta. Printed, not
    // asserted; the writes are read back so the optimizer cannot remove them.
    {
        llama_persona_tokens her(7, 1, 1 << 20, 128);
        std::vector<llama_token> prompt(100000, 5);
        CHECK(her.append(header(7, 1, 1, 0, id(1)), prompt.data(), prompt.size()).status == status::appended);
        std::vector<llama_token> delta(2048, 6);
        volatile llama_token sink = 0;
        const auto t0 = std::chrono::steady_clock::now();
        uint64_t n = 0;
        for (uint64_t i = 0; i < 100; i++) {
            const auto r = her.append(header(7, 1, i + 2, her.frontier(), id(i + 2)), delta.data(), delta.size());
            sink = her.data()[r.to - 1];
            n += r.status == status::appended;
        }
        const double ns = std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count() / 100;
        (void) sink;
        CHECK(n == 100);
        uint64_t sum = 0;
        for (uint64_t i = 0; i < her.frontier(); i++) {
            sum += (uint64_t) her.data()[i];
        }
        CHECK(sum == 100000ull * 5 + 100ull * 2048 * 6);
        fprintf(stderr, "append of a 2048-token delta onto a %llu-token history: %.0f ns each\n",
                (unsigned long long) prompt.size(), ns);
    }
    fprintf(stderr, "test-persona-tokens: ok\n");
    return 0;
}
