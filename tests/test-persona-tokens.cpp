// A persona's token history appends only a request's new ids, in one validated transaction
// (src/llama-persona-tokens.h, CONTINUUM-ENGINE 4.14).

#include "../src/llama-persona-tokens.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK(cond) do { if (!(cond)) { fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #cond); exit(1); } } while (0)

using status = llama_persona_append_status;

int main() {
    const llama_persona_request_id act1{ 1, 1 }, act2{ 1, 2 }, act3{ 1, 3 };

    // what this catches: her history rebuilt per act. Three acts: each request appends only its new ids
    // after what she generated; her history is exactly served + generated + new, in order.
    {
        llama_persona_tokens her(7, 1, 64);
        const llama_token prompt[] = { 1, 40, 9, 2, 7 };
        CHECK(her.append({ 7, 1, 0, act1 }, prompt, 5).status == status::appended);
        std::vector<llama_token> expect(prompt, prompt + 5);
        const llama_persona_request_id acts[] = { act2, act3, { 1, 4 } };
        for (int a = 0; a < 3; a++) {
            for (llama_token g : { (llama_token) (50 + a), (llama_token) (60 + a) }) { // her reasoning and call
                CHECK(her.generated(g));
                expect.push_back(g);
            }
            const llama_token delta[] = { 9, 1, (llama_token) (70 + a), 9, 2, 7 };
            const auto r = her.append({ 7, 1, her.frontier(), acts[a] }, delta, 6);
            CHECK(r.status == status::appended && r.from + 6 == r.to && r.to == her.frontier());
            expect.insert(expect.end(), delta, delta + 6);
        }
        CHECK(her.frontier() == expect.size());
        CHECK(std::equal(expect.begin(), expect.end(), her.data()));
    }
    // what this catches: a lost response read as permission to append again. The same request returns
    // its receipt and writes nothing; the same identity with another delta is refused.
    {
        llama_persona_tokens her(7, 1, 16);
        const llama_token delta[] = { 1, 2, 3 };
        const auto first = her.append({ 7, 1, 0, act1 }, delta, 3);
        const auto again = her.append({ 7, 1, 0, act1 }, delta, 3);
        CHECK(again.status == status::replayed && again.from == first.from && again.to == first.to && her.frontier() == 3);
        const llama_token other[] = { 1, 2, 4 };
        CHECK(her.append({ 7, 1, 0, act1 }, other, 3).status == status::changed && her.frontier() == 3);
    }
    // what this catches: a stale or foreign request appended to her history.
    {
        llama_persona_tokens her(7, 2, 16);
        const llama_token delta[] = { 1 };
        CHECK(her.append({ 8, 2, 0, act1 }, delta, 1).status == status::wrong_persona);
        CHECK(her.append({ 7, 1, 0, act1 }, delta, 1).status == status::wrong_epoch);
        CHECK(her.append({ 7, 2, 5, act1 }, delta, 1).status == status::moved);
        CHECK(her.append({ 7, 2, 0, {} }, delta, 1).status == status::empty_request);
        CHECK(her.frontier() == 0);
    }
    // what this catches: an append past her reserved capacity growing it in place (an allocation and a copy
    // of what she holds on the hot path): refused, her owner grows first.
    {
        llama_persona_tokens her(7, 1, 4);
        const llama_token delta[] = { 1, 2, 3, 4, 5 };
        CHECK(her.append({ 7, 1, 0, act1 }, delta, 5).status == status::no_room && her.frontier() == 0);
    }
    // The cost of an act's append at her real size: a 100k-token history, a 2k-token delta. Printed, not
    // asserted: the receipt for the design's "a memcpy of the delta" (no prefix touched).
    {
        llama_persona_tokens her(7, 1, 1 << 20);
        std::vector<llama_token> prompt(100000, 5);
        CHECK(her.append({ 7, 1, 0, act1 }, prompt.data(), prompt.size()).status == status::appended);
        std::vector<llama_token> delta(2048, 6);
        volatile llama_token sink = 0;
        const auto t0 = std::chrono::steady_clock::now();
        uint64_t n = 0;
        for (uint64_t i = 0; i < 100; i++) {
            const auto r = her.append({ 7, 1, her.frontier(), { 2, i + 1 } }, delta.data(), delta.size());
            sink = her.data()[r.to - 1];
            n += r.status == status::appended;
        }
        const double ns = std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count() / 100;
        CHECK(n == 100);
        // Read what was written, or the optimizer removes the copies and the number is a lie.
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
