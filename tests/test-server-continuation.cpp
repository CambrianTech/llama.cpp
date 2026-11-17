// A persona's next request continues the tokens her slot holds (tools/server/server-continuation.h).
// Synthetic tokens: 1 = <|im_start|>user, 2 = <|im_start|>assistant, 9 = <|im_end|> (end of turn),
// 3 = "\n", 7 = her generation prompt's <think>, 40.. = words.

#include "server-continuation.h"

#include <cstdio>
#include <cstdlib>

static bool is_eot(llama_token t) { return t == 9; }

static common_chat_msg_spans spans_of(const llama_tokens & tokens) {
    common_chat_msg_spans spans;
    std::vector<std::pair<common_chat_role, size_t>> at;
    for (size_t i = 0; i < tokens.size(); i++) {
        if (tokens[i] == 1) at.emplace_back(COMMON_CHAT_ROLE_USER, i);
        if (tokens[i] == 2) at.emplace_back(COMMON_CHAT_ROLE_ASSISTANT, i);
    }
    for (size_t k = 0; k < at.size(); k++) {
        const size_t end = k + 1 < at.size() ? at[k + 1].second : tokens.size();
        spans.add(at[k].first, at[k].second, end - at[k].second);
    }
    return spans;
}

static continuation run(const llama_tokens & held, const std::vector<continuation_response> & responses,
                        const std::string & continues, const llama_tokens & sent) {
    const auto spans = spans_of(sent);
    return continue_from({ held, responses }, { continues, sent, spans }, is_eot);
}

#define CHECK(cond) do { if (!(cond)) { fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #cond); exit(1); } } while (0)

int main() {
    // The prompt she was served: a user turn, then her generation prompt.
    const llama_tokens served = { 1, 40, 41, 9, 3, 2, 3, 7 };
    // What she generated after it: her reasoning (50, 51), her words (60), her call (61). No end of turn.
    llama_tokens held = served;
    held.insert(held.end(), { 50, 51, 60, 61 });
    const std::vector<continuation_response> act = { { "act-1", served.size(), held.size() } };

    // Her next request re-renders her response WITHOUT her reasoning (60, 61), then her result and the new turn.
    const llama_tokens sent = { 1, 40, 41, 9, 3, 2, 3, 60, 61, 9, 3, 1, 70, 71, 9, 3, 2, 3, 7 };

    // what this catches: her generated tokens (her reasoning) discarded and recomputed every act.
    {
        const auto c = run(held, act, "act-1", sent);
        CHECK(c.miss == continuation_miss::none);
        llama_tokens expect = held;
        expect.insert(expect.end(), { 9, 3, 1, 70, 71, 9, 3, 2, 3, 7 });
        CHECK(c.tokens == expect);
        CHECK(c.from == 9); // the end of turn closing her response
        CHECK(c.kept == held.size() && c.generated == 4);
        CHECK(c.shift == (int64_t) held.size() - 9);
        // A position past her response moves with the tail; one inside it is unknown; one before is kept.
        CHECK(continued_position(c, 11) == 11 + c.shift);
        CHECK(continued_position(c, 7) == -1);
        CHECK(continued_position(c, 3) == 3);
    }
    // what this catches: a warm pass during her tool (one token generated after her response's glue)
    // making her real request miss, or its token leaking into her prompt. Both continue act-1: the
    // slot's lineage still holds act-1, and only act-1's tokens are kept; the cache finds the glue.
    {
        llama_tokens warmed = held;
        warmed.insert(warmed.end(), { 9, 3, 2, 3, 7, 80 });
        const std::vector<continuation_response> lineage = { act[0], { "warm-1", warmed.size() - 1, warmed.size() } };
        const auto c = run(warmed, lineage, "act-1", sent);
        CHECK(c.miss == continuation_miss::none);
        CHECK(c.kept == held.size());
        llama_tokens expect = held;
        expect.insert(expect.end(), { 9, 3, 1, 70, 71, 9, 3, 2, 3, 7 });
        CHECK(c.tokens == expect);
    }
    // what this catches: a request naming no response is served as sent.
    CHECK(run(held, act, "", sent).miss == continuation_miss::not_requested);
    // what this catches: another response's tokens standing in for the one the request continues.
    CHECK(run(held, act, "act-2", sent).miss == continuation_miss::other_response);
    // what this catches: a lineage that outlived its tokens (the slot was cut shorter than her response).
    CHECK(run(served, act, "act-1", sent).miss == continuation_miss::other_response);
    // what this catches: a slot that holds someone else's history (a different first turn) used as hers.
    {
        llama_tokens theirs = held;
        theirs[1] = 99;
        CHECK(run(theirs, act, "act-1", sent).miss == continuation_miss::not_hers);
    }
    // what this catches: a response that generated nothing treated as holding her response.
    CHECK(run(held, { { "act-1", held.size(), held.size() } }, "act-1", sent).miss == continuation_miss::not_generated);
    // what this catches: a response whose served prompt ended before the request's response begins.
    CHECK(run(held, { { "act-1", 3, held.size() } }, "act-1", sent).miss == continuation_miss::not_hers);
    // what this catches: the generation prompt (the request's last span) taken for her response.
    CHECK(run(held, act, "act-1", { 1, 40, 41, 9, 3, 2, 3, 7 }).miss == continuation_miss::no_response);
    // what this catches: a template whose response carries no end of turn spliced at a guessed point.
    CHECK(run(held, act, "act-1", { 1, 40, 41, 9, 3, 2, 3, 60, 61, 1, 70, 2, 3 }).miss == continuation_miss::no_end_of_turn);
    fprintf(stderr, "test-server-continuation: ok\n");
    return 0;
}
