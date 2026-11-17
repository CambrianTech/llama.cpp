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

#define CHECK(cond) do { if (!(cond)) { fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #cond); exit(1); } } while (0)

int main() {
    // The prompt she was served: a user turn, then her generation prompt.
    const llama_tokens served = { 1, 40, 41, 9, 3, 2, 3, 7 };
    // What she generated after it: her reasoning (50, 51), her words (60), her call (61). No end of turn.
    llama_tokens held = served;
    held.insert(held.end(), { 50, 51, 60, 61 });
    const std::string response = "chatcmpl-a";

    // Her next request re-renders her response WITHOUT her reasoning (60, 61), then her result and the new turn.
    const llama_tokens sent = { 1, 40, 41, 9, 3, 2, 3, 60, 61, 9, 3, 1, 70, 71, 9, 3, 2, 3, 7 };
    const auto spans = spans_of(sent);

    // what this catches: her generated tokens (her reasoning) discarded and recomputed every act.
    {
        const auto c = continue_from({ held, served.size(), response }, { response, sent, spans }, is_eot);
        CHECK(c.miss == continuation_miss::none);
        llama_tokens expect = held;
        expect.insert(expect.end(), { 9, 3, 1, 70, 71, 9, 3, 2, 3, 7 });
        CHECK(c.tokens == expect);
        CHECK(c.from == 9); // the end of turn closing her response
        CHECK(c.shift == (int64_t) held.size() - 9);
        // A position past her response moves with the tail; one inside it is unknown; one before is kept.
        CHECK(continued_position(c, 11) == 11 + c.shift);
        CHECK(continued_position(c, 7) == -1);
        CHECK(continued_position(c, 3) == 3);
    }
    // what this catches: a request naming no response is served as sent.
    {
        const std::string none;
        CHECK(continue_from({ held, served.size(), response }, { none, sent, spans }, is_eot).miss == continuation_miss::not_requested);
    }
    // what this catches: another response's tokens standing in for the one the request carries.
    {
        const std::string other = "chatcmpl-b";
        CHECK(continue_from({ held, served.size(), response }, { other, sent, spans }, is_eot).miss == continuation_miss::other_response);
    }
    // what this catches: a slot that holds someone else's history (a different first turn) used as hers.
    {
        llama_tokens theirs = held;
        theirs[1] = 99;
        CHECK(continue_from({ theirs, served.size(), response }, { response, sent, spans }, is_eot).miss == continuation_miss::not_hers);
    }
    // what this catches: a slot that served the prompt but generated nothing being treated as holding her response.
    {
        CHECK(continue_from({ served, served.size(), response }, { response, sent, spans }, is_eot).miss == continuation_miss::not_generated);
    }
    // what this catches: the generation prompt (the request's last span) taken for her response.
    {
        const llama_tokens asking = { 1, 40, 41, 9, 3, 2, 3, 7 };
        CHECK(continue_from({ held, served.size(), response }, { response, asking, spans_of(asking) }, is_eot).miss == continuation_miss::no_response);
    }
    // what this catches: a request with no assistant message spliced anyway.
    {
        const llama_tokens no_reply = { 1, 40, 41, 9, 3 };
        CHECK(continue_from({ held, served.size(), response }, { response, no_reply, spans_of(no_reply) }, is_eot).miss == continuation_miss::no_response);
    }
    // what this catches: a template whose response carries no end of turn spliced at a guessed point.
    {
        const llama_tokens open = { 1, 40, 41, 9, 3, 2, 3, 60, 61, 1, 70, 2, 3 };
        CHECK(continue_from({ held, served.size(), response }, { response, open, spans_of(open) }, is_eot).miss == continuation_miss::no_end_of_turn);
    }
    fprintf(stderr, "test-server-continuation: ok\n");
    return 0;
}
