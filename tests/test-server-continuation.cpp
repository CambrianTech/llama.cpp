// A persona's next request carries only what is new, and continues the tokens her slot holds
// (tools/server/server-continuation.h). Synthetic tokens: 1 = <|im_start|>user,
// 2 = <|im_start|>assistant, 9 = <|im_end|> (end of turn), 3 = "\n", 7 = <think>, 40.. = words.

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

// The server's rendering of a continuing request: the stub (a user line, an assistant turn), then
// the request's new messages and the generation prompt; its delta starts at the stub's end of turn.
static llama_tokens delta_of(const llama_tokens & new_messages) {
    llama_tokens rendered = { 1, 30, 9, 3, 2, 31, 9, 3 };
    rendered.insert(rendered.end(), new_messages.begin(), new_messages.end());
    rendered.insert(rendered.end(), { 2, 3, 7 });
    const auto d = delta_after_stub(rendered, spans_of(rendered), is_eot);
    if (d.miss != continuation_miss::none) {
        fprintf(stderr, "delta missed: %s\n", continuation_miss_name(d.miss));
        exit(1);
    }
    return llama_tokens(rendered.begin() + d.from, rendered.end());
}

#define CHECK(cond) do { if (!(cond)) { fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #cond); exit(1); } } while (0)

int main() {
    // what this catches: the stub itself (or anything before its end of turn) entering her tokens.
    {
        const llama_tokens delta = delta_of({ 1, 70, 9, 3 });
        CHECK((delta == llama_tokens{ 9, 3, 1, 70, 9, 3, 2, 3, 7 })); // her turn's glue, the new turn, the generation prompt
    }
    // what this catches: Codex's act-3 miss. Three consecutive acts, each generating reasoning the
    // caller never sends back: every act continues from her own tokens, nothing of hers is
    // re-rendered or compared, and her slot after act 3 is exactly what she generated plus what was new.
    {
        llama_tokens slot = { 1, 40, 9, 3, 2, 3, 7 }; // act 1's prompt
        std::vector<continuation_response> lineage;
        llama_tokens expect = slot;
        for (llama_token act = 1; act <= 3; act++) {
            const size_t served = slot.size();
            const llama_tokens generated = { (llama_token) (50 + act), (llama_token) (60 + act) }; // her reasoning, then her call
            slot.insert(slot.end(), generated.begin(), generated.end());
            expect.insert(expect.end(), generated.begin(), generated.end());
            lineage.push_back({ "act-" + std::to_string(act), served, slot.size() });
            // Her next request: only her result and the room's new line.
            const llama_tokens delta = delta_of({ 1, (llama_token) (70 + act), 9, 3 });
            const auto c = continue_from(slot, lineage, "act-" + std::to_string(act), delta);
            CHECK(c.miss == continuation_miss::none);
            CHECK(c.kept == slot.size() && c.generated == 2);
            expect.insert(expect.end(), delta.begin(), delta.end());
            CHECK(c.tokens == expect);
            slot = c.tokens; // served: her slot holds the continued prompt, her next generation follows it
        }
    }
    // what this catches: a warm pass during her tool (one token generated after her response's
    // glue) making her real request miss, or its token leaking into her prompt.
    {
        llama_tokens slot = { 1, 40, 9, 3, 2, 3, 7, 51, 61 };
        std::vector<continuation_response> lineage = { { "act-1", 7, slot.size() } };
        slot.insert(slot.end(), { 9, 3, 2, 3, 7, 80 });
        lineage.push_back({ "warm-1", slot.size() - 1, slot.size() });
        const llama_tokens delta = delta_of({ 1, 71, 9, 3 });
        const auto c = continue_from(slot, lineage, "act-1", delta);
        CHECK(c.miss == continuation_miss::none && c.kept == 9);
        llama_tokens expect = { 1, 40, 9, 3, 2, 3, 7, 51, 61 };
        expect.insert(expect.end(), delta.begin(), delta.end());
        CHECK(c.tokens == expect);
    }
    const llama_tokens held = { 1, 40, 9, 3, 2, 3, 7, 51, 61 };
    const std::vector<continuation_response> act = { { "act-1", 7, held.size() } };
    // what this catches: a request naming no response treated as a continuation.
    CHECK(continue_from(held, act, "", { 9 }).miss == continuation_miss::not_requested);
    // what this catches: another response's tokens standing in for the one the request continues.
    CHECK(continue_from(held, act, "act-2", { 9 }).miss == continuation_miss::other_response);
    // what this catches: a lineage that outlived its tokens (the slot was cut shorter than her response).
    CHECK(continue_from(llama_tokens(held.begin(), held.begin() + 5), act, "act-1", { 9 }).miss == continuation_miss::other_response);
    // what this catches: a response that generated nothing treated as holding her response.
    CHECK(continue_from(held, { { "act-1", held.size(), held.size() } }, "act-1", { 9 }).miss == continuation_miss::not_generated);
    // what this catches: a template with no assistant delimiter cut at a guessed point.
    {
        const llama_tokens no_assistant = { 1, 30, 9, 3, 1, 70, 9, 3 };
        CHECK(delta_after_stub(no_assistant, spans_of(no_assistant), is_eot).miss == continuation_miss::no_stub);
    }
    // what this catches: a stub turn with no end of turn cut at a guessed point.
    {
        const llama_tokens open = { 1, 30, 9, 3, 2, 31, 1, 70 };
        CHECK(delta_after_stub(open, spans_of(open), is_eot).miss == continuation_miss::no_end_of_turn);
    }
    fprintf(stderr, "test-server-continuation: ok\n");
    return 0;
}
