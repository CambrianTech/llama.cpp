#pragma once

#include "chat.h"
#include "common.h"
#include "llama.h"

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

// Continuum fork: a persona's next request continues the tokens her slot already holds, and carries
// only what is new (CONTINUUM-ENGINE 4.14, "her token history is the engine's").
//
// After her act her slot holds [the prompt she was served][the tokens she generated]: her reasoning,
// her words and her call, exactly as sampled, without the end-of-turn token that stopped her. A
// request that names the request her response answered (`continues`) carries ONLY the messages after
// that response. The server renders them behind a short stub of her response (a user line, then an
// assistant turn whose tool calls the new tool results answer), never her response itself, and takes
// the tokens from the end-of-turn that closes the stub: the template's own glue that closes her turn,
// then the new messages. Her request is then her held tokens up to her response's end plus that delta.
// Nothing she holds is re-rendered, re-tokenized, compared or recomputed.
//
// Anything that does not hold exactly is a typed miss, and the request is refused with it
// (`continuation_refused`): her held tokens and lineage are kept. A cold start is the caller's own
// request that names no continuation.

enum class continuation_miss {
    none,           // continued
    not_requested,  // the request names no response it continues
    other_response, // her slot's lineage holds no response by that id
    has_media,      // a multimodal prompt is not continued
    no_stub,        // the template's delimiters found no assistant turn for the stub
    no_end_of_turn, // the stub's assistant turn carries no end-of-turn token
    not_generated,  // the response it names generated nothing
};

inline const char * continuation_miss_name(continuation_miss miss) {
    switch (miss) {
        case continuation_miss::none:           return "none";
        case continuation_miss::not_requested:  return "not_requested";
        case continuation_miss::other_response: return "other_response";
        case continuation_miss::has_media:      return "has_media";
        case continuation_miss::no_stub:        return "no_stub";
        case continuation_miss::no_end_of_turn: return "no_end_of_turn";
        case continuation_miss::not_generated:  return "not_generated";
    }
    return "unknown";
}

// One response her slot holds: the caller's id for the request it answered, where the prompt she was
// served for it ends and where her response ends in the slot's tokens. A slot holds a lineage of them:
// a continuation keeps every earlier one as its prefix, any other request starts a new lineage.
struct continuation_response {
    std::string id;
    size_t      n_served = 0;
    size_t      n_end    = 0;
};

// Where a continuing request's new tokens start in its rendering (the stub, then its new messages):
// at the end-of-turn closing the stub's assistant turn, which closes her held response in her tokens.
struct continuation_delta {
    continuation_miss miss = continuation_miss::none;
    size_t            from = 0;
};

inline continuation_delta delta_after_stub(const llama_tokens & tokens, const common_chat_msg_spans & spans,
                                           const std::function<bool(llama_token)> & is_end_of_turn) {
    continuation_delta out;
    const common_chat_msg_span * stub = nullptr;
    for (const auto & span : spans.spans) {
        if (span.role == COMMON_CHAT_ROLE_ASSISTANT) {
            stub = &span;
            break;
        }
    }
    if (stub == nullptr) {
        out.miss = continuation_miss::no_stub;
        return out;
    }
    for (size_t i = stub->pos + stub->len; i > stub->pos; --i) {
        if (is_end_of_turn(tokens[i - 1])) {
            out.from = i - 1;
            return out;
        }
    }
    out.miss = continuation_miss::no_end_of_turn;
    return out;
}

struct continuation {
    continuation_miss miss      = continuation_miss::not_requested;
    llama_tokens      tokens;        // her held tokens up to her response's end, then the delta
    size_t            kept      = 0; // her held tokens kept: the prompt she was served and her response
    size_t            generated = 0; // of those, the tokens she generated
};

// Her request: the held tokens of the response it continues, then its delta. `responses` is her slot's
// lineage, oldest first; a continuation of an earlier one (her real request after a warm pass that
// continued the same response) keeps only that response's tokens, and the prompt cache finds the rest.
inline continuation continue_from(const llama_tokens & held, const std::vector<continuation_response> & responses,
                                  const std::string & continues, const llama_tokens & delta) {
    continuation out;
    if (continues.empty()) {
        return out;
    }
    const continuation_response * answered = nullptr;
    for (const auto & r : responses) {
        if (r.id == continues) {
            answered = &r;
        }
    }
    if (answered == nullptr || answered->n_end > held.size()) {
        out.miss = continuation_miss::other_response;
        return out;
    }
    if (answered->n_end <= answered->n_served) {
        out.miss = continuation_miss::not_generated;
        return out;
    }
    out.miss      = continuation_miss::none;
    out.kept      = answered->n_end;
    out.generated = answered->n_end - answered->n_served;
    out.tokens.reserve(answered->n_end + delta.size());
    out.tokens.insert(out.tokens.end(), held.begin(), held.begin() + answered->n_end);
    out.tokens.insert(out.tokens.end(), delta.begin(), delta.end());
    return out;
}
