#pragma once

#include "chat.h"
#include "common.h"
#include "llama.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

// Continuum fork: a persona's next request continues the tokens her slot already holds, never
// re-rendered, re-tokenized or recomputed (CONTINUUM-ENGINE 4.14, "her token history is the
// engine's").
//
// After her act her slot holds [the prompt she was served][the tokens she generated]: her reasoning,
// her words and her call, exactly as sampled, without the end-of-turn token that stopped her. Her
// next request re-renders that response from its parsed message, without her reasoning, so its tokens
// differ from hers at her own response; a hybrid model's recurrent state cannot truncate there and
// rewinds to a checkpoint (the M5, 2026-11-17: 2,048 tokens recomputed on every act). A request that
// names the response it continues (`continues`, the caller's id for the request her last response answered) is
// instead served as her held tokens followed by the request's tokens from the end-of-turn that closes
// that response: the template's own glue, then everything new.
//
// Anything that does not hold exactly is a typed miss, said by the caller: the request is then served
// as sent.

enum class continuation_miss {
    none,           // continued
    not_requested,  // the request names no response it continues
    other_response, // her slot's last response is not the one the request continues
    has_media,      // a multimodal prompt is not spliced (checked by the caller: its tokens are not plain)
    no_response,    // the request holds no assistant message to continue after
    not_hers,       // her slot does not hold the request's history up to that response
    not_generated,  // her slot generated nothing after the prompt she was served
    no_end_of_turn, // the response's rendering carries no end-of-turn token
};

inline const char * continuation_miss_name(continuation_miss miss) {
    switch (miss) {
        case continuation_miss::none:           return "none";
        case continuation_miss::not_requested:  return "not_requested";
        case continuation_miss::other_response: return "other_response";
        case continuation_miss::has_media:      return "has_media";
        case continuation_miss::no_response:    return "no_response";
        case continuation_miss::not_hers:       return "not_hers";
        case continuation_miss::not_generated:  return "not_generated";
        case continuation_miss::no_end_of_turn: return "no_end_of_turn";
    }
    return "unknown";
}

// What her slot holds from her last response.
// One response her slot holds: the caller's id for the request it answered, where the prompt she was
// served for it ends and where her response ends in the slot's tokens. A slot holds a lineage of them:
// a continuation keeps every earlier one as its prefix, any other request starts a new lineage.
struct continuation_response {
    std::string id;
    size_t      n_served = 0;
    size_t      n_end    = 0;
};

struct continuation_held {
    const llama_tokens &                       tokens;    // her slot's tokens
    const std::vector<continuation_response> & responses; // the responses they hold, oldest first
};

// The request as rendered and tokenized.
struct continuation_request {
    const std::string &           continues; // the id of the request whose response it continues, or empty
    const llama_tokens &          tokens;
    const common_chat_msg_spans & spans; // its messages' token spans
};

struct continuation {
    continuation_miss miss = continuation_miss::not_requested;
    llama_tokens      tokens; // her held tokens, then the request's from the response's end-of-turn
    size_t            response  = 0; // where her response starts in the request's tokens
    size_t            kept      = 0; // her held tokens kept: the prompt she was served and her response
    size_t            generated = 0; // of those, the tokens she generated
    size_t            from  = 0; // where in the request's tokens the kept tail starts
    int64_t           shift = 0; // how far every request position at or after `from` moved
};

inline continuation continue_from(const continuation_held & held, const continuation_request & request,
                                  const std::function<bool(llama_token)> & is_end_of_turn) {
    continuation out;
    if (request.continues.empty()) {
        return out;
    }
    const continuation_response * answered = nullptr;
    for (const auto & r : held.responses) {
        if (r.id == request.continues) {
            answered = &r;
        }
    }
    if (answered == nullptr || answered->n_end > held.tokens.size()) {
        out.miss = continuation_miss::other_response;
        return out;
    }
    // Her response is the request's newest assistant message that something follows: everything after
    // it is new. The request's own last span is the generation prompt for her next turn, never it.
    const common_chat_msg_span * response = nullptr;
    const auto &                 spans    = request.spans.spans;
    for (size_t i = 0; i + 1 < spans.size(); i++) {
        if (spans[i].role == COMMON_CHAT_ROLE_ASSISTANT) {
            response = &spans[i];
        }
    }
    if (response == nullptr) {
        out.miss = continuation_miss::no_response;
        return out;
    }
    // Her slot holds the request's history up to her response: the prompt she was served began with it.
    const size_t start = response->pos;
    if (answered->n_served < start ||
        !std::equal(request.tokens.begin(), request.tokens.begin() + start, held.tokens.begin())) {
        out.miss = continuation_miss::not_hers;
        return out;
    }
    if (answered->n_end <= answered->n_served) {
        out.miss = continuation_miss::not_generated;
        return out;
    }
    // The end-of-turn that closes her response in the request: what her generation stopped on.
    size_t end_of_turn = start + response->len;
    for (size_t i = start + response->len; i > start; --i) {
        if (is_end_of_turn(request.tokens[i - 1])) {
            end_of_turn = i - 1;
            break;
        }
    }
    if (end_of_turn == start + response->len) {
        out.miss = continuation_miss::no_end_of_turn;
        return out;
    }
    out.miss = continuation_miss::none;
    out.response  = start;
    out.kept      = answered->n_end;
    out.generated = answered->n_end - answered->n_served;
    out.from = end_of_turn;
    // Her tokens up to her response's end: whatever the slot holds after it (a warm pass's glue) is
    // found again by the prompt cache, never trusted here.
    out.shift = (int64_t) answered->n_end - (int64_t) end_of_turn;
    out.tokens.reserve(answered->n_end + request.tokens.size() - end_of_turn);
    out.tokens.insert(out.tokens.end(), held.tokens.begin(), held.tokens.begin() + answered->n_end);
    out.tokens.insert(out.tokens.end(), request.tokens.begin() + end_of_turn, request.tokens.end());
    return out;
}

// A request position after the splice: unchanged before the kept tail, moved by `shift` within it,
// and unknown (-1) inside her response, which her held tokens replace.
inline int64_t continued_position(const continuation & c, int64_t pos) {
    if (pos < 0 || c.miss != continuation_miss::none || (size_t) pos <= c.response) {
        return pos;
    }
    if ((size_t) pos < c.from) {
        return -1;
    }
    return pos + c.shift;
}
