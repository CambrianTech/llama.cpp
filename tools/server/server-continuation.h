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
// names the response it continues (`continues`, the completion id her last response carried) is
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
struct continuation_held {
    const llama_tokens & tokens;   // the prompt she was served, then what she generated
    size_t               n_served; // how many of `tokens` are the prompt she was served
    const std::string &  response; // the completion id that response carried
};

// The request as rendered and tokenized.
struct continuation_request {
    const std::string &           continues; // the completion id it continues, or empty
    const llama_tokens &          tokens;
    const common_chat_msg_spans & spans; // its messages' token spans
};

struct continuation {
    continuation_miss miss = continuation_miss::not_requested;
    llama_tokens      tokens; // her held tokens, then the request's from the response's end-of-turn
    size_t            response = 0; // where her response starts in the request's tokens
    size_t            from  = 0; // where in the request's tokens the kept tail starts
    int64_t           shift = 0; // how far every request position at or after `from` moved
};

inline continuation continue_from(const continuation_held & held, const continuation_request & request,
                                  const std::function<bool(llama_token)> & is_end_of_turn) {
    continuation out;
    if (request.continues.empty()) {
        return out;
    }
    if (request.continues != held.response) {
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
    if (held.n_served < start || held.tokens.size() < start ||
        !std::equal(request.tokens.begin(), request.tokens.begin() + start, held.tokens.begin())) {
        out.miss = continuation_miss::not_hers;
        return out;
    }
    if (held.tokens.size() <= held.n_served) {
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
    out.response = start;
    out.from = end_of_turn;
    out.shift = (int64_t) held.tokens.size() - (int64_t) end_of_turn;
    out.tokens.reserve(held.tokens.size() + request.tokens.size() - end_of_turn);
    out.tokens.insert(out.tokens.end(), held.tokens.begin(), held.tokens.end());
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
