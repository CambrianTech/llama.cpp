#pragma once

// Continuum fork: a MIND, named by the request (the core's PageKey::Mind, a 128-bit id), never guessed from her
// tokens. Text exists only at the JSON edge: a request's "page" is parsed here once, into two integers.
//
// Switching minds costs almost nothing (Joel, 2026-11-16: "This is the entire difference between a multi mind
// system and a POS"): a mind leaving a lane is PARKED as a sequence of the unified KV pool (seq_cp + seq_rm:
// attention cells re-tagged, the recurrent state moved by id, nothing serialized), and a mind arriving is moved
// back the same way, continuing at her own end. server-context.cpp holds the parked table (sized once at load
// from the context's sequence capacity) and performs the moves.

#include <cstdint>
#include <random>
#include <string>

// This engine incarnation's token, drawn once per process and reported on /props (resident_minds.incarnation).
// The core sends it back with a prefetch, so a stream meant for a replaced engine is refused: issued by the engine
// itself, never inferred from an OS pid (which the OS reuses).
inline uint64_t mind_incarnation() {
    static const uint64_t token = [] {
        std::random_device rd;
        const uint64_t t = ((uint64_t) rd() << 32) ^ (uint64_t) rd();
        return t != 0 ? t : 1;
    }();
    return token;
}

struct mind_page {
    uint64_t hi = 0, lo = 0;
    bool valid() const { return hi != 0 || lo != 0; }
    bool operator==(const mind_page & o) const { return hi == o.hi && lo == o.lo; }
    bool operator!=(const mind_page & o) const { return !(*this == o); }
};

// "page": 32 hexadecimal digits, dashes allowed (a UUID). Anything else is no page (the request is anonymous).
inline mind_page mind_page_parse(const std::string & s) {
    mind_page p;
    int n = 0;
    for (char c : s) {
        if (c == '-') {
            continue;
        }
        int v;
        if (c >= '0' && c <= '9') {
            v = c - '0';
        } else if (c >= 'a' && c <= 'f') {
            v = c - 'a' + 10;
        } else if (c >= 'A' && c <= 'F') {
            v = c - 'A' + 10;
        } else {
            return mind_page{};
        }
        if (n < 16) {
            p.hi = (p.hi << 4) | (uint64_t) v;
        } else if (n < 32) {
            p.lo = (p.lo << 4) | (uint64_t) v;
        } else {
            return mind_page{};
        }
        ++n;
    }
    return n == 32 ? p : mind_page{};
}
