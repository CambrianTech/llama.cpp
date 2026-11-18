#pragma once

#include <cstddef>
#include <iterator>

// Whether a media chunk occupies any position at or after `from`, given the chunks keyed by their
// first position (`starts`, ordered) and each chunk's length in positions (`length(chunk)`). A chunk
// that starts before `from` and spans past it counts: its positions hold no token ids either.
template <typename Starts, typename Length>
bool media_reaches(const Starts & starts, size_t from, Length length) {
    if (starts.empty()) {
        return false;
    }
    auto it = starts.lower_bound(from);
    if (it != starts.end()) {
        return true; // a chunk starts in the range
    }
    it = std::prev(it); // the last chunk starts before `from`
    return it->first + length(it->second) > from;
}
