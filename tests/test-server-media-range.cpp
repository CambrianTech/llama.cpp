#include "../tools/server/server-media-range.h"

#include <cstdlib>
#include <iostream>
#include <map>

static void require(bool condition, const char * message) {
    if (!condition) {
        std::cerr << message << '\n';
        std::exit(1);
    }
}

// What this catches (the M5, 2026-11-17 22:09:33Z): a learning step reading a served range as token ids
// when a media chunk lies in it. On a vision-loaded server get_tokens() aborted the whole engine; the
// learn path now reads the range in place and refuses when a chunk starts in it or spans into it.
int main() {
    const auto length = [](size_t n) { return n; };
    const std::map<size_t, size_t> none;
    require(!media_reaches(none, 0, length), "no media: every range is token ids");

    // one image of 16 positions at 100..115
    const std::map<size_t, size_t> image{{100, 16}};
    require(!media_reaches(image, 116, length), "a range after the image holds only ids");
    require(media_reaches(image, 115, length), "a range starting on the image's last position holds media");
    require(media_reaches(image, 100, length), "a range starting at the image holds media");
    require(media_reaches(image, 40, length), "a range before the image reaches it");

    // two images; a range between them still reaches the second
    const std::map<size_t, size_t> two{{10, 4}, {50, 4}};
    require(media_reaches(two, 20, length), "a later image lies in the range");
    require(!media_reaches(two, 54, length), "past both images: ids only");
    std::cout << "media range: ok\n";
    return 0;
}
