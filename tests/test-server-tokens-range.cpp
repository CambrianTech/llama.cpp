// The accessor a learning step reads her served range through, on a REAL server_tokens of a
// media-capable server (has_mtmd = true): the M5 engine aborted in get_tokens() on exactly such a
// server (2026-11-17 22:09:33Z), with or without media in the prompt.
#include "server-common.h"
#include "mtmd.h"

#include <cstdlib>
#include <iostream>

static void require(bool condition, const char * message) {
    if (!condition) {
        std::cerr << message << '\n';
        std::exit(1);
    }
}

int main() {
    // A text-only history on a media-capable server: the ids are read in place, at any offset.
    {
        server_tokens text(llama_tokens{10, 11, 12, 13}, /* has_mtmd */ true);
        const llama_token * from1 = text.text_from(1);
        require(from1 != nullptr && from1[0] == 11 && from1[2] == 13, "text-only ids are read from the offset");
        require(text.text_from(0) + 1 == from1, "in place: the same storage, no copy");
        require(text.text_from(4) != nullptr, "the end is a valid, empty range");
        require(text.text_from(5) == nullptr, "past the end is refused");
        require(text.size() == 4 && text[0] == 10 && text[3] == 13, "reading changes nothing");
    }
    // The fixture's text ids, then its image, then 3 more ids; every offset from the chunks' own counts.
    {
        mtmd::input_chunks chunks(mtmd_test_create_input_chunks());
        require(chunks.size() == 2, "fixture: a text chunk and an image chunk");
        const size_t n_text  = mtmd_input_chunk_get_n_tokens(chunks[0]);
        const size_t n_image = mtmd_input_chunk_get_n_tokens(chunks[1]);
        require(n_text > 0 && n_image > 0, "fixture: real text and a real image");
        const size_t image_end = n_text + n_image; // one past the image's last position
        server_tokens history(chunks, /* has_mtmd */ true);
        history.push_back(7);
        history.push_back(8);
        history.push_back(9);
        require(history.size() == image_end + 3, "the history is the text, the image and 3 ids");
        require(history.text_from(0) == nullptr, "a range holding the image is refused");
        require(history.text_from(n_text) == nullptr, "a range starting at the image is refused");
        require(history.text_from(image_end - 1) == nullptr, "a range starting on the image's last position is refused");
        const llama_token * after = history.text_from(image_end);
        require(after != nullptr && after[0] == 7 && after[2] == 9, "media before the range is allowed: ids after it are read");
        require(history.text_from(image_end + 3) != nullptr && history.text_from(image_end + 4) == nullptr, "bounds");
        require(history.size() == image_end + 3, "reading changes nothing");
    }
    std::cout << "server_tokens range: ok\n";
    return 0;
}
