// what this catches: the tokenizer's fragment cache (llama-vocab.cpp) returning different tokens than BPE
// would, across a growing append-only chat prompt, a changed message, and text that only differs by
// whitespace; needs -m <model> (vocab only). The receipt is the per-act time it prints.
#include "llama.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static void require(bool condition, const char * message) {
    if (!condition) {
        std::fprintf(stderr, "FAILED: %s\n", message);
        std::exit(1);
    }
}

static std::vector<llama_token> tokenize(const llama_vocab * v, const std::string & text) {
    std::vector<llama_token> out(text.size() + 16);
    const int n = llama_tokenize(v, text.data(), (int) text.size(), out.data(), (int) out.size(), true, true);
    require(n >= 0, "tokenize fits");
    out.resize(n);
    return out;
}

static std::string msg(int i, const char * body) {
    return std::string("<|im_start|>") + (i % 2 ? "assistant\n" : "user\n") + body + " (" + std::to_string(i) + ")<|im_end|>\n";
}

int main(int argc, char ** argv) {
    require(argc >= 3 && std::string(argv[1]) == "-m", "usage: -m <model>");
    llama_backend_init();
    auto mp       = llama_model_default_params();
    mp.vocab_only = true;
    llama_model * m = llama_model_load_from_file(argv[2], mp);
    require(m != nullptr, "model loads");
    const llama_vocab * v = llama_model_get_vocab(m);

    // a second vocabulary from the same file has its own, empty cache: the ground truth
    llama_model * truth_m = llama_model_load_from_file(argv[2], mp);
    const llama_vocab * truth = llama_model_get_vocab(truth_m);

    std::string chat;
    for (int i = 0; i < 40; ++i) {
        chat += msg(i, "Aiko read request_body.rs and ran cargo check; three  warnings\tremain.");
    }
    for (int act = 0; act < 6; ++act) {
        chat += msg(100 + act, act % 2 ? "A new reply arrived." : "A new  reply arrived.");
        const auto t0  = std::chrono::steady_clock::now();
        const auto got = tokenize(v, chat);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        llama_model * fresh_m = llama_model_load_from_file(argv[2], mp);
        const auto want = tokenize(llama_model_get_vocab(fresh_m), chat);
        llama_model_free(fresh_m);
        require(got == want, "cached tokens equal a fresh tokenizer's, every act");
        std::printf("act %d: %zu tokens in %.2f ms\n", act, got.size(), ms);
    }
    // a changed message is new text: it is tokenized, never served from the cache
    std::string edited = chat;
    edited.replace(edited.find("three"), 5, "four");
    require(tokenize(v, edited) == tokenize(truth, edited), "an edited message tokenizes as new text");

    llama_model_free(truth_m);
    llama_model_free(m);
    std::printf("test-tokenizer-fragment-cache: ok\n");
    return 0;
}
