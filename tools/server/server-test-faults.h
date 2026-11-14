#pragma once

#include <cstdint>
#include <string>
#ifdef LLAMA_SERVER_TEST_FAULTS
#include <cstdlib>
#include <fstream>
#endif

// Continuum fork: failures the server tests inject through the environment, so a test can show
// that a failure leaves the server consistent. Compiled in only by the server test build
// (cmake -DLLAMA_SERVER_TEST_FAULTS=ON). In any other build `enabled` is false, every use sits
// under `if constexpr (server_test_faults::enabled)` and is discarded, and no environment is
// read, so a stray variable can never fail a real slot.
struct server_test_faults {
#ifdef LLAMA_SERVER_TEST_FAULTS
    static constexpr bool enabled = true;
#else
    static constexpr bool enabled = false;
#endif

    int32_t prompt_checkpoints = 0; // LLAMA_SERVER_TEST_FAIL_CHECKPOINTS: the next N prompt checkpoint copies fail to allocate
    int32_t spec_checkpoints   = 0; // LLAMA_SERVER_TEST_FAIL_SPEC_CHECKPOINTS: the next N speculative checkpoint copies fail to allocate
    int32_t prompt_fills       = 0; // LLAMA_SERVER_TEST_FAIL_PROMPT_FILLS: the next N prompt fills throw after adding their tokens
    // LLAMA_SERVER_TEST_PAGE_IO_GATE: a path (a FIFO the test holds) each KV page switch's file IO
    // reads one byte from before it starts, so a test holds IO in flight without sleeping: opening a
    // FIFO's write end returns once the IO is waiting, and the byte written releases it
    std::string page_io_gate;

    static server_test_faults from_env() {
        server_test_faults faults;
#ifdef LLAMA_SERVER_TEST_FAULTS
        faults.prompt_checkpoints = read("LLAMA_SERVER_TEST_FAIL_CHECKPOINTS");
        faults.spec_checkpoints   = read("LLAMA_SERVER_TEST_FAIL_SPEC_CHECKPOINTS");
        faults.prompt_fills       = read("LLAMA_SERVER_TEST_FAIL_PROMPT_FILLS");
        if (const char * gate = std::getenv("LLAMA_SERVER_TEST_PAGE_IO_GATE")) {
            faults.page_io_gate = gate;
        }
#endif
        return faults;
    }

    bool any() const {
        return prompt_checkpoints || spec_checkpoints || prompt_fills || !page_io_gate.empty();
    }

    // Blocks the calling page IO worker until the test releases it through `gate` (see page_io_gate).
    static void hold(const std::string & gate) {
#ifdef LLAMA_SERVER_TEST_FAULTS
        if (!gate.empty()) {
            std::ifstream in(gate, std::ios::binary);
            char released = 0;
            in.get(released);
        }
#else
        (void) gate;
#endif
    }

    // true when this occurrence is one to fail, counting it
    static bool take(int32_t & remaining) {
        if (remaining <= 0) {
            return false;
        }
        --remaining;
        return true;
    }

#ifdef LLAMA_SERVER_TEST_FAULTS
private:
    static int32_t read(const char * name) {
        const char * value = std::getenv(name);
        return value ? std::atoi(value) : 0;
    }
#endif
};
