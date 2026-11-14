#pragma once

#include <cstdint>
#ifdef LLAMA_SERVER_TEST_FAULTS
#include <cstdlib>
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
    // LLAMA_SERVER_TEST_STEP_DELAY_MS: every decode step that ran a batch then waits N ms, so a test
    // can act while generations are in progress on a model that would otherwise finish in milliseconds
    // (the reconfigure test); a pace, not a failure
    int32_t step_delay_ms      = 0;

    static server_test_faults from_env() {
        server_test_faults faults;
#ifdef LLAMA_SERVER_TEST_FAULTS
        faults.prompt_checkpoints = read("LLAMA_SERVER_TEST_FAIL_CHECKPOINTS");
        faults.spec_checkpoints   = read("LLAMA_SERVER_TEST_FAIL_SPEC_CHECKPOINTS");
        faults.prompt_fills       = read("LLAMA_SERVER_TEST_FAIL_PROMPT_FILLS");
        faults.step_delay_ms      = read("LLAMA_SERVER_TEST_STEP_DELAY_MS");
#endif
        return faults;
    }

    bool any() const {
        return prompt_checkpoints || spec_checkpoints || prompt_fills || step_delay_ms;
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
