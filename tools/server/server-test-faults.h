#pragma once

#include <cstdint>
#include <cstdlib>

// Continuum fork: failures the server tests inject through the environment, so a test can show
// that a failure leaves the server consistent. Every count is zero unless its variable is set.
struct server_test_faults {
    int32_t checkpoint_copies = 0; // LLAMA_SERVER_TEST_FAIL_CHECKPOINTS: the next N prompt checkpoint copies fail to allocate
    int32_t prompt_fills      = 0; // LLAMA_SERVER_TEST_FAIL_PROMPT_FILLS: the next N prompt fills throw after adding their tokens

    static server_test_faults from_env() {
        server_test_faults faults;
        faults.checkpoint_copies = read("LLAMA_SERVER_TEST_FAIL_CHECKPOINTS");
        faults.prompt_fills      = read("LLAMA_SERVER_TEST_FAIL_PROMPT_FILLS");
        return faults;
    }

    // true when this occurrence is one to fail, counting it
    static bool take(int32_t & remaining) {
        if (remaining <= 0) {
            return false;
        }
        --remaining;
        return true;
    }

private:
    static int32_t read(const char * name) {
        const char * value = std::getenv(name);
        return value ? std::atoi(value) : 0;
    }
};
