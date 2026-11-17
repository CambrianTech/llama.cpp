#include "llama-memory.h"

llama_memory_status llama_memory_status_combine(llama_memory_status s0, llama_memory_status s1) {
    bool has_update = false;

    switch (s0) {
        case LLAMA_MEMORY_STATUS_SUCCESS:
            {
                has_update = true;
                break;
            }
        case LLAMA_MEMORY_STATUS_NO_UPDATE:
            {
                break;
            }
        case LLAMA_MEMORY_STATUS_FAILED_PREPARE:
        case LLAMA_MEMORY_STATUS_FAILED_COMPUTE:
            {
                return s0;
            }
    }

    switch (s1) {
        case LLAMA_MEMORY_STATUS_SUCCESS:
            {
                has_update = true;
                break;
            }
        case LLAMA_MEMORY_STATUS_NO_UPDATE:
            {
                break;
            }
        case LLAMA_MEMORY_STATUS_FAILED_PREPARE:
        case LLAMA_MEMORY_STATUS_FAILED_COMPUTE:
            {
                return s1;
            }
    }

    // if either status has an update, then the combined status has an update
    return has_update ? LLAMA_MEMORY_STATUS_SUCCESS : LLAMA_MEMORY_STATUS_NO_UPDATE;
}

bool llama_memory_status_is_fail(llama_memory_status status) {
    switch (status) {
        case LLAMA_MEMORY_STATUS_SUCCESS:
        case LLAMA_MEMORY_STATUS_NO_UPDATE:
            {
                return false;
            }
        case LLAMA_MEMORY_STATUS_FAILED_PREPARE:
        case LLAMA_MEMORY_STATUS_FAILED_COMPUTE:
            {
                return true;
            }
    }

    return false;
}

const char * llama_persona_move_name(llama_persona_move m) {
    switch (m) {
        case llama_persona_move::done:               return "done";
        case llama_persona_move::unsupported:        return "unsupported";
        case llama_persona_move::no_such_stream:     return "no_such_stream";
        case llama_persona_move::transposed_v:       return "transposed_v";
        case llama_persona_move::stream_not_empty:   return "stream_not_empty";
        case llama_persona_move::cells_not_prepared: return "cells_not_prepared";
        case llama_persona_move::layout_mismatch:    return "layout_mismatch";
        case llama_persona_move::invalid_seq:        return "invalid_seq";
        case llama_persona_move::seq_holds_state:    return "seq_holds_state";
        case llama_persona_move::shared_cell:        return "shared_cell";
        case llama_persona_move::no_free_cell:       return "no_free_cell";
    }
    return "unknown";
}
