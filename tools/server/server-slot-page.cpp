#include "server-slot-page.h"

#include "server-common.h"

server_slot_page_size server_slot_page_measure(llama_context * ctx_tgt, llama_context * ctx_dft, llama_seq_id seq) {
    server_slot_page_size size;
    size.main =           llama_state_seq_get_size_ext(ctx_tgt, seq, LLAMA_STATE_SEQ_FLAGS_NONE);
    size.drft = ctx_dft ? llama_state_seq_get_size_ext(ctx_dft, seq, LLAMA_STATE_SEQ_FLAGS_NONE) : 0;
    return size;
}

bool server_slot_page_read(llama_context * ctx_tgt, llama_context * ctx_dft, llama_seq_id seq, server_prompt_data & data) {
    const size_t n_main = llama_state_seq_get_data_ext(ctx_tgt, data.main.data(), data.main.size(), seq, LLAMA_STATE_SEQ_FLAGS_NONE);
    if (n_main != data.main.size()) {
        SRV_ERR("seq %d: target state gave %zu of %zu bytes\n", seq, n_main, data.main.size());
        return false;
    }
    if (data.drft.empty()) {
        return true;
    }
    if (ctx_dft == nullptr) {
        SRV_ERR("seq %d: a draft state was sized with no draft context\n", seq);
        return false;
    }
    const size_t n_drft = llama_state_seq_get_data_ext(ctx_dft, data.drft.data(), data.drft.size(), seq, LLAMA_STATE_SEQ_FLAGS_NONE);
    if (n_drft != data.drft.size()) {
        SRV_ERR("seq %d: draft state gave %zu of %zu bytes\n", seq, n_drft, data.drft.size());
        return false;
    }
    return true;
}

bool server_slot_page_write(llama_context * ctx_tgt, llama_context * ctx_dft, llama_seq_id seq, const server_prompt_data & data) {
    auto empty_both = [&]() {
        llama_memory_seq_rm(llama_get_memory(ctx_tgt), seq, -1, -1);
        if (ctx_dft) {
            llama_memory_seq_rm(llama_get_memory(ctx_dft), seq, -1, -1);
        }
    };

    const size_t n_main = llama_state_seq_set_data_ext(ctx_tgt, data.main.data(), data.main.size(), seq, LLAMA_STATE_SEQ_FLAGS_NONE);
    if (n_main != data.main.size()) {
        SRV_ERR("seq %d: failed to restore the target state with size %zu\n", seq, data.main.size());
        empty_both();
        return false;
    }
    if (data.drft.empty()) {
        return true;
    }
    if (ctx_dft == nullptr) {
        SRV_ERR("seq %d: the page holds a draft state and there is no draft context\n", seq);
        empty_both();
        return false;
    }
    const size_t n_drft = llama_state_seq_set_data_ext(ctx_dft, data.drft.data(), data.drft.size(), seq, LLAMA_STATE_SEQ_FLAGS_NONE);
    if (n_drft != data.drft.size()) {
        SRV_WRN("seq %d: failed to restore the draft state with size %zu\n", seq, data.drft.size());
        empty_both();
        return false;
    }
    return true;
}
