// Continuum: a sequence's state as a file and as a file body (what llama-server pages out and back), sized and
// written by the one writer so a size can never disagree with what is written. Included by llama-context.cpp after
// its file-local I/O classes (llama_io_write_dummy, _host, _file; llama_io_read_host), so upstream's context merges
// without touching this.

size_t llama_context::state_seq_write_file(llama_io_write_i & io, llama_seq_id seq_id, const llama_token * tokens, size_t n_token_count) {
    const uint32_t magic = LLAMA_STATE_SEQ_MAGIC;
    const uint32_t version = LLAMA_STATE_SEQ_VERSION;
    const uint32_t count = (uint32_t) n_token_count;
    io.write(&magic, sizeof(magic));
    io.write(&version, sizeof(version));
    io.write(&count, sizeof(count));
    io.write(tokens, sizeof(llama_token) * n_token_count);
    state_seq_write_data(io, seq_id, 0);
    return io.n_bytes();
}

size_t llama_context::state_seq_file_size(llama_seq_id seq_id, const llama_token * tokens, size_t n_token_count) {
    llama_io_write_dummy io(false);
    return state_seq_write_file(io, seq_id, tokens, n_token_count);
}

size_t llama_context::state_seq_save_file(llama_seq_id seq_id, const char * filepath, const llama_token * tokens, size_t n_token_count) {
    llama_file file(filepath, "wb");
    llama_io_write_file io(&file);
    const size_t written = state_seq_write_file(io, seq_id, tokens, n_token_count);
    GGML_ASSERT(written == file.tell());
    return written;
}

size_t llama_context::state_seq_file_body_size(llama_seq_id seq_id) {
    llama_io_write_dummy io(false);
    try {
        return state_seq_write_data(io, seq_id, 0);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error sizing sequence state: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_seq_get_file_body(llama_seq_id seq_id, uint8_t * dst, size_t size) {
    try {
        size_t n = 0;
        {
            // the host writer copies the tensors out when it goes out of scope
            llama_io_write_host io(dst, size);
            n = state_seq_write_data(io, seq_id, 0);
        }
        return n;
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error copying sequence state: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_seq_set_file_body(llama_seq_id seq_id, const uint8_t * src, size_t size) {
    try {
        size_t n = 0;
        {
            // the host reader copies the tensors in when it goes out of scope
            llama_io_read_host io(src, size);
            n = state_seq_read_data(io, seq_id, 0);
        }
        return n;
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading sequence state: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_state_seq_file_body_size(llama_context * ctx, llama_seq_id seq_id) {
    ctx->synchronize();

    return ctx->state_seq_file_body_size(seq_id);
}

size_t llama_state_seq_get_file_body(llama_context * ctx, llama_seq_id seq_id, uint8_t * dst, size_t size) {
    ctx->synchronize();

    return ctx->state_seq_get_file_body(seq_id, dst, size);
}

size_t llama_state_seq_set_file_body(llama_context * ctx, llama_seq_id dest_seq_id, const uint8_t * src, size_t size) {
    ctx->synchronize();

    return ctx->state_seq_set_file_body(dest_seq_id, src, size);
}

size_t llama_state_seq_file_size(llama_context * ctx, llama_seq_id seq_id, const llama_token * tokens, size_t n_token_count) {
    ctx->synchronize();
    try {
        return ctx->state_seq_file_size(seq_id, tokens, n_token_count);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error sizing sequence state file: %s\n", __func__, err.what());
        return 0;
    }
}
