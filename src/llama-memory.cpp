#include "llama-memory.h"

#include "llama-ext.h"
#include "llama-io.h"

#include <set>
#include <stdexcept>
#include <string>
#include <vector>

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
        case llama_persona_move::wrong_persona:      return "wrong_persona";
        case llama_persona_move::invalid_rollback:   return "invalid_rollback";
        case llama_persona_move::wrong_owner:        return "wrong_owner";
        case llama_persona_move::out_of_memory:      return "out_of_memory";
        case llama_persona_move::no_empty_stream:    return "no_empty_stream";
        case llama_persona_move::no_checkpoint:      return "no_checkpoint";
        case llama_persona_move::checkpoint_ahead:   return "checkpoint_ahead";
        case llama_persona_move::checkpoint_stale:   return "checkpoint_stale";
        case llama_persona_move::io_error:           return "io_error";
        case llama_persona_move::file_mismatch:      return "file_mismatch";
        case llama_persona_move::pool_full:          return "pool_full";
        case llama_persona_move::pending_shift:      return "pending_shift";
    }
    return "unknown";
}

//
// llama_memory_bufs
//

ggml_backend_buffer_ptr llama_memory_bufs::alloc_buf(ggml_context * ctx, ggml_backend_buffer_type_t buft, bool no_alloc, const char * name) {
    ggml_backend_buffer_ptr buf;
    if (no_alloc) {
        buf.reset(ggml_backend_buft_alloc_buffer(buft, /*size =*/ 0)); // dummy buffer
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
            t->buffer = buf.get(); // set dummy buffer so that the backend scheduler won't try to allocate the tensors
        }
    } else {
        buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft)); // real buffer
    }
    if (!buf) {
        throw std::runtime_error(std::string("failed to allocate buffer for ") + name);
    }
    return buf;
}

ggml_backend_buffer_t llama_memory_bufs::emplace(ggml_context_ptr ctx, ggml_backend_buffer_type_t buft, const char * name) {
    ggml_backend_buffer_ptr buf = alloc_buf(ctx.get(), buft, no_alloc, name);

    // initialize the buffer to avoid NaNs in the padding
    ggml_backend_buffer_clear(buf.get(), 0);

    bufs.emplace_back(std::move(ctx), std::move(buf));

    return bufs.back().second.get();
}

std::map<ggml_backend_buffer_type_t, size_t> llama_memory_bufs::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> ret;
    for (const auto & [ctx, buf] : bufs) {
        ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(buf.get());

        if (no_alloc) {
            GGML_ASSERT(ggml_backend_buffer_get_base(buf.get()) == nullptr);
            ret[buft] += ggml_backend_alloc_ctx_tensors_from_buft_size(ctx.get(), buft);
        } else {
            ret[buft] += ggml_backend_buffer_get_size(buf.get());
        }
    }
    return ret;
}

// collects the tensors that a state write visits, without keeping the data
struct llama_memory_tensors_writer : public llama_io_write_i {
    std::set<ggml_tensor *> tensors;
    size_t size = 0;

    void write(const void *, size_t n) override {
        size += n;
    }

    void write_tensor(ggml_tensor * tensor, size_t, size_t n) override {
        tensors.insert(tensor);
        size += n;
    }

    size_t n_bytes() override {
        return size;
    }
};

std::vector<ggml_backend_buffer_t> llama_memory_get_buffers(const llama_memory_t mem) {
    std::set<ggml_backend_buffer_t> buffers;
    for (ggml_tensor * tensor : llama_memory_get_tensors(mem)) {
        buffers.insert(tensor->buffer);
    }

    return std::vector<ggml_backend_buffer_t>(buffers.begin(), buffers.end());
}

std::vector<ggml_tensor *> llama_memory_get_tensors(const llama_memory_t mem) {
    if (mem == nullptr) {
        return {};
    }

    llama_memory_tensors_writer writer;

    mem->state_write(writer);

    return std::vector<ggml_tensor *>(writer.tensors.begin(), writer.tensors.end());
}
