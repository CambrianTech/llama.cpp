#pragma once

#include "llama.h"
#include "llama-graph.h"
#include "llama-persona-blocks.h"

#include <map>
#include <vector>
#include <string>
#include <memory>
#include <functional>
#include <vector>

struct llama_ubatch;

class llama_batch_allocr;

class llama_io_write_i;
class llama_io_read_i;

struct llama_memory_params {
    // kv cache
    ggml_type type_k;
    ggml_type type_v;

    // use full-size SWA cache
    bool swa_full;

    llama_context_type ctx_type;

    llama_memory_t mem_other;

    // only simulate the memory allocations
    bool no_alloc;
};

// the tensor contexts and backend buffers of a memory instance
// with no_alloc, dummy buffers are attached instead of allocating the tensor data, so the memory use can still be measured
struct llama_memory_bufs {
    explicit llama_memory_bufs(bool no_alloc) : no_alloc(no_alloc) {}

    // allocate the tensors of ctx on buft, or attach a dummy buffer if no_alloc is set
    // returns the buffer, for logging
    ggml_backend_buffer_t emplace(ggml_context_ptr ctx, ggml_backend_buffer_type_t buft, const char * name);

    // the memory used by the tensors: the buffer sizes, or the simulated sizes if no_alloc is set
    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const;

    // allocate a single ctx buffer - for the memory instances that keep their own buffer layout
    static ggml_backend_buffer_ptr alloc_buf(ggml_context * ctx, ggml_backend_buffer_type_t buft, bool no_alloc, const char * name);

    bool empty() const { return bufs.empty(); }

    auto begin()       { return bufs.begin(); }
    auto end()         { return bufs.end();   }
    auto begin() const { return bufs.begin(); }
    auto end()   const { return bufs.end();   }

    std::vector<std::pair<ggml_context_ptr, ggml_backend_buffer_ptr>> bufs;

    const bool no_alloc;
};

enum llama_memory_status {
    LLAMA_MEMORY_STATUS_SUCCESS = 0,
    LLAMA_MEMORY_STATUS_NO_UPDATE,
    LLAMA_MEMORY_STATUS_FAILED_PREPARE,
    LLAMA_MEMORY_STATUS_FAILED_COMPUTE,
};

// helper function for combining the status of two memory contexts
// useful for implementing hybrid memory types (e.g. iSWA)
llama_memory_status llama_memory_status_combine(llama_memory_status s0, llama_memory_status s1);

// helper function for checking if a memory status indicates a failure
bool llama_memory_status_is_fail(llama_memory_status status);

// the interface for managing the memory context during batch processing
// this interface is implemented per memory type. see:
//   - llama_kv_cache_context
//   - llama_kv_cache_iswa_context
//   ...
//
// the only method that should mutate the memory and the memory context is llama_memory_i::apply()
struct llama_memory_context_i {
    virtual ~llama_memory_context_i() = default;

    // consume the current ubatch from the context and proceed to the next one
    // return false if we are done
    virtual bool next() = 0;

    // apply the memory state for the current ubatch to the memory object
    // return false on failure
    virtual bool apply() = 0;

    // get the current ubatch
    virtual const llama_ubatch & get_ubatch() const = 0;

    // get the status of the memory context - used for error handling and checking if any updates would be applied
    virtual llama_memory_status get_status() const = 0;
};

using llama_memory_context_ptr = std::unique_ptr<llama_memory_context_i>;

// general concept of LLM memory
// the KV cache is a type of LLM memory, but there can be other types
// One cell holder's occupancy: the cells holding at least one sequence, of the cells it has. A
// composite memory reports each holder it owns, its name prefixed ("attn.base", "recr"), so a
// reader sees which holder bounds what; nothing is summed across holders of different kinds.
struct llama_memory_cells {
    std::string holder;
    uint32_t    used;
    uint32_t    size;
};

// Prefix every holder in `children` with `name` ("kv" -> "attn.kv" is never produced: a plain cache
// reports holder "", which becomes `name`).
inline void llama_memory_cells_append(std::vector<llama_memory_cells> & out, const char * name, std::vector<llama_memory_cells> children) {
    for (auto & c : children) {
        c.holder = c.holder.empty() ? std::string(name) : std::string(name) + "." + c.holder;
        out.push_back(std::move(c));
    }
}

struct llama_memory_i {
    // this callback is used to filter out layers that should not be included in the cache
    using layer_filter_cb = std::function<bool(int32_t il)>;

    // this callback is used to specify which layers should reuse memory from other layers
    // return negative value to indicate that the layer il should not reuse memory
    using layer_reuse_cb = std::function<int32_t(int32_t il)>;

    using layer_share_cb = std::function<int32_t(int32_t il)>;

    virtual ~llama_memory_i() = default;

    // Read-only admission for a validated batch; nullptr means admitted.
    virtual const char * training_error(const llama_batch & batch, bool flash_attn) const = 0;

    // split the input batch into a set of ubatches and verify that they can fit into the cache
    // return a context object containing the ubatches and memory state required to process them
    // check the llama_memory_context_i::get_status() for the result
    // embd_all: keep all the tokens of a sequence in one ubatch (all tokens are output, or the attention is non-causal)
    virtual llama_memory_context_ptr init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) = 0;

    // simulate full cache, used for allocating worst-case compute buffers
    virtual llama_memory_context_ptr init_full() = 0;

    // Full served capacity for one learning sequence, without applying/cache mutation.
    // Unsupported memory owners refuse explicitly rather than building multi-stream backward.
    virtual llama_memory_context_ptr init_full_single_stream() { return nullptr; }


    // prepare for any pending memory updates, such as shifts, copies, etc.
    // status == LLAMA_MEMORY_STATUS_NO_UPDATE if there is nothing to update
    virtual llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) = 0;

    // Read-only: init_update may transfer pending copy ownership. Unknown owners defer.
    virtual bool has_pending_update() const { return true; }

    // getters
    virtual bool get_can_shift() const = 0;

    //
    // ops
    //

    // if data == true, the data buffers will also be cleared together with the metadata
    virtual void clear(bool data) = 0;

    virtual bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) = 0;
    virtual void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) = 0;
    virtual void seq_keep(llama_seq_id seq_id) = 0;
    virtual void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) = 0;
    virtual void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) = 0;

    virtual llama_pos seq_pos_min(llama_seq_id seq_id) const = 0;
    virtual llama_pos seq_pos_max(llama_seq_id seq_id) const = 0;

    // the physical attention stream holding this sequence (Continuum fork: a persona's sequence can be bound to any
    // stream, so a stream is never assumed from the sequence id). Memories with no stream map hold it by identity.
    virtual uint32_t seq_stream(llama_seq_id seq_id) const { return (uint32_t) seq_id; }

    // Continuum fork (CONTINUUM-ENGINE 4.14): a persona's state moved raw between this memory and host memory, two
    // phases. persona_prepare runs once per persona (sizes her blocks, records the layout). check_* decide every
    // condition without touching the memory, and reserve the destination and every byte and node the commit needs;
    // commit_* only copy, swap and insert reserved nodes: nothing allocates, so a commit cannot fail. Callers go
    // through llama_context::persona_save / persona_load, which fence the queued device work first.
    virtual llama_persona_move persona_prepare(llama_persona_blocks & blocks) const {
        (void) blocks;
        return llama_persona_move::unsupported;
    }
    virtual llama_persona_move persona_check_save(llama_seq_id seq_id, llama_persona_blocks & out) const {
        (void) seq_id; (void) out;
        return llama_persona_move::unsupported;
    }
    virtual llama_persona_move persona_check_load(llama_seq_id seq_id, uint32_t stream, const llama_persona_blocks & in,
                                                  llama_persona_slot & slot) const {
        (void) seq_id; (void) stream; (void) in; (void) slot;
        return llama_persona_move::unsupported;
    }
    // An empty stream for a load to place her in, chosen by the memory that owns the streams (the caller never numbers
    // them); -1 when none is empty or the memory has no streams to move into.
    virtual int32_t persona_empty_stream() const { return -1; }
    // Continuum fork (UNIFIED-PAGING.md, slice 2): one mind's KV storage built over memory the caller owns (a mapped
    // file), so her KV is a block of the block layer; 0 bytes / unsupported where the memory has no attention storage
    virtual size_t persona_storage_bytes() const { return 0; }
    virtual bool persona_storage_viewable() const { return false; }
    virtual llama_persona_move persona_reserve_storage(llama_persona_blocks & blocks) const {
        (void) blocks;
        return llama_persona_move::unsupported;
    }
    virtual llama_persona_move persona_attach_storage(llama_persona_blocks & blocks, void * base, size_t bytes) const {
        (void) blocks; (void) base; (void) bytes;
        return llama_persona_move::unsupported;
    }
    virtual void persona_commit_save(llama_seq_id seq_id, llama_persona_blocks & out) { (void) seq_id; (void) out; }
    virtual void persona_commit_load(llama_seq_id seq_id, llama_persona_slot & slot, llama_persona_blocks & in) {
        (void) seq_id; (void) slot; (void) in;
    }
    // A checkpoint of a BOUND sequence, the same two phases: check reserves every byte (sizes `out`), commit copies her
    // rows out and leaves her in place. A rewind writes them back and cuts what she holds past the checkpoint.
    virtual llama_persona_move persona_check_checkpoint(llama_seq_id seq_id, llama_persona_checkpoint_blocks & out) const {
        (void) seq_id; (void) out;
        return llama_persona_move::unsupported;
    }
    virtual void persona_commit_checkpoint(llama_seq_id seq_id, llama_persona_checkpoint_blocks & out) {
        (void) seq_id; (void) out;
    }
    virtual llama_persona_move persona_check_rewind(llama_seq_id seq_id, const llama_persona_checkpoint_blocks & in) const {
        (void) seq_id; (void) in;
        return llama_persona_move::unsupported;
    }
    virtual void persona_commit_rewind(llama_seq_id seq_id, const llama_persona_checkpoint_blocks & in) {
        (void) seq_id; (void) in;
    }

    virtual std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const = 0;

    // the occupancy of every cell holder this memory owns (see llama_memory_cells)
    virtual std::vector<llama_memory_cells> cell_usage() const = 0;

    //
    // state write/read
    //

    virtual void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const = 0;
    virtual void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) = 0;
};

using llama_memory_ptr = std::unique_ptr<llama_memory_i>;
