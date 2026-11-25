// Continuum fork (CONTINUUM-ENGINE 4.14): a mind's KV moves through a context (save, load, bind, checkpoint, rewind)
// and their public seam, kept beside llama-context.cpp so upstream's context merges without touching it.
#include "llama-context.h"

#include "llama-impl.h"
#include "llama-memory.h"
#include "llama-persona.h"

#include <new>
#include <stdexcept>

llama_persona_move llama_context::persona_checkpoint(llama_seq_id seq_id, llama_persona_checkpoint_blocks & out, const llama_persona_identity & who) {
    // Her buffers are reused take after take (no copy of the previous checkpoint). Any refused take, an unsupported
    // memory included, leaves no checkpoint at all, so a later rewind refuses (no_checkpoint) instead of restoring a
    // stale one (Codex on #153).
    out.taken = false;
    if (!memory) {
        return llama_persona_move::unsupported;
    }
    synchronize(); // the raw copies read device memory a queued decode may still be writing
    const llama_persona_move checked = memory->persona_check_checkpoint(seq_id, out);
    if (checked != llama_persona_move::done) {
        return checked;
    }
    memory->persona_commit_checkpoint(seq_id, out);
    out.owner = who;
    out.taken = true;
    return llama_persona_move::done;
}

llama_persona_move llama_context::persona_rewind(llama_seq_id seq_id, const llama_persona_checkpoint_blocks & in, const llama_persona_identity & who) {
    if (!memory) {
        return llama_persona_move::unsupported;
    }
    if (in.taken && (in.owner.persona != who.persona || in.owner.epoch != who.epoch)) {
        return llama_persona_move::wrong_owner; // refused before anything is written
    }
    synchronize();
    const llama_persona_move checked = memory->persona_check_rewind(seq_id, in);
    if (checked != llama_persona_move::done) {
        return checked; // refused before anything is written: she stays exactly as she was
    }
    memory->persona_commit_rewind(seq_id, in);
    return llama_persona_move::done;
}

llama_persona_move llama_context::persona_prepare(llama_persona_blocks & blocks) const {
    return memory ? memory->persona_prepare(blocks) : llama_persona_move::unsupported;
}

llama_persona_move llama_context::persona_save(llama_seq_id seq_id, llama_persona_blocks & out, const llama_persona_identity & who) {
    if (!memory) {
        return llama_persona_move::unsupported;
    }
    synchronize(); // a decode returns before its writes land; the raw copies read the same device memory
    const llama_persona_move checked = memory->persona_check_save(seq_id, out);
    if (checked != llama_persona_move::done) {
        return checked;
    }
    memory->persona_commit_save(seq_id, out);
    persona_moved();
    out.owned       = true; // a completed save stamps her blocks with her owner
    out.owner       = who.persona;
    out.owner_epoch = who.epoch;
    return llama_persona_move::done;
}

llama_persona_move llama_context::persona_load(llama_seq_id seq_id, uint32_t stream, llama_persona_blocks & in, const llama_persona_identity & who) {
    if (!memory) {
        return llama_persona_move::unsupported;
    }
    if (!in.belongs_to(who)) {
        return llama_persona_move::wrong_owner; // refused before anything is written
    }
    synchronize();
    llama_persona_slot slot;
    const llama_persona_move checked = memory->persona_check_load(seq_id, stream, in, slot);
    if (checked != llama_persona_move::done) {
        return checked;
    }
    memory->persona_commit_load(seq_id, slot, in);
    persona_moved();
    return llama_persona_move::done;
}

llama_persona_move llama_context::persona_bind(llama_seq_id seq_id, llama_persona_blocks & in, const llama_persona_identity & who) {
    if (!memory) {
        return llama_persona_move::unsupported;
    }
    const int32_t stream = memory->persona_empty_stream();
    if (stream < 0) {
        return llama_persona_move::no_empty_stream;
    }
    return persona_load(seq_id, (uint32_t) stream, in, who);
}

// Continuum fork (CONTINUUM-ENGINE 4.14): the public seam of a persona's move (llama-persona-blocks.h); the persona
// object itself is in llama-persona.h.

// Her owner grows her history ahead of need, between acts: an append on the hot path only writes into it.
int32_t llama_persona_grow_tokens(llama_persona * persona, size_t more_tokens, size_t more_receipts) {
    return (int32_t) (persona->tokens->grow(more_tokens, more_receipts) ? llama_persona_move::done : llama_persona_move::out_of_memory);
}

// As she cools her owner gives back the room she does not hold; nothing she holds is dropped.
int32_t llama_persona_shrink_tokens(llama_persona * persona) {
    return (int32_t) (persona->tokens->shrink() ? llama_persona_move::done : llama_persona_move::out_of_memory);
}

llama_persona_append_receipt llama_persona_append(llama_persona * persona, const llama_persona_append_header * header,
                                                  const llama_token * delta, size_t n) {
    llama_persona_append_receipt out = { (int32_t) llama_persona_append_status::invalid, 0, 0, 0 };
    if (header == nullptr) {
        return out;
    }
    llama_persona_append_request h;
    h.persona    = header->persona;
    h.epoch      = header->epoch;
    h.seq        = header->seq;
    h.frontier   = header->frontier;
    h.request.hi = header->request_hi;
    h.request.lo = header->request_lo;
    const llama_persona_receipt r = persona->tokens->append(h, delta, n);
    out.status = (int32_t) r.status;
    out.seq    = r.seq;
    out.from   = r.from;
    out.to     = r.to;
    return out;
}

const llama_token * llama_persona_token_ids(const llama_persona * persona, uint64_t * n) {
    *n = persona->tokens->frontier();
    return persona->tokens->data();
}

const char * llama_persona_append_status_name(int32_t status) {
    return llama_persona_append_status_name((llama_persona_append_status) status);
}

llama_persona * llama_persona_init(llama_context * ctx, uint64_t persona_id, uint64_t epoch) {
    auto * persona = new (std::nothrow) llama_persona();
    if (persona == nullptr) {
        return nullptr;
    }
    persona->tokens.reset(new (std::nothrow) llama_persona_tokens(persona_id, epoch));
    if (persona->tokens == nullptr || ctx->persona_prepare(persona->blocks) != llama_persona_move::done) {
        delete persona;
        return nullptr;
    }
    return persona;
}

void llama_persona_free(llama_persona * persona) {
    delete persona;
}

// Her blocks move only under her owner (the persona and epoch of her token history): the native save stamps them,
// the native load and bind refuse blocks stamped by anyone else, so no path skips the check.
static llama_persona_identity identity_of(const llama_persona * persona) {
    return { persona->tokens->persona(), persona->tokens->epoch() };
}

int32_t llama_persona_save(llama_context * ctx, llama_seq_id seq_id, llama_persona * persona) {
    return (int32_t) ctx->persona_save(seq_id, persona->blocks, identity_of(persona));
}

int32_t llama_persona_load(llama_context * ctx, llama_seq_id seq_id, uint32_t stream, llama_persona * persona) {
    return (int32_t) ctx->persona_load(seq_id, stream, persona->blocks, identity_of(persona));
}

int32_t llama_persona_bind(llama_context * ctx, llama_seq_id seq_id, llama_persona * persona) {
    return (int32_t) ctx->persona_bind(seq_id, persona->blocks, identity_of(persona));
}

// Whose blocks these are, as stamped by the last completed save (false before any).
bool llama_persona_owner(const llama_persona * persona, uint64_t * persona_id, uint64_t * epoch) {
    if (!persona->blocks.owned) {
        return false;
    }
    *persona_id = persona->blocks.owner;
    *epoch      = persona->blocks.owner_epoch;
    return true;
}

struct llama_persona_checkpoint {
    llama_persona_checkpoint_blocks blocks;
};

llama_persona_checkpoint * llama_persona_checkpoint_init(void) {
    return new (std::nothrow) llama_persona_checkpoint();
}

void llama_persona_checkpoint_free(llama_persona_checkpoint * checkpoint) {
    delete checkpoint;
}

int32_t llama_persona_checkpoint_take(llama_context * ctx, llama_seq_id seq_id, uint64_t persona_id, uint64_t epoch, llama_persona_checkpoint * out) {
    return (int32_t) ctx->persona_checkpoint(seq_id, out->blocks, llama_persona_identity { persona_id, epoch });
}

int32_t llama_persona_checkpoint_rewind(llama_context * ctx, llama_seq_id seq_id, uint64_t persona_id, uint64_t epoch, const llama_persona_checkpoint * in) {
    return (int32_t) ctx->persona_rewind(seq_id, in->blocks, llama_persona_identity { persona_id, epoch });
}

llama_pos llama_persona_checkpoint_pos(const llama_persona_checkpoint * checkpoint) {
    return checkpoint->blocks.taken ? checkpoint->blocks.pos : -1;
}

size_t llama_persona_checkpoint_bytes(const llama_persona_checkpoint * checkpoint) {
    // what its buffers hold allocated (capacity, not the last take's size): what the governor must count
    size_t bytes = 0;
    for (const auto & row : checkpoint->blocks.rs.r) {
        bytes += row.capacity();
    }
    for (const auto & row : checkpoint->blocks.rs.s) {
        bytes += row.capacity();
    }
    return bytes;
}

const char * llama_persona_status_name(int32_t status) {
    return llama_persona_move_name((llama_persona_move) status);
}

// A move by pointer leaves every built graph's KV views at addresses the stream no longer owns, so none is reused:
// the next decode builds against the stream's new storage (the same reset a memory update takes, llama-context.cpp).
void llama_context::persona_moved() {
    for (auto & res : gf_res_prev) {
        res->reset();
    }
    gf_res_prev_active = nullptr;
}

size_t llama_persona_storage_bytes(const llama_context * ctx) {
    const llama_memory_t memory = llama_get_memory(ctx);
    return memory ? memory->persona_storage_bytes() : 0;
}

int32_t llama_persona_storage_attach(llama_context * ctx, llama_persona * persona, void * base, size_t bytes) {
    const llama_memory_t memory = llama_get_memory(ctx);
    if (memory == nullptr) {
        return (int32_t) llama_persona_move::unsupported;
    }
    return (int32_t) memory->persona_attach_storage(persona->blocks, base, bytes);
}

bool llama_persona_storage_viewable(const llama_context * ctx) {
    const llama_memory_t memory = llama_get_memory(ctx);
    return memory && memory->persona_storage_viewable();
}

int32_t llama_persona_storage_reserve(llama_context * ctx, llama_persona * persona) {
    const llama_memory_t memory = llama_get_memory(ctx);
    if (memory == nullptr) {
        return (int32_t) llama_persona_move::unsupported;
    }
    return (int32_t) memory->persona_reserve_storage(persona->blocks);
}
