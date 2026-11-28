// Continuum: a decode's outputs kept readable while the next decode is queued behind it (llama-output-ring.h)

#include "llama-output-ring.h"

#include "llama-context.h"
#include "llama-model.h"

#include <utility>

bool llama_context::set_output_ring(bool on) {
    synchronize();
    auto & ring = output_ring;
    ring.on = false;
    ring.spare.reset();
    for (auto & slot : ring.slots) {
        slot = llama_output_ticket_slot{};
    }
    if (!on) {
        return true;
    }
    // the output copies run on the output device's backend, so its events say when they landed
    ggml_backend_dev_t dev = model.dev_output();
    for (auto & backend : backends) {
        if (ggml_backend_get_device(backend.get()) == dev) {
            ring.backend = backend.get();
        }
    }
    if (ring.backend == nullptr) {
        return false;
    }
    for (auto & slot : ring.slots) {
        slot.event.reset(ggml_backend_event_new(dev));
        if (!slot.event) {
            return false;
        }
    }
    ring.on = true;
    return true;
}

// before a decode reserves its outputs: it writes the spare, so the latest decode's outputs stay where they are and
// become the one before; what the decode before that held is overwritten
void llama_context::output_ring_swap() {
    auto & ring = output_ring;
    if (!ring.on) {
        return;
    }
    buf_output.swap(ring.spare);
    std::swap(ring.slots[0], ring.slots[1]);
    ring.slots[0].ticket = -1; // being written
}

// once the decode's output copies are queued
void llama_context::output_ring_record() {
    auto & ring = output_ring;
    if (!ring.on) {
        return;
    }
    auto & slot = ring.slots[0];
    slot.ticket    = ring.decodes++;
    slot.n_outputs = n_outputs;
    slot.logits    = logits.data;
    ggml_backend_event_record(slot.event.get(), ring.backend);
}

int64_t llama_context::output_ticket() const {
    return output_ring.on ? output_ring.slots[0].ticket : -1;
}

float * llama_context::ticket_logits(int64_t ticket, int32_t j) {
    for (auto & slot : output_ring.slots) {
        if (ticket < 0 || slot.ticket != ticket) {
            continue;
        }
        if (j < 0 || j >= slot.n_outputs || slot.logits == nullptr) {
            return nullptr;
        }
        ggml_backend_event_synchronize(slot.event.get()); // this decode's outputs only, never one queued after it
        return slot.logits + (size_t) j * model.vocab.n_tokens();
    }
    return nullptr;
}

bool llama_set_output_ring(llama_context * ctx, bool on) {
    return ctx->set_output_ring(on);
}

int64_t llama_output_ticket(const llama_context * ctx) {
    return ctx->output_ticket();
}

float * llama_ticket_logits(llama_context * ctx, int64_t ticket, int32_t j) {
    return ctx->ticket_logits(ticket, j);
}
