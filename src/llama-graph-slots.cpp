// Continuum: each mind's decode graph kept built and allocated across a switch (llama-graph-slots.h)

#include "llama-graph-slots.h"

#include "llama-context.h"
#include "llama-model.h"

#include <algorithm>

// the context and the slot exchange scheduler, graph results, active graph and adapter set: entering and leaving are
// the same four pointer swaps
void llama_context::graph_slot_swap(llama_graph_slot & slot) {
    sched.swap(slot.sched);
    gf_res_prev.swap(slot.gf_res_prev);
    std::swap(gf_res_prev_active, slot.gf_res_prev_active);
    loras.swap(slot.loras);
}

void llama_context::graph_slot_leave() {
    if (graph_slots.active) {
        graph_slot_swap(*graph_slots.active);
        graph_slots.active = nullptr;
    }
}

// a slot is freed only once the backends finished with its buffers (its last step may still be in flight)
static void graph_slot_free(std::unique_ptr<llama_graph_slot> & slot) {
    ggml_backend_sched_synchronize(slot->sched.get());
    slot.reset();
}

// whom a sequence serves: the owner's name for her, or the sequence itself when the owner named no one
static uint64_t graph_slot_key(const llama_graph_slots & gs, llama_seq_id seq) {
    const auto it = gs.named.find(seq);
    return it != gs.named.end() ? it->second : (UINT64_C(1) << 63) | (uint64_t) (uint32_t) seq;
}

// Each mind's steps keep their graph in her own slot (keyed by whom the ubatch's sequences serve). Whether her kept
// graph serves this step is the graph's own reuse check (its K/V extent, adapter set, pin and every other input),
// never a key derived here; when it cannot, her step rebuilds in her slot, and no other mind's graph is touched.
void llama_context::graph_slot_enter(const llama_ubatch & ubatch) {
    auto & gs = graph_slots;

    // one token per sequence, nothing to learn, and no live owner holding this context's scheduler or its graph
    const bool step = gs.in_decode && gs.capacity > 0 && ubatch.n_tokens == ubatch.n_seqs_unq && ubatch.n_labels() == 0 &&
        !opt_serving_continuation && !(opt_reply && opt_reply->graph_meta);
    if (!step) {
        graph_slot_leave();
        return;
    }

    gs.key.clear();
    for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
        gs.key.push_back(graph_slot_key(gs, ubatch.seq_id_unq[s]));
    }
    if (gs.active && gs.active->key == gs.key) { // the next ubatch of the same decode
        gs.active->used = ++gs.clock;
        return;
    }
    graph_slot_leave();

    // a slot made before an adapter was freed may name an address a new adapter now holds
    const uint64_t frees = model.lora_frees.load(std::memory_order_acquire);
    for (auto & slot : gs.slots) {
        if (slot->lora_frees != frees) {
            graph_slot_free(slot);
        }
    }
    gs.slots.erase(std::remove(gs.slots.begin(), gs.slots.end(), nullptr), gs.slots.end());

    auto own = std::find_if(gs.slots.begin(), gs.slots.end(), [&](const auto & slot) { return slot->key == gs.key; });
    llama_graph_slot * slot = nullptr;
    if (own != gs.slots.end()) {
        slot = own->get();
    } else if (gs.slots.size() < gs.capacity) {
        auto made = std::make_unique<llama_graph_slot>();
        made->sched.reset(ggml_backend_sched_new(backend_ptrs.data(), backend_buft.data(), backend_ptrs.size(),
                sched_params_cur.max_nodes, sched_params_cur.parallel, sched_params_cur.op_offload));
        ggml_backend_sched_set_copy_callback(made->sched.get(), sched_copy_experts, this);
        ggml_backend_sched_set_realloc_callback(made->sched.get(), sched_graph_realloc, this);
        for (auto & res : made->gf_res_prev) {
            res = std::make_unique<llm_graph_result>(sched_params_cur.max_nodes);
        }
        made->loras      = std::make_unique<llama_adapter_loras>();
        made->lora_frees = frees;
        gs.slots.push_back(std::move(made));
        slot = gs.slots.back().get();
    } else { // a mind with no slot takes the one served longest ago
        slot = std::min_element(gs.slots.begin(), gs.slots.end(), [](const auto & a, const auto & b) { return a->used < b->used; })->get();
    }

    // her kept graph wears the set it was built with; a step wearing another (by value: addresses and scales, never
    // dereferenced) rebuilds with a copy of the set it wears
    const bool same_set = slot->key == gs.key && *slot->loras == *loras;
    if (!same_set) {
        slot->key   = gs.key;
        slot->loras = std::make_unique<llama_adapter_loras>(*loras);
        slot->gf_res_prev_active = nullptr;
    }
    slot->used = ++gs.clock;
    gs.active  = slot;
    graph_slot_swap(*slot);
}

void llama_context::set_graph_slot_key(llama_seq_id seq, uint64_t key) {
    graph_slots.named[seq] = key;
}

// the scheduler was re-reserved (its node budget or parameters changed): every slot was made for the old one
void llama_context::graph_slots_drop() {
    graph_slot_leave();
    for (auto & slot : graph_slots.slots) {
        graph_slot_free(slot);
    }
    graph_slots.slots.clear();
}

// no slot's graph may be reused across a change it cannot see; the schedulers and their buffers stay
void llama_context::graph_slots_forget() {
    for (auto & slot : graph_slots.slots) {
        for (auto & res : slot->gf_res_prev) {
            res->reset();
        }
        slot->gf_res_prev_active = nullptr;
    }
}

void llama_context::set_graph_slots(size_t n) {
    graph_slot_leave();
    graph_slots.capacity = n;
    auto & slots = graph_slots.slots;
    while (slots.size() > n) {
        auto lru = std::min_element(slots.begin(), slots.end(), [](const auto & a, const auto & b) { return a->used < b->used; });
        graph_slot_free(*lru);
        slots.erase(lru);
    }
}

size_t llama_context::graph_slots_bytes() const {
    size_t bytes = 0;
    for (const auto & slot : graph_slots.slots) {
        ggml_backend_sched * s = slot.get() == graph_slots.active ? sched.get() : slot->sched.get();
        for (auto * backend : backend_ptrs) {
            bytes += ggml_backend_sched_get_buffer_size(s, backend);
        }
    }
    return bytes;
}

// every slot's compute buffers, counted beside the context's own
void llama_context::graph_slots_breakdown(std::map<ggml_backend_buffer_type_t, llama_memory_breakdown_data> & ret, ggml_backend_t backend) const {
    for (const auto & slot : graph_slots.slots) {
        ggml_backend_sched * s = slot.get() == graph_slots.active ? sched.get() : slot->sched.get();
        ret[ggml_backend_sched_get_buffer_type(s, backend)].compute +=
            ggml_backend_sched_get_buffer_size(s, backend);
    }
}

void llama_set_graph_slots(llama_context * ctx, size_t n) {
    ctx->set_graph_slots(n);
}

size_t llama_graph_slots_bytes(const llama_context * ctx) {
    return ctx->graph_slots_bytes();
}

void llama_set_graph_slot_key(llama_context * ctx, llama_seq_id seq, uint64_t key) {
    ctx->set_graph_slot_key(seq, key);
}
