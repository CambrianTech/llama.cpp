// Continuum: a mind's KV moves (CONTINUUM-ENGINE 4.14): her stream's rows saved, loaded, checkpointed and
// rewound as one unit, kept beside llama-kv-cache.cpp so upstream's cache merges without touching it.
#include "llama-kv-cache.h"

#include "llama-impl.h"
#include "llama-io.h"
#include "llama-model.h"
#include "llama-context.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <stdexcept>

static bool cells_prepared(const llama_kv_cells & cells, uint32_t size) {
    return cells.size() == size && cells.get_used() == 0 && !cells.get_has_shift();
}

// this memory's attention layout: what her rows' sizes and meaning depend on
static uint64_t kv_layout_mix(uint64_t h, uint64_t v) { return (h ^ v) * 0x100000001b3ull; }

uint64_t llama_kv_cache::compute_persona_layout(bool with_size) const {
    uint64_t h = kv_layout_mix(0xcbf29ce484222325ull, layers.size());
    // the stream size is part of a live move's layout; a file restores into any stream her rows fit
    h = with_size ? kv_layout_mix(h, get_size()) : h;
    for (const auto & layer : layers) {
        h = kv_layout_mix(h, layer.il);
        h = kv_layout_mix(h, layer.k ? (uint64_t) layer.k->type : 0xffu);
        h = kv_layout_mix(h, layer.k ? (uint64_t) layer.k->ne[0] : 0);
        h = kv_layout_mix(h, layer.v ? (uint64_t) layer.v->type : 0xffu);
        h = kv_layout_mix(h, layer.v ? (uint64_t) layer.v->ne[0] : 0);
    }
    return h;
}

// Continuum fork: a persona move copies one sequence's own stream, so it needs every stream to belong to
// exactly one sequence. That holds with a stream per sequence, and with one stream and one sequence (the
// Core's context: the mind swapped in holds the whole of it). It does not hold for one stream shared by
// several sequences (unified KV), where her rows are interleaved with others'.
bool llama_kv_cache::persona_streams_are_own() const {
    return n_stream > 1 || n_seq_max == 1;
}

llama_persona_move llama_kv_cache::persona_prepare(llama_persona_blocks & blocks) const {
    if (!persona_streams_are_own()) {
        return llama_persona_move::no_such_stream;
    }
    blocks.kv_layout = persona_layout();
    blocks.kv_rows   = persona_rows_id;
    blocks.kv.cells.resize(get_size()); // once per persona, at her creation
    if (persona_by_pointer() && !persona_reserve_storage(blocks.kv.storage)) {
        return llama_persona_move::out_of_memory; // her own storage, moved by pointer from here on
    }
    blocks.kv.k.resize(layers.size());
    blocks.kv.v.resize(layers.size());
    // the row sizes a save copies (persona_check_save), recorded for a file read to check before it allocates
    blocks.geometry.k_row.resize(layers.size());
    blocks.geometry.v_row.resize(layers.size());
    for (size_t i = 0; i < layers.size(); ++i) {
        const auto & layer = layers[i];
        blocks.geometry.k_row[i] = ggml_row_size(layer.k->type, layer.k->ne[0]);
        blocks.geometry.v_row[i] = layer.v ? ggml_row_size(layer.v->type, layer.v->ne[0]) : 0;
    }
    return llama_persona_move::done;
}

llama_persona_move llama_kv_cache::persona_check_save(llama_seq_id seq_id, llama_persona_blocks & out) const {
    if (!persona_streams_are_own()) {
        return llama_persona_move::no_such_stream;
    }
    if (seq_id < 0 || (size_t) seq_id >= seq_to_stream.size()) {
        return llama_persona_move::invalid_seq;
    }
    if (v_trans) {
        return llama_persona_move::transposed_v;
    }
    if (out.kv_layout != persona_layout()) {
        return llama_persona_move::layout_mismatch;
    }
    if (!cells_prepared(out.kv.cells, get_size()) || out.kv.k.size() != layers.size() || out.kv.v.size() != layers.size()) {
        return llama_persona_move::cells_not_prepared;
    }
    const uint32_t n_rows = v_cells[seq_to_stream[seq_id]].used_max_p1();
    if (persona_by_pointer()) {
        out.kv.n_rows = n_rows; // her rows stay where she wrote them: the commit exchanges storage
        return out.kv.storage.held() ? llama_persona_move::done : llama_persona_move::cells_not_prepared;
    }
    // reserve: every byte the commit copies, so the commit never allocates
    for (size_t i = 0; i < layers.size(); ++i) {
        const auto & layer = layers[i];
        out.kv.k[i].resize(ggml_row_size(layer.k->type, layer.k->ne[0]) * n_rows);
        out.kv.v[i].resize(layer.v ? ggml_row_size(layer.v->type, layer.v->ne[0]) * n_rows : 0);
    }
    out.kv.n_rows = n_rows;
    return llama_persona_move::done;
}

void llama_kv_cache::persona_commit_save(llama_seq_id seq_id, llama_persona_blocks & out) {
    const uint32_t s = seq_to_stream[seq_id];
    auto & cells = v_cells[s];
    if (persona_by_pointer()) {
        persona_exchange_storage(out.kv.storage); // she keeps the storage she decoded in
        out.seq = seq_id;
        out.kv.cells.swap(cells);
        v_heads[s] = 0;
        return;
    }
    for (size_t i = 0; i < layers.size(); ++i) {
        const auto & layer = layers[i];
        if (!out.kv.k[i].empty()) {
            ggml_backend_tensor_get(layer.k_stream[s], out.kv.k[i].data(), 0, out.kv.k[i].size());
        }
        if (!out.kv.v[i].empty()) {
            ggml_backend_tensor_get(layer.v_stream[s], out.kv.v[i].data(), 0, out.kv.v[i].size());
        }
    }
    out.seq = seq_id;
    out.kv.cells.swap(cells); // hers out; the stream takes the prepared empty object
    v_heads[s] = 0;
}

// Attention KV is never copied by a checkpoint: it stays in place, and only her position is recorded, so a rewind can
// cut what she holds past it.
llama_persona_move llama_kv_cache::persona_check_checkpoint(llama_seq_id seq_id, llama_persona_checkpoint_blocks & out) const {
    if (seq_id < 0 || (size_t) seq_id >= seq_to_stream.size()) {
        return llama_persona_move::invalid_seq;
    }
    out.pos = v_cells[seq_to_stream[seq_id]].seq_pos_max(seq_id);
    auto & cells = v_cells[seq_to_stream[seq_id]];
    out.lineage = cells.lineage();
    try {
        out.mark = std::make_shared<const llama_pos>(out.pos);
        cells.prepare_checkpoint();
    } catch (const std::bad_alloc &) {
        return llama_persona_move::out_of_memory;
    }
    return llama_persona_move::done;
}

void llama_kv_cache::persona_commit_checkpoint(llama_seq_id seq_id, llama_persona_checkpoint_blocks & out) {
    out.seq = seq_id;
    v_cells[seq_to_stream[seq_id]].mark_checkpoint(out.mark);
}

llama_persona_move llama_kv_cache::persona_check_rewind(llama_seq_id seq_id, const llama_persona_checkpoint_blocks & in) const {
    if (seq_id < 0 || (size_t) seq_id >= seq_to_stream.size()) {
        return llama_persona_move::invalid_seq;
    }
    if (!in.taken) {
        return llama_persona_move::no_checkpoint;
    }
    if (in.pos > v_cells[seq_to_stream[seq_id]].seq_pos_max(seq_id)) {
        return llama_persona_move::checkpoint_ahead;
    }
    // the cells must be the ones it was taken from, uncut at or below its position since (Codex on #153: a slot
    // reused by another mind must never receive these rows)
    const auto & cells = v_cells[seq_to_stream[seq_id]];
    if (cells.lineage() != in.lineage || !cells.holds_checkpoint(in.mark)) {
        return llama_persona_move::checkpoint_stale;
    }
    return llama_persona_move::done;
}

void llama_kv_cache::persona_commit_rewind(llama_seq_id seq_id, const llama_persona_checkpoint_blocks & in) {
    // what she holds past the checkpoint is cut; what she held up to it is untouched in place
    seq_rm(seq_id, in.pos + 1, -1);
}

int32_t llama_kv_cache::persona_empty_stream() const {
    if (!persona_streams_are_own()) {
        return -1; // unified: one stream holds every sequence, so there is nothing empty to bind into
    }
    for (uint32_t s = 0; s < n_stream; ++s) {
        if (v_cells[s].get_used() == 0) {
            return (int32_t) s;
        }
    }
    return -1;
}

llama_persona_move llama_kv_cache::persona_check_load(llama_seq_id seq_id, uint32_t stream,
                                                     const llama_persona_blocks & in, llama_persona_slot & slot) const {
    if (!persona_streams_are_own() || stream >= n_stream) {
        return llama_persona_move::no_such_stream;
    }
    if (seq_id < 0 || (size_t) seq_id >= seq_to_stream.size()) {
        return llama_persona_move::invalid_seq;
    }
    if (v_trans) {
        return llama_persona_move::transposed_v;
    }
    if (v_cells[stream].get_used() != 0) {
        return llama_persona_move::stream_not_empty;
    }
    if (v_cells[seq_to_stream[seq_id]].seq_pos_max(seq_id) >= 0) {
        return llama_persona_move::seq_holds_state;
    }
    if (in.kv_layout != persona_layout()) {
        return llama_persona_move::layout_mismatch;
    }
    const auto & kv = in.kv;
    if (kv.cells.size() != get_size() || kv.n_rows > get_size() || kv.cells.used_max_p1() > kv.n_rows ||
        kv.k.size() != layers.size() || kv.v.size() != layers.size()) {
        return llama_persona_move::layout_mismatch;
    }
    if (persona_by_pointer()) {
        if (!kv.storage.held()) {
            return llama_persona_move::cells_not_prepared;
        }
        slot.stream = stream;
        return llama_persona_move::done;
    }
    for (size_t i = 0; i < layers.size(); ++i) {
        const auto & layer = layers[i];
        const size_t k_bytes = ggml_row_size(layer.k->type, layer.k->ne[0]) * kv.n_rows;
        const size_t v_bytes = layer.v ? ggml_row_size(layer.v->type, layer.v->ne[0]) * kv.n_rows : 0;
        if (kv.k[i].size() != k_bytes || kv.v[i].size() != v_bytes) {
            return llama_persona_move::layout_mismatch;
        }
    }
    slot.stream = stream;
    return llama_persona_move::done;
}

void llama_kv_cache::persona_commit_load(llama_seq_id seq_id, llama_persona_slot & slot, llama_persona_blocks & in) {
    const uint32_t s = slot.stream;
    if (persona_by_pointer()) {
        persona_exchange_storage(in.kv.storage); // hers becomes the stream's; she keeps the stream's empty one
    }
    for (size_t i = 0; i < layers.size() && !persona_by_pointer(); ++i) {
        const auto & layer = layers[i];
        if (!in.kv.k[i].empty()) {
            ggml_backend_tensor_set(layer.k_stream[s], in.kv.k[i].data(), 0, in.kv.k[i].size());
        }
        if (!in.kv.v[i].empty()) {
            ggml_backend_tensor_set(layer.v_stream[s], in.kv.v[i].data(), 0, in.kv.v[i].size());
        }
    }
    in.kv.cells.swap(v_cells[s]); // hers in; her blocks keep the stream's empty object for her next save
    v_cells[s].seq_rename(in.seq, seq_id); // her cells answer to the sequence she is bound as now
    v_heads[s] = 0;
    // bind her own sequence id to stream s; the sequence that held s takes her old stream (both empty of others)
    const uint32_t old = seq_to_stream[seq_id];
    for (auto & st : seq_to_stream) {
        if (st == s) {
            st = old;
            break;
        }
    }
    seq_to_stream[seq_id] = s;
}


llama_memory_context_ptr llama_kv_cache::init_full_single_stream() {
    return std::make_unique<llama_kv_cache_context>(this, true);
}


// One holder per stream: a sequence draws only from its own stream's cells, so streams are never
// summed (a sum would report cells a request cannot use). A unified cache is one stream, one holder.
std::vector<llama_memory_cells> llama_kv_cache::cell_usage() const {
    std::vector<llama_memory_cells> out;
    for (size_t s = 0; s < v_cells.size(); ++s) {
        const auto & cells = v_cells[s];
        out.push_back({ v_cells.size() == 1 ? std::string() : "s" + std::to_string(s), cells.get_used(), cells.size() });
    }
    return out;
}
