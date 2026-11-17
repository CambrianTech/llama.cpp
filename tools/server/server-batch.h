#pragma once

// Continuum fork: the serving loop's batch, out of server-context.cpp so it can be tested without a model
// (tests/test-server-batch.cpp). A pure move; CONTINUUM-ENGINE 4.11.10 adds removing one slot's tokens from a frame.

#include "common.h"
#include "llama.h"

#include <algorithm>
#include <vector>

struct server_slot; // forward declaration

struct server_batch {
    llama_batch batch;
    bool batch_rendered = false;

    struct token {
        int32_t id_slot;
        llama_token token;
        llama_pos pos;
        bool output;
        bool is_prompt; // for stats tracking
    };
    std::vector<token> tokens;
    int32_t n_tokens_alloc = 0;
    int32_t n_embd = 0;

    // track if given slot can be batched with slots already in the batch
    server_slot * slot_batched = nullptr;

    // in embd mode, we temporarily swap out the tokens arr and restore it on clear()
    bool has_embd = false;
    llama_token * tokens_ptr = nullptr;
    std::vector<float> embd;

    float  alora_scale       = -1.0f;
    size_t alora_disabled_id = 0;

    server_batch() {
        batch.pos = nullptr; // sentinel: uninitialized batch
    }

    ~server_batch() {
        if (batch.pos != nullptr) {
            clear();
            llama_batch_free(batch);
        }
    }

    void init(int32_t n_tokens_alloc, int32_t n_embd) {
        this->n_tokens_alloc = n_tokens_alloc;
        this->n_embd = n_embd;
        batch = llama_batch_init(n_tokens_alloc, 0, 1);
        tokens_ptr = batch.token;
        tokens.reserve(n_tokens_alloc);
    }

    bool add(int32_t id_slot, llama_token token, llama_pos pos, bool output, bool is_prompt) {
        GGML_ASSERT(!has_embd); // cannot mix tokens + embd in same batch
        GGML_ASSERT(batch.pos != nullptr);
        if ((int32_t)tokens.size() >= n_tokens_alloc) {
            return false;
        }
        tokens.push_back({ id_slot, token, pos, output, is_prompt });
        return true;
    }

    bool add(int32_t id_slot, const std::vector<float> & embd_in, llama_pos pos, bool output, bool is_prompt) {
        GGML_ASSERT(batch.pos != nullptr);
        if ((int32_t)tokens.size() >= n_tokens_alloc) {
            return false;
        }
        tokens.push_back({ id_slot, LLAMA_TOKEN_NULL, pos, output, is_prompt });
        has_embd = true;
        embd.insert(embd.end(), embd_in.begin(), embd_in.end());
        return true;
    }

    void clear() {
        tokens.clear();
        embd.clear();
        common_batch_clear(batch);
        slot_batched      = nullptr;
        alora_scale       = -1.0f;
        alora_disabled_id = 0;
        batch_rendered    = false;
        has_embd          = false;
        if (batch.token == nullptr) {
            batch.token = tokens_ptr;
            batch.embd  = nullptr;
        }
    }

    int32_t size() const {
        return (int32_t)tokens.size();
    }

    // what a slot finds in the batch before it adds to it
    struct mark {
        int32_t       n_tokens;
        size_t        n_embd;
        bool          has_embd;
        server_slot * slot_batched;
        float         alora_scale;
        size_t        alora_disabled_id;
    };

    mark get_mark() const {
        return { size(), embd.size(), has_embd, slot_batched, alora_scale, alora_disabled_id };
    }

    // undo everything added since the mark, so a slot that fails leaves the batch as it found it
    void rollback(const mark & m) {
        GGML_ASSERT(!batch_rendered);
        GGML_ASSERT(m.n_tokens <= size() && m.n_embd <= embd.size());
        tokens.resize(m.n_tokens);
        embd.resize(m.n_embd);
        has_embd          = m.has_embd;
        slot_batched      = m.slot_batched;
        alora_scale       = m.alora_scale;
        alora_disabled_id = m.alora_disabled_id;
    }

    // Remove one slot's tokens at index >= `from` from a frame decode could not place (CONTINUUM-ENGINE 4.11.10:
    // that slot is preempted or deferred, the others decode). A frame decodes in pieces; when the piece at `from`
    // finds no room, the pieces before it are already placed and computed, so only tokens from `from` on can leave,
    // and the caller cuts the slot's own bookkeeping back to what was placed. Order is kept, embedding rows move with
    // their tokens, and the batch must be rendered again before it decodes. Returns old index -> new index (-1 for a
    // removed token): the caller remaps every index it holds into this batch (each slot's i_batch and spec_i_batch)
    // and clears slot_batched if it was this slot; the batch cannot, since it sees server_slot only as a declaration.
    std::vector<int32_t> remove_slot(int32_t id_slot, int32_t from) {
        std::vector<int32_t> moved(tokens.size(), -1);
        size_t kept = 0;
        for (size_t i = 0; i < tokens.size(); ++i) {
            if (tokens[i].id_slot == id_slot && (int32_t) i >= from) {
                continue;
            }
            if (has_embd && kept != i) {
                std::copy_n(embd.begin() + i * n_embd, n_embd, embd.begin() + kept * n_embd);
            }
            tokens[kept] = tokens[i];
            moved[i] = (int32_t) kept++;
        }
        tokens.resize(kept);
        if (has_embd) {
            embd.resize(kept * n_embd);
        }
        if (batch_rendered) {
            common_batch_clear(batch);
            if (batch.token == nullptr) {
                batch.token = tokens_ptr;
                batch.embd  = nullptr;
            }
            batch_rendered = false;
        }
        return moved;
    }

    void set_output(int32_t idx, bool output) {
        GGML_ASSERT(idx >= 0 && idx < (int32_t)tokens.size());
        tokens[idx].output = output;
    }

    void render() {
        GGML_ASSERT(!batch_rendered);
        GGML_ASSERT(batch.pos != nullptr);
        common_batch_clear(batch);
        for (int32_t i = 0; i < size(); i++) {
            const auto & t = tokens[i];
            common_batch_add(batch, t.token, t.pos, { t.id_slot }, t.output);
        }
        if (has_embd) {
            batch.token = nullptr; // will be restored on clear()
            batch.embd  = embd.data();
        }
        batch_rendered = true;
    }

    llama_batch get_view(int32_t off, int32_t n_tokens) const {
        GGML_ASSERT(batch.pos != nullptr);
        GGML_ASSERT(batch_rendered);
        GGML_ASSERT(off >= 0 && off < size());
        GGML_ASSERT(n_tokens > 0 && off + n_tokens <= size());

        auto * token = batch.token ? batch.token + off          : nullptr;
        auto * embd  = batch.embd  ? batch.embd  + off * n_embd : nullptr;

        llama_batch view = {
            n_tokens,
            token,
            embd,
            batch.pos      + off,
            batch.n_seq_id + off,
            batch.seq_id   + off,
            batch.logits   + off,
            batch.label ? batch.label + off : nullptr,
            batch.label && batch.label_weight ? batch.label_weight + off : nullptr,
        };

        return view;
    }
};
