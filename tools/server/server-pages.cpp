#include "server-pages.h"

#include "speculative.h"

#include <new>

size_t server_page_bytes(size_t n_state, size_t n_spec, size_t n_checkpoint_bytes) {
    return n_state + n_spec + n_checkpoint_bytes;
}

server_page_charge::~server_page_charge() {
    if (held) {
        held->fetch_sub(n);
    }
}

size_t server_page::n_checkpoint_bytes() const {
    size_t n = 0;
    for (const auto & c : prompt.checkpoints) {
        n += c.size();
    }
    return n;
}

size_t server_page::n_bytes() const {
    return server_page_bytes(data.size(), spec.size(), n_checkpoint_bytes());
}

server_page_shape server_page_measure(llama_context * ctx_tgt, llama_context * ctx_dft, common_speculative * spec, llama_seq_id seq) {
    server_page_shape shape;
    shape.n_tgt = llama_state_seq_get_size_ext(ctx_tgt, seq, LLAMA_STATE_SEQ_FLAGS_NONE);
    shape.n_dft = ctx_dft ? llama_state_seq_get_size_ext(ctx_dft, seq, LLAMA_STATE_SEQ_FLAGS_NONE) : 0;
    common_speculative_get_state(spec, seq, shape.spec);
    return shape;
}

server_page_ptr server_page_capture(llama_context * ctx_tgt, llama_context * ctx_dft, llama_seq_id seq, server_page_shape shape) {
    const size_t n_tgt = shape.n_tgt;
    const size_t n_dft = shape.n_dft;
    if (n_tgt == 0) {
        throw server_page_error(ERROR_TYPE_SERVER, "the target sequence state could not be sized");
    }
    server_page_ptr page;
    try {
        page = std::make_shared<server_page>();
        page->data.main.resize(n_tgt);
        page->data.drft.resize(n_dft);
    } catch (const std::bad_alloc &) {
        throw server_page_error(ERROR_TYPE_UNAVAILABLE, "host RAM for the page could not be allocated");
    }
    if (llama_state_seq_get_data_ext(ctx_tgt, page->data.main.data(), n_tgt, seq, LLAMA_STATE_SEQ_FLAGS_NONE) != n_tgt) {
        throw server_page_error(ERROR_TYPE_SERVER, "the target sequence state read short");
    }
    if (n_dft > 0 && llama_state_seq_get_data_ext(ctx_dft, page->data.drft.data(), n_dft, seq, LLAMA_STATE_SEQ_FLAGS_NONE) != n_dft) {
        throw server_page_error(ERROR_TYPE_SERVER, "the draft sequence state read short");
    }
    page->spec = std::move(shape.spec);
    return page;
}

void server_page_restore(const server_page & page, llama_context * ctx_tgt, llama_context * ctx_dft, common_speculative * spec, llama_seq_id seq) {
    const auto & tgt = page.data.main;
    const auto & dft = page.data.drft;
    if (llama_state_seq_set_data_ext(ctx_tgt, tgt.data(), tgt.size(), seq, LLAMA_STATE_SEQ_FLAGS_NONE) != tgt.size()) {
        throw server_page_error(ERROR_TYPE_SERVER, "the target sequence state did not fit the context (KV cache full?)");
    }
    if (!dft.empty()) {
        if (ctx_dft == nullptr) {
            throw server_page_error(ERROR_TYPE_INVALID_REQUEST, "the page carries a draft state and this server has no draft context");
        }
        if (llama_state_seq_set_data_ext(ctx_dft, dft.data(), dft.size(), seq, LLAMA_STATE_SEQ_FLAGS_NONE) != dft.size()) {
            throw server_page_error(ERROR_TYPE_SERVER, "the draft sequence state did not fit the draft context");
        }
    } else if (ctx_dft != nullptr) {
        throw server_page_error(ERROR_TYPE_INVALID_REQUEST, "the page carries no draft state and this server drafts: the draft would be cold");
    }
    if (spec != nullptr && !page.spec.empty()) {
        common_speculative_set_state(spec, seq, page.spec);
    }
}

//
// server_page_store
//

void server_page_store::admit(const std::string & name, size_t n_bytes_new) const {
    if (limit_bytes == 0) {
        return;
    }
    size_t live = n_bytes();
    const auto it = pages.find(name);
    // only this thread makes owners, so a sole owner seen here stays sole until the replacement frees it
    if (it != pages.end() && it->second.use_count() == 1) {
        live -= it->second->charge.n;
    }
    if (live + n_bytes_new > limit_bytes) {
        throw server_page_error(ERROR_TYPE_UNAVAILABLE, string_format(
            "the live pages hold %zu bytes; a page of %zu bytes would exceed the ceiling of %llu bytes: spill or drop a page first",
            live, n_bytes_new, (unsigned long long) limit_bytes));
    }
}

uint64_t server_page_store::put(const std::string & name, const server_page_ptr & page, server_prompt * from) {
    GGML_ASSERT(page->charge.held == nullptr && "a page is stored once");
    const auto give_back = [&]() {
        if (from != nullptr) {
            *from = std::move(page->prompt);
        }
    };
    if (from != nullptr) {
        page->prompt = std::move(*from);
    }
    std::map<std::string, server_page_ptr>::iterator it;
    try {
        admit(name, page->n_bytes());
        it = pages.try_emplace(name).first;
        if (it->second) {
            SRV_WRN("page '%s' replaced (%zu bytes, freed when no spill holds them)\n", name.c_str(), it->second->charge.n);
        }
    } catch (const std::bad_alloc &) {
        give_back();
        throw server_page_error(ERROR_TYPE_UNAVAILABLE, "host RAM for the page's entry could not be allocated");
    } catch (...) {
        give_back();
        throw;
    }
    // nothing below can fail: the page is stored whole or not at all
    page->gen         = next_gen++;
    page->charge.n    = page->n_bytes();
    page->charge.held = held;
    held->fetch_add(page->charge.n);
    it->second = page;
    return page->gen;
}

server_page_ptr server_page_store::find(const std::string & name) const {
    const auto it = pages.find(name);
    return it == pages.end() ? nullptr : it->second;
}

bool server_page_store::erase(const std::string & name, uint64_t gen) {
    const auto it = pages.find(name);
    if (it == pages.end() || (gen != 0 && it->second->gen != gen)) {
        return false;
    }
    pages.erase(it);
    return true;
}

void server_page_store::evict_lora(const llama_adapter_lora * adapter) {
    for (auto it = pages.begin(); it != pages.end();) {
        bool uses = false;
        for (const auto & l : it->second->lora) {
            uses = uses || (l.ptr == adapter && l.scale != 0.0f);
        }
        if (uses) {
            SRV_WRN("page '%s' dropped: its adapter is being unloaded (%zu bytes)\n", it->first.c_str(), it->second->charge.n);
            it = pages.erase(it);
        } else {
            ++it;
        }
    }
}

bool server_page_name_valid(const std::string & name) {
    if (name.empty() || name.size() > 255) {
        return false;
    }
    for (const unsigned char c : name) {
        if (c < 0x20 || c == 0x7f) {
            return false;
        }
    }
    return true;
}
