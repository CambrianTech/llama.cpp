#include "server-pages.h"

#include "../../src/llama-ext.h"
#include "speculative.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <system_error>

static constexpr uint32_t PAGE_CKPT_MAGIC = 0x434B5054u; // 'CKPT'
static constexpr uint32_t PAGE_LORA_MAGIC = 0x4C4F5241u; // 'LORA'
static constexpr uint32_t PAGE_DRFT_MAGIC = 0x44524654u; // 'DRFT'

size_t server_page::n_checkpoint_bytes() const {
    size_t n = 0;
    for (const auto & c : prompt.checkpoints) {
        n += c.size();
    }
    return n;
}

size_t server_page::n_bytes() const {
    return data.size() + spec.size() + n_checkpoint_bytes();
}

std::pair<size_t, size_t> server_page_state_size(llama_context * ctx_tgt, llama_context * ctx_dft, llama_seq_id seq) {
    return {
        llama_state_seq_get_size_ext(ctx_tgt, seq, LLAMA_STATE_SEQ_FLAGS_NONE),
        ctx_dft ? llama_state_seq_get_size_ext(ctx_dft, seq, LLAMA_STATE_SEQ_FLAGS_NONE) : 0,
    };
}

server_page_ptr server_page_capture(llama_context * ctx_tgt, llama_context * ctx_dft, common_speculative * spec, llama_seq_id seq) {
    const auto [n_tgt, n_dft] = server_page_state_size(ctx_tgt, ctx_dft, seq);
    if (n_tgt == 0) {
        throw server_page_error(ERROR_TYPE_SERVER, "the target sequence state could not be sized");
    }
    auto page = std::make_shared<server_page>();
    try {
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
    if (spec != nullptr) {
        common_speculative_get_state(spec, seq, page->spec);
    }
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
    size_t held = n_bytes();
    const auto it = pages.find(name);
    if (it != pages.end()) {
        held -= it->second->n_bytes();
    }
    if (held + n_bytes_new > limit_bytes) {
        throw server_page_error(ERROR_TYPE_UNAVAILABLE, string_format(
            "the page store holds %zu bytes; a page of %zu bytes would exceed its ceiling of %llu bytes: spill or drop a page first",
            held, n_bytes_new, (unsigned long long) limit_bytes));
    }
}

uint64_t server_page_store::put(const std::string & name, server_page_ptr page) {
    page->gen = next_gen++;
    const auto it = pages.find(name);
    if (it != pages.end()) {
        SRV_WRN("page '%s' replaced (%zu bytes freed)\n", name.c_str(), it->second->n_bytes());
    }
    pages[name] = std::move(page);
    return pages[name]->gen;
}

server_page_ptr server_page_store::find(const std::string & name) const {
    const auto it = pages.find(name);
    return it == pages.end() ? nullptr : it->second;
}

size_t server_page_store::erase(const std::string & name, uint64_t gen) {
    const auto it = pages.find(name);
    if (it == pages.end() || (gen != 0 && it->second->gen != gen)) {
        return 0;
    }
    const size_t n = it->second->n_bytes();
    pages.erase(it);
    return n;
}

void server_page_store::evict_lora(const llama_adapter_lora * adapter) {
    for (auto it = pages.begin(); it != pages.end();) {
        bool uses = false;
        for (const auto & l : it->second->lora) {
            uses = uses || (l.ptr == adapter && l.scale != 0.0f);
        }
        if (uses) {
            SRV_WRN("page '%s' dropped: its adapter is being unloaded (%zu bytes freed)\n", it->first.c_str(), it->second->n_bytes());
            it = pages.erase(it);
        } else {
            ++it;
        }
    }
}

size_t server_page_store::n_bytes() const {
    size_t n = 0;
    for (const auto & [name, page] : pages) {
        n += page->n_bytes();
    }
    return n;
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

//
// page files
//

std::pair<uint64_t, int64_t> lora_file_revision(const std::string & path) {
    std::error_code ec;
    const auto size = std::filesystem::file_size(path, ec);
    if (ec) {
        return {0, 0};
    }
    const auto mtime = std::filesystem::last_write_time(path, ec);
    if (ec) {
        return {0, 0};
    }
    return {(uint64_t) size, (int64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(mtime.time_since_epoch()).count()};
}

void server_page_write_checkpoints(const std::vector<const common_prompt_checkpoint *> & kept, const server_page_put & put) {
    const uint32_t magic   = PAGE_CKPT_MAGIC;
    const uint32_t version = 1;
    const uint32_t n_ckpt  = (uint32_t) kept.size();
    put(&magic, sizeof(magic)); put(&version, sizeof(version)); put(&n_ckpt, sizeof(n_ckpt));
    for (const auto * c : kept) {
        const int64_t  n_tokens = c->n_tokens;
        const int32_t  pos_min  = c->pos_min;
        const int32_t  pos_max  = c->pos_max;
        const uint64_t sz_tgt   = c->data_tgt.size();
        const uint64_t sz_dft   = c->data_dft.size();
        const uint64_t sz_spec  = c->data_spec.size();
        put(&n_tokens, sizeof(n_tokens)); put(&pos_min, sizeof(pos_min)); put(&pos_max, sizeof(pos_max));
        put(&sz_tgt, sizeof(sz_tgt)); put(&sz_dft, sizeof(sz_dft)); put(&sz_spec, sizeof(sz_spec));
        put(c->data_tgt.data(), c->data_tgt.size());
        put(c->data_dft.data(), c->data_dft.size());
        put(c->data_spec.data(), c->data_spec.size());
    }
}

void server_page_write_adapters(const std::vector<common_adapter_lora_info> & lora, const server_page_put & put) {
    const uint32_t magic   = PAGE_LORA_MAGIC;
    const uint32_t version = 2;
    const uint32_t n_lora  = (uint32_t) lora.size();
    put(&magic, sizeof(magic)); put(&version, sizeof(version)); put(&n_lora, sizeof(n_lora));
    for (const auto & l : lora) {
        const uint32_t n_path = (uint32_t) l.path.size();
        const auto     rev    = lora_file_revision(l.path);
        put(&n_path, sizeof(n_path));
        put(l.path.data(), n_path);
        put(&l.scale, sizeof(l.scale));
        put(&rev.first, sizeof(rev.first)); put(&rev.second, sizeof(rev.second));
    }
}

void server_page_write_draft(const uint8_t * dft, size_t n_dft, const std::vector<uint8_t> & spec, const server_page_put & put) {
    const uint32_t magic   = PAGE_DRFT_MAGIC;
    const uint32_t version = 1;
    const uint64_t sz_dft  = n_dft;
    const uint64_t sz_spec = spec.size();
    put(&magic, sizeof(magic)); put(&version, sizeof(version));
    put(&sz_dft, sizeof(sz_dft)); put(&sz_spec, sizeof(sz_spec));
    put(dft, n_dft);
    put(spec.data(), spec.size());
}

bool server_page_read_draft(const std::string & path, std::vector<uint8_t> & dft, std::vector<uint8_t> & spec, std::string & why) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) {
        why = "absent";
        return false;
    }
    const auto total = (uint64_t) in.tellg();
    in.seekg(0);
    uint32_t magic = 0, version = 0;
    uint64_t sz_dft = 0, sz_spec = 0;
    in.read(reinterpret_cast<char *>(&magic), sizeof(magic));
    in.read(reinterpret_cast<char *>(&version), sizeof(version));
    in.read(reinterpret_cast<char *>(&sz_dft), sizeof(sz_dft));
    in.read(reinterpret_cast<char *>(&sz_spec), sizeof(sz_spec));
    const uint64_t header = 2 * sizeof(uint32_t) + 2 * sizeof(uint64_t);
    if (!in || magic != PAGE_DRFT_MAGIC || version != 1 || sz_dft > total || sz_spec > total || header + sz_dft + sz_spec != total) {
        why = "malformed";
        return false;
    }
    dft.resize(sz_dft);
    spec.resize(sz_spec);
    in.read(reinterpret_cast<char *>(dft.data()), sz_dft);
    in.read(reinterpret_cast<char *>(spec.data()), sz_spec);
    if (!in) {
        why = "truncated";
        return false;
    }
    return true;
}

static bool write_sidecar(const std::string & path, const std::function<void(const server_page_put &)> & body, size_t & n_written) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    body([&](const void * ptr, size_t n) {
        out.write(reinterpret_cast<const char *>(ptr), n);
        n_written += n;
    });
    out.flush();
    return (bool) out;
}

size_t server_page_spill(const server_page & page, const std::string & filepath) {
    const std::string sidecars[] = { filepath + ".ckpt", filepath + ".lora", filepath + ".dft" };
    auto remove_all = [&]() {
        std::error_code ec;
        std::filesystem::remove(filepath, ec);
        for (const auto & p : sidecars) {
            std::filesystem::remove(p, ec);
        }
    };
    // no file of an older page under this name may survive beside the new state
    remove_all();

    std::vector<char> packed;
    try {
        packed = page.prompt.tokens.serialize();
    } catch (const std::exception & err) {
        throw server_page_error(ERROR_TYPE_NOT_SUPPORTED, err.what());
    }
    size_t n_written = llama_state_seq_save_file_from_data(filepath.c_str(),
        page.data.main.data(), page.data.main.size(),
        reinterpret_cast<const llama_token *>(packed.data()), packed.size() / sizeof(llama_token));
    if (n_written == 0) {
        remove_all();
        throw server_page_error(ERROR_TYPE_SERVER, "the page's sequence state could not be written: " + filepath);
    }

    std::vector<const common_prompt_checkpoint *> all;
    for (const auto & c : page.prompt.checkpoints) {
        all.push_back(&c);
    }
    const bool ok =
        write_sidecar(sidecars[0], [&](const server_page_put & put) { server_page_write_checkpoints(all, put); }, n_written) &&
        write_sidecar(sidecars[1], [&](const server_page_put & put) { server_page_write_adapters(page.lora, put); }, n_written) &&
        write_sidecar(sidecars[2], [&](const server_page_put & put) { server_page_write_draft(page.data.drft.data(), page.data.drft.size(), page.spec, put); }, n_written);
    if (!ok) {
        remove_all();
        throw server_page_error(ERROR_TYPE_SERVER, "a page sidecar could not be written beside " + filepath);
    }
    return n_written;
}
