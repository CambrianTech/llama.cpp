#include "server-page-files.h"

#include "../../src/llama-ext.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <set>
#include <system_error>

#if defined(_WIN32)
#   ifndef NOMINMAX
#       define NOMINMAX
#   endif
#   include <windows.h>
#   include <fcntl.h>
#   include <io.h>
#   include <process.h>
#else
#   include <cerrno>
#   include <fcntl.h>
#   include <signal.h>
#   include <unistd.h>
#endif

namespace fs = std::filesystem;

static constexpr uint32_t PAGE_CKPT_MAGIC = 0x434B5054u; // 'CKPT'
static constexpr uint32_t PAGE_LORA_MAGIC = 0x4C4F5241u; // 'LORA'
static constexpr uint32_t PAGE_DRFT_MAGIC = 0x44524654u; // 'DRFT'

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

//
// generation publication
//

// Every file a page may have. A generation names the ones it carries; the others are removed at its commit.
static const char * const PAGE_SUFFIXES[] = { "", ".ckpt", ".lora", ".dft" };

// Beside the page dir; ".." in it is a name fs_validate_filename refuses, so no page can take it.
static const char * const STAGING_DIR = "..page-staging";

// Publications in this process: the commit boundary every reader shares, and the stages being written.
struct page_publications {
    std::shared_mutex commit; // a commit (and a reader's repair) exclusive; a reader's read shared
    std::mutex        stages_mutex;
    std::set<std::string> stages;
    uint64_t next = 1;
};

static page_publications & publications() {
    static page_publications pubs;
    return pubs;
}

static long current_pid() {
#if defined(_WIN32)
    return (long) _getpid();
#else
    return (long) getpid();
#endif
}

static bool process_alive(long pid) {
#if defined(_WIN32)
    HANDLE h = OpenProcess(SYNCHRONIZE, FALSE, (DWORD) pid);
    if (h == nullptr) {
        return false;
    }
    const bool alive = WaitForSingleObject(h, 0) == WAIT_TIMEOUT;
    CloseHandle(h);
    return alive;
#else
    return kill((pid_t) pid, 0) == 0 || errno == EPERM;
#endif
}

// A stage is "<pid>-<n>": live while its publication runs in this process, or while its process runs.
static bool stage_live(const std::string & id) {
    const auto dash = id.find('-');
    long pid = -1;
    try {
        pid = std::stol(id.substr(0, dash));
    } catch (const std::exception &) {
        return false;
    }
    if (pid <= 0) {
        return false;
    }
    if (pid != current_pid()) {
        return process_alive(pid);
    }
    auto & pubs = publications();
    std::lock_guard<std::mutex> lock(pubs.stages_mutex);
    return pubs.stages.count(id) > 0;
}

// Makes the file's contents durable before anything that depends on them (the commit record).
static bool sync_file(const fs::path & path) {
#if defined(_WIN32)
    const int fd = _wopen(path.c_str(), _O_RDWR | _O_BINARY);
    if (fd < 0) {
        return false;
    }
    const bool ok = _commit(fd) == 0;
    _close(fd);
    return ok;
#else
    const int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        return false;
    }
#   if defined(__APPLE__)
    // macOS fsync stops at the drive's cache; only F_FULLFSYNC orders the data before the record
    bool ok = fcntl(fd, F_FULLFSYNC) == 0 || fsync(fd) == 0;
#   else
    bool ok = fsync(fd) == 0;
#   endif
    ok = close(fd) == 0 && ok;
    return ok;
#endif
}

static std::function<bool(const std::string &)> & sync_dir_fault() {
    static std::function<bool(const std::string &)> fault;
    return fault;
}

void server_page_files_inject_sync_fault(std::function<bool(const std::string & dir)> fault) {
    sync_dir_fault() = std::move(fault);
}

// Makes a dir's entries (a rename into it) durable, or throws: a rename not proven durable is never
// reported as published. Windows has no handle for this; NTFS journals the entries.
static void sync_dir(const fs::path & dir) {
    bool ok = !(sync_dir_fault() && sync_dir_fault()(dir.string()));
#if !defined(_WIN32)
    if (ok) {
        const int fd = open(dir.c_str(), O_RDONLY);
        ok = fd >= 0 && fsync(fd) == 0;
        ok = (fd < 0 || close(fd) == 0) && ok;
    }
#endif
    if (!ok) {
        throw server_page_error(ERROR_TYPE_SERVER, "a page directory could not be synced: " + dir.string());
    }
}

static fs::path staging_dir(const fs::path & file) {
    return file.parent_path() / STAGING_DIR;
}

// Its presence is the commit: the stage it names holds the generation, whole and synced.
static fs::path commit_record(const fs::path & file) {
    return staging_dir(file) / (file.filename().string() + ".commit");
}

// "<stage id>\n" then "f<suffix>\n" per file of the generation.
static bool read_commit_record(const fs::path & record, std::string & id, std::vector<std::string> & suffixes) {
    std::ifstream in(record);
    if (!in || !std::getline(in, id) || id.empty()) {
        return false;
    }
    for (std::string line; std::getline(in, line);) {
        if (line.empty() || line[0] != 'f') {
            return false;
        }
        suffixes.push_back(line.substr(1));
    }
    return !suffixes.empty();
}

// Writes the record inside the stage, syncs it, then renames it into place: the rename is the commit.
static void write_commit_record(const fs::path & file, const fs::path & stage, const std::string & id, const std::vector<std::string> & suffixes) {
    const fs::path tmp = stage / "..commit"; // ".." is in no page name
    {
        std::ofstream out(tmp, std::ios::trunc);
        out << id << '\n';
        for (const auto & s : suffixes) {
            out << 'f' << s << '\n';
        }
        out.flush();
        if (!out) {
            throw server_page_error(ERROR_TYPE_SERVER, "the page's commit record could not be written: " + tmp.string());
        }
    }
    std::error_code ec;
    if (!sync_file(tmp) || (fs::rename(tmp, commit_record(file), ec), ec)) {
        throw server_page_error(ERROR_TYPE_SERVER, "the page's commit record could not be placed: " + commit_record(file).string());
    }
}

// Moves a committed generation over the page's files, removes the files it lacks, then its stage and
// record. Idempotent: a file a crash already moved is not in the stage. Throws, keeping the record, so
// the next reader finishes it.
static void roll_forward(const fs::path & file, const fs::path & stage, const std::vector<std::string> & suffixes) {
    const std::string name = file.filename().string();
    std::error_code ec;
    for (const auto & s : suffixes) {
        const fs::path staged = stage / (name + s);
        if (fs::exists(staged, ec) && (fs::rename(staged, file.string() + s, ec), ec)) {
            throw server_page_error(ERROR_TYPE_SERVER, "a committed page file could not be moved into place: " + staged.string() + ": " + ec.message());
        }
    }
    for (const char * s : PAGE_SUFFIXES) {
        if (std::find(suffixes.begin(), suffixes.end(), s) == suffixes.end() && !fs::remove(file.string() + s, ec) && ec) {
            throw server_page_error(ERROR_TYPE_SERVER, "an older page file could not be removed: " + file.string() + s + ": " + ec.message());
        }
    }
    sync_dir(file.parent_path()); // throws, keeping the record: the moved names are not yet durable
    fs::remove_all(stage, ec);
    fs::remove(commit_record(file), ec);
    try {
        sync_dir(staging_dir(file));
    } catch (const server_page_error & err) {
        // the generation is durable; a record whose removal is lost names a stage already moved,
        // and the next reader's roll_forward of it changes nothing
        SRV_WRN("%s\n", err.what());
    }
}

// Removes every stage no commit record names and no live publication owns: what a crash before a
// commit left. Called under the exclusive commit lock.
static void remove_orphan_stages(const fs::path & staging) {
    std::error_code ec;
    if (!fs::is_directory(staging, ec)) {
        return;
    }
    std::set<std::string> committed;
    for (const auto & e : fs::directory_iterator(staging, ec)) {
        std::string id;
        std::vector<std::string> suffixes;
        if (e.path().extension() == ".commit" && read_commit_record(e.path(), id, suffixes)) {
            committed.insert(id);
        }
    }
    std::vector<fs::path> orphans;
    for (const auto & e : fs::directory_iterator(staging, ec)) {
        const std::string id = e.path().filename().string();
        if (e.is_directory(ec) && committed.count(id) == 0 && !stage_live(id)) {
            orphans.push_back(e.path());
        }
    }
    for (const auto & p : orphans) {
        SRV_WRN("removing a page generation a publication left uncommitted: %s\n", p.string().c_str());
        fs::remove_all(p, ec);
    }
}

// A publication's stage, live while it exists: uncommitted, it is removed with its owner; committed, it
// stays until its roll forward removes it.
struct stage_owner {
    std::string id;
    fs::path    stage;
    bool        committed = false;

    explicit stage_owner(const fs::path & staging) {
        auto & pubs = publications();
        std::lock_guard<std::mutex> lock(pubs.stages_mutex);
        id    = std::to_string(current_pid()) + "-" + std::to_string(pubs.next++);
        stage = staging / id;
        pubs.stages.insert(id);
    }

    ~stage_owner() {
        std::error_code ec;
        if (!committed) {
            fs::remove_all(stage, ec);
        }
        auto & pubs = publications();
        std::lock_guard<std::mutex> lock(pubs.stages_mutex);
        pubs.stages.erase(id);
    }

    stage_owner(const stage_owner &) = delete;
    stage_owner & operator=(const stage_owner &) = delete;
};

server_page_file server_page_sidecar(std::string suffix, std::function<void(const server_page_put &)> body) {
    return { std::move(suffix), [body = std::move(body)](const std::string & path) -> size_t {
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        size_t n = 0;
        body([&](const void * ptr, size_t len) {
            out.write(reinterpret_cast<const char *>(ptr), len);
            n += len;
        });
        out.flush();
        return out ? n : 0;
    } };
}

size_t server_page_publish(const std::string & filepath, const std::vector<server_page_file> & files) {
    const fs::path file(filepath);
    auto & pubs = publications();

    stage_owner owner(staging_dir(file));
    const fs::path & stage = owner.stage;

    std::error_code ec;
    fs::create_directories(stage, ec);
    if (ec) {
        throw server_page_error(ERROR_TYPE_SERVER, "the page could not be staged: " + stage.string() + ": " + ec.message());
    }
    size_t n_written = 0;
    std::vector<std::string> suffixes;
    for (const auto & f : files) {
        GGML_ASSERT(std::find(std::begin(PAGE_SUFFIXES), std::end(PAGE_SUFFIXES), f.suffix) != std::end(PAGE_SUFFIXES));
        const fs::path staged = stage / (file.filename().string() + f.suffix);
        size_t n = 0;
        try {
            n = f.write(staged.string());
        } catch (const std::exception & err) {
            throw server_page_error(ERROR_TYPE_SERVER, "a page file could not be written: " + staged.string() + ": " + err.what());
        }
        if (n == 0 || !sync_file(staged)) {
            throw server_page_error(ERROR_TYPE_SERVER, "a page file could not be written: " + staged.string());
        }
        n_written += n;
        suffixes.push_back(f.suffix);
    }
    sync_dir(stage);

    std::unique_lock<std::shared_mutex> lock(pubs.commit);
    write_commit_record(file, stage, owner.id, suffixes);
    owner.committed = true; // the record names the stage: from here a failure leaves it to the next reader
    sync_dir(staging_dir(file));
    roll_forward(file, stage, suffixes);
    return n_written;
}

size_t server_page_spill(const server_page & page, const std::string & filepath) {
    std::vector<char> packed;
    try {
        packed = page.prompt.tokens.serialize();
    } catch (const std::exception & err) {
        throw server_page_error(ERROR_TYPE_NOT_SUPPORTED, err.what());
    }
    std::vector<const common_prompt_checkpoint *> all;
    for (const auto & c : page.prompt.checkpoints) {
        all.push_back(&c);
    }
    return server_page_publish(filepath, {
        { "", [&](const std::string & path) {
            return llama_state_seq_save_file_from_data(path.c_str(), page.data.main.data(), page.data.main.size(),
                reinterpret_cast<const llama_token *>(packed.data()), packed.size() / sizeof(llama_token));
        } },
        server_page_sidecar(".ckpt", [&](const server_page_put & put) { server_page_write_checkpoints(all, put); }),
        server_page_sidecar(".lora", [&](const server_page_put & put) { server_page_write_adapters(page.lora, put); }),
        server_page_sidecar(".dft",  [&](const server_page_put & put) { server_page_write_draft(page.data.drft.data(), page.data.drft.size(), page.spec, put); }),
    });
}

server_page_reader::server_page_reader(const std::string & filepath) {
    const fs::path file(filepath);
    auto & pubs = publications();
    {
        std::unique_lock<std::shared_mutex> repair(pubs.commit);
        const fs::path record = commit_record(file);
        std::string id;
        std::vector<std::string> suffixes;
        if (read_commit_record(record, id, suffixes)) {
            SRV_WRN("completing the publication of %s that stopped after its commit\n", filepath.c_str());
            roll_forward(file, staging_dir(file) / id, suffixes);
        }
        remove_orphan_stages(staging_dir(file));
    }
    lock = std::shared_lock<std::shared_mutex>(pubs.commit);
}
