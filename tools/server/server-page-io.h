#pragma once

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <filesystem>
#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

// Continuum fork: a KV page's file IO runs off the server's task loop.
//
// The pinned fork ran every page save and restore, disk IO included, as one task on the single
// task loop, so a save or restore stalled decode for every other slot while it ran and switches
// issued together waited for each other (the M5, 2026-11-14 00:18:13Z: Benchy's restore waited
// 1,024 ms behind Aris's 1,012 ms). Now the loop only copies a sequence's state out of or into
// the context (bounded, proportional to its bytes); the file write or read runs on this pool, and
// its completion is a task posted back to the loop. Like texture streaming: the render thread
// never blocks on IO.
//
// Model-free: nothing here knows llama, so tests/test-server-page-io.cpp exercises it alone.

// Runs page file IO on a few worker threads. Jobs naming the same file run one at a time in the
// order submitted (a page has one writer or reader at a time, so a restore never reads a file a
// save is still writing); jobs on different files run at once.
//
// A job reports its own outcome (the server's posts its completion to the task loop); a job that
// throws ends the process rather than losing a page switch silently. Destruction runs every
// submitted job, then joins: a save the core already sent is never dropped at shutdown.
class server_page_io_pool {
public:
    explicit server_page_io_pool(size_t n_workers) {
        const size_t n = std::max<size_t>(n_workers, 1); // a pool runs at least the job it is given
        workers.reserve(n);
        for (size_t i = 0; i < n; ++i) {
            workers.emplace_back([this] { work(); });
        }
    }

    server_page_io_pool(const server_page_io_pool &) = delete;
    server_page_io_pool & operator=(const server_page_io_pool &) = delete;

    ~server_page_io_pool() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            stopping = true;
        }
        cv.notify_all();
        for (auto & w : workers) {
            w.join();
        }
    }

    size_t n_workers() const { return workers.size(); }

    void submit(const std::string & file, std::function<void()> job) {
        {
            std::lock_guard<std::mutex> lock(mutex);
            auto it = parked.find(file);
            if (it != parked.end()) {
                it->second.push_back(std::move(job)); // the file is in flight: run after it, in order
                return;
            }
            parked.emplace(file, std::deque<std::function<void()>>{});
            ready.push_back({ file, std::move(job) });
        }
        cv.notify_one();
    }

private:
    struct runnable {
        std::string           file;
        std::function<void()> job;
    };

    void work() {
        std::unique_lock<std::mutex> lock(mutex);
        while (true) {
            cv.wait(lock, [&] { return stopping || !ready.empty(); });
            if (ready.empty()) {
                return; // stopping, and nothing left to run: a running job hands its successor to its own worker
            }
            runnable next = std::move(ready.front());
            ready.pop_front();
            lock.unlock();
            next.job();
            lock.lock();
            auto it = parked.find(next.file);
            if (it->second.empty()) {
                parked.erase(it);
            } else {
                ready.push_back({ next.file, std::move(it->second.front()) });
                it->second.pop_front();
                cv.notify_one();
            }
        }
    }

    std::mutex              mutex;
    std::condition_variable cv;
    std::deque<runnable>    ready;
    // every file with a job in flight, and the jobs waiting behind it
    std::unordered_map<std::string, std::deque<std::function<void()>>> parked;
    bool                     stopping = false;
    std::vector<std::thread> workers;
};

// Reads a file's fields in order, streaming (a checkpoint sidecar is parsed straight into the
// checkpoints, never held twice). `ok` turns false at the first short read and stays false; `size`
// is the file's (64-bit on every OS), `at` the bytes read so far.
struct server_page_reader {
    FILE *   f    = nullptr;
    uint64_t size = 0;
    uint64_t at   = 0;
    bool     ok   = false;

    explicit server_page_reader(const std::string & path) {
        std::error_code ec;
        size = (uint64_t) std::filesystem::file_size(path, ec);
        f    = ec ? nullptr : std::fopen(path.c_str(), "rb");
        ok   = f != nullptr;
    }
    server_page_reader(const server_page_reader &) = delete;
    server_page_reader & operator=(const server_page_reader &) = delete;
    ~server_page_reader() {
        if (f != nullptr) {
            std::fclose(f);
        }
    }

    uint64_t remaining() const { return size - at; }

    void get(void * dst, size_t n) {
        ok = ok && n <= remaining() && (n == 0 || std::fread(dst, 1, n, f) == n);
        if (ok) {
            at += n;
        }
    }
};

// The bytes page switches hold in host memory, from their copy until their IO ends (a restore's
// until it is applied), admitted against ONE budget the core sets at runtime from its live host room
// (POST /page-io/budget). A switch is admitted before it allocates: at once when its bytes fit beside
// those in flight, or when nothing is in flight (a switch larger than the whole budget runs alone and
// never starves); otherwise it waits, in the order asked, and the release that makes room admits it
// (an event, never a poll). Until the core sets a budget it is 0: one switch's bytes at a time, still
// off the task loop. No number here is a guess: the budget is the core's measurement.
class server_page_admission {
public:
    struct snapshot_t {
        uint64_t budget    = 0;
        uint64_t in_flight = 0;
        size_t   waiting   = 0;
    };

    // Admits `bytes` now (true: the caller proceeds, `admitted` is dropped) or parks `admitted`, which
    // the release that makes room runs on its own thread, outside this lock.
    bool admit_or_park(uint64_t bytes, std::function<void()> admitted) {
        std::lock_guard<std::mutex> lock(m);
        if (waiting.empty() && fits(bytes)) {
            in_flight += bytes;
            return true;
        }
        if (!closed) {
            waiting.push_back({ bytes, std::move(admitted) });
        }
        return false;
    }

    void release(uint64_t bytes) {
        std::vector<std::function<void()>> run;
        {
            std::lock_guard<std::mutex> lock(m);
            in_flight -= std::min(bytes, in_flight);
            drain(run);
        }
        for (auto & f : run) {
            f();
        }
    }

    // A held reservation re-measured when its switch runs (decode may have grown the slot since).
    void resize(uint64_t from, uint64_t to) {
        std::vector<std::function<void()>> run;
        {
            std::lock_guard<std::mutex> lock(m);
            in_flight = in_flight - std::min(from, in_flight) + to;
            drain(run);
        }
        for (auto & f : run) {
            f();
        }
    }

    void set_budget(uint64_t bytes) {
        std::vector<std::function<void()>> run;
        {
            std::lock_guard<std::mutex> lock(m);
            budget = bytes;
            drain(run);
        }
        for (auto & f : run) {
            f();
        }
    }

    // At shutdown: waiting switches are dropped unrun, and no later release admits anything.
    void close() {
        std::deque<parked> dropped;
        {
            std::lock_guard<std::mutex> lock(m);
            closed = true;
            dropped.swap(waiting);
        }
    }

    snapshot_t snapshot() const {
        std::lock_guard<std::mutex> lock(m);
        return { budget, in_flight, waiting.size() };
    }

private:
    struct parked {
        uint64_t              bytes;
        std::function<void()> admitted;
    };

    bool fits(uint64_t bytes) const { return in_flight == 0 || bytes <= budget - std::min(budget, in_flight); }

    void drain(std::vector<std::function<void()>> & run) {
        while (!closed && !waiting.empty() && fits(waiting.front().bytes)) {
            in_flight += waiting.front().bytes;
            run.push_back(std::move(waiting.front().admitted));
            waiting.pop_front();
        }
    }

    mutable std::mutex m;
    uint64_t           budget    = 0;
    uint64_t           in_flight = 0;
    std::deque<parked> waiting;
    bool               closed    = false;
};

// The bytes one admitted switch holds, returned to the budget once, by whichever owner is done with
// them first (the worker after a write, the loop after an apply or a drop, or the destructor).
class server_page_reservation {
public:
    server_page_reservation() = default;
    server_page_reservation(const server_page_reservation &) = delete;
    server_page_reservation & operator=(const server_page_reservation &) = delete;
    ~server_page_reservation() { release(); }

    void hold(server_page_admission * a, uint64_t bytes) {
        admission = a;
        held.store(bytes);
    }
    void resize(uint64_t to) {
        const uint64_t from = held.exchange(to);
        if (admission != nullptr) {
            admission->resize(from, to);
        }
    }
    void release() {
        const uint64_t bytes = held.exchange(0);
        if (bytes > 0 && admission != nullptr) {
            admission->release(bytes);
        }
    }
    uint64_t bytes() const { return held.load(); }

private:
    server_page_admission * admission = nullptr;
    std::atomic<uint64_t>   held{0};
};

// A sequence state file: magic, version, token count, the tokens, then the state body (the bytes
// llama_state_seq_get_file_body copies out). The framing is the library's own
// (llama_state_seq_save_file), so pages written before this change restore and pages written by
// it restore on an engine without it.
struct server_page_file {
    uint32_t magic;
    uint32_t version;

    // Writes the page; returns its bytes, or 0 with `why`.
    size_t write(const std::string & path, const int32_t * tokens, size_t n_tokens,
                 const std::vector<uint8_t> & body, std::string & why) const {
        FILE * f = std::fopen(path.c_str(), "wb");
        if (f == nullptr) {
            why = "cannot open " + path + " for writing";
            return 0;
        }
        const uint32_t count = (uint32_t) n_tokens;
        bool ok = n_tokens <= UINT32_MAX;
        ok = ok && std::fwrite(&magic,   sizeof(magic),   1, f) == 1;
        ok = ok && std::fwrite(&version, sizeof(version), 1, f) == 1;
        ok = ok && std::fwrite(&count,   sizeof(count),   1, f) == 1;
        ok = ok && (n_tokens == 0 || std::fwrite(tokens, sizeof(int32_t), n_tokens, f) == n_tokens);
        ok = ok && (body.empty() || std::fwrite(body.data(), 1, body.size(), f) == body.size());
        ok = (std::fclose(f) == 0) && ok;
        if (!ok) {
            why = "short write to " + path;
            return 0;
        }
        return sizeof(magic) + sizeof(version) + sizeof(count) + n_tokens * sizeof(int32_t) + body.size();
    }

    struct contents {
        std::vector<int32_t> tokens;
        std::vector<uint8_t> body;
        size_t               n_bytes = 0; // the whole file
    };

    // Reads a page whole. `progress(done, total)` is told after each part of the body and returning
    // false cancels the read. nullopt with `why` on any defect: an unknown framing, a token count
    // larger than the file, a short read, a cancel.
    std::optional<contents> read(const std::string & path, const std::function<bool(uint64_t, uint64_t)> & progress,
                                 std::string & why) const {
        server_page_reader in(path);
        if (!in.ok) {
            why = "cannot open " + path;
            return std::nullopt;
        }
        uint32_t head[3] = { 0, 0, 0 };
        in.get(head, sizeof(head));
        if (!in.ok) {
            why = "truncated framing in " + path;
            return std::nullopt;
        }
        if (head[0] != magic || head[1] != version) {
            why = "unknown (magic, version) in " + path;
            return std::nullopt;
        }
        if ((uint64_t) head[2] > in.remaining() / sizeof(int32_t)) {
            why = "token count exceeds the file size in " + path;
            return std::nullopt;
        }
        contents out;
        out.tokens.resize(head[2]);
        in.get(out.tokens.data(), out.tokens.size() * sizeof(int32_t));
        out.body.resize((size_t) in.remaining());
        // progress is told 64 times over the body: a cancel takes effect within 1/64 of the read
        const size_t part = std::max<size_t>(out.body.size() / 64, 1);
        for (size_t done = 0; in.ok && done < out.body.size();) {
            const size_t n = std::min(part, out.body.size() - done);
            in.get(out.body.data() + done, n);
            done += n;
            if (in.ok && progress && !progress(in.at, in.size)) {
                why = "state file load cancelled";
                return std::nullopt;
            }
        }
        if (!in.ok) {
            why = "short read in " + path;
            return std::nullopt;
        }
        out.n_bytes = (size_t) in.size;
        return out;
    }
};

// A whole small buffer (a page's sidecar) out to a file; false when it cannot be written.
inline bool server_page_write_bytes(const std::string & path, const std::vector<uint8_t> & bytes) {
    FILE * f = std::fopen(path.c_str(), "wb");
    if (f == nullptr) {
        return false;
    }
    const bool ok = bytes.empty() || std::fwrite(bytes.data(), 1, bytes.size(), f) == bytes.size();
    return (std::fclose(f) == 0) && ok;
}

// Appends a sidecar's fields to a buffer.
inline void server_page_put(std::vector<uint8_t> & out, const void * src, size_t n) {
    const auto * p = (const uint8_t *) src;
    out.insert(out.end(), p, p + n);
}
