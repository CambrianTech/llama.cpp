#pragma once

// Continuum fork: the includes, private types (server_subproc, server_lru_sched) and helpers that several
// server-models translation units share; out of server-models.cpp (pure move; the 1,000-line rule).

#include "server-common.h"
#include "http.h"
#include "server-models.h"
#include "server-context.h"
#include "server-stream.h"

#include "build-info.h"
#include "preset.h"
#include "download.h"
#include "http.h"
#include "subproc.h"

#include <cpp-httplib/httplib.h> // TODO: remove this once we use HTTP client from download.h
#include <optional>

#include <functional>
#include <optional>
#include <algorithm>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <cstring>
#include <cstdlib>
#include <atomic>
#include <chrono>
#include <queue>
#include <filesystem>
#include <random>
#include <sstream>
#include <cstring>

#ifndef _WIN32
extern char **environ;
#endif

#if defined(__APPLE__) && defined(__MACH__)
// macOS: use _NSGetExecutablePath to get the executable path
#include <mach-o/dyld.h>
#include <limits.h>
#endif

#define DEFAULT_STOP_TIMEOUT 10 // seconds

#define CMD_ROUTER_TO_CHILD_EXIT  "cmd_router_to_child:exit"
#define CMD_CHILD_TO_ROUTER_STATE "cmd_child_to_router:state:" // followed by json string

// address for child process, this is needed because router may run on 0.0.0.0
// ref: https://github.com/ggml-org/llama.cpp/issues/17862
#define CHILD_ADDR "127.0.0.1"

struct server_subproc {
    common_subproc sproc; // not yet spawned while in DOWNLOADING state
    std::atomic<bool> stopped{false}; // set to cancel a download or signal child process exit

    bool is_alive() {
        return sproc.alive();
    }

    void request_exit() {
        FILE * stdin_file = sproc.stdin_file();
        if (stdin_file) {
            fprintf(stdin_file, "%s\n", CMD_ROUTER_TO_CHILD_EXIT);
            fflush(stdin_file);
        }
        stopped.store(true, std::memory_order_relaxed);
    }

    void terminate() {
        sproc.terminate();
    }
};

struct server_lru_sched {
    server_lru_sched(server_models & models) : models(models) {}

    bool has_capacity(std::unique_lock<std::mutex> & lk) {
        check_lock(lk);
        return models.base_params.models_max <= 0
            || count_running() < (size_t) models.base_params.models_max;
    }

    // returns "" if no model can be given up
    std::string pick_victim(std::unique_lock<std::mutex> & lk, const std::string & exclude) {
        check_lock(lk);
        std::string victim;
        int64_t victim_last_used = 0;
        for (const auto & m : models.mapping) {
            if (m.first == exclude) {
                continue;
            }
            // a busy model is mid-request, one still coming up has no request to finish
            if (m.second.req_count != 0 || !m.second.meta.is_ready_or_sleep()) {
                continue;
            }
            if (victim.empty() || m.second.meta.last_used < victim_last_used) {
                victim           = m.first;
                victim_last_used = m.second.meta.last_used;
            }
        }
        return victim;
    }

    // requests wanting the same model share one entry, so they all need only one slot
    // and all get unblocked by the single load that entry performs
    void join(std::unique_lock<std::mutex> & lk, const std::string & model_id) {
        check_lock(lk);
        if (entry_t * e = find(model_id)) {
            e->n_waiters++;
            SRV_INF("request for name=%s joined the queue, %d waiting\n", model_id.c_str(), e->n_waiters);
            return;
        }
        queue.push_back({ model_id, 1, false, false });
        SRV_INF("models_max reached, request for name=%s queued at position %zu\n",
                model_id.c_str(), queue.size());
    }

    void leave(std::unique_lock<std::mutex> & lk, const std::string & model_id) {
        check_lock(lk);
        for (auto it = queue.begin(); it != queue.end(); ++it) {
            if (it->model_id == model_id) {
                if (--it->n_waiters <= 0) {
                    queue.erase(it); // last one waiting for this model went away
                }
                return;
            }
        }
    }

    bool queue_empty(std::unique_lock<std::mutex> & lk) {
        check_lock(lk);
        return queue.empty();
    }

    // true if it is this model's turn to load, and nobody is loading it yet
    bool try_claim(std::unique_lock<std::mutex> & lk, const std::string & model_id) {
        check_lock(lk);
        if (queue.empty() || queue.front().model_id != model_id || queue.front().loading) {
            return false;
        }
        if (!has_capacity(lk)) {
            return false;
        }
        queue.front().loading = true;
        return true;
    }

    // ok means the model is up: drop the entry, the other waiters just watch its status now
    void claim_done(std::unique_lock<std::mutex> & lk, const std::string & model_id, bool ok) {
        check_lock(lk);
        for (auto it = queue.begin(); it != queue.end(); ++it) {
            if (it->model_id == model_id) {
                if (ok) {
                    queue.erase(it);
                } else {
                    it->loading = false;
                }
                return;
            }
        }
    }

    // a model is on its way out for this entry, so other requests do not also give up one
    void mark_slot_pending(std::unique_lock<std::mutex> & lk, const std::string & model_id) {
        check_lock(lk);
        if (entry_t * e = find(model_id)) {
            e->slot_pending = true;
        }
    }

    // model_id went idle: give up its slot if a queued request needs one
    // thread-safe, caller must NOT hold models.mutex
    void on_model_idle(const std::string & model_id) {
        if (models.base_params.models_max <= 0) {
            return; // no limit, nothing is ever queued
        }
        {
            std::unique_lock<std::mutex> lk(models.mutex);
            if (queue.empty()) {
                return;
            }
            size_t promised     = 0;
            bool   has_unserved = false;
            for (const auto & e : queue) {
                if (e.needs_slot()) {
                    has_unserved = true;
                } else {
                    promised++;
                }
            }
            if (!has_unserved) {
                return;
            }
            if ((int) count_running() - (int) promised < models.base_params.models_max) {
                return; // a slot is already on its way
            }
            // never give up a model that a queued request wants
            for (const auto & e : queue) {
                if (e.model_id == model_id) {
                    return;
                }
            }
            auto it = models.mapping.find(model_id);
            if (it == models.mapping.end() || it->second.req_count != 0 || !it->second.meta.is_ready_or_sleep()) {
                return;
            }
            for (auto & e : queue) {
                if (!e.slot_pending) {
                    e.slot_pending = true;
                    break;
                }
            }
        }
        SRV_INF("model name=%s went idle, giving up its slot to a queued request\n", model_id.c_str());
        models.unload(model_id);
    }

  private:
    struct entry_t {
        std::string model_id;
        int  n_waiters;    // requests waiting for this model
        bool slot_pending; // a model is already being evicted for this entry
        bool loading;      // one of the waiters is doing the load right now

        // a slot is already coming, or already taken by the load in flight
        bool needs_slot() const { return !slot_pending && !loading; }
    };

    entry_t * find(const std::string & model_id) {
        for (auto & e : queue) {
            if (e.model_id == model_id) {
                return &e;
            }
        }
        return nullptr;
    }

    void check_lock(std::unique_lock<std::mutex> & lk) {
        GGML_ASSERT(lk.owns_lock() && lk.mutex() == &models.mutex);
    }

    size_t count_running() {
        size_t count = 0;
        for (const auto & m : models.mapping) {
            if (m.second.meta.is_running()) {
                count++;
            }
        }
        return count;
    }

    server_models & models;
    std::deque<entry_t> queue;
};

// short loopback budget for the resumable stream router to child JSON calls (probe, lookup,
// delete). distinct from params.timeout_read/write which only applies to the generation proxy
inline constexpr int STREAM_LOOKUP_TIMEOUT_MS = 250;

inline std::filesystem::path get_server_exec_path() {
#if defined(_WIN32)
    wchar_t buf[32768] = { 0 };  // Large buffer to handle long paths
    DWORD len = GetModuleFileNameW(nullptr, buf, _countof(buf));
    if (len == 0 || len >= _countof(buf)) {
        throw std::runtime_error("GetModuleFileNameW failed or path too long");
    }
    return std::filesystem::path(buf);
#elif defined(__APPLE__) && defined(__MACH__)
    char small_path[PATH_MAX];
    uint32_t size = sizeof(small_path);

    if (_NSGetExecutablePath(small_path, &size) == 0) {
        // resolve any symlinks to get absolute path
        try {
            return std::filesystem::canonical(std::filesystem::path(small_path));
        } catch (...) {
            return std::filesystem::path(small_path);
        }
    } else {
        // buffer was too small, allocate required size and call again
        std::vector<char> buf(size);
        if (_NSGetExecutablePath(buf.data(), &size) == 0) {
            try {
                return std::filesystem::canonical(std::filesystem::path(buf.data()));
            } catch (...) {
                return std::filesystem::path(buf.data());
            }
        }
        throw std::runtime_error("_NSGetExecutablePath failed after buffer resize");
    }
#else
    char path[FILENAME_MAX];
    ssize_t count = readlink("/proc/self/exe", path, FILENAME_MAX);
    if (count <= 0) {
        throw std::runtime_error("failed to resolve /proc/self/exe");
    }
    return std::filesystem::path(std::string(path, count));
#endif
}

inline void unset_reserved_args(common_preset & preset, bool unset_model_args) {
    preset.unset_option("LLAMA_ARG_SSL_KEY_FILE");
    preset.unset_option("LLAMA_ARG_SSL_CERT_FILE");
    preset.unset_option("LLAMA_API_KEY");
    preset.unset_option("LLAMA_ARG_MODELS_DIR");
    preset.unset_option("LLAMA_ARG_MODELS_MAX");
    preset.unset_option("LLAMA_ARG_MODELS_PRESET");
    preset.unset_option("LLAMA_ARG_MODELS_AUTOLOAD");
    if (unset_model_args) {
        preset.unset_option("LLAMA_ARG_MODEL");
        preset.unset_option("LLAMA_ARG_MMPROJ");
        preset.unset_option("LLAMA_ARG_ALIAS");
        preset.unset_option("LLAMA_ARG_HF_REPO");
    }
}

#ifdef _WIN32
inline std::string wide_to_utf8(const wchar_t * ws) {
    if (!ws || !*ws) {
        return {};
    }

    const int len = static_cast<int>(std::wcslen(ws));
    const int bytes = WideCharToMultiByte(CP_UTF8, 0, ws, len, nullptr, 0, nullptr, nullptr);
    if (bytes == 0) {
        return {};
    }

    std::string utf8(bytes, '\0');
    WideCharToMultiByte(CP_UTF8, 0, ws, len, utf8.data(), bytes, nullptr, nullptr);

    return utf8;
}
#endif

inline std::vector<std::string> get_environment() {
    std::vector<std::string> env;

#ifdef _WIN32
    LPWCH env_block = GetEnvironmentStringsW();
    if (!env_block) {
        return env;
    }
    for (LPWCH e = env_block; *e; e += wcslen(e) + 1) {
        env.emplace_back(wide_to_utf8(e));
    }
    FreeEnvironmentStringsW(env_block);
#else
    if (environ == nullptr) {
        return env;
    }
    for (char ** e = environ; *e != nullptr; e++) {
        env.emplace_back(*e);
    }
#endif

    return env;
}
