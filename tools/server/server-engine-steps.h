#pragma once

// The engine's step capture (continuum CONTINUUM-ENGINE 4.10 P5 slice 1): every step the task loop
// runs, published to ONE subscriber on its own stream (`GET /engine/steps`), never through the
// learning events stream. The subscriber sets the ring's bound when it subscribes (its memory to
// spend); with no subscriber nothing is recorded. Writing never blocks the task loop: a full ring
// counts what it could not keep and publishes that count in position as a `missed` record, so the
// reader knows exactly where its capture has a hole.

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>

#include "server-common.h"

struct engine_step_capture {
    // Open a capture of at most `capacity` records; refused while another is open.
    bool open(size_t capacity, int64_t origin_us) {
        std::lock_guard<std::mutex> lock(mtx);
        if (subscribed || capacity == 0) {
            return false;
        }
        subscribed   = true;
        bound        = capacity;
        next_seq     = 0;
        lost         = 0;
        records.clear();
        this->origin_us = origin_us;
        return true;
    }

    void close() {
        std::lock_guard<std::mutex> lock(mtx);
        subscribed = false;
        records.clear();
    }

    bool active() {
        std::lock_guard<std::mutex> lock(mtx);
        return subscribed;
    }

    int64_t origin() {
        std::lock_guard<std::mutex> lock(mtx);
        return origin_us;
    }

    // Record one step record (its `seq` is stamped here). Never blocks; counts loss when full.
    void push(json record) {
        std::lock_guard<std::mutex> lock(mtx);
        if (!subscribed) {
            return;
        }
        if (lost > 0 && records.size() < bound) {
            records.push_back({{"kind", "missed"}, {"seq", next_seq++}, {"count", lost}});
            lost = 0;
        }
        if (records.size() >= bound) {
            lost++; // published as one `missed` record when room returns; no sequence number spent
            return;
        }
        record["seq"] = next_seq++;
        records.push_back(std::move(record));
        ready.notify_one();
    }

    // The next record as one SSE frame, waiting at most `wait_s`; empty when none arrived.
    std::string next_frame(int wait_s) {
        std::unique_lock<std::mutex> lock(mtx);
        ready.wait_for(lock, std::chrono::seconds(wait_s), [&] { return !records.empty() || !subscribed; });
        if (records.empty()) {
            return std::string();
        }
        std::string frame = "data: " + records.front().dump() + "\n\n";
        records.pop_front();
        return frame;
    }

private:
    std::mutex mtx;
    std::condition_variable ready;
    bool subscribed = false;
    size_t bound = 0;
    uint64_t next_seq = 0;
    uint64_t lost = 0;
    int64_t origin_us = 0;
    std::deque<json> records;
};
