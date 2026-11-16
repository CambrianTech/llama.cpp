#pragma once

// The engine's step capture (continuum CONTINUUM-ENGINE 4.10 P5 slice 1): every step the task loop
// runs, published to ONE subscriber on its own stream (`GET /engine/steps`), never through the
// learning events stream. The subscriber sets the ring's bound when it subscribes (its memory to
// spend); with no subscriber nothing is recorded. Writing never blocks the task loop: a full ring
// counts what it could not keep and publishes that count in position as a `missed` record, so the
// reader knows exactly where its capture has a hole.
//
// The task loop only copies typed fields into a record the ring allocated when the subscriber opened
// it (inline room for every slot): no allocation, no JSON on the loop. The text frame is formatted at
// the edge, on the HTTP reader's thread (next_frame).

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

// One slot's share of a step.
struct engine_step_work {
    int32_t  task           = 0;
    int32_t  depth          = 0; // tokens in her context when the step began
    int32_t  prefilled      = 0;
    uint64_t decoded        = 0;
    bool     accepted_known = false; // a task that finished in the step has its draft counters reset
    uint64_t accepted       = 0;
};

struct engine_step_record {
    enum kind_t : uint8_t { began, ended, missed } kind = began;
    int64_t  at_us = 0;
    uint64_t step  = 0;
    uint64_t seq   = 0;
    uint64_t count = 0; // missed: how many records were lost here
    // capacity reserved once at open(); a push copies within it
    std::vector<int32_t>          chosen;
    std::vector<engine_step_work> work;
};

struct engine_step_capture {
    // Open a capture of at most `capacity` records for up to `n_slots` slots per step; refused while
    // another is open. Every record's storage is allocated here, once.
    bool open(size_t capacity, size_t n_slots, int64_t origin_us) {
        std::lock_guard<std::mutex> lock(mtx);
        if (subscribed || capacity == 0) {
            return false;
        }
        ring.assign(capacity, engine_step_record{});
        for (auto & r : ring) {
            r.chosen.reserve(n_slots);
            r.work.reserve(n_slots);
        }
        subscribed      = true;
        head            = 0;
        size            = 0;
        next_seq        = 0;
        lost            = 0;
        this->origin_us = origin_us;
        return true;
    }

    void close() {
        std::lock_guard<std::mutex> lock(mtx);
        subscribed = false;
        size       = 0;
        ready.notify_all();
    }

    bool active() {
        std::lock_guard<std::mutex> lock(mtx);
        return subscribed;
    }

    int64_t origin() {
        std::lock_guard<std::mutex> lock(mtx);
        return origin_us;
    }

    // Record a `began` step: the tasks it chose.
    void push_began(int64_t at_us, uint64_t step, const int32_t * chosen, size_t n) {
        std::lock_guard<std::mutex> lock(mtx);
        engine_step_record * r = claim(engine_step_record::began, at_us, step);
        if (r != nullptr) {
            r->chosen.assign(chosen, chosen + n);
        }
    }

    // Record an `ended` step: each chosen slot's work.
    void push_ended(int64_t at_us, uint64_t step, const engine_step_work * work, size_t n) {
        std::lock_guard<std::mutex> lock(mtx);
        engine_step_record * r = claim(engine_step_record::ended, at_us, step);
        if (r != nullptr) {
            r->work.assign(work, work + n);
        }
    }

    // The next record as one SSE frame, formatted here (the reader's thread), waiting at most
    // `wait_s`; empty when none arrived.
    std::string next_frame(int wait_s) {
        std::unique_lock<std::mutex> lock(mtx);
        ready.wait_for(lock, std::chrono::seconds(wait_s), [&] { return size > 0 || !subscribed; });
        if (size == 0) {
            return std::string();
        }
        const engine_step_record & r = ring[head];
        std::string frame = "data: {\"kind\":\"";
        frame += kind_name(r.kind);
        frame += "\"";
        if (r.kind == engine_step_record::missed) {
            frame += ",\"count\":" + std::to_string(r.count);
        } else {
            frame += ",\"at_us\":" + std::to_string(r.at_us) + ",\"step\":" + std::to_string(r.step);
            if (r.kind == engine_step_record::began) {
                frame += ",\"chosen\":[";
                for (size_t i = 0; i < r.chosen.size(); ++i) {
                    frame += (i ? "," : "") + std::to_string(r.chosen[i]);
                }
                frame += "]";
            } else {
                frame += ",\"work\":[";
                for (size_t i = 0; i < r.work.size(); ++i) {
                    const auto & w = r.work[i];
                    frame += (i ? ",{" : "{");
                    frame += "\"task\":" + std::to_string(w.task) + ",\"depth\":" + std::to_string(w.depth) +
                             ",\"prefilled\":" + std::to_string(w.prefilled) + ",\"decoded\":" + std::to_string(w.decoded) +
                             ",\"accepted_draft\":" + (w.accepted_known ? std::to_string(w.accepted) : std::string("null")) + "}";
                }
                frame += "]";
            }
        }
        frame += ",\"seq\":" + std::to_string(r.seq) + "}\n\n";
        head = (head + 1) % ring.size();
        --size;
        return frame;
    }

private:
    static const char * kind_name(engine_step_record::kind_t kind) {
        switch (kind) {
            case engine_step_record::began:  return "began";
            case engine_step_record::ended:  return "ended";
            case engine_step_record::missed: return "missed";
        }
        return "missed";
    }

    // The next free record, its header stamped; nullptr (and one more lost) when the ring is full. A
    // `missed` record for earlier losses is placed first when room returns. Under mtx.
    engine_step_record * claim(engine_step_record::kind_t kind, int64_t at_us, uint64_t step) {
        if (!subscribed) {
            return nullptr;
        }
        if (lost > 0 && size < ring.size()) {
            engine_step_record & m = ring[(head + size++) % ring.size()];
            m.kind  = engine_step_record::missed;
            m.seq   = next_seq++;
            m.count = lost;
            m.chosen.clear();
            m.work.clear();
            lost = 0;
        }
        if (size >= ring.size()) {
            lost++; // published as one `missed` record when room returns; no sequence number spent
            return nullptr;
        }
        engine_step_record & r = ring[(head + size++) % ring.size()];
        r.kind  = kind;
        r.at_us = at_us;
        r.step  = step;
        r.seq   = next_seq++;
        r.count = 0;
        r.chosen.clear();
        r.work.clear();
        ready.notify_one();
        return &r;
    }

    std::mutex                      mtx;
    std::condition_variable         ready;
    bool                            subscribed = false;
    std::vector<engine_step_record> ring;
    size_t                          head     = 0;
    size_t                          size     = 0;
    uint64_t                        next_seq = 0;
    uint64_t                        lost     = 0;
    int64_t                         origin_us = 0;
};
