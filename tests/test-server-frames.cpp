// what this catches: the frames-and-references pattern (server-frames.h) without a model: a
// single-producer/single-consumer ring delivers every frame once and in order across threads with no
// lock; many rings run at once; a full ring is backpressure (push refuses, the producer sleeps on an
// atomic until the consumer drains); close wakes a sleeping consumer; and a memory_mover starts every
// transfer without waiting on any, aliases a move within one physical domain (nothing copied),
// refuses an unsupported pair typed (never a silent copy), and streams completions back from two
// producers (the pumping thread and the backend's engine) through two rings and one doorbell.
#include "../tools/server/server-frames.h"

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <mutex>
#include <thread>

static void require(bool condition, const char * message) {
    if (!condition) {
        std::cerr << "FAILED: " << message << '\n';
        std::exit(1);
    }
}

// One producer and one consumer over `n` frames through a small ring: order and completeness.
static void stream(frame_ring<uint64_t> & ring, uint64_t n) {
    std::thread producer([&] {
        for (uint64_t i = 0; i < n; ++i) {
            while (!ring.push(i)) {
                ring.wait_for_room(); // backpressure: sleep until the consumer drains
            }
        }
        ring.close();
    });
    uint64_t expect = 0;
    uint64_t batch[256];
    while (true) {
        const size_t got = ring.drain(batch);
        for (size_t i = 0; i < got; ++i) {
            require(batch[i] == expect++, "every frame once, in order");
        }
        if (got == 0) {
            if (ring.closed() && ring.pushed() == ring.taken()) {
                break;
            }
            ring.wait_for_frames();
        }
    }
    producer.join();
    require(expect == n, "the consumer saw every frame");
}

static void one_stream_delivers_every_frame_in_order() {
    frame_ring<uint64_t> ring(64);
    stream(ring, 2'000'000);
}

static void many_streams_run_at_once() {
    std::vector<std::unique_ptr<frame_ring<uint64_t>>> rings;
    for (int i = 0; i < 8; ++i) {
        rings.push_back(std::make_unique<frame_ring<uint64_t>>(128));
    }
    std::vector<std::thread> pairs;
    for (auto & r : rings) {
        pairs.emplace_back([&r] { stream(*r, 250'000); });
    }
    for (auto & t : pairs) {
        t.join();
    }
}

static void a_full_ring_is_backpressure() {
    frame_ring<uint64_t> ring(4);
    for (uint64_t i = 0; i < 4; ++i) {
        require(ring.push(i), "fits");
    }
    require(!ring.push(99), "a full ring refuses the push at once: backpressure is visible");
    uint64_t out[2];
    require(ring.drain(out) == 2 && out[0] == 0 && out[1] == 1, "the consumer drains in order");
    require(ring.push(4) && ring.push(5), "room returns as soon as it drains");
}

static void close_wakes_a_waiting_consumer() {
    frame_ring<uint64_t> ring(8);
    std::thread consumer([&] { ring.wait_for_frames(); });
    ring.close();
    consumer.join();
    require(ring.closed(), "the stream ended and the consumer woke");
}

// A stand-in backend: host and device are one physical domain (unified memory), so a move between
// them aliases; a disk -> host move runs on its own engine (a worker standing in for DMA) and
// completes later from that one context; anything else is refused.
class test_mover final : public memory_mover {
public:
    test_mover(frame_ring<transfer_frame> & in, size_t done_capacity)
        : memory_mover(in, done_capacity), engine_([this](std::stop_token stop) { run(stop); }) {}
    ~test_mover() override {
        engine_.request_stop();
        work_.close();
        engine_.join();
    }

private:
    bool aliases(const buffer_ref & a, const buffer_ref & b) const override {
        const auto physical = [](memory_domain d) { return d == memory_domain::disk ? 1 : 0; };
        return physical(a.domain) == physical(b.domain);
    }
    bool start(const transfer_frame & t) override {
        if (!(t.src.domain == memory_domain::disk && t.dst.domain == memory_domain::host)) {
            return false;
        }
        while (!work_.push(t)) {
            work_.wait_for_room();
        }
        return true; // started, not finished: the caller never waits here
    }
    void run(std::stop_token stop) {
        transfer_frame batch[32];
        while (!stop.stop_requested() || work_.pushed() != work_.taken()) {
            const size_t n = work_.drain(batch);
            for (size_t i = 0; i < n; ++i) {
                complete(batch[i].ticket, transfer_status::moved); // the engine's completion context
            }
            if (n == 0) {
                if (work_.closed()) break;
                work_.wait_for_frames();
            }
        }
    }

    frame_ring<transfer_frame> work_{ 64 };
    std::jthread               engine_;
};

static void a_mover_streams_moves_without_waiting() {
    frame_ring<transfer_frame>      in(1024);
    std::map<uint64_t, transfer_status> done;
    {
        test_mover mover(in, 1024);
        const buffer_ref disk{ memory_domain::disk, 1, 7, 0, 4096 };
        const buffer_ref host{ memory_domain::host, 0, 3, 0, 4096 };
        const buffer_ref dev{ memory_domain::device, 2, 9, 0, 4096 };
        uint64_t ticket = 0;
        for (int i = 0; i < 300; ++i) {
            require(in.push({ ++ticket, disk, host }), "a page read is queued");   // moved later
            require(in.push({ ++ticket, host, dev }), "a host->device move is queued"); // aliased now
            require(in.push({ ++ticket, dev, disk }), "a device->disk move is queued"); // refused (unsupported here)
            mover.pump();
        }
        while (done.size() < 900) {
            const uint64_t bell = mover.doorbell(); // read BEFORE draining: a ring after it wakes us
            transfer_done_frame batch[64];
            const size_t n = mover.drain_done(batch);
            for (size_t i = 0; i < n; ++i) {
                require(done.emplace(batch[i].ticket, batch[i].status).second, "each completion once");
            }
            if (n == 0) mover.wait_done(bell);
        }
    }
    size_t moved = 0, aliased = 0, refused = 0;
    for (auto & [t, s] : done) {
        const int kind = (int) ((t - 1) % 3);
        if (kind == 0) { require(s == transfer_status::moved, "a disk read moves on the engine"); moved++; }
        if (kind == 1) { require(s == transfer_status::aliased, "within one physical domain nothing is copied"); aliased++; }
        if (kind == 2) { require(s == transfer_status::refused, "an unsupported pair is refused typed, never copied by the CPU"); refused++; }
    }
    require(moved == 300 && aliased == 300 && refused == 300, "every transfer completed exactly once");
}

int main() {
    one_stream_delivers_every_frame_in_order();
    many_streams_run_at_once();
    a_full_ring_is_backpressure();
    close_wakes_a_waiting_consumer();
    a_mover_streams_moves_without_waiting();
    std::cout << "test-server-frames: ok\n";
    return 0;
}
