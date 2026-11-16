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
#include <chrono>
#include <cstdio>
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
        : memory_mover(in, done_capacity, 1), engine_([this](std::stop_token stop) { run(stop); }) {}
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
    start_result start(const transfer_frame & t) override {
        if (!(t.src.domain == memory_domain::disk && t.dst.domain == memory_domain::host)) {
            return start_result::refused;
        }
        // started, not finished; a full engine queue is would_block, never a wait
        return work_.push(t) ? start_result::started : start_result::would_block;
    }
    void run(std::stop_token stop) {
        transfer_frame batch[32];
        while (!stop.stop_requested() || work_.pushed() != work_.taken()) {
            const size_t n = work_.drain(batch);
            for (size_t i = 0; i < n; ++i) {
                complete(0, batch[i].ticket, transfer_status::moved, transfer_done_frame::NO_SLOT); // its one context
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
        const buffer_ref disk{ memory_domain::disk, 1, 7, 0, 0, 4096 };
        const buffer_ref host{ memory_domain::host, 0, 3, 0, 0, 4096 };
        const buffer_ref dev{ memory_domain::device, 2, 9, 0, 0, 4096 };
        uint64_t ticket = 0;
        for (int i = 0; i < 300; ++i) {
            require(in.push({ ++ticket, disk, host }), "a page read is queued");   // moved later
            require(in.push({ ++ticket, host, dev }), "a host->device move is queued"); // aliased now
            require(in.push({ ++ticket, dev, disk }), "a device->disk move is queued"); // refused (unsupported here)
            mover.pump();
        }
        while (done.size() < 900) {
            const uint64_t bell = mover.doorbell(); // read BEFORE draining: a ring after it wakes us
            mover.pump();                           // resume anything that was waiting for room
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

// A test pool: issues pins on its bytes and counts the ones out (the custody a mover must hold and return).
struct pin_pool {
    std::atomic<int> out{0};
    static void unpin(void * pool, uint64_t) noexcept { static_cast<pin_pool *>(pool)->out.fetch_sub(1); }
    buffer_pin pin(std::byte * data, uint64_t bytes) {
        out.fetch_add(1);
        return buffer_pin(data, bytes, this, &unpin, 0);
    }
};

// The parallel copier: many transfers in flight, each split across lanes; bytes arrive exactly; every
// ticket completes once; a size mismatch and a disk pair are refused typed.
static void run_copies(size_t lanes, size_t chunk, size_t n_transfers, size_t bytes_each, double * gbps) {
    std::vector<std::vector<std::byte>> src(n_transfers), dst(n_transfers);
    for (size_t i = 0; i < n_transfers; ++i) {
        src[i].resize(bytes_each);
        dst[i].assign(bytes_each, std::byte{0});
        for (size_t b = 0; b < bytes_each; b += 4096) src[i][b] = std::byte(uint8_t(i * 31 + b / 4096));
    }
    // pool 0 = sources, pool 1 = destinations; buffer = index
    pin_pool pool;
    auto resolve = [&](const buffer_ref & r) -> buffer_pin {
        auto & v = r.pool == 0 ? src : dst;
        // the custody contract: a live generation (0 here) and in bounds, else no pin
        if (r.generation != 0 || r.buffer >= v.size() || r.offset + r.bytes > v[r.buffer].size()) return {};
        return pool.pin(v[r.buffer].data() + r.offset, v[r.buffer].size() - r.offset);
    };
    frame_ring<transfer_frame> in(n_transfers + 8);
    std::map<uint64_t, transfer_status> done;
    const auto t0 = std::chrono::steady_clock::now();
    {
        parallel_copier mover(in, resolve, lanes, chunk, 64);
        for (size_t i = 0; i < n_transfers; ++i) {
            const buffer_ref s{ memory_domain::host, 0, (uint32_t) i, 0, 0, bytes_each };
            const buffer_ref d{ memory_domain::host, 1, (uint32_t) i, 0, 0, bytes_each };
            require(in.push({ i + 1, s, d }), "queued");
        }
        const buffer_ref bad_s{ memory_domain::host, 0, 0, 0, 0, bytes_each };
        const buffer_ref bad_d{ memory_domain::host, 1, 0, 0, 0, bytes_each - 1 };
        require(in.push({ n_transfers + 1, bad_s, bad_d }), "queued");                     // size mismatch
        require(in.push({ n_transfers + 2, bad_s, { memory_domain::disk, 9, 0, 0, 0, bytes_each } }), "queued"); // disk pair
        while (done.size() < n_transfers + 2) {
            const uint64_t bell = mover.doorbell();
            mover.pump(); // starts what the lanes have room for and returns: never waits
            transfer_done_frame batch[64];
            const size_t n = mover.drain_done(batch);
            for (size_t i = 0; i < n; ++i) {
                require(done.emplace(batch[i].ticket, batch[i].status).second, "each ticket completes once");
            }
            if (n == 0) mover.wait_done(bell);
        }
    }
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (gbps) *gbps = (double) n_transfers * (double) bytes_each / secs / 1e9;
    for (size_t i = 0; i < n_transfers; ++i) {
        require(done[i + 1] == transfer_status::moved, "every copy moved");
        require(std::memcmp(src[i].data(), dst[i].data(), bytes_each) == 0, "the bytes arrived exactly");
    }
    require(done[n_transfers + 1] == transfer_status::refused, "a size mismatch is refused, never truncated");
    require(done[n_transfers + 2] == transfer_status::refused, "a disk pair is refused by the host copier");
    require(pool.out.load() == 0, "every pin returned");
}

// what this catches (Codex on #134): a consumer's close raced the producer's tail store, erasing the closed
// bit or stranding a frame whose push returned true; and a producer waiting for room never woke on close
static void a_close_never_strands_an_accepted_frame() {
    for (int round = 0; round < 2000; ++round) {
        frame_ring<uint64_t> ring(8);
        std::atomic<uint64_t> accepted{0};
        std::atomic<bool> producer_done{false};
        std::thread producer([&] {
            for (uint64_t i = 1; i < 1000; ++i) {
                if (ring.push(i)) {
                    accepted.fetch_add(1);
                } else if (ring.closed() || !ring.wait_for_room()) {
                    break; // closed: wait_for_room returns false rather than sleeping forever
                }
            }
            producer_done = true;
        });
        uint64_t drained = 0, buf[8];
        for (int k = 0; k < round % 7; ++k) drained += ring.drain(buf);
        ring.close();
        producer.join();
        while (size_t n = ring.drain(buf)) drained += n;
        require(ring.closed(), "the closed bit survives a racing push");
        require(drained == accepted.load(), "every frame whose push returned true is drained");
    }
}

// Codex's #134 review, one regression each.
using done_map = std::map<uint64_t, transfer_status>;
static void drain_all(memory_mover & mover, done_map & done, size_t want) {
    while (done.size() < want) {
        const uint64_t bell = mover.doorbell();
        mover.pump();
        transfer_done_frame batch[4];
        const size_t n = mover.drain_done(batch);
        for (size_t i = 0; i < n; ++i) {
            require(done.emplace(batch[i].ticket, batch[i].status).second, "each ticket has exactly one terminal frame");
        }
        if (n == 0) mover.wait_done(bell);
    }
}

// what this catches: settlements published into a full completion ring slept inside pump, while the same
// caller could only drain after pump returned (capacity 2 + 3 aliases deadlocked)
static void a_full_completion_ring_is_backpressure_not_a_deadlock() {
    std::vector<std::byte> a(4096), b(4096);
    pin_pool pool;
    auto resolve = [&](const buffer_ref & r) -> buffer_pin {
        auto & v = r.buffer == 0 ? a : b;
        if (r.generation != 0 || r.offset + r.bytes > v.size()) return {};
        return pool.pin(v.data() + r.offset, v.size() - r.offset);
    };
    frame_ring<transfer_frame> in(16);
    parallel_copier mover(in, resolve, 2, 4096, 2);
    const buffer_ref x{ memory_domain::host, 0, 0, 0, 0, 4096 };
    for (uint64_t t = 1; t <= 5; ++t) require(in.push({ t, x, x }), "queued");
    mover.pump(); // must return with settlements held, never sleep
    require(mover.blocked(), "unsettled frames are reported as backpressure");
    done_map done;
    drain_all(mover, done, 5);
    for (auto & [t, s] : done) require(s == transfer_status::aliased, "each alias settles once");
}

// what this catches: an alias succeeded before validation (unequal lengths, a stale generation), and
// overlapping ranges in one buffer reached parallel memcpy (UB)
static void invalid_aliases_and_overlaps_are_refused() {
    std::vector<std::byte> a(1 << 16);
    pin_pool pool;
    auto resolve = [&](const buffer_ref & r) -> buffer_pin {
        if (r.generation != 0 || r.offset + r.bytes > a.size()) return {};
        // buffers 0 and 1 are two ids over the SAME bytes (a physical alias the ids do not show)
        return pool.pin(a.data() + r.offset, a.size() - r.offset);
    };
    frame_ring<transfer_frame> in(16);
    parallel_copier mover(in, resolve, 4, 4096, 8);
    const auto ref = [](uint32_t gen, uint64_t off, uint64_t n) { return buffer_ref{ memory_domain::host, 0, 0, gen, off, n }; };
    require(in.push({ 1, ref(0, 0, 4096), ref(0, 0, 8192) }), "q");   // same start, unequal lengths
    require(in.push({ 2, ref(1, 0, 4096), ref(1, 0, 4096) }), "q");   // stale generation named twice
    require(in.push({ 3, ref(0, 0, 8192), ref(0, 4096, 8192) }), "q"); // overlapping ranges
    require(in.push({ 4, ref(0, 0, 4096), ref(0, 1 << 16, 4096) }), "q"); // out of bounds
    require(in.push({ 5, ref(0, 0, 4096), ref(0, 8192, 4096) }), "q");  // disjoint: a real copy
    const auto other = [](uint64_t off, uint64_t n) { return buffer_ref{ memory_domain::host, 0, 1, 0, off, n }; };
    require(in.push({ 6, ref(0, 0, 4096), other(0, 4096) }), "q");    // another id, the SAME bytes
    require(in.push({ 7, ref(0, 0, 4096), other(2048, 4096) }), "q"); // another id, overlapping bytes
    done_map done;
    drain_all(mover, done, 7);
    require(done[1] == transfer_status::refused && done[2] == transfer_status::refused, "an alias is never unvalidated");
    require(done[3] == transfer_status::refused, "overlap is refused, never raced");
    require(done[4] == transfer_status::refused, "out of bounds is refused");
    require(done[5] == transfer_status::moved, "a disjoint copy in one buffer moves");
    require(done[6] == transfer_status::aliased, "two ids over the same physical bytes alias, decided on addresses");
    require(done[7] == transfer_status::refused, "two ids over overlapping physical bytes are refused");
    require(pool.out.load() == 0, "every pin went back to its pool once its completion was drained");
}

// what this catches: the destructor joined lanes blocked on full completion rings, and tickets never handed
// to a lane had no terminal outcome
// what this catches (Codex on #134): a generation alone did not keep the pool's custody; the pins must be
// held from start until the completion is DRAINED, then returned
static void a_transfer_holds_its_pins_until_drained() {
    std::vector<std::byte> a(1 << 20), b(1 << 20);
    pin_pool pool;
    auto resolve = [&](const buffer_ref & r) -> buffer_pin {
        auto & v = r.buffer == 0 ? a : b;
        return pool.pin(v.data(), v.size());
    };
    frame_ring<transfer_frame> in(4);
    parallel_copier mover(in, resolve, 2, 64 * 1024, 2);
    require(in.push({ 1, { memory_domain::host, 0, 0, 0, 0, 1 << 20 }, { memory_domain::host, 0, 1, 0, 0, 1 << 20 } }), "q");
    mover.pump();
    while (mover.doorbell() == 0) mover.wait_done(0);   // the copy landed and was published...
    require(pool.out.load() == 2, "...yet both pins stay out until the completion is drained");
    done_map done;
    drain_all(mover, done, 1);
    require(done[1] == transfer_status::moved && pool.out.load() == 0, "drained: moved, and both pins returned");
}

static void teardown_with_work_outstanding_ends_every_ticket() {
    const size_t n = 64, bytes = 1 << 20;
    std::vector<std::vector<std::byte>> src(n, std::vector<std::byte>(bytes)), dst(n, std::vector<std::byte>(bytes));
    pin_pool pool;
    auto resolve = [&](const buffer_ref & r) -> buffer_pin {
        auto & v = r.pool == 0 ? src : dst;
        if (r.generation != 0 || r.buffer >= v.size() || r.offset + r.bytes > v[r.buffer].size()) return {};
        return pool.pin(v[r.buffer].data() + r.offset, v[r.buffer].size() - r.offset);
    };
    frame_ring<transfer_frame> in(n);
    {
        parallel_copier mover(in, resolve, 3, 64 * 1024, 2); // two slots: most transfers stay queued
        for (uint64_t i = 0; i < n; ++i) {
            require(in.push({ i + 1, { memory_domain::host, 0, (uint32_t) i, 0, 0, bytes }, { memory_domain::host, 1, (uint32_t) i, 0, 0, bytes } }), "q");
        }
        mover.pump();
        mover.shutdown();
        done_map done;
        drain_all(mover, done, n); // every ticket ends: moved/failed by a lane, or cancelled unhanded
        size_t cancelled = 0;
        for (auto & [t, s] : done) {
            require(s == transfer_status::moved || s == transfer_status::failed || s == transfer_status::cancelled, "a terminal status");
            cancelled += s == transfer_status::cancelled;
        }
        require(cancelled > 0, "transfers no lane took are cancelled, typed");
        require(pool.out.load() == 0, "after every terminal frame is drained, no pin is held");
        require(!in.push({ n + 1, {}, {} }), "a shut-down mover takes nothing more");
    } // the destructor joins without hanging
    {
        parallel_copier mover(in, resolve, 3, 64 * 1024, 2);
    } // and a fresh mover over the closed ring tears down at once
}

static void the_parallel_copier_moves_many_at_once() {
    run_copies(/*lanes=*/6, /*chunk=*/64 * 1024, /*transfers=*/200, /*bytes=*/1 << 20, nullptr);
}

int main(int argc, char ** argv) {
    if (argc > 1 && std::strcmp(argv[1], "--bandwidth") == 0) {
        // the receipt: one lane against the lanes at the measured knee, same transfers
        for (size_t lanes : { (size_t) 1, (size_t) 4, (size_t) 8, (size_t) 12 }) {
            double g = 0;
            run_copies(lanes, 4 << 20, 16, 64 << 20, &g);
            std::printf("parallel_copier lanes %2zu: %6.1f GB/s (16 x 64 MiB)\n", lanes, g);
        }
        return 0;
    }
    one_stream_delivers_every_frame_in_order();
    many_streams_run_at_once();
    a_full_ring_is_backpressure();
    close_wakes_a_waiting_consumer();
    a_mover_streams_moves_without_waiting();
    a_close_never_strands_an_accepted_frame();
    a_full_completion_ring_is_backpressure_not_a_deadlock();
    invalid_aliases_and_overlaps_are_refused();
    a_transfer_holds_its_pins_until_drained();
    teardown_with_work_outstanding_ends_every_ticket();
    the_parallel_copier_moves_many_at_once();
    std::cout << "test-server-frames: ok\n";
    return 0;
}
