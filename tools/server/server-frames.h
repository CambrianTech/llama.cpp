#pragma once

// Continuum fork: STREAMS OF FRAMES and MOVING BYTES BY REFERENCE, one pattern reused everywhere
// (CONTINUUM-ENGINE 4.7.12). Joel, 2026-11-16: "Fast tokenless memory transfer and DMA must be encoded
// into a pattern or parent class"; "Streaming not promises"; "pass references and ids, don't
// tokenize, do in parallel, don't wait. Run many things simultaneously."
//
//   frame_ring<T>   a lock-free single-producer / single-consumer ring of FRAMES. A frame is small and
//                   trivially copyable (ids, references, counters), never a payload. A full ring is
//                   backpressure the producer sees at once (push returns false); an empty ring sleeps
//                   the consumer on an atomic wait (a futex), never a lock, a timer or a promise. Work
//                   flows as streams: requests in one ring, results out another, many rings at once.
//
//   buffer_ref      bytes NAMED, never carried: a memory domain, the owning pool, a buffer id, an
//                   offset and a length. What crosses a ring, a thread or a process is the reference.
//
//   memory_mover    the one base class for moving bytes between domains. It consumes a stream of
//                   transfer frames and produces a stream of completion frames; a backend implements
//                   only `start` (enqueue the move on its own engine: a GPU blit, a backend async copy,
//                   a disk read into a pool buffer) and reports completion. Within one physical domain
//                   (unified memory) a "move" is a new reference: nothing is copied. Nothing is ever
//                   encoded or decoded on the way.
//
// Pure and header-only, proven by tests/test-server-frames.cpp without a model.

#include <algorithm>
#include <atomic>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>
#include <vector>

// ---- the stream ------------------------------------------------------------------------------------

template <typename T>
class frame_ring {
    static_assert(std::is_trivially_copyable_v<T>, "a frame is ids and references, never a payload with an owner");

public:
    // `capacity` rounds up to a power of two; the ring is allocated once.
    explicit frame_ring(size_t capacity) : slots_(std::bit_ceil(capacity < 2 ? size_t(2) : capacity)), mask_(slots_.size() - 1) {}

    frame_ring(const frame_ring &)             = delete;
    frame_ring & operator=(const frame_ring &) = delete;

    // Producer side. False when full: backpressure, decided by the producer, never hidden.
    bool push(const T & frame) {
        const uint64_t raw = tail_.load(std::memory_order_relaxed);
        if (raw & CLOSED) {
            return false; // a closed stream takes nothing more
        }
        const uint64_t tail = raw;
        if (tail - head_cache_ >= slots_.size()) {
            head_cache_ = head_.load(std::memory_order_acquire);
            if (tail - head_cache_ >= slots_.size()) {
                return false;
            }
        }
        slots_[tail & mask_] = frame;
        tail_.store(tail + 1, std::memory_order_release);
        tail_.notify_one();
        return true;
    }

    // Consumer side: take up to out.size() frames at once; returns how many.
    size_t drain(std::span<T> out) {
        const uint64_t head = head_.load(std::memory_order_relaxed);
        uint64_t tail = tail_cache_;
        if (tail == head) {
            tail = tail_cache_ = tail_.load(std::memory_order_acquire) & COUNT;
        }
        const size_t n = (size_t) std::min<uint64_t>(tail - head, out.size());
        for (size_t i = 0; i < n; ++i) {
            out[i] = slots_[(head + i) & mask_];
        }
        if (n > 0) {
            head_.store(head + n, std::memory_order_release);
            head_.notify_one(); // a producer waiting for room wakes
        }
        return n;
    }

    // Producer side: sleep until there is room for one frame (an atomic wait on the consumer's head).
    void wait_for_room() const {
        const uint64_t tail = tail_.load(std::memory_order_relaxed);
        uint64_t head = head_.load(std::memory_order_acquire);
        while (tail - head >= slots_.size()) {
            head_.wait(head, std::memory_order_acquire);
            head = head_.load(std::memory_order_acquire);
        }
    }

    // Consumer side: sleep until a frame is pushed or the stream is closed. An atomic wait returns only
    // when the watched WORD changes, so the closed flag lives in the same word as the count (its top
    // bit): closing changes the word the consumer sleeps on.
    void wait_for_frames() const {
        const uint64_t head = head_.load(std::memory_order_relaxed);
        uint64_t seen = tail_.load(std::memory_order_acquire);
        while ((seen & COUNT) == head && !(seen & CLOSED)) {
            tail_.wait(seen, std::memory_order_acquire);
            seen = tail_.load(std::memory_order_acquire);
        }
    }

    // The producer ends the stream; a waiting consumer wakes (the word it sleeps on changes).
    void close() {
        tail_.fetch_or(CLOSED, std::memory_order_release);
        tail_.notify_all();
    }
    bool closed() const { return (tail_.load(std::memory_order_acquire) & CLOSED) != 0; }

    size_t   capacity() const { return slots_.size(); }
    uint64_t pushed()   const { return tail_.load(std::memory_order_acquire) & COUNT; }
    uint64_t taken()    const { return head_.load(std::memory_order_acquire); }

private:
    static constexpr uint64_t CLOSED = uint64_t(1) << 63;
    static constexpr uint64_t COUNT  = CLOSED - 1;

    std::vector<T>                     slots_;
    const uint64_t                     mask_;
    alignas(64) std::atomic<uint64_t>  head_{0};       // consumer writes
    alignas(64) std::atomic<uint64_t>  tail_{0};       // producer writes: count | CLOSED
    alignas(64) uint64_t               head_cache_ = 0; // producer's view of head
    alignas(64) uint64_t               tail_cache_ = 0; // consumer's view of tail
};

// ---- bytes by reference ----------------------------------------------------------------------------

enum class memory_domain : uint8_t {
    host,    // host-addressable memory
    device,  // backend (GPU) buffers; on unified memory the same physical pages as host
    disk,    // a page file
};

struct buffer_ref {
    memory_domain domain = memory_domain::host;
    uint32_t      pool   = 0; // the owning pool (the host pool, a device pool, a page store)
    uint32_t      buffer = 0; // the buffer within that pool
    uint64_t      offset = 0;
    uint64_t      bytes  = 0;
};
static_assert(std::is_trivially_copyable_v<buffer_ref>);

// ---- moving bytes ----------------------------------------------------------------------------------

struct transfer_frame {
    uint64_t   ticket = 0;
    buffer_ref src;
    buffer_ref dst;
};

enum class transfer_status : uint8_t {
    moved,      // the backend moved the bytes
    aliased,    // same physical domain: dst now names src's bytes, nothing was copied
    refused,    // the backend cannot move between these domains (typed, never a silent CPU copy)
    failed,     // the backend reported an error
};

struct transfer_done_frame {
    uint64_t        ticket = 0;
    transfer_status status = transfer_status::moved;
};

// The one base for every byte mover. A subclass is only the diff: how its engine starts a move and
// when that move completes. Completions come from TWO producers, each with its own single-producer
// ring: the pumping thread settles what needs no engine (aliased, refused), and the backend's own
// completion context finishes real moves. One doorbell counter, bumped by both, wakes the consumer:
// fan-in without a lock.
class memory_mover {
public:
    memory_mover(frame_ring<transfer_frame> & in, size_t done_capacity) : in_(in), settled_(done_capacity), finished_(done_capacity) {}
    virtual ~memory_mover() = default;

    // Drain every pending transfer frame and start each on the backend's engine without waiting on
    // any of them. Returns how many were started on the engine.
    size_t pump() {
        transfer_frame batch[64];
        size_t started = 0;
        for (size_t n; (n = in_.drain(batch)) > 0;) {
            for (size_t i = 0; i < n; ++i) {
                if (aliases(batch[i].src, batch[i].dst)) {
                    settle(batch[i].ticket, transfer_status::aliased);
                } else if (!start(batch[i])) {
                    settle(batch[i].ticket, transfer_status::refused);
                } else {
                    started++;
                }
            }
        }
        return started;
    }

    // Consumer side: take completions from both producers; returns how many.
    size_t drain_done(std::span<transfer_done_frame> out) {
        const size_t a = settled_.drain(out);
        return a + finished_.drain(out.subspan(a));
    }

    // Consumer side: sleep until either producer rings the doorbell (an atomic wait, no lock).
    void wait_done(uint64_t seen) const {
        while (doorbell_.load(std::memory_order_acquire) == seen) {
            doorbell_.wait(seen, std::memory_order_acquire);
        }
    }
    uint64_t doorbell() const { return doorbell_.load(std::memory_order_acquire); }

    uint64_t backpressured() const { return backpressured_.load(std::memory_order_relaxed); }

protected:
    // Enqueue the move on this backend's own engine; false when it cannot move between these domains.
    virtual bool start(const transfer_frame & t) = 0;

    // Whether dst can simply name src's bytes (one physical domain): no copy at all.
    virtual bool aliases(const buffer_ref & src, const buffer_ref & dst) const = 0;

    // The backend's ONE completion context reports a finished move.
    void complete(uint64_t ticket, transfer_status status) { publish(finished_, ticket, status); }

private:
    void settle(uint64_t ticket, transfer_status status) { publish(settled_, ticket, status); }

    void publish(frame_ring<transfer_done_frame> & ring, uint64_t ticket, transfer_status status) {
        while (!ring.push(transfer_done_frame{ ticket, status })) {
            // the consumer of completions is behind: counted, then sleep until it drains (no spin)
            backpressured_.fetch_add(1, std::memory_order_relaxed);
            ring.wait_for_room();
        }
        doorbell_.fetch_add(1, std::memory_order_release);
        doorbell_.notify_all();
    }

    frame_ring<transfer_frame> &    in_;
    frame_ring<transfer_done_frame> settled_;  // producer: the pumping thread
    frame_ring<transfer_done_frame> finished_; // producer: the backend's completion context
    std::atomic<uint64_t>           doorbell_{0};
    std::atomic<uint64_t>           backpressured_{0};
};
