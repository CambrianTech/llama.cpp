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
#include <cstring>
#include <functional>
#include <memory>
#include <optional>
#include <stop_token>
#include <thread>
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

// The one base for every byte mover. A subclass is only the diff: how its engines start a move and
// when each part completes. Moving is PARALLEL: a backend owns several completion contexts (lanes: CPU
// copy threads, GPU blit queues, disk readers), each completing into its OWN single-producer ring, and
// the pumping thread settles what needs no engine (aliased, refused) into one more. One doorbell
// counter, bumped by every producer, wakes the consumer: fan-in without a lock.
class memory_mover {
public:
    memory_mover(frame_ring<transfer_frame> & in, size_t done_capacity, size_t completion_contexts)
        : in_(in), settled_(done_capacity) {
        for (size_t i = 0; i < std::max<size_t>(completion_contexts, 1); ++i) {
            finished_.push_back(std::make_unique<frame_ring<transfer_done_frame>>(done_capacity));
        }
    }
    virtual ~memory_mover() = default;

    // Start as many queued transfers as the engines have room for, and RETURN: pump never waits. A
    // transfer the engines cannot take yet stays pending here (the backend remembers how far it got)
    // and resumes on the next pump; the caller drains completions in between, which is what makes
    // room. Returns how many transfers were fully handed to an engine.
    size_t pump() {
        size_t started = 0;
        while (true) {
            if (!pending_) {
                transfer_frame f;
                if (in_.drain(std::span<transfer_frame>(&f, 1)) == 0) {
                    return started;
                }
                if (aliases(f.src, f.dst)) {
                    settle(f.ticket, transfer_status::aliased);
                    continue;
                }
                pending_ = f;
            }
            switch (start(*pending_)) {
                case start_result::started:     started++; pending_.reset(); break;
                case start_result::refused:     settle(pending_->ticket, transfer_status::refused); pending_.reset(); break;
                case start_result::would_block: return started; // engines full: drain completions, pump again
            }
        }
    }

    // A transfer is waiting for engine room (the caller should drain completions, then pump again).
    bool blocked() const { return pending_.has_value(); }

    // Consumer side: take completions from every producer; returns how many.
    size_t drain_done(std::span<transfer_done_frame> out) {
        size_t n = settled_.drain(out);
        for (auto & ring : finished_) {
            n += ring->drain(out.subspan(n));
        }
        return n;
    }

    // Consumer side: sleep until any producer rings the doorbell (read doorbell() BEFORE draining).
    void wait_done(uint64_t seen) const {
        while (doorbell_.load(std::memory_order_acquire) == seen) {
            doorbell_.wait(seen, std::memory_order_acquire);
        }
    }
    uint64_t doorbell() const { return doorbell_.load(std::memory_order_acquire); }

    uint64_t backpressured() const { return backpressured_.load(std::memory_order_relaxed); }

protected:
    enum class start_result : uint8_t { started, refused, would_block };

    // Hand the move to this backend's engines, as much as they have room for, without waiting.
    // would_block: called again with the same frame on the next pump (the backend keeps its own
    // progress); refused: it cannot move between these domains.
    virtual start_result start(const transfer_frame & t) = 0;

    // Whether dst can simply name src's bytes (one physical domain): no copy at all.
    virtual bool aliases(const buffer_ref & src, const buffer_ref & dst) const = 0;

    // Completion context `context` (one lane) reports a finished move.
    void complete(size_t context, uint64_t ticket, transfer_status status) { publish(*finished_[context], ticket, status); }

private:
    void settle(uint64_t ticket, transfer_status status) { publish(settled_, ticket, status); }

    void publish(frame_ring<transfer_done_frame> & ring, uint64_t ticket, transfer_status status) {
        while (!ring.push(transfer_done_frame{ ticket, status })) {
            backpressured_.fetch_add(1, std::memory_order_relaxed);
            ring.wait_for_room(); // the consumer is behind: sleep until it drains (no spin)
        }
        doorbell_.fetch_add(1, std::memory_order_release);
        doorbell_.notify_all();
    }

    frame_ring<transfer_frame> &                                   in_;
    std::optional<transfer_frame>                                  pending_; // pumping thread only
    frame_ring<transfer_done_frame>                                settled_;  // producer: the pumping thread
    std::vector<std::unique_ptr<frame_ring<transfer_done_frame>>>  finished_; // producer: one lane each
    std::atomic<uint64_t>                                          doorbell_{0};
    std::atomic<uint64_t>                                          backpressured_{0};
};

// ---- the host backend: a parallel copier ------------------------------------------------------------

// Host-to-host (and file-mapped) moves done by CPU lanes in parallel: a transfer is split into chunks
// spread across the lanes, every lane copies its chunks at once, and the LAST chunk of a transfer to
// land publishes its completion. Lane count and chunk size come from the node's measured copy
// bandwidth curve (where adding a lane stops adding bandwidth), never a constant here.
class parallel_copier final : public memory_mover {
public:
    using resolver = std::function<std::byte *(const buffer_ref &)>; // a pool's id -> its bytes

    parallel_copier(frame_ring<transfer_frame> & in, resolver resolve, size_t lanes, size_t chunk_bytes, size_t in_flight)
        : memory_mover(in, in_flight, lanes), resolve_(std::move(resolve)), chunk_(std::max<size_t>(chunk_bytes, 4096)),
          slots_(std::max<size_t>(in_flight, 1)) {
        for (size_t i = 0; i < std::max<size_t>(lanes, 1); ++i) {
            lanes_.push_back(std::make_unique<lane>(in_flight * 4));
        }
        for (size_t i = 0; i < lanes_.size(); ++i) {
            lanes_[i]->worker = std::jthread([this, i](std::stop_token stop) { run_lane(i, stop); });
        }
    }

    ~parallel_copier() override {
        for (auto & l : lanes_) {
            l->work.close();
        }
        for (auto & l : lanes_) {
            l->worker.join();
        }
    }

    size_t lanes() const { return lanes_.size(); }

private:
    struct chunk_frame {
        uint32_t        slot;
        const std::byte * src;
        std::byte *     dst;
        uint64_t        bytes;
    };
    struct lane {
        explicit lane(size_t capacity) : work(capacity) {}
        frame_ring<chunk_frame> work;
        std::jthread            worker;
    };
    struct slot {
        std::atomic<bool>     in_use{false};
        std::atomic<uint32_t> remaining{0};
        uint64_t              ticket = 0;
    };

    bool aliases(const buffer_ref & a, const buffer_ref & b) const override {
        return a.domain == b.domain && a.pool == b.pool && a.buffer == b.buffer && a.offset == b.offset;
    }

    start_result start(const transfer_frame & t) override {
        if (!progress_) {
            if (t.src.domain == memory_domain::disk || t.dst.domain == memory_domain::disk || t.src.bytes != t.dst.bytes) {
                return start_result::refused; // disk moves belong to a reader backend; a size mismatch is never truncated
            }
            std::byte * src = resolve_(t.src);
            std::byte * dst = resolve_(t.dst);
            if (src == nullptr || dst == nullptr) {
                return start_result::refused;
            }
            const auto s = claim_slot();
            if (!s) {
                return start_result::would_block; // every in-flight slot is busy
            }
            const uint64_t n_chunks = std::max<uint64_t>(1, (t.src.bytes + chunk_ - 1) / chunk_);
            slots_[*s].ticket = t.ticket;
            slots_[*s].remaining.store((uint32_t) n_chunks, std::memory_order_release);
            progress_ = progress{ *s, src, dst, t.src.bytes, n_chunks, 0 };
        }
        // push the remaining chunks, round robin across lanes, as far as the lanes have room
        while (progress_->next < progress_->n_chunks) {
            const uint64_t off = progress_->next * chunk_;
            const chunk_frame f{ progress_->slot, progress_->src + off, progress_->dst + off,
                                 std::min<uint64_t>(chunk_, progress_->bytes - off) };
            if (!lanes_[next_lane_ % lanes_.size()]->work.push(f)) {
                return start_result::would_block; // that lane is full; resume here next pump
            }
            next_lane_++;
            progress_->next++;
        }
        progress_.reset();
        return start_result::started;
    }

    // A free in-flight slot, or none (never waits); the lane that lands a transfer's last chunk frees it.
    std::optional<uint32_t> claim_slot() {
        for (size_t i = 0; i < slots_.size(); ++i) {
            const uint32_t s = (uint32_t) ((cursor_ + i) % slots_.size());
            bool free = false;
            if (slots_[s].in_use.compare_exchange_strong(free, true, std::memory_order_acq_rel)) {
                cursor_ = s + 1;
                return s;
            }
        }
        return std::nullopt;
    }

    void run_lane(size_t index, std::stop_token) {
        auto & l = *lanes_[index];
        chunk_frame batch[32];
        while (true) {
            const size_t n = l.work.drain(batch);
            for (size_t i = 0; i < n; ++i) {
                std::memcpy(batch[i].dst, batch[i].src, batch[i].bytes);
                slot & s = slots_[batch[i].slot];
                if (s.remaining.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                    const uint64_t ticket = s.ticket;
                    s.in_use.store(false, std::memory_order_release);
                    complete(index, ticket, transfer_status::moved);
                }
            }
            if (n == 0) {
                if (l.work.closed() && l.work.pushed() == l.work.taken()) {
                    return;
                }
                l.work.wait_for_frames();
            }
        }
    }

    resolver                            resolve_;
    size_t                              chunk_;
    std::vector<slot>                   slots_;
    std::vector<std::unique_ptr<lane>>  lanes_;
    struct progress {
        uint32_t          slot;
        const std::byte * src;
        std::byte *       dst;
        uint64_t          bytes;
        uint64_t          n_chunks;
        uint64_t          next;
    };
    std::optional<progress>             progress_;      // pumping thread only: a transfer part-handed out
    size_t                              next_lane_ = 0; // pumping thread only
    size_t                              cursor_    = 0; // pumping thread only
};
