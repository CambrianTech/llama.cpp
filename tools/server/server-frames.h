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
#include <chrono>
#include <cstdlib>
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
    // A frame is accepted iff its publication (a CAS on the tail word) lands before a close: a close can
    // never be erased by a racing push, and a push that returned true is always drained (Codex on #134).
    bool push(const T & frame) {
        uint64_t raw = tail_.load(std::memory_order_relaxed);
        if (raw & CLOSED) {
            return false; // a closed stream takes nothing more
        }
        const uint64_t tail = raw;
        if (tail - head_cache_ >= slots_.size()) {
            head_cache_ = head_.load(std::memory_order_acquire) & COUNT;
            if (tail - head_cache_ >= slots_.size()) {
                return false;
            }
        }
        slots_[tail & mask_] = frame;
        if (!tail_.compare_exchange_strong(raw, tail + 1, std::memory_order_release, std::memory_order_relaxed)) {
            return false; // closed between the check and the publication: not accepted
        }
        tail_.notify_one();
        return true;
    }

    // Consumer side: take up to out.size() frames at once; returns how many.
    size_t drain(std::span<T> out) {
        const uint64_t head = head_.load(std::memory_order_relaxed) & COUNT;
        uint64_t tail = tail_cache_;
        if (tail == head) {
            tail = tail_cache_ = tail_.load(std::memory_order_acquire) & COUNT;
        }
        const size_t n = (size_t) std::min<uint64_t>(tail - head, out.size());
        for (size_t i = 0; i < n; ++i) {
            out[i] = slots_[(head + i) & mask_];
        }
        if (n > 0) {
            head_.fetch_add(n, std::memory_order_release); // an RMW: a racing close's bit is never erased
            head_.notify_one(); // a producer waiting for room wakes
        }
        return n;
    }

    // Producer side: sleep until there is room for one frame or the stream is closed (an atomic wait on the
    // consumer's head word, which carries the closed bit too). False: closed, nothing more will be taken.
    bool wait_for_room() const {
        const uint64_t tail = tail_.load(std::memory_order_relaxed) & COUNT;
        uint64_t raw = head_.load(std::memory_order_acquire);
        while (!(raw & CLOSED) && tail - (raw & COUNT) >= slots_.size()) {
            head_.wait(raw, std::memory_order_acquire);
            raw = head_.load(std::memory_order_acquire);
        }
        return !(raw & CLOSED);
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

    // Either side ends the stream; both words carry the bit, so a waiting consumer AND a producer waiting
    // for room wake (the word each sleeps on changes). Frames accepted before it are still drained.
    void close() {
        tail_.fetch_or(CLOSED, std::memory_order_acq_rel);
        head_.fetch_or(CLOSED, std::memory_order_acq_rel);
        tail_.notify_all();
        head_.notify_all();
    }
    bool closed() const { return (tail_.load(std::memory_order_acquire) & CLOSED) != 0; }

    size_t   capacity() const { return slots_.size(); }
    uint64_t pushed()   const { return tail_.load(std::memory_order_acquire) & COUNT; }
    uint64_t taken()    const { return head_.load(std::memory_order_acquire) & COUNT; }

private:
    static constexpr uint64_t CLOSED = uint64_t(1) << 63;
    static constexpr uint64_t COUNT  = CLOSED - 1;

    std::vector<T>                     slots_;
    const uint64_t                     mask_;
    alignas(64) std::atomic<uint64_t>  head_{0};       // consumer advances (fetch_add): count | CLOSED
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

// Custody (Codex on #134): the pool that owns `buffer` keeps it live and unrecycled from the push of a
// transfer naming it until that transfer's completion frame is DRAINED; `generation` is the buffer's
// incarnation in its pool, bumped on every reuse, so a reference that outlived its buffer is refused by
// the resolver, never resolved to the next occupant's bytes.
struct buffer_ref {
    memory_domain domain     = memory_domain::host;
    uint32_t      pool       = 0; // the owning pool (the host pool, a device pool, a page store)
    uint32_t      buffer     = 0; // the buffer within that pool
    uint32_t      generation = 0; // the buffer's incarnation (see custody above)
    uint64_t      offset     = 0;
    uint64_t      bytes      = 0;
};
static_assert(std::is_trivially_copyable_v<buffer_ref>);

// The owning pool's custody of a resolved reference (Codex on #134: a raw pointer plus a generation does
// not keep the bytes live). A pool resolves a buffer_ref into a pin: the bytes' address, held live and
// unrecycled until the pin is released (its destructor calls the pool's own unpin with the pool's token).
// A mover holds a transfer's pins until its completion is drained. Nothing here owns memory: the pool does.
class buffer_pin {
public:
    using unpin_fn = void (*)(void * pool, uint64_t token) noexcept;

    buffer_pin() = default;
    buffer_pin(std::byte * data, uint64_t bytes, void * pool, unpin_fn unpin, uint64_t token)
        : data_(data), bytes_(bytes), pool_(pool), unpin_(unpin), token_(token) {}
    buffer_pin(buffer_pin && o) noexcept { take(o); }
    buffer_pin & operator=(buffer_pin && o) noexcept {
        if (this != &o) {
            release();
            take(o);
        }
        return *this;
    }
    buffer_pin(const buffer_pin &)             = delete;
    buffer_pin & operator=(const buffer_pin &) = delete;
    ~buffer_pin() { release(); }

    explicit operator bool() const { return data_ != nullptr; }
    std::byte * data() const { return data_; }
    uint64_t    bytes() const { return bytes_; }

    void release() noexcept {
        if (unpin_ != nullptr) {
            unpin_(pool_, token_);
        }
        data_ = nullptr; bytes_ = 0; pool_ = nullptr; unpin_ = nullptr; token_ = 0;
    }

private:
    void take(buffer_pin & o) noexcept {
        data_ = o.data_; bytes_ = o.bytes_; pool_ = o.pool_; unpin_ = o.unpin_; token_ = o.token_;
        o.data_ = nullptr; o.bytes_ = 0; o.pool_ = nullptr; o.unpin_ = nullptr; o.token_ = 0;
    }

    std::byte * data_  = nullptr;
    uint64_t    bytes_ = 0;
    void *      pool_  = nullptr;
    unpin_fn    unpin_ = nullptr;
    uint64_t    token_ = 0;
};

// ---- moving bytes ----------------------------------------------------------------------------------

struct transfer_frame {
    uint64_t   ticket = 0;
    buffer_ref src;
    buffer_ref dst;
};

enum class transfer_status : uint8_t {
    moved,      // the backend moved the bytes
    aliased,    // same physical domain: dst now names src's bytes, nothing was copied
    refused,    // invalid (stale, out of bounds, unequal lengths, overlapping) or an unsupported pair:
                // typed, never a silent CPU copy
    failed,     // the backend reported an error
    cancelled,  // the mover shut down before the transfer was handed to an engine: nothing was written
};

struct transfer_done_frame {
    uint64_t        ticket = 0;
    transfer_status status = transfer_status::moved;
    uint32_t        slot   = NO_SLOT; // the backend's in-flight slot, retired when this frame is drained
    static constexpr uint32_t NO_SLOT = UINT32_MAX;
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
            if (!unsettled_.empty()) {
                return started; // settlements wait for the caller's drain: take nothing new (bounded)
            }
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
                case start_result::aliased:     settle(pending_->ticket, transfer_status::aliased, take_terminal_slot()); pending_.reset(); break;
                case start_result::would_block: return started; // engines full: drain completions, pump again
            }
        }
    }

    // A transfer is waiting for engine room (the caller should drain completions, then pump again).
    bool blocked() const { return pending_.has_value() || !unsettled_.empty(); }

    // Consumer side, on the pumping thread: take completions from every producer; returns how many. A
    // drained frame retires its backend slot, which is what bounds the completions outstanding.
    size_t drain_done(std::span<transfer_done_frame> out) {
        size_t n = settled_.drain(out);
        while (n < out.size() && !unsettled_.empty()) {
            out[n++] = unsettled_.front();
            unsettled_.erase(unsettled_.begin());
        }
        for (auto & ring : finished_) {
            n += ring->drain(out.subspan(n));
        }
        for (size_t i = 0; i < n; ++i) {
            if (out[i].slot != transfer_done_frame::NO_SLOT) {
                retire(out[i].slot);
            }
        }
        return n;
    }

    // Shutdown, on the pumping thread: every transfer still queued or part-handed that no engine took is
    // settled `cancelled` (nothing written); transfers an engine took complete as they land. After it,
    // every ticket pushed has exactly one terminal frame to drain. Pushing more is refused (closed).
    void shutdown() {
        in_.close();
        if (pending_) {
            if (const auto ended = cancel_pending()) {
                settle(pending_->ticket, *ended, take_terminal_slot());
            } // else an engine holds part of it and reports its one terminal frame
            pending_.reset();
        }
        transfer_frame f;
        while (in_.drain(std::span<transfer_frame>(&f, 1)) == 1) {
            settle(f.ticket, transfer_status::cancelled);
        }
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
    // aliased: the backend found both references resolve to the same physical bytes (nothing to move)
    enum class start_result : uint8_t { started, refused, would_block, aliased };

    // Hand the move to this backend's engines, as much as they have room for, without waiting.
    // would_block: called again with the same frame on the next pump (the backend keeps its own
    // progress); refused: it cannot move between these domains.
    virtual start_result start(const transfer_frame & t) = 0;

    // Whether dst can simply name src's bytes (one physical domain): no copy at all. Only after both
    // references are validated (live generation, in bounds, equal lengths): an alias never vouches for
    // a reference the backend would refuse to copy.
    virtual bool aliases(const buffer_ref & src, const buffer_ref & dst) const = 0;

    // The transfer pending when shutdown came: the backend stops handing it out. Its terminal status when
    // no engine holds any of it (settled here), or nullopt when an engine does (that engine reports it).
    virtual std::optional<transfer_status> cancel_pending() { return transfer_status::cancelled; }

    // A drained completion's slot may be reused.
    virtual void retire(uint32_t /*slot*/) {}

    // Set by a backend, on the pumping thread, when the terminal it hands back (start's `aliased`, or the status
    // cancel_pending returns) still holds a slot with custody: the base settles that frame with the slot, so the
    // pins return only when the frame is drained (Codex on #134).
    uint32_t terminal_slot_ = transfer_done_frame::NO_SLOT;

    // Completion context `context` (one lane) reports a finished move. The backend guarantees room: its
    // completions outstanding never exceed the ring's capacity (a slot is retired only on drain), so a
    // lane never waits on the consumer.
    void complete(size_t context, uint64_t ticket, transfer_status status, uint32_t slot) {
        const bool pushed = finished_[context]->push(transfer_done_frame{ ticket, status, slot });
        if (!pushed) {
            std::abort(); // the slot bound above is the contract; breaking it is a bug, never a wait
        }
        ring();
    }

private:
    // On the pumping thread: into the settled ring, or held for the caller's drain when it is full
    // (never a wait: the same thread drains).
    // `slot`: the backend slot that still holds the transfer's custody (pins), retired when this frame is drained.
    void settle(uint64_t ticket, transfer_status status, uint32_t slot = transfer_done_frame::NO_SLOT) {
        if (!settled_.push(transfer_done_frame{ ticket, status, slot })) {
            backpressured_.fetch_add(1, std::memory_order_relaxed);
            unsettled_.push_back(transfer_done_frame{ ticket, status, slot });
        }
        ring();
    }

    uint32_t take_terminal_slot() {
        const uint32_t s = terminal_slot_;
        terminal_slot_   = transfer_done_frame::NO_SLOT;
        return s;
    }

    void ring() {
        doorbell_.fetch_add(1, std::memory_order_release);
        doorbell_.notify_all();
    }

    frame_ring<transfer_frame> &                                   in_;
    std::optional<transfer_frame>                                  pending_; // pumping thread only
    std::vector<transfer_done_frame>                               unsettled_; // pumping thread only
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
//
// `Resolver`: callable (const buffer_ref &) -> buffer_pin, the owning pool's own resolution: an empty pin
// for a stale generation, an unknown buffer or bytes out of bounds. Aliasing and overlap are decided on the
// resolved PHYSICAL ranges, so two different ids naming the same bytes are caught.
template <typename Resolver>
class parallel_copier final : public memory_mover {
public:
    parallel_copier(frame_ring<transfer_frame> & in, Resolver resolve, size_t lanes, size_t chunk_bytes, size_t in_flight)
        // every lane's completion ring holds every slot: a slot is retired only when its completion is
        // drained, so a lane can never find its ring full
        : memory_mover(in, std::max<size_t>(in_flight, 1), lanes), resolve_(std::move(resolve)), chunk_(std::max<size_t>(chunk_bytes, 4096)),
          slots_(std::max<size_t>(in_flight, 1)) {
        for (size_t i = 0; i < std::max<size_t>(lanes, 1); ++i) {
            lanes_.push_back(std::make_unique<lane>(in_flight * 4));
        }
        for (size_t i = 0; i < lanes_.size(); ++i) {
            lanes_[i]->worker = std::jthread([this, i](std::stop_token stop) { run_lane(i, stop); });
        }
    }

    // Teardown never hangs: shutdown() cancels what no lane took, the lanes land the chunks they hold
    // (their completion pushes cannot block), then they are joined. Completions not drained go with the
    // mover; the caller drains before destroying it when it needs every terminal frame.
    ~parallel_copier() override {
        shutdown();
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
        std::atomic<bool>     in_use{false};   // claimed at start; free again only when its completion is drained
        std::atomic<uint32_t> remaining{0};
        std::atomic<uint8_t>  status{0};       // moved, or failed when shutdown cut it part-handed
        uint64_t              ticket = 0;
        buffer_pin            src, dst;        // the pools' custody, held until the completion is drained
    };

    // Decided in start() on the resolved physical ranges, never on ids.
    bool aliases(const buffer_ref &, const buffer_ref &) const override { return false; }

    void free_slot(slot & s) {
        s.src.release();
        s.dst.release();
        s.in_use.store(false, std::memory_order_release);
    }

    std::optional<transfer_status> cancel_pending() override {
        if (!progress_) {
            return transfer_status::cancelled; // never claimed a slot
        }
        const auto p = *progress_;
        progress_.reset();
        slot & s = slots_[p.slot];
        if (p.next == 0) {
            terminal_slot_ = p.slot; // no chunk handed, nothing written; the pins go back when this frame is drained
            return transfer_status::cancelled;
        }
        // part-handed: its handed chunks land and the last one reports `failed` (a partial destination);
        // the chunks never handed are subtracted so that last one is found
        s.status.store((uint8_t) transfer_status::failed, std::memory_order_relaxed);
        const uint32_t not_handed = (uint32_t) (p.n_chunks - p.next);
        if (s.remaining.fetch_sub(not_handed, std::memory_order_acq_rel) == not_handed) {
            // every handed chunk already landed before the subtraction: none will report it, so it ends here,
            // still holding its pins until the frame is drained
            terminal_slot_ = p.slot;
            return transfer_status::failed;
        }
        return std::nullopt;
    }

    void retire(uint32_t s) override { free_slot(slots_[s]); }

    start_result start(const transfer_frame & t) override {
        if (!progress_) {
            if (t.src.domain == memory_domain::disk || t.dst.domain == memory_domain::disk || t.src.bytes != t.dst.bytes) {
                return start_result::refused; // disk moves belong to a reader backend; a size mismatch is never truncated
            }
            buffer_pin src = resolve_(t.src);
            buffer_pin dst = resolve_(t.dst);
            if (!src || !dst || src.bytes() < t.src.bytes || dst.bytes() < t.dst.bytes) {
                return start_result::refused; // stale generation, unknown buffer or out of bounds
            }
            // physical ranges, whatever ids named them
            const std::byte * a = src.data();
            const std::byte * b = dst.data();
            const uint64_t    n = t.src.bytes;
            if (a != b && a < b + n && b < a + n) {
                return start_result::refused; // independent parallel chunks over overlapping ranges race (UB)
            }
            const auto s = claim_slot();
            if (!s) {
                return start_result::would_block; // every in-flight slot is busy (the pins go back; re-resolved next pump)
            }
            if (a == b) {
                // the same bytes: nothing to move, but the pins are held in the slot until the frame is drained
                slots_[*s].src = std::move(src);
                slots_[*s].dst = std::move(dst);
                terminal_slot_ = *s;
                return start_result::aliased;
            }
            std::byte * src_bytes = src.data();
            std::byte * dst_bytes = dst.data();
            slots_[*s].src = std::move(src);
            slots_[*s].dst = std::move(dst);
            const uint64_t n_chunks = std::max<uint64_t>(1, (t.src.bytes + chunk_ - 1) / chunk_);
            slots_[*s].ticket = t.ticket;
            slots_[*s].status.store((uint8_t) transfer_status::moved, std::memory_order_relaxed);
            slots_[*s].remaining.store((uint32_t) n_chunks, std::memory_order_release);
            progress_ = progress{ *s, src_bytes, dst_bytes, t.src.bytes, n_chunks, 0 };
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
                    // the slot stays claimed until this completion is drained (retire): that is the bound
                    complete(index, s.ticket, (transfer_status) s.status.load(std::memory_order_relaxed), batch[i].slot);
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

    Resolver                            resolve_;
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

// ---- how many copy lanes: the first guess, then the measurement ------------------------------------

// The first guess for CPU copy lanes is the node's EFFICIENCY core count: copying is bandwidth-bound,
// the efficiency cores reach the memory ceiling (measured on the M5 Pro: 12 lanes = 118.9 GB/s, the
// knee; 18 = 109) and the performance cores stay free for the decode thread. Read from the OS; empty
// when the OS does not report core classes (then the probe below starts from the core count it can
// see). measure_copy_lanes() refines it on the node's own bandwidth curve.
#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif
inline std::optional<size_t> efficiency_cores() {
#if defined(__APPLE__)
    int n = 0;
    size_t len = sizeof(n);
    if (sysctlbyname("hw.perflevel1.physicalcpu", &n, &len, nullptr, 0) == 0 && n > 0) {
        return (size_t) n;
    }
#endif
    return std::nullopt; // Linux/Windows core classes: the host adapter's job (continuum-host), not guessed here
}

// The node's copy-lane count, MEASURED: copy `probe_bytes` with each candidate lane count (best of three,
// the probe shares the machine) and keep the FIRST GUESS (the efficiency-core count) unless another count
// beats it by more than the margin `1/knee`: noise never moves the choice, a real knee elsewhere does.
// Runs once at startup, off the decode thread. Returns the chosen count and its GB/s.
struct copy_lane_measurement {
    size_t lanes = 1;
    double gbps  = 0.0;
};
inline copy_lane_measurement measure_copy_lanes(size_t guess, std::span<const size_t> others, size_t probe_bytes, double knee) {
    std::vector<std::byte> src(probe_bytes, std::byte{1}), dst(probe_bytes);
    auto measure = [&](size_t lanes) {
        lanes = std::max<size_t>(lanes, 1);
        double best = 0.0;
        for (int rep = 0; rep < 3; ++rep) {
            const auto t0 = std::chrono::steady_clock::now();
            {
                std::vector<std::jthread> ws;
                const size_t chunk = probe_bytes / lanes;
                for (size_t i = 0; i < lanes; ++i) {
                    const size_t off = i * chunk, n = i + 1 == lanes ? probe_bytes - off : chunk;
                    ws.emplace_back([&, off, n] { std::memcpy(dst.data() + off, src.data() + off, n); });
                }
            } // joins
            const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            best = std::max(best, (double) probe_bytes / secs / 1e9);
        }
        return copy_lane_measurement{ lanes, best };
    };
    copy_lane_measurement pick = measure(guess);
    for (size_t lanes : others) {
        const auto m = measure(lanes);
        if (m.gbps * knee > pick.gbps) {
            pick = m; // clearly better than the guess, not noise
        }
    }
    return pick;
}
