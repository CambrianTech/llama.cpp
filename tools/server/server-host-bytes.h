#pragma once

// Continuum fork: a resident's bytes as a HANDLE the host pool owns (CONTINUUM-ENGINE 4.7.6, 4.7.7).
//
// A checkpoint or cache entry is an immutable, reference-counted buffer whose last handle releases it
// to the pool. Every list that holds residents (a slot's checkpoints, a cache entry's copy of them)
// holds handles, never vectors, so:
//   - copying a checkpoint into the cache costs a reference, not a second copy of its bytes;
//   - every erase, clear and pop at every call site releases correctly by construction (RAII), with
//     no release call to forget;
//   - a release may happen on any thread: the pool is reached through a thread-safe owner.
//
// The bytes are reserved before they are written (host_pool::reserve) and committed once complete;
// a reservation that is dropped without a commit rolls back. Pure and header-only, proven by
// tests/test-server-host-bytes.cpp without a model.

#include "server-host-pool.h"

#include <cstdint>
#include <expected>
#include <memory>
#include <mutex>
#include <span>
#include <utility>
#include <vector>

// The pool, reachable from any thread. Every operation is one short critical section; the owner frees
// the residents it names outside the lock.
class host_pool_shared {
public:
    host_pool_shared(host_pool_instance instance, size_t launch_budget) : pool_(instance, launch_budget) {}

    template <typename F>
    auto with(F && f) {
        std::scoped_lock lock(mutex_);
        return std::forward<F>(f)(pool_);
    }

private:
    std::mutex mutex_;
    host_pool  pool_;
};

// Committed bytes of one resident. Immutable once built; released to the pool when the last handle
// goes. Never copied: shared by handle.
class host_bytes {
public:
    host_bytes(const host_bytes &)             = delete;
    host_bytes & operator=(const host_bytes &) = delete;

    ~host_bytes() {
        if (auto pool = pool_.lock()) {
            pool->with([&](host_pool & p) { p.release(id_); });
        }
    }

    std::span<const uint8_t> bytes() const { return data_; }
    size_t                   size()  const { return data_.size(); }
    uint64_t                 id()    const { return id_; }
    host_pool_kind           kind()  const { return kind_; }

private:
    friend class host_bytes_writer;
    host_bytes(std::weak_ptr<host_pool_shared> pool, uint64_t id, host_pool_kind kind, std::vector<uint8_t> data)
        : pool_(std::move(pool)), id_(id), kind_(kind), data_(std::move(data)) {}

    std::weak_ptr<host_pool_shared> pool_; // weak: a resident never keeps the pool alive
    uint64_t                        id_;
    host_pool_kind                  kind_;
    std::vector<uint8_t>            data_;
};

using host_bytes_handle = std::shared_ptr<const host_bytes>;

// Why a write did not produce bytes.
enum class host_bytes_refusal : uint8_t {
    oversized,            // larger than the whole grant
    pinned_pressure,      // fits the grant, but pinned and reserved bytes leave no room
    write_failed,         // the copy into the reservation failed
    larger_than_reserved, // the writer produced more than it reserved
    pool_gone,            // the engine is shutting down
};

// A reservation being filled. The writer reserves its whole peak up front, writes into the buffer,
// then finish() commits exactly what it wrote; dropping it unfinished rolls the reservation back.
class host_bytes_writer {
public:
    // Reserve `peak` bytes of `kind`. `evict` receives the residents the owner must free now (even when
    // the reservation is refused), so the pool's room is real before any byte is written.
    static std::expected<host_bytes_writer, host_bytes_refusal> reserve(
            const std::shared_ptr<host_pool_shared> & pool, host_pool_kind kind, size_t peak, std::vector<uint64_t> & evict) {
        auto result = pool->with([&](host_pool & p) { return p.reserve(kind, peak); });
        evict.insert(evict.end(), result.evict.begin(), result.evict.end());
        if (!result.admitted) {
            return std::unexpected(result.admitted.error() == host_pool_refusal::oversized
                                   ? host_bytes_refusal::oversized : host_bytes_refusal::pinned_pressure);
        }
        return host_bytes_writer(pool, *result.admitted);
    }

    host_bytes_writer(host_bytes_writer && o) noexcept
        : pool_(std::move(o.pool_)), reservation_(o.reservation_), data_(std::move(o.data_)), live_(std::exchange(o.live_, false)) {}
    host_bytes_writer & operator=(host_bytes_writer &&) = delete;
    host_bytes_writer(const host_bytes_writer &)          = delete;

    ~host_bytes_writer() { abandon(); }

    // The buffer to write into: sized to the reservation; finish() keeps only `written` bytes.
    std::span<uint8_t> buffer() {
        if (data_.size() != reservation_.bytes) {
            data_.resize(reservation_.bytes);
        }
        return data_;
    }

    // Commit the first `written` bytes as an immutable resident at `value`.
    std::expected<host_bytes_handle, host_bytes_refusal> finish(size_t written, host_pool_value value) && {
        auto pool = pool_.lock();
        if (!pool) {
            live_ = false;
            return std::unexpected(host_bytes_refusal::pool_gone);
        }
        auto committed = pool->with([&](host_pool & p) { return p.commit(reservation_, written, value); });
        if (!committed) {
            abandon();
            return std::unexpected(host_bytes_refusal::larger_than_reserved);
        }
        live_ = false;
        data_.resize(written);
        data_.shrink_to_fit();
        return host_bytes_handle(new host_bytes(pool_, reservation_.id, reservation_.kind, std::move(data_)));
    }

    // The copy failed: return the reservation.
    void fail() && { abandon(); }

private:
    host_bytes_writer(std::weak_ptr<host_pool_shared> pool, host_pool_reservation r) : pool_(std::move(pool)), reservation_(r) {}

    void abandon() {
        if (!live_) {
            return;
        }
        live_ = false;
        if (auto pool = pool_.lock()) {
            pool->with([&](host_pool & p) { p.rollback(reservation_); });
        }
    }

    std::weak_ptr<host_pool_shared> pool_;
    host_pool_reservation           reservation_;
    std::vector<uint8_t>            data_;
    bool                            live_ = true;
};
