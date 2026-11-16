// what this catches: a resident's bytes as a pool-owned handle (server-host-bytes.h) without a model:
// the last handle releases its bytes to the pool from any thread, sharing a resident costs a reference
// and never its bytes twice, a writer reserves its peak before writing and rolls back if dropped or
// failed, a commit larger than its reservation is refused, and a resident outliving the pool is safe.
#include "../tools/server/server-host-bytes.h"

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <list>
#include <thread>

static void require(bool condition, const char * message) {
    if (!condition) {
        std::cerr << "FAILED: " << message << '\n';
        std::exit(1);
    }
}

static const host_pool_instance ENGINE{ 7, 1794830000 };

static size_t committed(const std::shared_ptr<host_pool_shared> & pool) {
    return pool->with([](host_pool & p) { return p.committed_bytes(); });
}
static size_t reserved(const std::shared_ptr<host_pool_shared> & pool) {
    return pool->with([](host_pool & p) { return p.reserved_bytes(); });
}

static host_bytes_handle write(const std::shared_ptr<host_pool_shared> & pool, host_pool_kind kind, size_t n, uint8_t fill) {
    std::vector<uint64_t> evict;
    auto writer = host_bytes_writer::reserve(pool, kind, n, evict);
    require(writer.has_value(), "the reservation is admitted");
    auto buf = writer->buffer();
    std::memset(buf.data(), fill, buf.size());
    auto handle = std::move(*writer).finish(n, host_pool_value::unmeasured());
    require(handle.has_value(), "the bytes commit");
    return *handle;
}

static void the_last_handle_releases_and_sharing_never_copies() {
    auto pool = std::make_shared<host_pool_shared>(ENGINE, 1000);
    std::list<host_bytes_handle> slot;     // a slot's checkpoints
    std::list<host_bytes_handle> cached;   // a cache entry's copy of them
    slot.push_back(write(pool, host_pool_kind::checkpoint, 400, 0xAB));
    require(committed(pool) == 400, "one resident, its bytes charged once");
    cached = slot;                         // the cache takes the slot's checkpoints
    require(committed(pool) == 400 && cached.front().get() == slot.front().get(), "sharing is a reference: no second copy, no second charge");
    require(cached.front()->bytes()[0] == 0xAB, "the cache reads the same bytes");
    slot.clear();                          // the slot moves on
    require(committed(pool) == 400, "still held while the cache holds it");
    cached.pop_front();                    // the cache evicts it
    require(committed(pool) == 0, "the last handle released it to the pool");
}

static void a_writer_reserves_first_and_rolls_back_unless_finished() {
    auto pool = std::make_shared<host_pool_shared>(ENGINE, 1000);
    {
        std::vector<uint64_t> evict;
        auto writer = host_bytes_writer::reserve(pool, host_pool_kind::cache, 700, evict);
        require(writer.has_value(), "admitted");
        require(reserved(pool) == 700 && committed(pool) == 0, "the peak is reserved before any byte is written");
        std::vector<uint64_t> evict2;
        auto second = host_bytes_writer::reserve(pool, host_pool_kind::cache, 400, evict2);
        require(!second && second.error() == host_bytes_refusal::pinned_pressure, "a reserved peak blocks an overflow: typed refusal");
    } // dropped unfinished
    require(reserved(pool) == 0 && committed(pool) == 0, "a dropped writer rolls its reservation back");
    {
        std::vector<uint64_t> evict;
        auto writer = host_bytes_writer::reserve(pool, host_pool_kind::cache, 300, evict);
        std::move(*writer).fail();
    }
    require(reserved(pool) == 0, "a failed copy returns its reservation");
    std::vector<uint64_t> evict;
    auto writer = host_bytes_writer::reserve(pool, host_pool_kind::page, 200, evict);
    writer->buffer();
    auto over = std::move(*writer).finish(201, host_pool_value::unmeasured());
    require(!over && over.error() == host_bytes_refusal::larger_than_reserved, "more than reserved is refused");
    require(reserved(pool) == 0 && committed(pool) == 0, "...and its reservation returns");
}

static void releases_from_another_thread_are_safe() {
    auto pool = std::make_shared<host_pool_shared>(ENGINE, 1 << 20);
    std::vector<host_bytes_handle> handles;
    for (int i = 0; i < 200; ++i) {
        handles.push_back(write(pool, host_pool_kind::checkpoint, 1000, (uint8_t) i));
    }
    require(committed(pool) == 200 * 1000, "200 residents charged");
    std::thread t([moved = std::move(handles)]() mutable { moved.clear(); });
    t.join();
    require(committed(pool) == 0, "handles dropped on another thread release every byte");
}

static void a_resident_may_outlive_the_pool() {
    host_bytes_handle survivor;
    {
        auto pool = std::make_shared<host_pool_shared>(ENGINE, 1000);
        survivor = write(pool, host_pool_kind::cache, 100, 1);
    }
    require(survivor->size() == 100, "the bytes stay readable after the pool is gone");
    survivor.reset(); // releasing into a gone pool is a no-op, never a crash
}

int main() {
    the_last_handle_releases_and_sharing_never_copies();
    a_writer_reserves_first_and_rolls_back_unless_finished();
    releases_from_another_thread_are_safe();
    a_resident_may_outlive_the_pool();
    std::cout << "test-server-host-bytes: ok\n";
    return 0;
}
