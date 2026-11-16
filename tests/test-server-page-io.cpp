#include "../tools/server/server-page-io.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <iostream>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

// Model-free validator of KV page IO off the task loop (server-page-io.h). No sleeps: every wait is
// on the event that proves the claim, with a deadline that only turns a hang into a failure.

static void require(bool condition, const char * message) {
    if (!condition) {
        std::cerr << message << '\n';
        std::exit(1);
    }
}

using clock_type = std::chrono::steady_clock;
static constexpr auto deadline = std::chrono::seconds(60); // a hang fails, it never passes

// A latch the test opens; jobs wait on it.
struct gate {
    std::mutex              m;
    std::condition_variable cv;
    bool                    open = false;
    int                     waiting = 0;

    void wait() {
        std::unique_lock<std::mutex> lock(m);
        ++waiting;
        cv.notify_all();
        require(cv.wait_for(lock, deadline, [&] { return open; }), "a held job was never released");
    }
    void wait_for_waiting(int n) {
        std::unique_lock<std::mutex> lock(m);
        require(cv.wait_for(lock, deadline, [&] { return waiting >= n; }), "jobs never reached the gate");
    }
    void release() {
        { std::lock_guard<std::mutex> lock(m); open = true; }
        cv.notify_all();
    }
};

// what this catches: page switches on different files serialized behind each other (the pinned
// fork: one task loop, so four residents' saves cost the ~13.5 s sum, dyno 11b). Each job waits
// until all N are running at once, so a pool that ran them one at a time could never finish one:
// every save's interval contains the instant all N were in flight, so the wall time is one save's
// span and not the sum of N spans. Proven by the ordering of events, not by comparing durations
// (a clock's resolution and thread start-up made `wall < sum` flaky on Windows runners).
static void saves_of_different_files_overlap() {
    const int n = 4;
    server_page_io_pool pool(n);
    gate all_running;
    std::atomic<int> running{0}, peak{0};
    std::vector<clock_type::time_point> started(n), ended(n);
    std::mutex done_m;
    std::condition_variable done_cv;
    int done = 0;
    for (int i = 0; i < n; ++i) {
        pool.submit("slot" + std::to_string(i) + ".bin", [&, i] {
            started[i] = clock_type::now();
            const int now = ++running;
            for (int p = peak.load(); now > p && !peak.compare_exchange_weak(p, now);) {}
            all_running.wait();
            --running;
            ended[i] = clock_type::now();
            { std::lock_guard<std::mutex> lock(done_m); ++done; }
            done_cv.notify_all();
        });
    }
    all_running.wait_for_waiting(n); // all four are inside their IO at once
    const auto all_in_flight = clock_type::now();
    all_running.release();
    {
        std::unique_lock<std::mutex> lock(done_m);
        require(done_cv.wait_for(lock, deadline, [&] { return done == n; }), "the saves never completed");
    }
    require(peak.load() == n, "every save's IO ran at once");
    for (int i = 0; i < n; ++i) {
        require(started[i] <= all_in_flight && all_in_flight <= ended[i],
                "every save was in flight at one instant: the wall is one span, not the sum of them");
    }
}

// what this catches: a restore reading a page file a save is still writing. Jobs naming one file run
// one at a time, in the order submitted; a job on another file is not held behind them.
static void one_file_has_one_writer_or_reader_at_a_time() {
    server_page_io_pool pool(4);
    gate first_held;
    std::mutex m;
    std::condition_variable cv;
    std::vector<int> order;
    int in_flight = 0, max_in_flight = 0;
    bool other_done = false;
    auto on_page = [&](int id, bool hold) {
        return [&, id, hold] {
            {
                std::lock_guard<std::mutex> lock(m);
                max_in_flight = std::max(max_in_flight, ++in_flight);
                order.push_back(id);
            }
            if (hold) {
                first_held.wait();
            }
            { std::lock_guard<std::mutex> lock(m); --in_flight; }
            cv.notify_all();
        };
    };
    pool.submit("page.bin", on_page(1, true));
    first_held.wait_for_waiting(1);
    pool.submit("page.bin", on_page(2, false));
    pool.submit("page.bin", on_page(3, false));
    pool.submit("other.bin", [&] {
        { std::lock_guard<std::mutex> lock(m); other_done = true; }
        cv.notify_all();
    });
    {
        std::unique_lock<std::mutex> lock(m);
        require(cv.wait_for(lock, deadline, [&] { return other_done; }), "another file's IO waited behind a held one");
        require(order.size() == 1, "nothing behind a held file ran before it finished");
    }
    first_held.release();
    {
        std::unique_lock<std::mutex> lock(m);
        require(cv.wait_for(lock, deadline, [&] { return order.size() == 3 && in_flight == 0; }), "the parked switches never ran");
    }
    require(max_in_flight == 1, "one file never has two switches in flight");
    require(order == std::vector<int>({ 1, 2, 3 }), "one file's switches run in the order submitted");
}

// what this catches: decode stalled while a page's file is written. A loop shaped like the server's
// (it takes events off its queue and decodes between them) keeps decoding while a save's IO is held
// on a worker, and the save's completion reaches it as an event, never polled.
static void decode_progresses_while_a_save_is_in_flight() {
    std::mutex m;
    std::condition_variable cv;
    std::deque<std::string> events;
    gate disk;
    server_page_io_pool pool(1);
    pool.submit("slot0.bin", [&] {
        disk.wait(); // the file write, held until the test has seen decode go on
        { std::lock_guard<std::mutex> lock(m); events.push_back("save done"); }
        cv.notify_all();
    });
    disk.wait_for_waiting(1);
    int decoded = 0;
    bool save_done = false;
    const int steps_while_held = 32;
    while (!save_done) {
        // a decode step for slot 1
        ++decoded;
        if (decoded == steps_while_held) {
            {
                std::lock_guard<std::mutex> held(disk.m);
                require(disk.waiting == 1 && !disk.open, "the save's IO is still in flight");
            }
            disk.release();
        }
        std::unique_lock<std::mutex> lock(m);
        if (decoded >= steps_while_held) {
            require(cv.wait_for(lock, deadline, [&] { return !events.empty(); }), "the save's completion never arrived");
        }
        while (!events.empty()) {
            save_done = save_done || events.front() == "save done";
            events.pop_front();
        }
    }
    require(decoded >= steps_while_held, "slot 1 decoded while slot 0's page was written");
}

// what this catches: a page file the library cannot read, or a defect read as a page. The framing is
// the library's (magic, version, token count, tokens, then the state body), so old pages restore.
static void a_page_file_round_trips_and_refuses_defects() {
    const auto dir = std::filesystem::temp_directory_path() / ("page-io-" + std::to_string(std::rand()));
    std::filesystem::create_directories(dir);
    const std::string path = (dir / "page.bin").string();
    const server_page_file file { 0x67677371u, 3 };
    const std::vector<int32_t> tokens = { -1, 1, 84, 7, 8 };
    std::vector<uint8_t> body(1 << 20);
    for (size_t i = 0; i < body.size(); ++i) {
        body[i] = (uint8_t) (i * 31);
    }
    std::string why;
    const size_t written = file.write(path, tokens.data(), tokens.size(), body, why);
    require(written == 12 + tokens.size() * 4 + body.size() && why.empty(), "the page is written whole");
    require(std::filesystem::file_size(path) == written, "the receipt's bytes are the file's");

    uint64_t told = 0, total = 0;
    auto page = file.read(path, [&](uint64_t done, uint64_t t) { told = done; total = t; return true; }, why);
    require(page && page->tokens == tokens && page->body == body && page->n_bytes == written, "the page reads back as written");
    require(told == written && total == written, "progress ends at the file's size");

    int calls = 0;
    require(!file.read(path, [&](uint64_t, uint64_t) { return ++calls < 3; }, why) && why == "state file load cancelled",
            "a cancel ends the read");
    require(!server_page_file{ 0x67677371u, 2 }.read(path, nullptr, why), "another version is refused");
    require(!file.read((dir / "missing.bin").string(), nullptr, why), "a missing page is refused");

    // a token count past the end of the file
    const std::vector<int32_t> none;
    file.write(path, none.data(), 0, {}, why);
    {
        FILE * f = std::fopen(path.c_str(), "r+b");
        const uint32_t lie = 1000;
        std::fseek(f, 8, SEEK_SET);
        std::fwrite(&lie, sizeof(lie), 1, f);
        std::fclose(f);
    }
    require(!file.read(path, nullptr, why), "a token count larger than the file is refused");
    std::filesystem::remove_all(dir);
}

// what this catches: a save the core already sent dropped at shutdown.
static void stopping_runs_every_submitted_switch() {
    std::atomic<int> ran{0};
    {
        server_page_io_pool pool(2);
        for (int i = 0; i < 16; ++i) {
            pool.submit(i % 2 ? "a.bin" : "b.bin", [&] { ++ran; });
        }
    }
    require(ran.load() == 16, "every submitted switch ran before the pool stopped");
}

// what this catches (Fable on #118: the M5 at 0.1 GB free): in-flight page bytes unbounded against
// host memory. Switches are admitted against one budget before they allocate; one that does not fit
// waits, in order, and the release that makes room admits it in that release's own call (an event).
static void page_bytes_are_admitted_against_one_budget() {
    server_page_admission budget;
    std::vector<int> admitted;
    auto later = [&](int id) { return [&admitted, id] { admitted.push_back(id); }; };

    // unset (0): one switch at a time, however large; nothing ever starves
    require(budget.admit_or_park(1600, later(1)), "with nothing in flight a switch is admitted, whatever its size");
    require(!budget.admit_or_park(10, later(2)), "with no budget a second switch waits");
    budget.release(1600);
    require(admitted == std::vector<int>({ 2 }), "the release admits the waiter at once");
    budget.release(10);

    // the core's budget: switches that fit run together
    budget.set_budget(1000);
    require(budget.admit_or_park(600, later(3)) && budget.admit_or_park(400, later(4)), "switches that fit are admitted together");
    require(!budget.admit_or_park(300, later(5)), "a switch past the budget waits");
    require(!budget.admit_or_park(1, later(6)), "and one behind it waits its turn, never jumps the queue");
    require(budget.snapshot().in_flight == 1000 && budget.snapshot().waiting == 2, "the budget counts what is held and waiting");
    budget.release(400);
    require(admitted == std::vector<int>({ 2, 5, 6 }), "one release admits every waiter that now fits, in order");
    budget.release(600 + 300 + 1);

    // a switch larger than the whole budget runs alone once the others are done
    require(budget.admit_or_park(1, later(7)), "a small switch is admitted");
    require(!budget.admit_or_park(5000, later(8)), "a switch larger than the budget waits for an empty pipe");
    budget.release(1);
    require(admitted.back() == 8 && budget.snapshot().in_flight == 5000, "then runs alone");
    budget.release(5000);

    // raising the budget admits at once; a re-measured reservation counts its new size
    require(budget.admit_or_park(1000, later(9)), "a full budget");
    require(!budget.admit_or_park(500, later(10)), "waits");
    budget.set_budget(1500);
    require(admitted.back() == 10, "raising the budget admits the waiter in the raise");
    budget.resize(1000, 1100);
    require(budget.snapshot().in_flight == 1600, "a reservation re-measured when its switch runs counts its new size");
    budget.release(1100);
    budget.release(500);
    require(budget.snapshot().in_flight == 0 && budget.snapshot().waiting == 0, "everything returned");
}

// what this catches (Fable on #118): a restore buffered for a slot that is then given a different
// page (the mind left) holding its bytes until that slot's new turn ends. Dropping it returns its
// bytes at once, admitting the switch waiting behind it; the later release of the same reservation
// (its completion, its destructor) returns nothing twice.
static void a_dropped_restore_returns_its_bytes_at_once() {
    server_page_admission budget;
    budget.set_budget(1000);
    bool next_admitted = false;
    {
        server_page_reservation restore;
        require(budget.admit_or_park(900, [] {}), "the restore is admitted");
        restore.hold(&budget, 900);
        require(!budget.admit_or_park(500, [&] { next_admitted = true; }), "the next switch waits behind the buffered restore");
        restore.release(); // its slot was given another task: dropped
        require(next_admitted, "the drop admits the waiter at once, not when the slot frees");
        require(budget.snapshot().in_flight == 500, "only the admitted switch is held");
        restore.release(); // the completion arrives later and releases again
    } // and the destructor
    require(budget.snapshot().in_flight == 500, "a reservation is returned once");
    budget.release(500);
}

int main() {
    saves_of_different_files_overlap();
    one_file_has_one_writer_or_reader_at_a_time();
    decode_progresses_while_a_save_is_in_flight();
    a_page_file_round_trips_and_refuses_defects();
    stopping_runs_every_submitted_switch();
    page_bytes_are_admitted_against_one_budget();
    a_dropped_restore_returns_its_bytes_at_once();
    std::cout << "server page io: ok\n";
    return 0;
}
