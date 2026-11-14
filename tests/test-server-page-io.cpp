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
// until all N are running at once, so if the pool ran them one at a time none could finish: with a
// lane per job the wall is one job's time and the sum of their times is N of them.
static void saves_of_different_files_overlap() {
    const int n = 4;
    server_page_io_pool pool(n);
    gate all_running;
    std::atomic<int> running{0}, peak{0};
    std::vector<double> took_ms(n, 0.0);
    std::mutex done_m;
    std::condition_variable done_cv;
    int done = 0;
    const auto t0 = clock_type::now();
    for (int i = 0; i < n; ++i) {
        pool.submit("slot" + std::to_string(i) + ".bin", [&, i] {
            const auto start = clock_type::now();
            const int now = ++running;
            for (int p = peak.load(); now > p && !peak.compare_exchange_weak(p, now);) {}
            all_running.wait();
            --running;
            took_ms[i] = std::chrono::duration<double, std::milli>(clock_type::now() - start).count();
            { std::lock_guard<std::mutex> lock(done_m); ++done; }
            done_cv.notify_all();
        });
    }
    all_running.wait_for_waiting(n); // all four are inside their IO at once
    all_running.release();
    {
        std::unique_lock<std::mutex> lock(done_m);
        require(done_cv.wait_for(lock, deadline, [&] { return done == n; }), "the saves never completed");
    }
    const double wall = std::chrono::duration<double, std::milli>(clock_type::now() - t0).count();
    double sum = 0;
    for (double ms : took_ms) {
        sum += ms;
    }
    require(peak.load() == n, "every save's IO ran at once");
    require(wall < sum, "concurrent saves take less wall time than the sum of their times");
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

int main() {
    saves_of_different_files_overlap();
    one_file_has_one_writer_or_reader_at_a_time();
    decode_progresses_while_a_save_is_in_flight();
    a_page_file_round_trips_and_refuses_defects();
    stopping_runs_every_submitted_switch();
    std::cout << "server page io: ok\n";
    return 0;
}
