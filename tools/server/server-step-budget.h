#pragma once

// Continuum fork: the decode loop's HOST timings, measured on every step (CONTINUUM-ENGINE 4.7.9).
//
// These are what the host thread does, never what the GPU does: llama_decode may enqueue command
// buffers before it returns, and a sync wait is the host blocked, not device execution. GPU busy time
// comes from the backend's own command-buffer timestamps (/props.gpu_time, `ggml_backend_gpu_time`);
// utilization is that busy time over wall time, and these timings say where the host spends a step.
// Three timestamps per step, all on the decode thread:
//
//   begin      llama_decode is called
//   submitted  llama_decode returns
//   synced     llama_synchronize returns
//
//   decode_call_us = submitted - begin     host building and encoding the step
//   sync_wait_us   = synced - submitted    host blocked until the outputs are readable
//   inter_step_us  = next begin - synced   host work between steps while work is pending
//                                          (sampling, slot bookkeeping, checkpoints, the next batch)
//
// A step without outputs is not synchronized (prefill chunks stream on): its whole span, from begin
// to the next begin, is `unsynced_us` and nothing else, so it never enters the synchronized timings.
// Time while no work is pending is not counted at all: an engine waiting for requests is not stalling.
// An attempt that never reaches end() (llama_decode returned non-zero and the loop retries, or a throw
// left the step) is closed by the next begin as `failed_steps` / `failed_us` (its begin to that next
// begin), so its cost is never lost and never mixed into the good steps' timings (Codex on #133).
//
// The counters are cumulative and monotonic, written by the decode thread with relaxed atomics and
// read from any thread without a lock; a reader differences two reads into a window, so the engine
// holds no window size and no threshold.
//
// Pure and header-only, proven by tests/test-server-step-budget.cpp without a model.

#include <atomic>
#include <cstdint>
#include <optional>

// Which steps a sample describes: decode-only steps set the per-stream tok/s; steps that carry
// prompt tokens show prefill contending with decode.
enum class step_kind : uint8_t { decode, with_prefill };

struct step_budget_counters {
    uint64_t steps          = 0; // synchronized steps
    uint64_t tokens         = 0; // tokens in synchronized steps
    uint64_t decode_call_us = 0;
    uint64_t sync_wait_us   = 0;
    uint64_t inter_step_us  = 0;
    uint64_t unsynced_steps = 0;
    uint64_t unsynced_us    = 0;
    uint64_t failed_steps   = 0;
    uint64_t failed_us      = 0;

    step_budget_counters operator-(const step_budget_counters & earlier) const {
        return { steps - earlier.steps, tokens - earlier.tokens, decode_call_us - earlier.decode_call_us,
                 sync_wait_us - earlier.sync_wait_us, inter_step_us - earlier.inter_step_us,
                 unsynced_steps - earlier.unsynced_steps, unsynced_us - earlier.unsynced_us,
                 failed_steps - earlier.failed_steps, failed_us - earlier.failed_us };
    }
};

struct step_budget_snapshot {
    step_budget_counters decode;
    step_budget_counters with_prefill;
};

class step_budget {
public:
    // The decode thread, around each llama_decode call. `now_us` is a monotonic clock.
    void begin(int64_t now_us, step_kind kind, uint32_t n_tokens) {
        if (open_) {
            // the previous attempt never ended: a failed or retried decode
            auto & c = of(kind_);
            add(c.failed_steps, 1);
            add(c.failed_us, span(begin_us_, now_us));
        }
        close_previous(now_us);
        open_       = true;
        kind_       = kind;
        begin_us_   = now_us;
        n_tokens_   = n_tokens;
        submitted_  = 0;
        synced_     = std::nullopt;
    }

    void submitted(int64_t now_us) { submitted_ = now_us; }

    void synced(int64_t now_us) { synced_ = now_us; }

    // The step's results are consumed. `work_pending`: some sequence still needs another step, so the
    // time until the next begin is a stall the engine owes, not an idle engine.
    void end(int64_t now_us, bool work_pending) {
        if (!open_) {
            return;
        }
        auto & c = of(kind_);
        if (synced_) {
            add(c.steps, 1);
            add(c.tokens, n_tokens_);
            add(c.decode_call_us, span(begin_us_, submitted_));
            add(c.sync_wait_us, span(submitted_, *synced_));
            add(c.inter_step_us, span(*synced_, now_us)); // sampling and bookkeeping after the sync
        } else {
            add(c.unsynced_steps, 1);
            add(c.unsynced_us, span(begin_us_, now_us)); // the whole step: neither idle nor busy is proven
        }
        end_us_       = now_us;
        work_pending_ = work_pending;
        synced_kind_  = synced_.has_value();
        open_         = false;
        after_end_    = true;
    }

    step_budget_snapshot snapshot() const { return { read(decode_), read(with_prefill_) }; }

private:
    struct atomic_counters {
        std::atomic<uint64_t> steps{0}, tokens{0}, decode_call_us{0}, sync_wait_us{0}, inter_step_us{0},
                              unsynced_steps{0}, unsynced_us{0}, failed_steps{0}, failed_us{0};
    };

    // The time from the previous step's end to this begin: a stall when work was pending, attributed
    // to the previous step's kind; nothing when the engine had no work.
    void close_previous(int64_t now_us) {
        if (!after_end_) {
            return;
        }
        after_end_ = false;
        if (!work_pending_) {
            return;
        }
        auto & c = of(kind_);
        if (synced_kind_) {
            add(c.inter_step_us, span(end_us_, now_us));
        } else {
            add(c.unsynced_us, span(end_us_, now_us));
        }
    }

    atomic_counters & of(step_kind k) { return k == step_kind::decode ? decode_ : with_prefill_; }

    static uint64_t span(int64_t from, int64_t to) { return to > from ? uint64_t(to - from) : 0; }
    static void add(std::atomic<uint64_t> & a, uint64_t v) { a.fetch_add(v, std::memory_order_relaxed); }
    static step_budget_counters read(const atomic_counters & a) {
        const auto r = [](const std::atomic<uint64_t> & v) { return v.load(std::memory_order_relaxed); };
        return { r(a.steps), r(a.tokens), r(a.decode_call_us), r(a.sync_wait_us), r(a.inter_step_us),
                 r(a.unsynced_steps), r(a.unsynced_us), r(a.failed_steps), r(a.failed_us) };
    }

    atomic_counters decode_;
    atomic_counters with_prefill_;

    // decode-thread state
    bool                   open_         = false;
    bool                   after_end_    = false;
    bool                   work_pending_ = false;
    bool                   synced_kind_  = false;
    step_kind              kind_         = step_kind::decode;
    int64_t                begin_us_     = 0;
    int64_t                submitted_    = 0;
    int64_t                end_us_       = 0;
    uint32_t               n_tokens_     = 0;
    std::optional<int64_t> synced_;
};
