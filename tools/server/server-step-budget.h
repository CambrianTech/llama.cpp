#pragma once

// Continuum fork: the decode loop's frame budget, measured on EVERY step (CONTINUUM-ENGINE 4.7.9).
//
// A one-off `sample` proved the GPU idles ~40-45% of the decode loop. This makes that a number the
// engine reports continuously, so a regression is seen the step it happens and every speed change
// carries its own before/after. Three timestamps per step, all already on the decode thread:
//
//   begin      llama_decode is called            (CPU starts building and submitting the graph)
//   submitted  llama_decode returns              (the graph is queued on the GPU)
//   synced     llama_synchronize returns         (the GPU finished; outputs are readable)
//
//   build = submitted - begin           CPU work with the GPU waiting on it (inputs, mask, graph)
//   gpu   = synced - submitted          the GPU computing (an upper bound of its idle-free time:
//                                       it may finish early and wait for the sync call)
//   gap   = next begin - synced         CPU work between steps (sampling, slot bookkeeping,
//                                       checkpoints, the next batch) with the GPU idle
//
// A step without outputs is not synchronized (prefill chunks stream on); its time until the next
// step is counted as `unsynced`, never as idle or busy, because neither is proven. A gap is counted
// only when work was pending at the step's end: an engine waiting for requests is not stalling.
//
// The counters are cumulative and monotonic (microseconds and counts), written by the decode thread
// with relaxed atomics and read from any thread without a lock; a reader differences two snapshots to
// get a window, so the engine holds no window size and no threshold.
//
//   GPU idle share = (build + gap) / (build + gpu + gap)
//
// Pure and header-only, proven by tests/test-server-step-budget.cpp without a model.

#include <atomic>
#include <cstdint>
#include <optional>

// Which steps a sample describes: decode-only steps set the per-stream tok/s; steps that carry
// prompt tokens show prefill contending with decode.
enum class step_kind : uint8_t { decode, with_prefill };

struct step_budget_counters {
    uint64_t steps        = 0;
    uint64_t tokens       = 0;
    uint64_t build_us     = 0;
    uint64_t gpu_us       = 0;
    uint64_t gap_us       = 0;
    uint64_t unsynced_us  = 0;

    // (build + gap) / (build + gpu + gap); empty when nothing synchronized was measured.
    std::optional<double> gpu_idle_share() const {
        const uint64_t total = build_us + gpu_us + gap_us;
        if (total == 0) {
            return std::nullopt;
        }
        return double(build_us + gap_us) / double(total);
    }

    step_budget_counters operator-(const step_budget_counters & earlier) const {
        return { steps - earlier.steps, tokens - earlier.tokens, build_us - earlier.build_us,
                 gpu_us - earlier.gpu_us, gap_us - earlier.gap_us, unsynced_us - earlier.unsynced_us };
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
        add(c.steps, 1);
        add(c.tokens, n_tokens_);
        add(c.build_us, span(begin_us_, submitted_));
        if (synced_) {
            add(c.gpu_us, span(submitted_, *synced_));
            add(c.gap_us, span(*synced_, now_us)); // sampling and bookkeeping after the sync
        } else {
            add(c.unsynced_us, span(submitted_, now_us));
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
        std::atomic<uint64_t> steps{0}, tokens{0}, build_us{0}, gpu_us{0}, gap_us{0}, unsynced_us{0};
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
            add(c.gap_us, span(end_us_, now_us));
        } else {
            add(c.unsynced_us, span(end_us_, now_us));
        }
    }

    atomic_counters & of(step_kind k) { return k == step_kind::decode ? decode_ : with_prefill_; }

    static uint64_t span(int64_t from, int64_t to) { return to > from ? uint64_t(to - from) : 0; }
    static void add(std::atomic<uint64_t> & a, uint64_t v) { a.fetch_add(v, std::memory_order_relaxed); }
    static step_budget_counters read(const atomic_counters & a) {
        return { a.steps.load(std::memory_order_relaxed), a.tokens.load(std::memory_order_relaxed),
                 a.build_us.load(std::memory_order_relaxed), a.gpu_us.load(std::memory_order_relaxed),
                 a.gap_us.load(std::memory_order_relaxed), a.unsynced_us.load(std::memory_order_relaxed) };
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
