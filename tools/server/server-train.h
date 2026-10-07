#pragma once

// In-process LoRA-only training on the model this server is already serving: the dream stage's
// first form. A second llama_context is created on the SAME loaded llama_model (the weights are
// shared, never copied), a fresh adapter is attached to that context only, and ggml-opt trains
// the adapter's A/B while the serving slots keep decoding on their own context. The base stays
// at its served quantization, frozen: the adapter is fit against exactly the numerics it will
// be served over (QLoRA by construction). One training run at a time.
//
//   POST /train  {"text": "...", "out": "adapter.gguf", "rank": 8, "alpha": 16,
//                 "targets": "attn_q,attn_v", "window": 256, "epochs": 1, "lr": 1e-5,
//                 "val_split": 0.1, "seed": 42}
//   GET  /train  -> idle | starting | running | done | cancelled | error,
//                   with per-epoch train/eval loss, train tok/s, and the adapter path
//   POST /train/cancel -> the run stops at the next training window and writes no adapter
//   POST /train/pause {"out": "adapter.gguf"} -> GET /train reports paused=true once acknowledged
//   POST /train/resume {"out": "adapter.gguf"} -> continue the same optimizer, adapter and cursor
// Use a unique output name per job; controls refuse an output name belonging to another job.
//
// Pause is acknowledged before the next training OR evaluation window, including the first.
// It cannot interrupt an in-flight window. The retained training allocation is not reclaimed:
// this is a compute pause on the same quantized base, not unloading or starting another model.
// Resume still respects automatic serving-slot yielding. Cancel wakes a paused worker and
// discards its partial adapter. Controls are refused once finalization starts.
//
//   POST /train  {"examples": [{"prompt": "...", "completion": "..."} | {"messages": [...]}, ...], ...}
//                -> each example is rendered through THIS model's chat template as a closed
//                   conversation (no generation prompt), control tokens parsed as tokens, and is
//                   ONE training window (padded; an example longer than the window is refused).
//                   Only the assistant turns carry loss (each turn's content and its end-of-turn
//                   token): the adapter learns her answers, not the prompts, the room or the
//                   system text. Losses are reported per trainable token. "text" and "examples"
//                   are exclusive; "text" trains on every token.
//
// "memory_budget_mib": what training may add on the GPU. Pass it wherever the driver's free figure
// is not physical (Windows/WDDM reports nearly the whole card free beside other processes): a graph
// over it is refused before allocation instead of spilling into host memory.
//
// "parse_special": true tokenizes control tokens in the text (<|im_start|> ...) as the tokens
// they name, which is what a corpus rendered through the model's chat template needs; the
// default (false) trains on the text exactly as written.

#include "common.h"
#include "train-segment.h"

#include "json.h"
#include "ggml-opt.h"

#include <atomic>
#include <cmath>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <string>
#include <thread>

struct llama_model;
struct llama_context;

// A bare `.gguf` file name resolved inside `dir` (the server's --train-dir), or "" when the
// name carries a directory, "..", a control character or another extension. The one rule
// for every path the server writes or reads there: /train's "out" and /lora-adapters/load.
std::string confine_out(const std::string & dir, const std::string & name);

class server_trainer {
public:
    // serving: read between training windows; the trainer yields while serving is busy, so a
    // turn never waits behind more than one window (the pacing below says for how long)
    // render_chat(messages, add_generation_prompt): messages (OpenAI shape) through the model's
    // chat template; throws on a template error. Empty = "examples" are refused.
    // (messages, tools, add_generation_prompt) -> the prompt text serving would frame
    using render_fn = std::function<std::string(const common_json &, const common_json &, bool)>;
    // What the trainer reads of serving, between windows (each an atomic load in the server):
    struct serving_view {
        std::function<int()>     busy_slots;       // slots working right now
        std::function<int64_t()> idle_ms;          // how long no slot has been working (0 while one is)
        std::function<int64_t()> tokens_generated; // every token a slot has accepted since start
    };
    server_trainer(llama_model * model, const common_params & params_base, serving_view serving,
                   render_fn render_chat);

    // THE SLOWDOWN BOUND, as a pure function. After a window of d_ms, the yield to busy slots
    // that keeps her decode rate over the cycle (window + yield) at or above (1 - T) of her rate
    // with no training running:  (r_w * d + r0 * y) / (d + y) >= r0 * (1 - T)
    //                       =>   y >= d * (r0 * (1 - T) - r_w) / (r0 * T)
    // r0: her tokens/ms while slots were busy and no window ran; r_w: her tokens/ms while a
    // window ran. 0 when the window already costs her less than T. Unmeasured (r0 <= 0) takes
    // the worst case r_w = 0, y = d * (1 - T) / T, so the bound holds before it has a sample.
    static int64_t yield_for_slowdown(int64_t d_ms, int64_t max_slowdown_ppm, double r0, double r_w);

    // "examples", tokenized: one sequence per example with its loss mask (1 = the token is
    // predicted with loss: the assistant's content and end-of-turn)
    struct examples_data {
        std::vector<std::vector<llama_token>> tokens;
        std::vector<std::vector<uint8_t>>     loss;
        // "fit": examples cut to fit the window ("middle": oldest history dropped; "left": the
        // tail that ends the last trained turn kept), and examples that could not fit (skipped)
        int64_t truncated = 0;
        int64_t skipped   = 0;
    };
    ~server_trainer();

    // Starts a run on a worker thread; refuses (ok=false) while one is running or on bad input.
    common_json start(const common_json & body_in);
    common_json status() const;
    // Stops the running job at its next training window (no adapter is written); ok=false when
    // nothing is running.
    common_json cancel();
    common_json pause(const std::string & out);
    common_json resume(const std::string & out);

private:
    void run(common_json req, examples_data ex);
    static bool before_window(bool train, void * user_data);
    // The step boundary: a new graph begins, so the segment plan starts over (then the same
    // yield as every segment boundary).
    static bool before_step(bool train, void * user_data);
    // The training context's sched eval callback: cuts each step into segments of about one of
    // her decode steps and yields to serving between them (train-segment.h, continuum S4).
    static bool on_eval(ggml_tensor * t, bool ask, void * user_data);
    static void on_batch(bool train, ggml_opt_context_t, ggml_opt_dataset_t, ggml_opt_result_t,
                         int64_t ibatch, int64_t ibatch_max, int64_t);

    llama_model * model;
    common_params params_base;
    serving_view serving;
    render_fn render_chat;
    // serving counts as busy while a slot works and until it has been idle IDLE_RELEASE_MS
    bool serving_busy() const;
    std::atomic<bool>    yield_to_turns{true};
    std::atomic<int64_t> yielded_ms{0};
    // THE SHARE. Yielding while any slot is busy is starvation on a lane whose slots never go
    // idle (continuum card 36c3c00a: a 27B lane with residents, 12 minutes at batch 0). The
    // trainer owns a share of the lane's time instead: after a window that took d ms it yields
    // to busy slots for at least d * (1 - share) / share, then takes the next window whether
    // or not serving is busy; idle serving releases it at once. share_ppm = 1_000_000 never
    // yields (the old "yield": false); 0 is refused at parse.
    std::atomic<int64_t> share_ppm{250000};
    // The bound the caller actually means (Joel: "negligible impact on inference latency ... a
    // slowdown if clever can be unnoticed"): her decode rate while training runs stays within
    // max_slowdown_ppm of her rate without it. 0 = off, and the share above paces the windows.
    // When on, the yield after each window comes from yield_for_slowdown on her MEASURED rates.
    std::atomic<int64_t> max_slowdown_ppm{0};
    // How long the slots must stay idle before an owed yield is released: longer than the gap
    // between one request's end and the next one's arrival in an agent's loop, shorter than
    // any pause a person would call idle.
    static constexpr int64_t IDLE_RELEASE_MS = 250;
    // Her rates as the controller sees them: recent evidence, each sample discounted by the
    // evidence that came after it (time constant RATE_EVIDENCE_MS of measured stretch), so a
    // change in her load moves the bound within a minute instead of being outvoted by the run.
    struct rate_estimate {
        double ms = 0, tokens = 0;
        void add(double d_ms, double d_tokens) {
            const double keep = std::exp(-d_ms / (double) RATE_EVIDENCE_MS);
            ms = ms * keep + d_ms;
            tokens = tokens * keep + d_tokens;
        }
        double per_ms() const { return ms > 0 ? tokens / ms : 0.0; }
    };
    static constexpr int64_t RATE_EVIDENCE_MS = 60000;
    // A bound that stops yielding stops measuring her rate without a window, and a stale rate
    // would let training run on while her load changed. So when that rate has gone unsampled
    // for PROBE_EVERY_MS, at least PROBE_MS is owed: about 1.6% of the trainer's time, the
    // price of the bound staying true.
    static constexpr int64_t PROBE_EVERY_MS = 30000;
    static constexpr int64_t PROBE_MS = 500;
    rate_estimate rate_no_window;   // guarded by mu
    rate_estimate rate_in_window;   // guarded by mu
    std::chrono::steady_clock::time_point last_no_window_sample{};
    // Her decoded tokens and the time they took, summed over the run, only across segments
    // that were busy end to end (a segment with an idle slot measures the load, not her rate).
    // Guarded by mu. Ratios of sums, never an average of per-segment rates: a 30 ms yield with
    // one token is one token of evidence, not a rate of 33 tok/s.
    int64_t busy_window_ms{0};
    int64_t busy_window_tokens{0};
    int64_t busy_yield_ms{0};
    int64_t busy_yield_tokens{0};
    int64_t window_tokens_start{0};
    bool    window_busy_start{false};
    std::atomic<int64_t> window_ms_last{0};
    std::atomic<int64_t> windows{0};
    std::atomic<int64_t> windows_while_busy{0};
    std::chrono::steady_clock::time_point window_started{};
    // The segments a step is cut into (touched only on the training thread) and the duration
    // each aims at, refreshed at every boundary from her measured rate (read per node, so atomic).
    train_segmenter      segmenter;
    std::atomic<double>  segment_target_ms{TRAIN_SEGMENT_DEFAULT_MS};
    // The window's duration is serving's latency bound while training is on (the share
    // bounds how OFTEN a window runs, not how long one takes; Joel: a brief, unnoticed
    // slowdown). Every window's duration is kept here, under `mu`, so status reports the
    // distribution exactly (max, p50, p95), never a polled sample that misses the long one.
    std::vector<int64_t> window_ms_samples;
    int64_t window_ms_max{0};

    std::thread worker;
    std::atomic<bool> running{false};
    std::atomic<bool> cancel_requested{false};
    mutable std::mutex mu;
    std::condition_variable control_changed;
    bool pause_requested = false; // guarded by mu
    bool paused = false;          // worker acknowledged at a window boundary
    bool waiting_for_serving = false;
    bool finishing = false;      // reject controls while committing the result; guarded by mu
    common_json state;                  // guarded by mu
    llama_context * ctx_live = nullptr; // guarded by mu: the training context while an epoch can run
};

// "fit": "left" for one tokenized example (see server-train.cpp): keep the window+1 tokens that
// end the last trained turn (the leading BOS kept when `bos`), mask a trained turn cut at the
// front, and return false when the last trained turn alone does not fit.
bool train_fit_left(std::vector<llama_token> & toks, std::vector<uint8_t> & loss, int64_t window, bool bos);

// "fit": "middle" for one conversation (see server-train.cpp): the index of the oldest history
// message to drop (after the leading system messages, before the user message her last trained
// turn answers), or npos when nothing droppable is left.
size_t train_fit_droppable(const std::vector<std::string> & roles, const std::vector<bool> & trained);
