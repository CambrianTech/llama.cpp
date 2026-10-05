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

#include "json.h"
#include "ggml-opt.h"

#include <atomic>
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
    // busy_slots: how many serving slots are working right now (read between training batches;
    // the trainer yields while it is non-zero, so a turn never waits behind more than one batch)
    // render_chat(messages, add_generation_prompt): messages (OpenAI shape) through the model's
    // chat template; throws on a template error. Empty = "examples" are refused.
    // (messages, tools, add_generation_prompt) -> the prompt text serving would frame
    using render_fn = std::function<std::string(const common_json &, const common_json &, bool)>;
    server_trainer(llama_model * model, const common_params & params_base, std::function<int()> busy_slots,
                   render_fn render_chat);

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
    static void on_batch(bool train, ggml_opt_context_t, ggml_opt_dataset_t, ggml_opt_result_t,
                         int64_t ibatch, int64_t ibatch_max, int64_t);

    llama_model * model;
    common_params params_base;
    std::function<int()> busy_slots;
    render_fn render_chat;
    std::atomic<bool>    yield_to_turns{true};
    std::atomic<int64_t> yielded_ms{0};
    // THE SHARE. Yielding while any slot is busy is starvation on a lane whose slots never go
    // idle (continuum card 36c3c00a: a 27B lane with residents, 12 minutes at batch 0). The
    // trainer owns a share of the lane's time instead: after a window that took d ms it yields
    // to busy slots for at least d * (1 - share) / share, then takes the next window whether
    // or not serving is busy; idle serving releases it at once. share_ppm = 1_000_000 never
    // yields (the old "yield": false); 0 is refused at parse.
    std::atomic<int64_t> share_ppm{250000};
    std::atomic<int64_t> window_ms_last{0};
    std::atomic<int64_t> windows{0};
    std::atomic<int64_t> windows_while_busy{0};
    std::chrono::steady_clock::time_point window_started{};

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
