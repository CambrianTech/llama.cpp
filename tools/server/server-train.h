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
//   GET  /train  -> the run's state: idle | running (epoch, batch/of) | done | cancelled | error,
//                   with per-epoch train/eval loss, train tok/s, and the adapter path
//   POST /train/cancel -> the run stops at the next training window and writes no adapter
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
    };
    ~server_trainer();

    // Starts a run on a worker thread; refuses (ok=false) while one is running or on bad input.
    common_json start(const common_json & body_in);
    common_json status() const;
    // Stops the running job at its next training window (no adapter is written); ok=false when
    // nothing is running.
    common_json cancel();

private:
    void run(common_json req, examples_data ex);
    static void on_batch(bool train, ggml_opt_context_t, ggml_opt_dataset_t, ggml_opt_result_t,
                         int64_t ibatch, int64_t ibatch_max, int64_t);

    llama_model * model;
    common_params params_base;
    std::function<int()> busy_slots;
    render_fn render_chat;
    std::atomic<bool>    yield_to_turns{true};
    std::atomic<int64_t> yielded_ms{0};

    std::thread worker;
    std::atomic<bool> running{false};
    std::atomic<bool> cancel_requested{false};
    mutable std::mutex mu;
    common_json state;                  // guarded by mu
    llama_context * ctx_live = nullptr; // guarded by mu: the training context while an epoch can run
};
