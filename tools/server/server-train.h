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
//   GET  /train  -> the run's state: idle | running (epoch, batch/of) | done | error, with
//                   per-epoch train/eval loss, train tok/s, and the adapter path

#include "common.h"

#include "json.h"

#include <atomic>
#include <mutex>
#include <string>
#include <thread>

struct llama_model;

class server_trainer {
public:
    server_trainer(llama_model * model, const common_params & params_base);
    ~server_trainer();

    // Starts a run on a worker thread; refuses (ok=false) while one is running or on bad input.
    common_json start(const common_json & body);
    common_json status() const;

private:
    void run(common_json req);

    llama_model * model;
    common_params params_base;

    std::thread worker;
    std::atomic<bool> running{false};
    mutable std::mutex mu;
    common_json state;  // guarded by mu
};
