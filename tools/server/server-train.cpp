#include "server-train.h"

#include "common.h"
#include "log.h"
#include "llama.h"
#include "ggml-opt.h"

#include <chrono>
#include <sstream>

using json = nlohmann::ordered_json;

// ggml-opt's epoch callback carries no user data; one run at a time, so the progress of the
// current run is kept here for status() to read.
static std::atomic<int64_t> g_train_batch{0};
static std::atomic<int64_t> g_train_batch_max{0};

static void train_progress(bool train, ggml_opt_context_t, ggml_opt_dataset_t, ggml_opt_result_t,
                           int64_t ibatch, int64_t ibatch_max, int64_t) {
    if (train) {
        g_train_batch.store(ibatch);
        g_train_batch_max.store(ibatch_max);
    }
}

static std::vector<std::string> split_targets(const std::string & list) {
    std::vector<std::string> out;
    std::stringstream ss(list);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) {
            out.push_back(item);
        }
    }
    return out;
}

server_trainer::server_trainer(llama_model * model, const common_params & params_base)
    : model(model), params_base(params_base) {
    state = json{{"state", "idle"}};
}

server_trainer::~server_trainer() {
    if (worker.joinable()) {
        worker.join();
    }
}

json server_trainer::start(const json & body) {
    if (!body.contains("text") || !body.at("text").is_string() || body.at("text").get<std::string>().empty()) {
        return json{{"ok", false}, {"error", "\"text\" (the training corpus) is required"}};
    }
    if (!body.contains("out") || !body.at("out").is_string() || body.at("out").get<std::string>().empty()) {
        return json{{"ok", false}, {"error", "\"out\" (the adapter file to write) is required"}};
    }
    bool expected = false;
    if (!running.compare_exchange_strong(expected, true)) {
        return json{{"ok", false}, {"error", "a training run is already in progress"}, {"status", status()}};
    }
    if (worker.joinable()) {
        worker.join();
    }
    {
        std::lock_guard<std::mutex> lock(mu);
        state = json{{"state", "starting"}, {"out", body.at("out")}, {"epochs", json::array()}};
    }
    g_train_batch.store(0);
    g_train_batch_max.store(0);
    worker = std::thread(&server_trainer::run, this, body);
    return json{{"ok", true}, {"status", status()}};
}

json server_trainer::status() const {
    std::lock_guard<std::mutex> lock(mu);
    json s = state;
    if (s.value("state", "") == "running") {
        s["batch"]     = g_train_batch.load();
        s["batch_max"] = g_train_batch_max.load();
    }
    return s;
}

void server_trainer::run(json req) {
    auto fail = [&](const std::string & why) {
        LOG_ERR("%s: training run failed: %s\n", __func__, why.c_str());
        std::lock_guard<std::mutex> lock(mu);
        state["state"] = "error";
        state["error"] = why;
        running.store(false);
    };

    const std::string text    = req.at("text");
    const std::string out     = req.at("out");
    const int32_t     rank    = req.value("rank", 8);
    const float       alpha   = req.value("alpha", 16.0f);
    const std::string targets = req.value("targets", std::string("attn_q,attn_v"));
    const uint32_t    window  = req.value("window", 256u);
    const unsigned    epochs  = req.value("epochs", 1u);
    const float       lr0     = req.value("lr", 1e-5f);
    const float       val     = req.value("val_split", 0.1f);
    const uint32_t    seed    = req.value("seed", 42u);

    // The training context: the SAME model, its own graph. One ubatch is the whole window
    // (the training graph attends to this ubatch's K/V directly); flash attention has no
    // backward; the KV cache types are F32 because OUT_PROD has no F16 path.
    llama_context_params cparams = common_context_params_to_llama(params_base);
    cparams.n_ctx           = window;
    cparams.n_batch         = window;
    cparams.n_ubatch        = window;
    cparams.n_seq_max       = 1;
    cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cparams.type_k          = GGML_TYPE_F32;
    cparams.type_v          = GGML_TYPE_F32;
    cparams.embeddings      = false;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (ctx == nullptr) {
        fail("could not create the training context (window " + std::to_string(window) + "): likely out of device memory");
        return;
    }

    const std::string init_path = out + ".init.gguf";
    if (!common_lora_write_fresh(model, init_path, rank, alpha, split_targets(targets), seed)) {
        llama_free(ctx);
        fail("could not write the fresh adapter (do the targets exist in this model?)");
        return;
    }
    llama_adapter_lora * adapter = llama_adapter_lora_init(model, init_path.c_str());
    if (adapter == nullptr) {
        llama_free(ctx);
        fail("could not load the fresh adapter");
        return;
    }
    float scale = 1.0f;
    if (llama_set_adapters_lora(ctx, &adapter, 1, &scale) != 0) {
        llama_adapter_lora_free(adapter);
        llama_free(ctx);
        fail("could not attach the adapter to the training context");
        return;
    }

    std::vector<llama_token> tokens = common_tokenize(ctx, text, true);
    if ((int64_t) tokens.size() < 2 * (int64_t) window) {
        llama_adapter_lora_free(adapter);
        llama_free(ctx);
        fail("the corpus tokenizes to " + std::to_string(tokens.size()) + " tokens; at least two windows are needed");
        return;
    }
    ggml_opt_dataset_t dataset = common_opt_dataset_init(ctx, tokens, window / 2);

    lr_opt lr;
    lr.lr0    = lr0;
    lr.epochs = epochs;
    lr.init();

    llama_opt_params lopt{
        /*n_ctx_train     =*/ 0,
        /*param_filter    =*/ llama_opt_param_filter_all,
        /*param_filter_ud =*/ nullptr,
        /*get_opt_pars    =*/ common_opt_lr_pars,
        /*get_opt_pars_ud =*/ &lr,
        /*optimizer_type  =*/ GGML_OPT_OPTIMIZER_TYPE_ADAMW,
        /*adapter         =*/ adapter,
    };
    llama_opt_init(ctx, model, lopt);

    const int64_t idata_split  = (int64_t) (ggml_opt_dataset_ndata(dataset) * (1.0f - val));
    const int64_t train_tokens = idata_split * (int64_t) window;
    ggml_opt_result_t result_train = ggml_opt_result_init();
    ggml_opt_result_t result_eval  = ggml_opt_result_init();
    {
        std::lock_guard<std::mutex> lock(mu);
        state["state"]        = "running";
        state["window"]       = window;
        state["tokens"]       = (int64_t) tokens.size();
        state["train_tokens"] = train_tokens;
    }
    LOG_INF("%s: training on the served model: %zu tokens, window %u, %u epochs, adapter -> %s\n",
            __func__, tokens.size(), window, epochs, out.c_str());

    for (lr.epoch = 0; lr.epoch < lr.epochs; ++lr.epoch) {
        const int64_t t0 = ggml_time_us();
        llama_opt_epoch(ctx, dataset, result_train, result_eval, idata_split, train_progress, nullptr);
        const double seconds = (ggml_time_us() - t0) / 1e6;
        double loss_train = 0.0, unc_train = 0.0, loss_eval = 0.0, unc_eval = 0.0;
        ggml_opt_result_loss(result_train, &loss_train, &unc_train);
        ggml_opt_result_loss(result_eval,  &loss_eval,  &unc_eval);
        const double tok_s = seconds > 0 ? train_tokens / seconds : 0.0;
        LOG_INF("%s: epoch %u: train_loss=%.5f eval_loss=%.5f seconds=%.1f train_tok_s=%.1f\n",
                __func__, lr.epoch, loss_train, loss_eval, seconds, tok_s);
        {
            std::lock_guard<std::mutex> lock(mu);
            state["epochs"].push_back(json{{"epoch", lr.epoch}, {"train_loss", loss_train}, {"eval_loss", loss_eval},
                                           {"seconds", seconds}, {"train_tok_s", tok_s}});
        }
        ggml_opt_result_reset(result_train);
        ggml_opt_result_reset(result_eval);
    }
    ggml_opt_result_free(result_train);
    ggml_opt_result_free(result_eval);
    ggml_opt_dataset_free(dataset);

    const int32_t saved = llama_adapter_lora_save(adapter, out.c_str());
    llama_adapter_lora_free(adapter);
    llama_free(ctx);

    std::lock_guard<std::mutex> lock(mu);
    if (saved != 0) {
        state["state"] = "error";
        state["error"] = "training finished but the adapter could not be written to " + out;
    } else {
        state["state"] = "done";
        state["adapter"] = out;
    }
    running.store(false);
}
