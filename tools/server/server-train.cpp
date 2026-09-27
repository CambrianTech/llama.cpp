#include "server-train.h"

#include "common.h"
#include "log.h"
#include "llama.h"
#include "ggml-opt.h"

#include <cstdio>
#include <chrono>
#include <sstream>

using json = common_json;

// ggml-opt's epoch callback carries no user data; one run at a time, so the current run's
// trainer and progress are kept here.
static std::atomic<int64_t>       g_train_batch{0};
static std::atomic<int64_t>       g_train_batch_max{0};
static std::atomic<server_trainer *> g_trainer{nullptr};

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

server_trainer::server_trainer(llama_model * model, const common_params & params_base, std::function<int()> busy_slots)
    : model(model), params_base(params_base), busy_slots(std::move(busy_slots)) {
    state = json::object({{"state", "idle"}});
}

server_trainer::~server_trainer() {
    // a shutdown must not wait out a whole run
    cancel();
    if (worker.joinable()) {
        worker.join();
    }
}

// "out" names a FILE in the server's --train-dir, never a path: no separators, no "..", not
// absolute, not empty, and it must end in .gguf. Returns the resolved path, or "" when refused.
static std::string confine_out(const std::string & dir, const std::string & name) {
    if (name.empty() || name.size() > 200 || name == "." || name == "..") {
        return "";
    }
    for (char c : name) {
        if (c == '/' || c == '\\' || c == ':' || (unsigned char) c < 0x20) {
            return "";
        }
    }
    if (name.find("..") != std::string::npos || name.size() < 6 || name.compare(name.size() - 5, 5, ".gguf") != 0) {
        return "";
    }
    std::string d = dir;
    if (!d.empty() && d.back() != '/' && d.back() != '\\') {
        d += '/';
    }
    return d + name;
}

json server_trainer::start(const json & body) {
    // OFF unless the server was started with --train-dir: the route writes files, so an engine
    // opts in explicitly and every write is confined to that directory (Cormac on #14).
    if (params_base.train_dir.empty()) {
        return json::object({{"ok", false}, {"error", "training is disabled on this server (start it with --train-dir DIR to enable /train)"}});
    }
    if (!body.contains("text") || !body.at("text").is_string() || body.at("text").get<std::string>().empty()) {
        return json::object({{"ok", false}, {"error", "\"text\" (the training corpus) is required"}});
    }
    if (!body.contains("out") || !body.at("out").is_string() || body.at("out").get<std::string>().empty()) {
        return json::object({{"ok", false}, {"error", "\"out\" (the adapter file name to write) is required"}});
    }
    if (confine_out(params_base.train_dir, body.at("out").get<std::string>()).empty()) {
        return json::object({{"ok", false}, {"error", "\"out\" must be a bare file name ending in .gguf (no directories, no \"..\"); it is written inside the server's --train-dir"}});
    }
    // Every numeric input is checked here, before a thread starts: inside a serving process an
    // assert in the training path would take the server down (Cormac on #14).
    {
        auto num = [&](const char * key, double def) { return body.contains(key) && body.at(key).is_number() ? body.at(key).get<double>() : def; };
        for (const char * key : {"rank", "alpha", "window", "epochs", "lr", "val_split", "seed"}) {
            if (body.contains(key) && !body.at(key).is_number()) {
                return json::object({{"ok", false}, {"error", std::string("\"") + key + "\" must be a number"}});
            }
        }
        const double rank = num("rank", 8), alpha = num("alpha", 16), window = num("window", 256), epochs = num("epochs", 1);
        const double lr = num("lr", 1e-5), val = num("val_split", 0.1);
        std::string why;
        if (rank < 1 || rank > 256 || rank != (int64_t) rank)       why = "rank must be an integer in [1, 256]";
        else if (!(alpha > 0))                                       why = "alpha must be > 0";
        else if (window < 16 || window > 8192 || window != (int64_t) window) why = "window must be an integer in [16, 8192]";
        else if (epochs < 1 || epochs > 100 || epochs != (int64_t) epochs)   why = "epochs must be an integer in [1, 100]";
        else if (!(lr > 0 && lr <= 1))                               why = "lr must be in (0, 1]";
        else if (!(val >= 0 && val < 1))                             why = "val_split must be in [0, 1)";
        else if (body.contains("targets") && (!body.at("targets").is_string() || split_targets(body.at("targets").get<std::string>()).empty()))
                                                                     why = "targets must be a non-empty comma-separated list of module names";
        if (!why.empty()) {
            return json::object({{"ok", false}, {"error", why}});
        }
    }
    // REPACKED WEIGHTS GIVE WRONG GRADIENTS, SILENTLY (Cormac on the /train plan): the backward
    // reads quantized blocks in the standard layout, and a weight in an extra buffer type
    // (CPU_REPACK) is not in it — no crash, garbage gradients. Extra buffer types only apply to
    // weights kept on the CPU, so training is allowed when they are disabled, or when every layer
    // is offloaded and no tensor override pins a weight elsewhere; otherwise refuse and say so.
    {
        const int32_t n_layer  = llama_model_n_layer(model);
        const int32_t ngl      = params_base.n_gpu_layers;
        const bool    all_gpu  = ngl <= -2 || ngl > n_layer;
        // the list carries a {nullptr, nullptr} terminator when parsed from args: count real entries
        bool override = false;
        for (const auto & o : params_base.tensor_buft_overrides) {
            override = override || o.pattern != nullptr;
        }
        if (!params_base.no_extra_bufts && (!all_gpu || override)) {
            return json::object({{"ok", false}, {"error",
                "refusing to train: this server may hold base weights in a repacked CPU buffer (n_gpu_layers " +
                std::to_string(ngl) + " of " + std::to_string(n_layer) + " layers" + (override ? ", tensor overrides set" : "") +
                "), and the backward reads the standard layout, so gradients would be silently wrong. "
                "Serve with every layer offloaded (-ngl all) or with --no-repack."}});
        }
    }
    if (body.contains("parse_special") && !body.at("parse_special").is_boolean()) {
        return json::object({{"ok", false}, {"error", "\"parse_special\" must be true or false"}});
    }
    bool expected = false;
    if (!running.compare_exchange_strong(expected, true)) {
        return json::object({{"ok", false}, {"error", "a training run is already in progress"}, {"status", status()}});
    }
    if (worker.joinable()) {
        worker.join();
    }
    {
        std::lock_guard<std::mutex> lock(mu);
        state = json::object({{"state", "starting"}, {"out", body.at("out")}, {"epochs", json::array()}});
    }
    g_train_batch.store(0);
    g_train_batch_max.store(0);
    yielded_ms.store(0);
    cancel_requested.store(false);
    yield_to_turns.store(body.value("yield", true));
    g_trainer.store(this);
    worker = std::thread(&server_trainer::run, this, body);
    return json::object({{"ok", true}, {"status", status()}});
}

// Between training batches: the frame budget's first form. While any serving slot is working,
// the next batch waits, so training fills the gaps between turns and a turn waits behind at
// most the one batch already in flight. The time spent yielding is reported in status().
void server_trainer::on_batch(bool train, ggml_opt_context_t, ggml_opt_dataset_t, ggml_opt_result_t,
                              int64_t ibatch, int64_t ibatch_max, int64_t) {
    if (!train) {
        return;
    }
    g_train_batch.store(ibatch);
    g_train_batch_max.store(ibatch_max);
    server_trainer * self = g_trainer.load();
    if (self == nullptr || !self->yield_to_turns.load() || !self->busy_slots) {
        return;
    }
    const auto t0 = std::chrono::steady_clock::now();
    while (self->busy_slots() > 0 && !self->cancel_requested.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    self->yielded_ms += std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();
}

json server_trainer::cancel() {
    if (!running.load()) {
        return json::object({{"ok", false}, {"error", "no training run is in progress"}});
    }
    cancel_requested.store(true);
    std::lock_guard<std::mutex> lock(mu);
    if (ctx_live != nullptr) {
        llama_opt_stop(ctx_live, true);
    }
    state["cancel_requested"] = true;
    return json::object({{"ok", true}});
}

json server_trainer::status() const {
    std::lock_guard<std::mutex> lock(mu);
    json s = state;
    if (s.value("state", "") == "running") {
        s["batch"]     = g_train_batch.load();
        s["batch_max"] = g_train_batch_max.load();
    }
    s["yielded_ms"] = yielded_ms.load();
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

    const std::string text    = req.at("text").get<std::string>();
    const std::string out     = confine_out(params_base.train_dir, req.at("out").get<std::string>());
    const int32_t     rank    = (int32_t) req.value("rank", (int64_t) 8);
    const float       alpha   = (float) req.value("alpha", 16.0);
    const std::string targets = req.value("targets", std::string("attn_q,attn_v"));
    const uint32_t    window  = (uint32_t) req.value("window", (int64_t) 256);
    const unsigned    epochs  = (unsigned) req.value("epochs", (int64_t) 1);
    const float       lr0     = (float) req.value("lr", 1e-5);
    const float       val     = (float) req.value("val_split", 0.1);
    const uint32_t    seed    = (uint32_t) req.value("seed", (int64_t) 42);
    const bool        special = req.value("parse_special", false);

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
    // training reads logits for every token of the window; the serving params cap outputs per
    // ubatch to what sampling needs (a server-computed limit), which a training batch overruns
    cparams.n_outputs_max         = 0;   // = n_batch
    cparams.n_outputs_max_per_seq = 0;   // = n_outputs_max

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
    std::remove(init_path.c_str()); // loaded (or refused): the file has done its job
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

    std::vector<llama_token> tokens = common_tokenize(ctx, text, true, special);
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
    {
        // from here an epoch can run: a cancel reaches this context directly
        std::lock_guard<std::mutex> lock(mu);
        ctx_live = ctx;
        if (cancel_requested.load()) {
            llama_opt_stop(ctx, true);
        }
    }

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

    for (lr.epoch = 0; lr.epoch < lr.epochs && !cancel_requested.load(); ++lr.epoch) {
        const int64_t t0 = ggml_time_us();
        llama_opt_epoch(ctx, dataset, result_train, result_eval, idata_split, &server_trainer::on_batch, nullptr);
        const double seconds = (ggml_time_us() - t0) / 1e6;
        if (cancel_requested.load()) {
            break;
        }
        double loss_train = 0.0, unc_train = 0.0, loss_eval = 0.0, unc_eval = 0.0;
        ggml_opt_result_loss(result_train, &loss_train, &unc_train);
        ggml_opt_result_loss(result_eval,  &loss_eval,  &unc_eval);
        const double tok_s = seconds > 0 ? train_tokens / seconds : 0.0;
        LOG_INF("%s: epoch %u: train_loss=%.5f eval_loss=%.5f seconds=%.1f train_tok_s=%.1f\n",
                __func__, lr.epoch, loss_train, loss_eval, seconds, tok_s);
        {
            std::lock_guard<std::mutex> lock(mu);
            state["epochs"].push_back(json::object({{"epoch", lr.epoch}, {"train_loss", loss_train}, {"eval_loss", loss_eval},
                                           {"seconds", seconds}, {"train_tok_s", tok_s}}));
        }
        ggml_opt_result_reset(result_train);
        ggml_opt_result_reset(result_eval);
    }
    ggml_opt_result_free(result_train);
    ggml_opt_result_free(result_eval);
    ggml_opt_dataset_free(dataset);
    {
        std::lock_guard<std::mutex> lock(mu);
        ctx_live = nullptr;
    }

    // a cancelled run writes nothing: a partial adapter is not a result
    const bool    cancelled = cancel_requested.load();
    const int32_t saved     = cancelled ? 0 : llama_adapter_lora_save(adapter, out.c_str());
    llama_adapter_lora_free(adapter);
    llama_free(ctx);

    std::lock_guard<std::mutex> lock(mu);
    if (cancelled) {
        state["state"] = "cancelled";
    } else if (saved != 0) {
        state["state"] = "error";
        state["error"] = "training finished but the adapter could not be written to " + out;
    } else {
        state["state"] = "done";
        state["adapter"] = out;
    }
    running.store(false);
}
