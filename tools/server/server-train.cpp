#include "server-train.h"

#include "common.h"
#include "log.h"
#include "llama.h"
#include "ggml-opt.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <chrono>
#include <sstream>

using json = common_json;

// ggml-opt's epoch callback carries no user data; one run at a time, so the current run's
// progress is kept here.
static std::atomic<int64_t>       g_train_batch{0};
static std::atomic<int64_t>       g_train_batch_max{0};

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

server_trainer::server_trainer(llama_model * model, const common_params & params_base, serving_view serving,
                               render_fn render_chat)
    : model(model), params_base(params_base), serving(std::move(serving)), render_chat(std::move(render_chat)) {
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
std::string confine_out(const std::string & dir, const std::string & name) {
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

// "examples" -> one tokenized sequence per example with its loss mask. Each example is rendered
// through the model's chat template as a closed conversation; each assistant turn's span is
// located by rendering the conversation up to it WITH the generation prompt (where the turn's
// content starts) and through it without (where it ends, end-of-turn included). A template that
// does not render prefix-stably (e.g. one that rewrites earlier turns) is refused for that example:
// the span cannot be located, and guessing would train the wrong tokens.
//
// Returns false with `why` when the example is malformed (the run is refused); true otherwise,
// with `toks`/`loss` filled.
static bool render_example(const json & messages, const std::vector<bool> & trained, const json & tools,
                           const server_trainer::render_fn & render, const llama_vocab * vocab, const std::string & at,
                           std::vector<llama_token> & toks, std::vector<uint8_t> & loss, std::string & why) {
    auto head = [&](size_t n) { // the first n messages
        json h = json::array();
        for (size_t j = 0; j < n; ++j) {
            h.push_back(messages[j]);
        }
        return h;
    };
    std::string full;
    std::vector<std::pair<size_t, size_t>> spans; // [begin, end) in chars of `full`
    try {
        full = render(messages, tools, false);
        for (size_t k = 0; k < messages.size(); ++k) {
            if (!trained[k] || !messages[k].is_object() || messages[k].value("role", std::string()) != "assistant") {
                continue;
            }
            const std::string pre  = render(head(k), tools, true);
            const std::string upto = render(head(k + 1), tools, false);
            if (full.compare(0, upto.size(), upto) != 0 || upto.compare(0, pre.size(), pre) != 0 || pre.size() >= upto.size()) {
                why = at + ": the chat template does not render this conversation prefix-stably at message " + std::to_string(k)
                    + ", so the assistant turn cannot be located (templates that strip earlier turns' reasoning, e.g. Qwen3's"
                    + " <think>, do this: send such a conversation as single-turn examples)";
                return false;
            }
            spans.emplace_back(pre.size(), upto.size());
        }
    } catch (const std::exception & e) {
        why = at + " could not be rendered through the chat template: " + e.what();
        return false;
    }
    if (spans.empty()) {
        why = at + " has no assistant turn to learn";
        return false;
    }
    // tokenize segment by segment so every token is wholly inside or outside a span
    toks.clear();
    loss.clear();
    size_t pos = 0;
    auto add = [&](size_t end, bool trainable) {
        if (end <= pos) {
            return;
        }
        const auto seg = common_tokenize(vocab, full.substr(pos, end - pos), /*add_special =*/ pos == 0, /*parse_special =*/ true);
        toks.insert(toks.end(), seg.begin(), seg.end());
        loss.insert(loss.end(), seg.size(), trainable ? 1 : 0);
        pos = end;
    };
    for (const auto & [b, e] : spans) {
        add(b, false);
        add(e, true);
    }
    add(full.size(), false);
    return true;
}

// "fit": "left" (continuum, card for Kimi's first dream): a served conversation (system prompt,
// tool block, history) is far longer than a training window, and only her reply carries loss. An
// example longer than the window keeps its TAIL: the window+1 tokens ending at its last trained
// token, with the leading BOS kept when the vocab adds one. A trained turn whose start is cut off
// is masked out (a reply learned without its opening teaches the wrong thing), and an example
// whose LAST trained turn cannot fit whole is skipped and counted. Without "fit" a long example
// is refused, as before.
bool train_fit_left(std::vector<llama_token> & toks, std::vector<uint8_t> & loss, int64_t window, bool bos) {
    const int64_t cap = window + 1;
    int64_t last = (int64_t) loss.size() - 1;
    while (last >= 0 && loss[last] == 0) {
        --last;
    }
    int64_t first = last; // the first token of the last trained turn
    while (first > 0 && loss[first - 1] == 1) {
        --first;
    }
    const int64_t stop  = last + 1;
    const int64_t keep  = bos ? cap - 1 : cap;
    const int64_t start = std::max<int64_t>(bos ? 1 : 0, stop - keep);
    if (last < 0 || first < start) {
        return false; // the last trained turn alone does not fit
    }
    std::vector<llama_token> t;
    std::vector<uint8_t>     l;
    if (bos) {
        t.push_back(toks[0]);
        l.push_back(0);
    }
    t.insert(t.end(), toks.begin() + start, toks.begin() + stop);
    l.insert(l.end(), loss.begin() + start, loss.begin() + stop);
    // a trained turn cut at the front: mask what remains of it
    for (size_t k = bos ? 1 : 0; k < l.size() && l[k] == 1 && start > 0 && loss[start - 1] == 1; ++k) {
        l[k] = 0;
    }
    toks = std::move(t);
    loss = std::move(l);
    return true;
}

// "fit": "middle" (continuum, Kimi's first dream): a served conversation (system prompt, tool
// block, history) is longer than a training window, and only her reply carries loss. Serving
// always carries the system and tool head, so training keeps it too (Cormac on #29): the OLDEST
// history messages after the leading system messages are dropped, a tool result with the call
// that asked for it, until the rendered example fits. The last trained turn and the user message
// it answers are never dropped. Returns the index to drop from, or npos when nothing droppable is
// left (the head plus the last exchange alone overflow the window).
size_t train_fit_droppable(const std::vector<std::string> & roles, const std::vector<bool> & trained) {
    size_t head = 0;
    while (head < roles.size() && roles[head] == "system") {
        ++head;
    }
    size_t last = std::string::npos; // the last trained assistant turn
    for (size_t k = roles.size(); k-- > 0;) {
        if (trained[k] && roles[k] == "assistant") {
            last = k;
            break;
        }
    }
    if (last == std::string::npos) {
        return std::string::npos;
    }
    size_t anchor = last; // the user message that last turn answers
    while (anchor > head && roles[anchor] != "user") {
        --anchor;
    }
    return head < anchor ? head : std::string::npos;
}

static bool prepare_examples(const json & examples, const server_trainer::render_fn & render, const llama_vocab * vocab,
                             int64_t window, const std::string & fit, server_trainer::examples_data & out, std::string & why) {
    for (size_t i = 0; i < examples.size(); ++i) {
        const std::string at = "examples[" + std::to_string(i) + "]";
        const json & ex = examples[i];
        json messages;
        if (ex.is_object() && ex.contains("messages") && ex.at("messages").is_array() && !ex.at("messages").empty()) {
            messages = ex.at("messages");
        } else if (ex.is_object() && ex.contains("prompt") && ex.at("prompt").is_string()
                   && ex.contains("completion") && ex.at("completion").is_string()) {
            messages = json::array({ json::object({{"role", "user"},      {"content", ex.at("prompt")}}),
                                     json::object({{"role", "assistant"}, {"content", ex.at("completion")}}) });
        } else {
            why = at + " needs \"prompt\" and \"completion\" strings, or a non-empty \"messages\" array";
            return false;
        }
        // "tools": the tool definitions the turn was served with, so the rendered prompt is the
        // one the model saw (a served turn's system block lists them).
        // A "tools" that is not an array is refused, like a non-boolean "train": ignoring it
        // would train a prompt without the tool block the turn was served with (Cormac on #22).
        if (ex.is_object() && ex.contains("tools") && !ex.at("tools").is_array()) {
            why = at + ".tools must be an array";
            return false;
        }
        const json tools = ex.is_object() && ex.contains("tools") ? ex.at("tools") : json::array();
        // "train": false on a message keeps an assistant turn as CONTEXT without training it:
        // a lived turn carries her earlier replies as history, and those are not this lesson.
        // The key is ours, so it is read here and removed before the template sees the message.
        std::vector<bool> trained(messages.size(), true);
        for (size_t k = 0; k < messages.size(); ++k) {
            if (messages[k].is_object() && messages[k].contains("train")) {
                if (!messages[k].at("train").is_boolean()) {
                    why = at + ".messages[" + std::to_string(k) + "].train must be a boolean";
                    return false;
                }
                trained[k] = messages[k].at("train").get<bool>();
                messages[k].erase("train");
            }
        }
        std::vector<llama_token> toks;
        std::vector<uint8_t>     loss;
        if (!render_example(messages, trained, tools, render, vocab, at, toks, loss, why)) {
            return false;
        }
        bool cut = false;
        bool skip = false;
        if (fit == "left" && (int64_t) toks.size() > window + 1) {
            const bool bos = llama_vocab_get_add_bos(vocab) && !toks.empty() && toks[0] == llama_vocab_bos(vocab);
            if (!train_fit_left(toks, loss, window, bos)) {
                ++out.skipped;
                continue;
            }
            ++out.truncated;
        }
        while ((int64_t) toks.size() > window + 1) {
            if (fit.empty()) {
                why = at + " is " + std::to_string(toks.size()) + " tokens; one example is one window, and this window holds "
                    + std::to_string(window + 1) + " (raise \"window\", split the example, or send \"fit\")";
                return false;
            }
            std::vector<std::string> roles;
            for (const auto & m : messages) {
                roles.push_back(m.is_object() ? m.value("role", std::string()) : std::string());
            }
            const size_t d = train_fit_droppable(roles, trained);
            if (d == std::string::npos) {
                skip = true; // the head plus her last exchange alone overflow the window
                break;
            }
            // drop the oldest WHOLE exchange: from d up to the next user message, so the user
            // prompt, her reply, its tool calls and their results leave together and the
            // conversation keeps the role alternation strict templates enforce (Codex on #29)
            do {
                messages.erase(d); // a JSON array erases by index
                trained.erase(trained.begin() + d);
            } while (d < messages.size() && messages[d].is_object() && messages[d].value("role", std::string()) != "user");
            cut = true;
            if (!render_example(messages, trained, tools, render, vocab, at, toks, loss, why)) {
                return false;
            }
        }
        if (skip) {
            ++out.skipped;
            continue;
        }
        if (std::count(loss.begin() + 1, loss.end(), (uint8_t) 1) == 0) {
            why = at + ": its assistant turns tokenize to nothing to learn";
            return false;
        }
        out.truncated += cut ? 1 : 0;
        out.tokens.push_back(std::move(toks));
        out.loss.push_back(std::move(loss));
    }
    if (out.tokens.empty()) {
        why = "no example fits the window (" + std::to_string(window + 1)
            + " tokens) even with its oldest history dropped: the system and tool head plus her last exchange overflow it; raise \"window\"";
        return false;
    }
    return true;
}

json server_trainer::start(const json & body_in) {
    json body = body_in;
    examples_data prepared;
    // OFF unless the server was started with --train-dir: the route writes files, so an engine
    // opts in explicitly and every write is confined to that directory (Cormac on #14).
    if (params_base.train_dir.empty()) {
        return json::object({{"ok", false}, {"error", "training is disabled on this server (start it with --train-dir DIR to enable /train)"}});
    }
    // Every numeric input is checked first, before "examples" are prepared against "window" and
    // before a thread starts: inside a serving process an
    // assert in the training path would take the server down (Cormac on #14).
    {
        auto num = [&](const char * key, double def) { return body.contains(key) && body.at(key).is_number() ? body.at(key).get<double>() : def; };
        for (const char * key : {"rank", "alpha", "window", "epochs", "lr", "val_split", "seed", "memory_budget_mib", "top_layers", "share_ppm", "max_slowdown_ppm"}) {
            if (body.contains(key) && !body.at(key).is_number()) {
                return json::object({{"ok", false}, {"error", std::string("\"") + key + "\" must be a number"}});
            }
        }
        const double rank = num("rank", 8), alpha = num("alpha", 16), window = num("window", 256), epochs = num("epochs", 1);
        const double lr = num("lr", 1e-5), val = num("val_split", 0.1);
        std::string why;
        if (rank < 1 || rank > 256 || rank != (int64_t) rank)       why = "rank must be an integer in [1, 256]";
        else if (!(alpha > 0))                                       why = "alpha must be > 0";
        // a context rounds n_ctx up to a multiple of 256, and training needs the window to BE the
        // context (one ubatch per context): any other window reached a GGML_ASSERT in opt_init and
        // took the serving process down with it
        // No fixed ceiling (Joel, 2026-09-28: a training window is the same as inference, and huge):
        // the bound is the model's own trained context, and the memory gate is the measured graph
        // against the caller's budget, checked before anything is allocated.
        else if (window < 256 || window > (double) llama_model_n_ctx_train(model) || window != (int64_t) window || (int64_t) window % 256 != 0)
                                                                     why = "window must be a multiple of 256, at least 256 and at most the model's trained context ("
                                                                         + std::to_string(llama_model_n_ctx_train(model)) + ")";
        else if (epochs < 1 || epochs > 100 || epochs != (int64_t) epochs)   why = "epochs must be an integer in [1, 100]";
        else if (!(lr > 0 && lr <= 1))                               why = "lr must be in (0, 1]";
        else if (!(val >= 0 && val < 1))                             why = "val_split must be in [0, 1)";
        else if (body.contains("memory_budget_mib") && !(num("memory_budget_mib", 0) >= 1))
                                                                     why = "memory_budget_mib must be >= 1";
        else if (body.contains("share_ppm") && !(num("share_ppm", 0) >= 1 && num("share_ppm", 0) <= 1000000 && num("share_ppm", 0) == (int64_t) num("share_ppm", 0)))
                                                                     why = "share_ppm must be an integer in [1, 1000000]: the trainer's share of the lane's time, in parts per million";
        else if (body.contains("max_slowdown_ppm") && !(num("max_slowdown_ppm", 0) >= 1 && num("max_slowdown_ppm", 0) <= 999999 &&
                                                        num("max_slowdown_ppm", 0) == (int64_t) num("max_slowdown_ppm", 0)))
                                                                     why = "max_slowdown_ppm must be an integer in [1, 999999]: how much slower her decoding may run while training runs, in parts per million";
        else if (body.contains("max_slowdown_ppm") && body.contains("share_ppm"))
                                                                     why = "max_slowdown_ppm and share_ppm are two pacings of the same windows: send one";
        else if (body.contains("top_layers") && !(num("top_layers", 0) >= 1 && num("top_layers", 0) <= llama_model_n_layer(model) &&
                                                  num("top_layers", 0) == (int64_t) num("top_layers", 0)))
                                                                     why = "top_layers must be an integer in [1, " + std::to_string(llama_model_n_layer(model)) + "]";
        else if (body.contains("targets") && (!body.at("targets").is_string() || split_targets(body.at("targets").get<std::string>()).empty()))
                                                                     why = "targets must be a non-empty comma-separated list of module names";
        if (!why.empty()) {
            return json::object({{"ok", false}, {"error", why}});
        }
    }
    if (body.contains("examples")) {
        if (body.contains("text")) {
            return json::object({{"ok", false}, {"error", "give \"text\" or \"examples\", not both"}});
        }
        if (!body.at("examples").is_array() || body.at("examples").empty()) {
            return json::object({{"ok", false}, {"error", "\"examples\" must be a non-empty array"}});
        }
        if (body.contains("parse_special") && !(body.at("parse_special").is_boolean() && body.at("parse_special").get<bool>())) {
            return json::object({{"ok", false}, {"error", "\"examples\" are rendered through the chat template, so their control tokens are always parsed (drop \"parse_special\")"}});
        }
        if (!render_chat) {
            return json::object({{"ok", false}, {"error", "this server has no chat template to render \"examples\" with; send \"text\""}});
        }
        const std::string fit = body.contains("fit") && body.at("fit").is_string() ? body.at("fit").get<std::string>() : std::string();
        if (body.contains("fit") && fit != "middle" && fit != "left") {
            return json::object({{"ok", false}, {"error",
                "\"fit\" must be \"middle\" (keep the system and tool head and her reply, drop the oldest history), "
                "\"left\" (keep the tail that ends her reply), or absent (refuse a long example)"}});
        }
        std::string why;
        if (!prepare_examples(body.at("examples"), render_chat, llama_model_get_vocab(model),
                              (int64_t) body.value("window", 256.0), fit, prepared, why)) {
            return json::object({{"ok", false}, {"error", why}});
        }
        // The GRAPH is sized to the data, the ceiling to serving (Codex on continuum #4498): the
        // caller's window is the served context an example may use whole, but the training
        // context holds activations for every position of its window, so a 61,696-token window
        // around 15k-token examples reserves four times the memory any step touches. The
        // window used is the longest fitted example rounded up to the context granularity,
        // never above what was asked; status reports both.
        {
            size_t longest = 0;
            for (const auto & t : prepared.tokens) {
                longest = std::max(longest, t.size());
            }
            const int64_t asked = (int64_t) body.value("window", 256.0);
            const int64_t need  = std::max<int64_t>(256, (((int64_t) longest - 1) + 255) / 256 * 256);
            body["window_asked"] = asked;
            if (need < asked) {
                body["window"] = need;
            }
        }
        body["n_examples"] = prepared.tokens.size();
        body["n_truncated"] = prepared.truncated;
        body["n_skipped"] = prepared.skipped;
        body.erase("examples");
    }
    if (prepared.tokens.empty() && (!body.contains("text") || !body.at("text").is_string() || body.at("text").get<std::string>().empty())) {
        return json::object({{"ok", false}, {"error", "\"text\" or \"examples\" (the training corpus) is required"}});
    }
    if (!body.contains("out") || !body.at("out").is_string() || body.at("out").get<std::string>().empty()) {
        return json::object({{"ok", false}, {"error", "\"out\" (the adapter file name to write) is required"}});
    }
    if (confine_out(params_base.train_dir, body.at("out").get<std::string>()).empty()) {
        return json::object({{"ok", false}, {"error", "\"out\" must be a bare file name ending in .gguf (no directories, no \"..\"); it is written inside the server's --train-dir"}});
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
    std::unique_lock<std::mutex> lock(mu);
    bool expected = false;
    if (!running.compare_exchange_strong(expected, true)) {
        return json::object({{"ok", false}, {"error", "a training run is already in progress"}});
    }
    if (worker.joinable()) {
        worker.join();
    }
    state = json::object({{"state", "starting"}, {"out", body.at("out")}, {"epochs", json::array()}});
    state["window"] = body.value("window", (int64_t) 256);
    if (body.contains("window_asked")) {
        state["window_asked"] = body.at("window_asked");
    }
    if (body.contains("n_examples")) {
        state["examples"] = body.at("n_examples");
        state["examples_truncated"] = body.value("n_truncated", (int64_t) 0);
        state["examples_skipped"] = body.value("n_skipped", (int64_t) 0);
    }
    pause_requested = false;
    paused = false;
    waiting_for_serving = false;
    finishing = false;
    g_train_batch.store(0);
    g_train_batch_max.store(0);
    yielded_ms.store(0);
    cancel_requested.store(false);
    yield_to_turns.store(body.value("yield", true));
    share_ppm.store(body.value("share_ppm", (int64_t) 250000));
    max_slowdown_ppm.store(body.value("max_slowdown_ppm", (int64_t) 0));
    busy_window_ms = busy_window_tokens = busy_yield_ms = busy_yield_tokens = 0;
    window_spans.clear();
    turn_slowdown_pct.fill(0);
    turns_overlapped = 0;
    turn_slowdown_ppm_max = 0;
    rate_no_window = rate_estimate{};
    rate_in_window = rate_estimate{};
    last_no_window_sample = std::chrono::steady_clock::time_point{};
    window_tokens_start = 0;
    window_busy_start = false;
    window_ms_last.store(0);
    windows.store(0);
    windows_while_busy.store(0);
    window_started = std::chrono::steady_clock::time_point{};
    window_ms_samples.clear();
    window_ms_max = 0;
    worker = std::thread([this, body, ex = std::move(prepared)]() mutable { run(std::move(body), std::move(ex)); });
    lock.unlock();
    return json::object({{"ok", true}, {"status", status()}});
}

int64_t server_trainer::yield_for_slowdown(int64_t d_ms, int64_t max_slowdown_ppm, double r0, double r_w) {
    if (d_ms <= 0) {
        return 0;
    }
    const double t = (double) std::clamp<int64_t>(max_slowdown_ppm, 1, 999999) / 1e6;
    if (!(r0 > 0)) {
        return (int64_t) std::ceil((double) d_ms * (1 - t) / t);
    }
    const double y = (double) d_ms * (r0 * (1 - t) - std::max(r_w, 0.0)) / (r0 * t);
    return y > 0 ? (int64_t) std::ceil(y) : 0;
}

void server_trainer::on_turn(int64_t gen_start_us, int64_t gen_end_us, int64_t steps, int64_t oldest_open_us) {
    std::lock_guard<std::mutex> lock(mu);
    // pruned on every finished turn, measured or not: the turns still in flight are the only
    // ones a span can still be asked about, and none of them began before oldest_open_us
    window_spans.erase(std::remove_if(window_spans.begin(), window_spans.end(),
                                      [oldest_open_us](const std::pair<int64_t, int64_t> & w) { return w.second != 0 && w.second < oldest_open_us; }),
                       window_spans.end());
    if (steps < TURN_MIN_STEPS || gen_end_us <= gen_start_us) {
        return;
    }
    int64_t overlap_us = 0;
    for (const auto & [start, end] : window_spans) {
        const int64_t e = end == 0 ? gen_end_us : end; // the window still running
        overlap_us += std::max<int64_t>(0, std::min(e, gen_end_us) - std::max(start, gen_start_us));
    }
    const double ms = (double) (gen_end_us - gen_start_us) / 1000.0;
    if (overlap_us == 0) {
        turn_rate_clean.add(ms, (double) steps);
        turns_clean += 1;
        return;
    }
    const double clean = turn_rate_clean.per_ms();
    if (clean > 0) { // no clean turn yet: nothing to measure her against, never a guessed sample
        // the WHOLE turn's rate: a turn a window overlapped for 20% of its span reads diluted,
        // which is what she felt over that turn, not the in-window slowdown
        const double  slowdown = std::clamp(1.0 - ((double) steps / ms) / clean, 0.0, 1.0);
        const int64_t ppm      = (int64_t) std::llround(slowdown * 1e6);
        turn_slowdown_pct[(size_t) std::llround(slowdown * 100)] += 1;
        turns_overlapped += 1;
        turn_slowdown_ppm_max = std::max(turn_slowdown_ppm_max, ppm);
    }
}

bool server_trainer::serving_busy() const {
    if (!yield_to_turns.load() || !serving.busy_slots) {
        return false;
    }
    // A turn ending is not serving going idle: an agent's next request lands milliseconds later
    // (a tool result, the next act). A window taken in that gap holds her next turn for the
    // whole window (measured on the 0.8B: 8.6 s windows taken between two requests).
    return serving.busy_slots() > 0 || (serving.idle_ms && serving.idle_ms() < IDLE_RELEASE_MS);
}

bool server_trainer::before_window(bool, void * user_data) {
    auto & self = *static_cast<server_trainer *>(user_data);
    const auto t0 = std::chrono::steady_clock::now();
    const int64_t tokens_t0 = self.serving.tokens_generated ? self.serving.tokens_generated() : 0;
    // The previous window's cost: the time since this callback last returned. The yield is
    // owed against it (nothing is owed before the first window: it must run to be measured).
    const int64_t last_ms = self.window_started == std::chrono::steady_clock::time_point{}
        ? 0
        : std::chrono::duration_cast<std::chrono::milliseconds>(t0 - self.window_started).count();
    self.window_ms_last.store(last_ms);
    bool took_while_busy = false;
    std::unique_lock<std::mutex> lock(self.mu);
    // a slot WORKING (not the idle grace): her rate is only measured while she is decoding
    const bool working_t0 = self.serving.busy_slots && self.serving.busy_slots() > 0;
    if (last_ms > 0) { // 0 = no window yet (the first call), never a sample
        self.window_ms_samples.push_back(last_ms);
        self.window_ms_max = std::max(self.window_ms_max, last_ms);
        if (!self.window_spans.empty() && self.window_spans.back().second == 0) {
            self.window_spans.back().second = ggml_time_us();
        }
        if (self.window_busy_start && working_t0) { // her rate WITH a window running
            self.busy_window_ms += last_ms;
            self.busy_window_tokens += tokens_t0 - self.window_tokens_start;
            self.rate_in_window.add((double) last_ms, (double) (tokens_t0 - self.window_tokens_start));
        }
    }
    const int64_t slowdown = self.max_slowdown_ppm.load();
    // The yield owed for the window just taken, from her rates as measured SO FAR: recomputed
    // on every look, so the first yield (no rate yet: the worst case) shortens as soon as her
    // rate without a window has been seen.
    // decided once, at the window boundary: the samples this yield takes must not cancel it
    const bool probe = slowdown > 0 && last_ms > 0 &&
                       (self.last_no_window_sample == std::chrono::steady_clock::time_point{} ||
                        std::chrono::duration_cast<std::chrono::milliseconds>(t0 - self.last_no_window_sample).count() > PROBE_EVERY_MS);
    const auto owed = [&]() -> int64_t {
        if (slowdown > 0) {
            const int64_t y = yield_for_slowdown(last_ms, slowdown, self.rate_no_window.per_ms(), self.rate_in_window.per_ms());
            return probe ? std::max(y, PROBE_MS) : y;
        }
        const int64_t share = std::clamp<int64_t>(self.share_ppm.load(), 1, 1000000);
        // d * (1 - s) / s, in ms: the yield owed to serving for the window just taken.
        return last_ms * (1000000 - share) / share;
    };
    // Her rate with no window running, sampled between consecutive looks that both saw a slot
    // working: any busy stretch of a yield counts, not only a yield busy end to end (bursty
    // turns would otherwise never yield a sample).
    auto    prev_t       = t0;
    int64_t prev_tokens  = tokens_t0;
    bool    prev_working = working_t0;
    int64_t sampled_ms   = 0; // this yield's measured stretch of her rate with no window
    while (!self.cancel_requested.load()) {
        self.paused = self.pause_requested;
        const auto    now     = std::chrono::steady_clock::now();
        const int64_t tokens  = self.serving.tokens_generated ? self.serving.tokens_generated() : 0;
        const bool    working = self.serving.busy_slots && self.serving.busy_slots() > 0;
        if (prev_working && working && !self.paused) {
            const int64_t d_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - prev_t).count();
            self.busy_yield_ms += d_ms;
            self.busy_yield_tokens += tokens - prev_tokens;
            self.rate_no_window.add((double) d_ms, (double) (tokens - prev_tokens));
            sampled_ms += d_ms;
        }
        prev_t = now;
        prev_tokens = tokens;
        prev_working = working;
        const bool busy = self.serving_busy();
        const int64_t yielded_so_far = std::chrono::duration_cast<std::chrono::milliseconds>(now - t0).count();
        // Busy serving holds the trainer only until the yield is paid; then the window is taken.
        self.waiting_for_serving = busy && yielded_so_far < owed();
        if (!self.paused && !self.waiting_for_serving) {
            took_while_busy = busy;
            break;
        }
        if (self.paused) {
            self.control_changed.wait(lock, [&self] { return !self.pause_requested || self.cancel_requested.load(); });
            prev_working = false; // a pause is not a measured stretch
        } else {
            // Serving publishes an atomic count; explicit pause/resume/cancel wake this wait.
            self.control_changed.wait_for(lock, std::chrono::milliseconds(10));
        }
    }
    const auto t1 = std::chrono::steady_clock::now();
    const int64_t yield_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    const int64_t tokens_t1 = self.serving.tokens_generated ? self.serving.tokens_generated() : 0;
    // fresh evidence is a probe's worth of it, never the instant a boundary looked at the slots
    if (sampled_ms >= PROBE_MS) {
        self.last_no_window_sample = t1;
    }
    self.paused = false;
    self.waiting_for_serving = false;
    self.yielded_ms += yield_ms;
    if (!self.cancel_requested.load()) {
        self.windows += 1;
        if (took_while_busy) {
            self.windows_while_busy += 1;
        }
        self.window_started = std::chrono::steady_clock::now();
        self.window_spans.emplace_back(ggml_time_us(), 0);
        self.window_tokens_start = tokens_t1;
        self.window_busy_start = self.serving.busy_slots && self.serving.busy_slots() > 0;
    }
    return !self.cancel_requested.load();
}

void server_trainer::on_batch(bool train, ggml_opt_context_t, ggml_opt_dataset_t, ggml_opt_result_t,
                              int64_t ibatch, int64_t ibatch_max, int64_t) {
    if (!train) {
        return;
    }
    g_train_batch.store(ibatch);
    g_train_batch_max.store(ibatch_max);
}

json server_trainer::cancel() {
    std::lock_guard<std::mutex> lock(mu);
    if (!running.load() || finishing) {
        return json::object({{"ok", false}, {"error", "no training run is in progress"}});
    }
    cancel_requested.store(true);
    if (ctx_live != nullptr) {
        llama_opt_stop(ctx_live, true);
    }
    state["cancel_requested"] = true;
    control_changed.notify_all();
    return json::object({{"ok", true}});
}

json server_trainer::pause(const std::string & out) {
    std::lock_guard<std::mutex> lock(mu);
    if (out.empty() || state.value("out", std::string()) != out) {
        return json::object({{"ok", false}, {"error", "training output does not match this job"}});
    }
    if (!running.load() || cancel_requested.load() || finishing) {
        return json::object({{"ok", false}, {"error", "no pausable training run"}});
    }
    pause_requested = true;
    control_changed.notify_all();
    return json::object({{"ok", true}, {"pause_requested", true}, {"paused", paused}});
}

json server_trainer::resume(const std::string & out) {
    std::lock_guard<std::mutex> lock(mu);
    if (out.empty() || state.value("out", std::string()) != out) {
        return json::object({{"ok", false}, {"error", "training output does not match this job"}});
    }
    if (!running.load() || cancel_requested.load() || finishing) {
        return json::object({{"ok", false}, {"error", "no resumable training run"}});
    }
    pause_requested = false;
    control_changed.notify_all();
    return json::object({{"ok", true}, {"pause_requested", false}, {"paused", paused}});
}

json server_trainer::status() const {
    std::lock_guard<std::mutex> lock(mu);
    json s = state;
    if (s.value("state", "") == "running") {
        s["batch"]     = g_train_batch.load();
        s["batch_max"] = g_train_batch_max.load();
    }
    s["yielded_ms"] = yielded_ms.load();
    s["share_ppm"] = share_ppm.load();
    s["max_slowdown_ppm"] = max_slowdown_ppm.load();
    // her measured decode rates (tokens/s, busy segments only) and the slowdown they realized
    // over the run: the receipt for the bound, absent until both have a sample
    if (busy_yield_ms > 0 && busy_window_ms > 0 && busy_yield_tokens > 0) {
        const double r0  = (double) busy_yield_tokens  / (double) busy_yield_ms;
        const double r_w = (double) busy_window_tokens / (double) busy_window_ms;
        const double cycle = (double) (busy_window_tokens + busy_yield_tokens) / (double) (busy_window_ms + busy_yield_ms);
        s["decode_tps_no_window"] = r0 * 1000.0;
        s["decode_tps_in_window"] = r_w * 1000.0;
        s["slowdown_ppm_realized"] = (int64_t) std::llround(std::max(0.0, 1.0 - cycle / r0) * 1e6);
    }
    // what the bound acts on right now (the recent evidence; see rate_estimate)
    if (rate_no_window.ms > 0) {
        s["decode_tps_no_window_recent"] = rate_no_window.per_ms() * 1000.0;
    }
    if (rate_in_window.ms > 0) {
        s["decode_tps_in_window_recent"] = rate_in_window.per_ms() * 1000.0;
    }
    // the per-turn receipt: each turn that overlapped a window, against her clean turns. A
    // sample is the whole turn's slowdown, so a turn a window only partly overlapped reads
    // diluted: a p95 here is what her turns felt, never the slowdown inside a window. p50/p95
    // to the 1% bucket (a fixed histogram), max exact.
    s["turns_clean"] = turns_clean;
    s["turns_overlapped"] = turns_overlapped;
    if (turn_rate_clean.ms > 0) {
        s["turn_tps_clean"] = turn_rate_clean.per_ms() * 1000.0;
    }
    if (turns_overlapped > 0) {
        const auto pct = [&](double p) {
            const int64_t rank = std::min(turns_overlapped - 1, (int64_t) (p * (double) turns_overlapped));
            int64_t seen = 0;
            for (size_t b = 0; b < turn_slowdown_pct.size(); ++b) {
                seen += turn_slowdown_pct[b];
                if (seen > rank) {
                    // a bucket rounds to the nearest 1%, which can sit above the exact max
                    return std::min((int64_t) b * 10000, turn_slowdown_ppm_max);
                }
            }
            return (int64_t) 1000000;
        };
        s["turn_slowdown_ppm_p50"] = pct(0.50);
        s["turn_slowdown_ppm_p95"] = pct(0.95);
        s["turn_slowdown_ppm_max"] = turn_slowdown_ppm_max;
    }
    s["busy_window_ms"] = busy_window_ms;
    s["busy_yield_ms"] = busy_yield_ms;
    s["windows"] = windows.load();
    s["windows_while_busy"] = windows_while_busy.load();
    s["window_ms_last"] = window_ms_last.load();
    s["window_ms_max"] = window_ms_max;
    if (!window_ms_samples.empty()) {
        std::vector<int64_t> sorted = window_ms_samples;
        std::sort(sorted.begin(), sorted.end());
        const auto pct = [&](double p) { return sorted[std::min(sorted.size() - 1, (size_t) (p * (double) sorted.size()))]; };
        s["window_ms_p50"] = pct(0.50);
        s["window_ms_p95"] = pct(0.95);
    }
    s["pause_requested"] = pause_requested;
    s["paused"] = paused;
    s["waiting_for_serving"] = waiting_for_serving;
    return s;
}

void server_trainer::run(json req, examples_data ex) {
    auto fail = [&](const std::string & why) {
        LOG_ERR("%s: training run failed: %s\n", __func__, why.c_str());
        std::lock_guard<std::mutex> lock(mu);
        state["state"] = "error";
        state["error"] = why;
        pause_requested = false;
        paused = false;
        waiting_for_serving = false;
        close_window_span();
        running.store(false);
    };

    const bool        by_example = !ex.tokens.empty();
    const std::string text    = by_example ? std::string() : req.at("text").get<std::string>();
    const std::string out     = confine_out(params_base.train_dir, req.at("out").get<std::string>());
    const int32_t     rank    = (int32_t) req.value("rank", (int64_t) 8);
    const float       alpha   = (float) req.value("alpha", 16.0);
    const std::string targets = req.value("targets", std::string("attn_q,attn_v"));
    // 0 = every block; K = only the last K (the backward pass stops at the lowest adapted one)
    const int32_t     top     = (int32_t) req.value("top_layers", (int64_t) 0);
    const uint32_t    window  = (uint32_t) req.value("window", (int64_t) 256);
    const unsigned    epochs  = (unsigned) req.value("epochs", (int64_t) 1);
    const float       lr0     = (float) req.value("lr", 1e-5);
    const float       val     = (float) req.value("val_split", 0.1);
    const uint32_t    seed    = (uint32_t) req.value("seed", (int64_t) 42);
    const bool        special = req.value("parse_special", false);
    // what training may add on the GPU, from a caller that knows physical memory (the core's
    // governed lease); without it the driver's own free figure, which on Windows is not physical
    const size_t      budget  = (size_t) (req.value("memory_budget_mib", 0.0) * 1024.0 * 1024.0);

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
    if (const uint32_t n_ctx = llama_n_ctx(ctx); n_ctx != window) {
        // never reach opt_init's one-ubatch-per-context assert inside a serving process
        llama_free(ctx);
        fail("the training context is " + std::to_string(n_ctx) + " tokens, not the window " + std::to_string(window));
        return;
    }

    const std::string init_path = out + ".init.gguf";
    if (!common_lora_write_fresh(model, init_path, rank, alpha, split_targets(targets), seed, top)) {
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
    {
        // the depth this run ACTUALLY adapts, reported back: a caller must not infer it from
        // what it asked for (an engine without top_layers ignores the field and adapts every
        // block), so it reads the effective depth here (Codex on #27)
        const int32_t n_layer = llama_model_n_layer(model);
        std::lock_guard<std::mutex> lock(mu);
        state["n_layer"]        = n_layer;
        state["layers_adapted"] = top > 0 && top < n_layer ? top : n_layer;
    }
    float scale = 1.0f;
    if (llama_set_adapters_lora(ctx, &adapter, 1, &scale) != 0) {
        llama_adapter_lora_free(adapter);
        llama_free(ctx);
        fail("could not attach the adapter to the training context");
        return;
    }

    // text: one stream, overlapping windows, every token carries loss. examples: one window per
    // example, only the assistant turns carry loss (the rest of each window is label -1).
    int64_t n_tokens = 0;
    std::vector<int64_t> trainable_prefix; // examples: trainable targets in examples [0, i)
    ggml_opt_dataset_t dataset = nullptr;
    if (by_example) {
        trainable_prefix.push_back(0);
        for (size_t i = 0; i < ex.tokens.size(); ++i) {
            n_tokens += (int64_t) ex.tokens[i].size();
            trainable_prefix.push_back(trainable_prefix.back() + std::count(ex.loss[i].begin() + 1, ex.loss[i].end(), (uint8_t) 1));
        }
        const llama_vocab * vocab = llama_model_get_vocab(model);
        const llama_token   eos   = llama_vocab_eos(vocab);
        dataset = common_opt_dataset_init_masked(window, ex.tokens, ex.loss, eos >= 0 ? eos : 0);
    } else {
        std::vector<llama_token> tokens = common_tokenize(ctx, text, true, special);
        if ((int64_t) tokens.size() < 2 * (int64_t) window) {
            llama_adapter_lora_free(adapter);
            llama_free(ctx);
            fail("the corpus tokenizes to " + std::to_string(tokens.size()) + " tokens; at least two windows are needed");
            return;
        }
        n_tokens = (int64_t) tokens.size();
        dataset  = common_opt_dataset_init(ctx, tokens, window / 2);
    }

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
    llama_opt_set_memory_budget(ctx, budget);
    llama_opt_init(ctx, model, lopt);
    llama_opt_set_step_callback(ctx, &server_trainer::before_window, this);
    {
        // from here an epoch can run: a cancel reaches this context directly
        std::lock_guard<std::mutex> lock(mu);
        ctx_live = ctx;
        if (cancel_requested.load()) {
            llama_opt_stop(ctx, true);
        }
    }

    const int64_t ndata        = ggml_opt_dataset_ndata(dataset);
    const int64_t idata_split  = (int64_t) (ndata * (1.0f - val));
    const int64_t train_tokens = idata_split * (int64_t) window; // positions the forward runs (padding included)
    if (idata_split < 1) {
        ggml_opt_dataset_free(dataset);
        {
            std::lock_guard<std::mutex> lock(mu);
            ctx_live = nullptr;
        }
        llama_adapter_lora_free(adapter);
        llama_free(ctx);
        fail("no training windows after the validation split (" + std::to_string(ndata) + " in all); send more data or lower \"val_split\"");
        return;
    }
    // ggml-opt averages the loss over every position of a window, masked ones counting zero; the
    // reported loss is per TRAINABLE token so it reads the same as an unmasked run's
    const int64_t trainable_train = by_example ? trainable_prefix[idata_split] : train_tokens;
    const int64_t trainable_eval  = by_example ? trainable_prefix.back() - trainable_prefix[idata_split]
                                               : (ndata - idata_split) * (int64_t) window;
    // The loss is a mean over the LABELLED positions by construction (the training graph has
    // outputs only where a label exists, llama_context::opt_epoch_iter), so it is already
    // per trained token. The old mean ran over every position of the window, zeros
    // included, and was rescaled here by window / trainable; applied to the new mean that
    // multiplied a 2.21 window into a reported 141.67 (the M5, 2026-10-05). Gradients
    // carry the same change: an example's step no longer shrinks with its label density.
    const double  scale_train = trainable_train > 0 ? 1.0 : 0.0;
    const double  scale_eval  = trainable_eval  > 0 ? 1.0 : 0.0;
    ggml_opt_result_t result_train = ggml_opt_result_init();
    ggml_opt_result_t result_eval  = ggml_opt_result_init();
    {
        std::lock_guard<std::mutex> lock(mu);
        state["state"]        = "running";
        state["window"]       = window;
        state["tokens"]       = n_tokens;
        state["train_tokens"] = train_tokens;
        if (by_example) {
            state["trainable_tokens"] = trainable_train;
        }
        // the held-out trainable tokens the eval loss is measured over; 0 means no held-out
        // evaluation ran (val_split 0, or a split that rounded to no windows)
        state["eval_trainable_tokens"] = trainable_eval;
    }
    LOG_INF("%s: training on the served model: %zu tokens, window %u, %u epochs, adapter -> %s\n",
            __func__, (size_t) n_tokens, window, epochs, out.c_str());

    for (lr.epoch = 0; lr.epoch < lr.epochs && !cancel_requested.load(); ++lr.epoch) {
        const int64_t t0 = ggml_time_us();
        llama_opt_epoch(ctx, dataset, result_train, result_eval, idata_split, &server_trainer::on_batch, nullptr);
        const double seconds = (ggml_time_us() - t0) / 1e6;
        if (cancel_requested.load() || llama_opt_failed(ctx)) {
            break;
        }
        double loss_train = 0.0, unc_train = 0.0, loss_eval = 0.0, unc_eval = 0.0;
        ggml_opt_result_loss(result_train, &loss_train, &unc_train);
        ggml_opt_result_loss(result_eval,  &loss_eval,  &unc_eval);
        loss_train *= scale_train;
        loss_eval  *= scale_eval;
        const double tok_s = seconds > 0 ? train_tokens / seconds : 0.0;
        LOG_INF("%s: epoch %u: train_loss=%.5f eval_loss=%.5f seconds=%.1f train_tok_s=%.1f\n",
                __func__, lr.epoch, loss_train, loss_eval, seconds, tok_s);
        {
            std::lock_guard<std::mutex> lock(mu);
            // no held-out tokens: no eval loss (null), never a 0.0 that reads as a perfect score
            state["epochs"].push_back(json::object({{"epoch", lr.epoch}, {"train_loss", loss_train},
                                           {"eval_loss", trainable_eval > 0 ? json(loss_eval) : json(nullptr)},
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
        // Close admission to control requests before committing the result. A late cancel
        // must not report success after the worker has decided to save the adapter.
        finishing = true;
        pause_requested = false;
        paused = false;
        waiting_for_serving = false;
    }

    // a cancelled run writes nothing: a partial adapter is not a result; nor does a run whose
    // training graph did not fit (the serving slots never shared that memory, so they go on)
    const bool    cancelled = cancel_requested.load();
    const bool    no_fit    = llama_opt_failed(ctx);
    // copied before llama_free: the refusal names a node the device cannot run, when that was it
    const std::string refusal = llama_opt_failure(ctx);
    const double  graph_mib = llama_opt_graph_bytes(ctx) / 1048576.0;
    const int32_t saved     = (cancelled || no_fit) ? 0 : llama_adapter_lora_save(adapter, out.c_str());
    llama_adapter_lora_free(adapter);
    llama_free(ctx);

    std::lock_guard<std::mutex> lock(mu);
    // what the training graph needed on the GPU, measured by its own preflight (also on a refusal:
    // it is the number to size the next attempt, or a caller's admission, by)
    state["graph_mib"] = graph_mib;
    if (no_fit) {
        state["state"] = "error";
        state["error"] = !refusal.empty() ? refusal + " (window " + std::to_string(window) + ")" :
                         "the training graph did not fit in device memory at window " + std::to_string(window)
                       + "; nothing was written and serving is unaffected (use a smaller window or fewer targets)";
    } else if (cancelled) {
        state["state"] = "cancelled";
    } else if (saved != 0) {
        state["state"] = "error";
        state["error"] = "training finished but the adapter could not be written to " + out;
    } else {
        state["state"] = "done";
        state["adapter"] = out;
    }
    close_window_span();
    running.store(false);
}

void server_trainer::close_window_span() {
    // a run that ends (done, failed, cancelled) ends its last window: an open span would read
    // every later turn as overlapped by training that no longer runs
    if (!window_spans.empty() && window_spans.back().second == 0) {
        window_spans.back().second = ggml_time_us();
    }
}
