// LoRA-only training on the loaded model: the base weights stay frozen (and may stay
// quantized), only a LoRA adapter's A/B tensors are optimized, and the result is a GGUF
// LoRA that llama-server loads with --lora. One copy of the base, the same kernels as
// inference.
//
//   llama-finetune-lora -m base.gguf -f train.txt -o adapter.gguf \
//       [--lora-rank 16] [--lora-alpha 32] [--lora-targets attn_q,attn_v,...] \
//       [--lora continue.gguf] [-epochs 2] [-lr 1e-4] [-val-split 0.1]
//
// A fresh adapter starts with A random and B zero, so the model's output is unchanged
// until the first step moves B. --lora continues training an existing adapter instead.
//
// Training runs on the GPU (-ngl 999) wherever the node or the grid has one. With no GPU on
// the node and none reachable on the grid, it runs on the CPU in the background (-ngl 0):
// slower, never wrong. The CPU backend is also the PARITY ORACLE the GPU kernels are checked
// against: a CPU run on the same data and seed gives the reference loss per step.

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "ggml.h"
#include "gguf.h"

#include <clocale>
#include <cmath>
#include <cstdio>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267)  // possible loss of data
#endif

static std::vector<std::string> split_list(const std::string & list) {
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

// Write a fresh adapter for every `blk.<i>.<target>.weight` the model has: A (n_in x rank)
// uniform in +-1/sqrt(n_in) (PEFT's kaiming-uniform bound), B (rank x n_out) zero. F32,
// because the optimizer steps F32 parameters.
static bool write_fresh_adapter(const llama_model * model, const std::string & path,
                                int32_t rank, float alpha, const std::vector<std::string> & targets) {
    struct shape { std::string name; int64_t n_in; int64_t n_out; };
    std::vector<shape> shapes;
    for (int32_t il = 0; il < llama_model_n_layer(model); ++il) {
        for (const auto & target : targets) {
            const std::string name = "blk." + std::to_string(il) + "." + target + ".weight";
            const ggml_tensor * w = llama_model_get_tensor(model, name.c_str());
            if (w != nullptr && ggml_n_dims(w) == 2) {
                shapes.push_back({ name, w->ne[0], w->ne[1] });
            }
        }
    }
    if (shapes.empty()) {
        LOG_ERR("%s: none of the targets (%zu names) exist in this model\n", __func__, targets.size());
        return false;
    }

    size_t data_size = 0;
    for (const auto & s : shapes) {
        data_size += (size_t) (s.n_in * rank + rank * s.n_out) * sizeof(float);
    }
    ggml_init_params params = {
        /*.mem_size   =*/ 2 * shapes.size() * ggml_tensor_overhead() + data_size,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ false,
    };
    ggml_context * ctx = ggml_init(params);
    gguf_context * out = gguf_init_empty();

    char arch[128];
    if (llama_model_meta_val_str(model, "general.architecture", arch, sizeof(arch)) < 0) {
        LOG_ERR("%s: the model names no general.architecture\n", __func__);
        return false;
    }
    gguf_set_val_str(out, "general.type", "adapter");
    gguf_set_val_str(out, "general.architecture", arch);
    gguf_set_val_str(out, "adapter.type", "lora");
    gguf_set_val_f32(out, "adapter.lora.alpha", alpha);

    std::mt19937 rng(42);
    for (const auto & s : shapes) {
        ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, s.n_in, rank);
        ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, rank, s.n_out);
        ggml_set_name(a, (s.name + ".lora_a").c_str());
        ggml_set_name(b, (s.name + ".lora_b").c_str());
        const float bound = 1.0f / std::sqrt((float) s.n_in);
        std::uniform_real_distribution<float> uniform(-bound, bound);
        float * ad = (float *) a->data;
        for (int64_t i = 0; i < ggml_nelements(a); ++i) {
            ad[i] = uniform(rng);
        }
        std::fill_n((float *) b->data, ggml_nelements(b), 0.0f);
        gguf_add_tensor(out, a);
        gguf_add_tensor(out, b);
    }

    const bool ok = gguf_write_to_file(out, path.c_str(), false);
    LOG_INF("%s: fresh adapter: %zu target tensors, rank %d, alpha %.1f -> %s\n",
            __func__, shapes.size(), rank, (double) alpha, path.c_str());
    gguf_free(out);
    ggml_free(ctx);
    return ok;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    params.escape = false;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_FINETUNE)) {
        return 1;
    }

    // Backward through a repacked weight would read blocks in the wrong layout: the base
    // must stay in the standard layout OUT_PROD dequantizes.
    params.no_extra_bufts = true;
    // FLASH_ATTN_EXT has no backward pass.
    params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    if (params.cache_type_k != GGML_TYPE_F32) {
        LOG_INF("%s: force changing k cache type to f32 due to a lack of f16 support for OUT_PROD\n", __func__);
        params.cache_type_k = GGML_TYPE_F32;
    }
    if (params.cache_type_v != GGML_TYPE_F32) {
        LOG_INF("%s: force changing v cache type to f32 due to a lack of f16 support for OUT_PROD\n", __func__);
        params.cache_type_v = GGML_TYPE_F32;
    }
    if (params.out_file.empty()) {
        LOG_ERR("%s: -o names the adapter file to write\n", __func__);
        return 1;
    }

    // The adapter to train is attached by us, not by common_init: keep any --lora aside.
    std::string continue_from;
    if (!params.lora_adapters.empty()) {
        continue_from = params.lora_adapters.front().path;
        params.lora_adapters.clear();
    }

    llama_backend_init();
    llama_numa_init(params.numa);
    auto llama_init = common_init_from_params(params);
    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();
    if (model == NULL) {
        LOG_ERR("%s: unable to load model\n", __func__);
        return 1;
    }
    LOG_INF("\n%s\n", common_params_get_system_info(params).c_str());

    std::string adapter_path = continue_from;
    if (adapter_path.empty()) {
        adapter_path = params.out_file + ".init.gguf";
        if (!write_fresh_adapter(model, adapter_path, params.lora_train_rank, params.lora_train_alpha,
                                 split_list(params.lora_train_targets))) {
            return 1;
        }
    }
    llama_adapter_lora * adapter = llama_adapter_lora_init(model, adapter_path.c_str());
    if (adapter == nullptr) {
        LOG_ERR("%s: failed to load the adapter %s\n", __func__, adapter_path.c_str());
        return 1;
    }
    float scale = 1.0f;
    if (llama_set_adapters_lora(ctx, &adapter, 1, &scale) != 0) {
        LOG_ERR("%s: failed to attach the adapter\n", __func__);
        return 1;
    }

    std::vector<llama_token> tokens  = common_tokenize(ctx, params.prompt, true);
    ggml_opt_dataset_t       dataset = common_opt_dataset_init(ctx, tokens, llama_n_ctx(ctx) / 2);

    struct lr_opt & lr = params.lr;
    LOG_INF("-optimizer %s -lr0 %.2g -wd %.2g -lr-min %.2g -min-epochs %.2g -epochs %d -period %.2g -val %.2g\n",
            ggml_opt_optimizer_name(params.optimizer), (double) lr.lr0, (double) lr.wd, (double) lr.lr_min, (double) lr.decay_epochs,
            (unsigned) lr.epochs, (double) params.n_batch / params.n_ubatch, (double) params.val_split);

    struct llama_opt_params lopt_params{
        /*n_ctx_train     =*/0,
        /*param_filter    =*/llama_opt_param_filter_all,
        /*param_filter_ud =*/nullptr,
        /*get_opt_pars    =*/common_opt_lr_pars,
        /*get_opt_pars_ud =*/&params.lr,
        /*optimizer_type  =*/params.optimizer,
        /*adapter         =*/adapter,
    };
    llama_opt_init(ctx, model, lopt_params);

    const int64_t idata_split = ggml_opt_dataset_ndata(dataset) * (1.0f - params.val_split);

    ggml_opt_result_t result_train = ggml_opt_result_init();
    ggml_opt_result_t result_eval  = ggml_opt_result_init();

    // Tokens one epoch trains on: every training sample is a full context window.
    const int64_t train_tokens = idata_split * (int64_t) llama_n_ctx(ctx);
    for (lr.epoch = 0; lr.epoch < lr.epochs; ++lr.epoch) {
        const int64_t t_start = ggml_time_us();
        llama_opt_epoch(ctx, dataset, result_train, result_eval, idata_split,
                        ggml_opt_epoch_callback_progress_bar, ggml_opt_epoch_callback_progress_bar);
        fprintf(stderr, "\n");
        const double seconds = (ggml_time_us() - t_start) / 1e6;

        double loss_train = 0.0, unc_train = 0.0, loss_eval = 0.0, unc_eval = 0.0;
        ggml_opt_result_loss(result_train, &loss_train, &unc_train);
        ggml_opt_result_loss(result_eval,  &loss_eval,  &unc_eval);
        // One line per epoch a reader can parse. Gates: the held-out loss falls, and train
        // tokens/s is at least a third of the lane's prefill tokens/s on the same node.
        LOG_INF("epoch %u: train_loss=%.5f eval_loss=%.5f train_tokens=%lld seconds=%.1f train_tok_s=%.1f\n",
                lr.epoch, loss_train, loss_eval, (long long) train_tokens, seconds,
                seconds > 0 ? train_tokens / seconds : 0.0);

        ggml_opt_result_reset(result_train);
        ggml_opt_result_reset(result_eval);
    }
    ggml_opt_result_free(result_train);
    ggml_opt_result_free(result_eval);

    const int32_t saved = llama_adapter_lora_save(adapter, params.out_file.c_str());

    llama_backend_free();

    return saved == 0 ? 0 : 1;
}
