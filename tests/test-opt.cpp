// TODO refactor

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-opt.h"
#include "gguf.h"

#include <cmath>
#include <cinttypes>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>

#define TEST_LOG(...)       printf(__VA_ARGS__)

static bool almost_equal(const double a, const double b, const double atol) {
    return fabs(a - b) < atol;
}

constexpr int64_t ne_datapoint = 2;
constexpr int64_t ne_label     = 1;
constexpr int64_t ndata        = 6;

struct helper_ctx_data {
    std::vector<ggml_opt_dataset_t>   datasets_supervised;
    std::vector<struct ggml_tensor *> data_batch;
    std::vector<struct ggml_tensor *> labels_batch;

    ggml_opt_dataset_t       dataset_unsupervised;
    struct ggml_context    * ctx_static;
    struct ggml_context    * ctx_compute;
    struct ggml_opt_params   opt_params;
    ggml_opt_context_t       opt_ctx;
    struct ggml_tensor     * inputs;
    struct ggml_tensor     * weights;
    struct ggml_tensor     * outputs;
    ggml_backend_buffer_t    buf;
    ggml_opt_result_t        result;
    ggml_opt_result_t        result2;
};

// These default values make it easier to check optimization results vs. expected values.
static ggml_opt_optimizer_params helper_get_test_opt_pars(void * userdata) {
    ggml_opt_optimizer_params result = ggml_opt_get_default_optimizer_params(userdata);

    result.adamw.alpha = 1.0f;
    result.adamw.beta1 = 0.0f;
    result.adamw.beta2 = 0.0f;
    result.adamw.eps   = 0.0f;
    result.adamw.wd    = 0.0f;
    result.sgd.wd      = 0.0f;
    result.sgd.alpha   = 1.0f;

    return result;
}

static helper_ctx_data helper_get_ctx_data(
        enum ggml_opt_optimizer_type optim,
        ggml_backend_sched_t    backend_sched,
        ggml_backend_t          backend,
        const bool              init_opt_ctx       = true,
        const bool              optimizer_defaults = true,
        int64_t                 nbatch_logical     = 1,
        int64_t                 nbatch_physical    = 1,
        enum ggml_opt_loss_type loss_type          = GGML_OPT_LOSS_TYPE_SUM) {
    std::vector<ggml_opt_dataset_t> datasets(ndata);
    for (int64_t ndata_shard = 1; ndata_shard <= ndata; ++ndata_shard) {
        ggml_opt_dataset_t dataset = ggml_opt_dataset_init(
            GGML_TYPE_F32, GGML_TYPE_F32, ne_datapoint, ne_label, ndata, ndata_shard);

        float * data   = ggml_get_data_f32(ggml_opt_dataset_data(  dataset));
        float * labels = ggml_get_data_f32(ggml_opt_dataset_labels(dataset));

        for (int64_t idata = 0; idata < ndata; ++idata) {
            for (int64_t id = 0; id < ne_datapoint; ++id) {
                data[  idata*ne_datapoint + id] =     16*idata + id;
            }
            for (int64_t il = 0; il < ne_label;     ++il) {
                labels[idata*ne_label     + il] = 16*(16*idata + il);
            }
        }

        datasets[ndata_shard-1] = dataset;
    }

    ggml_opt_dataset_t dataset_unsupervised = ggml_opt_dataset_init(
        GGML_TYPE_F32, GGML_TYPE_F32, 1, 0, ndata, /*ndata_shard =*/ 1);

    float * data = ggml_get_data_f32(ggml_opt_dataset_data(dataset_unsupervised));

    for (int64_t idata = 0; idata < ndata; ++idata) {
        data[idata] = idata;
    }

    struct ggml_context * ctx_static;
    struct ggml_context * ctx_compute;
    {
        struct ggml_init_params params = {
            /*.mem_size   =*/ (2*ndata + 2)*ggml_tensor_overhead(),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ctx_static = ggml_init(params);
    }
    {
        struct ggml_init_params params = {
            /*.mem_size   =*/ GGML_DEFAULT_GRAPH_SIZE*ggml_tensor_overhead() + 3*ggml_graph_overhead(),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ctx_compute = ggml_init(params);
    }

    std::vector<struct ggml_tensor *>   data_batch(ndata);
    std::vector<struct ggml_tensor *> labels_batch(ndata);
    for (int64_t ndata_batch = 1; ndata_batch <= ndata; ++ndata_batch) {
        data_batch[ndata_batch-1]   = ggml_new_tensor_1d(ctx_static, GGML_TYPE_F32, ndata_batch*ne_datapoint);
        labels_batch[ndata_batch-1] = ggml_new_tensor_1d(ctx_static, GGML_TYPE_F32, ndata_batch*ne_label);
    }

    struct ggml_tensor * inputs = ggml_new_tensor_1d(ctx_static, GGML_TYPE_F32, nbatch_physical);
    ggml_set_name(inputs, "inputs");

    struct ggml_tensor * weights = ggml_new_tensor_1d(ctx_static, GGML_TYPE_F32, 1);
    ggml_set_name(weights, "weights");
    ggml_set_param(weights);

    struct ggml_tensor * intermediary = ggml_add(ctx_compute, inputs, weights);

    struct ggml_tensor * outputs = ggml_scale(ctx_compute, intermediary, 1.0f);
    ggml_set_name(outputs, "outputs");

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx_static, backend);
    const float w0 = float(ndata)/2;
    ggml_backend_tensor_set(weights, &w0, 0, sizeof(float));

    GGML_ASSERT(nbatch_logical % nbatch_physical == 0);
    const int32_t opt_period = nbatch_logical / nbatch_physical;

    struct ggml_opt_params opt_params = ggml_opt_default_params(backend_sched, loss_type);
    opt_params.ctx_compute = ctx_compute;
    opt_params.inputs      = inputs;
    opt_params.outputs     = outputs;
    opt_params.opt_period  = opt_period;
    opt_params.optimizer   = optim;
    if (!optimizer_defaults) {
        opt_params.get_opt_pars = helper_get_test_opt_pars;
    }
    GGML_ASSERT(opt_params.get_opt_pars);
    ggml_opt_context_t opt_ctx = init_opt_ctx ? ggml_opt_init(opt_params) : nullptr;
    GGML_ASSERT(!opt_ctx || ggml_opt_context_optimizer_type(opt_ctx) == opt_params.optimizer);

    ggml_opt_result_t result  = ggml_opt_result_init();
    ggml_opt_result_t result2 = ggml_opt_result_init();

    return {datasets, data_batch, labels_batch, dataset_unsupervised, ctx_static, ctx_compute, opt_params, opt_ctx, inputs, weights, outputs, buf, result, result2};
}

static void helper_free_ctx_data(struct helper_ctx_data ctx_data) {
    ggml_opt_result_free(ctx_data.result);
    ggml_opt_result_free(ctx_data.result2);
    ggml_opt_free(ctx_data.opt_ctx);
    ggml_backend_buffer_free(ctx_data.buf);
    ggml_free(ctx_data.ctx_static);
    ggml_free(ctx_data.ctx_compute);
    for (ggml_opt_dataset_t dataset : ctx_data.datasets_supervised) {
        ggml_opt_dataset_free(dataset);
    }
    ggml_opt_dataset_free(ctx_data.dataset_unsupervised);
}

static void print_ok(bool subtest_ok) {
    printf(subtest_ok ? "\033[1;32mOK\033[0m\n" : "\033[1;31mFAIL\033[0m\n");
}

static void helper_after_test(
        enum ggml_opt_optimizer_type optim,
        const char * func, const bool high_level, const std::string options,
        const std::string subtest, const bool subtest_ok, int & ntest, int & npass) {
    printf("  %s(high_level=%s%s, subtest=%s, optimizer=%s): ",
           func, high_level ? "yes" : "no", options.c_str(), subtest.c_str(), ggml_opt_optimizer_name(optim));
    print_ok(subtest_ok);
    if (subtest_ok)
        npass++;
    ntest++;
}

static void print_ok(const char * func, bool subtest_ok, int & npass, int & ntest, const char * args = "") {
    printf("  %s(%s): ", func, args);
    print_ok(subtest_ok);
    if (subtest_ok)
        npass++;
    ++ntest;
}

static std::pair<int, int> test_dataset(
    enum ggml_opt_optimizer_type optim,
    ggml_backend_sched_t backend_sched, ggml_backend_t backend, const bool shuffle) {
    int ntest = 0;
    int npass = 0;

    struct helper_ctx_data cd = helper_get_ctx_data(optim, backend_sched, backend);

    for (int64_t ndata_shard = 1; ndata_shard <= ndata; ++ndata_shard) {
        ggml_opt_dataset_t dataset = cd.datasets_supervised[ndata_shard-1];

        if (shuffle) {
            ggml_opt_dataset_shuffle(cd.opt_ctx, dataset, -1);
        }

        for (int64_t ndata_batch = 1; ndata_batch <= ndata; ++ndata_batch) {
            if (ndata_batch % ndata_shard != 0) {
                continue;
            }
            bool subtest_ok = true;

            struct ggml_tensor *   data_batch =   cd.data_batch[ndata_batch-1];
            struct ggml_tensor * labels_batch = cd.labels_batch[ndata_batch-1];

            std::vector<float>   data(ggml_nelements(  data_batch));
            std::vector<float> labels(ggml_nelements(labels_batch));

            std::vector<int64_t> idata_shuffled;
            const int64_t nbatches = ndata / ndata_batch;
            for (int64_t ibatch = 0; ibatch < nbatches; ++ibatch) {
                ggml_opt_dataset_get_batch(dataset, data_batch, labels_batch, ibatch);

                ggml_backend_tensor_get(  data_batch,   data.data(), 0, ggml_nbytes(  data_batch));
                ggml_backend_tensor_get(labels_batch, labels.data(), 0, ggml_nbytes(labels_batch));

                for (int64_t idata_batch = 0; idata_batch < ndata_batch; ++idata_batch) {
                    const int64_t idata = ibatch*ndata_batch + idata_batch;
                    const int64_t idata_found = data[idata_batch*ne_datapoint] / 16;
                    subtest_ok = subtest_ok && (shuffle || idata_found == idata);
                    idata_shuffled.push_back(idata_found);

                    for (int64_t id = 0; id < ne_datapoint; ++id) {
                        if (data[  idata_batch*ne_datapoint + id] != 16*idata_found + id) {
                            subtest_ok = false;
                        }
                    }
                    for (int64_t il = 0; il < ne_label;     ++il) {
                        if (labels[idata_batch*ne_label     + il] != 16*(16*idata_found + il)) {
                            subtest_ok = false;
                        }
                    }
                }
            }

            if (!shuffle || ndata % ndata_batch == 0) {
                const int ndata_max = (ndata / ndata_batch) * ndata_batch;

                for (int64_t idata = 0; subtest_ok && idata < ndata_max; ++idata) {
                    int ninstances = 0;
                    for (int64_t id : idata_shuffled) {
                        ninstances += id == idata;
                    }
                    if (ninstances != 1) {
                        subtest_ok = false;
                    }
                }
            }

            printf("  %s(shuffle=%s, ndata_shard=%" PRId64 ", ndata_batch=%" PRId64 "): ",
                   __func__, shuffle ? "yes" : "no", ndata_shard, ndata_batch);
            if (subtest_ok) {
                printf("\033[1;32mOK\033[0m\n");
                npass++;
            } else {
                printf("\033[1;31mFAIL\033[0m\n");
            }
            ntest++;
        }
    }

    helper_free_ctx_data(cd);

    return std::make_pair(npass, ntest);
}

static std::pair<int, int> test_grad(
    enum ggml_opt_optimizer_type optim,
    ggml_backend_sched_t backend_sched, ggml_backend_t backend) {
    int ntest = 0;
    int npass = 0;

    struct helper_ctx_data cd = helper_get_ctx_data(optim, backend_sched, backend, /*init_opt_ctx =*/ true, /*optimizer_defaults =*/ false,
    /*nbatch_logical =*/ 999999, /*nbatch_physical =*/ 1);

    std::vector<float> grad_history(ndata);
    for (int64_t idata = 0; idata < ndata; ++idata) {
        grad_history[idata] = NAN;
    }

    for (int idata = 0; idata < ndata; ++idata) {
        const float idataf = idata;
        ggml_opt_alloc(cd.opt_ctx, /*backward =*/ true);
        // leaked
        ggml_backend_tensor_set(cd.inputs, &idataf, 0, ggml_nbytes(cd.inputs));
        ggml_opt_eval(cd.opt_ctx, cd.result);
        ggml_backend_tensor_get(ggml_opt_grad_acc(cd.opt_ctx, cd.weights), grad_history.data() + idata, 0, sizeof(float));
    }

    {
        bool subtest_ok = true;
        for (int idata = 0; idata < ndata; ++idata) {
            if (grad_history[idata] != idata + 1) {
                subtest_ok = false;
            }
        }
        printf("  %s(): ", __func__);
        if (subtest_ok) {
            printf("\033[1;32mOK\033[0m\n");
            npass++;
        } else {
            printf("\033[1;31mFAIL\033[0m\n");
        }
        ntest++;
    }

    helper_free_ctx_data(cd);

    return std::make_pair(npass, ntest);
}

static void helper_after_test_forward_backward(
        enum ggml_opt_optimizer_type optim,
        const char * func, const bool high_level, const bool shuffle,
        const std::string subtest, const bool subtest_ok, int & ntest, int & npass) {
    std::string options = ", shuffle=";
    options += shuffle ? "yes" : "no";
    helper_after_test(optim, func, high_level, options, subtest, subtest_ok, ntest, npass);
}

static std::pair<int, int> test_forward_backward(
        enum ggml_opt_optimizer_type optim,
        ggml_backend_sched_t backend_sched, ggml_backend_t backend, const bool high_level, const bool shuffle) {
    int ntest = 0;
    int npass = 0;

    struct helper_ctx_data cd = helper_get_ctx_data(optim, backend_sched, backend, /*init_opt_ctx =*/ true, /*optimizer_defaults =*/ false);
    struct ggml_tensor * loss = ggml_opt_loss(cd.opt_ctx);

    std::vector<float> loss_history(ndata);
    for (int64_t idata = 0; idata < ndata; ++idata) {
        loss_history[idata] = NAN;
    }

    {
        int64_t ndata;
        ggml_opt_result_ndata(cd.result, &ndata);
        double loss;
        double loss_unc;
        ggml_opt_result_loss(cd.result, &loss, &loss_unc);
        double accuracy;
        double accuracy_unc;
        ggml_opt_result_accuracy(cd.result, &accuracy, &accuracy_unc);
        const bool subtest_ok = ndata == 0 && almost_equal(loss, 0.0, 1e-6) && std::isnan(loss_unc) && std::isnan(accuracy) && std::isnan(accuracy_unc);
        helper_after_test_forward_backward(optim, __func__, high_level, shuffle, "results_initial", subtest_ok, ntest, npass);
    }

    if (high_level) {
        ggml_opt_dataset_t dataset = cd.dataset_unsupervised;
        if (shuffle) {
            ggml_opt_dataset_shuffle(cd.opt_ctx, dataset, -1);
        }
        ggml_opt_epoch(cd.opt_ctx, dataset, nullptr, cd.result, 0, nullptr, nullptr);
    } else {
        for (int idata = 0; idata < ndata; ++idata) {
            const float idataf = idata;
            ggml_opt_alloc(cd.opt_ctx, /*backward =*/ false);
            ggml_backend_tensor_set(cd.inputs, &idataf, 0, ggml_nbytes(cd.inputs));
            ggml_opt_eval(cd.opt_ctx, cd.result);
            ggml_backend_tensor_get(loss, loss_history.data() + idata, 0, sizeof(float));
        }
    }

    {
        float weights;
        ggml_backend_tensor_get(cd.weights, &weights, 0, sizeof(float));
        const bool subtest_ok = almost_equal(weights, ndata/2, 1e-10);
        helper_after_test_forward_backward(optim, __func__, high_level, shuffle, "weights_after_forward", subtest_ok, ntest, npass);
    }
    {
        constexpr double atol = 1e-10;

        int64_t ndata;
        ggml_opt_result_ndata(cd.result, &ndata);
        bool subtest_ok = ndata == 6;

        double loss;
        double loss_unc;
        ggml_opt_result_loss(cd.result, &loss, &loss_unc);
        subtest_ok = subtest_ok && almost_equal(loss, 33.0, atol) && almost_equal(loss_unc, sqrt(3.5), atol);

        double accuracy;
        double accuracy_unc;
        ggml_opt_result_accuracy(cd.result, &accuracy, &accuracy_unc);
        subtest_ok = subtest_ok && std::isnan(accuracy) && std::isnan(accuracy_unc);

        helper_after_test_forward_backward(optim, __func__, high_level, shuffle, "results_after_forward", subtest_ok, ntest, npass);
    }

    float w0;
    ggml_backend_tensor_get(cd.weights, &w0, 0, sizeof(float));
    for (int i = 0; i < 10; ++i) {
        ggml_opt_alloc(cd.opt_ctx, /*backward =*/ true);
        // leaked.
        ggml_opt_eval(cd.opt_ctx, cd.result);
    }
    ggml_backend_tensor_set(cd.weights, &w0, 0, sizeof(float));

    ggml_opt_reset(cd.opt_ctx, /*optimizer =*/ false);
    ggml_opt_result_reset(cd.result);

    for (int64_t idata = 0; idata < ndata; ++idata) {
        loss_history[idata] = NAN;
    }

    if (high_level) {
        ggml_opt_dataset_t dataset = cd.dataset_unsupervised;
        if (shuffle) {
            ggml_opt_dataset_shuffle(cd.opt_ctx, dataset, -1);
        }
        ggml_opt_epoch(cd.opt_ctx, dataset, cd.result, nullptr, ndata, nullptr, nullptr);
    } else {
        for (int idata = 0; idata < ndata; ++idata) {
            const float idataf = idata;
            ggml_opt_alloc(cd.opt_ctx, /*backward =*/ true);
            ggml_backend_tensor_set(cd.inputs, &idataf, 0, ggml_nbytes(cd.inputs));
            ggml_opt_eval(cd.opt_ctx, cd.result);
            ggml_backend_tensor_get(loss, loss_history.data() + idata, 0, sizeof(float));
        }
    }

    {
        float weights;
        ggml_backend_tensor_get(cd.weights, &weights, 0, sizeof(float));
        const bool subtest_ok = almost_equal(weights, -ndata * 0.5, 1e-10);
        helper_after_test_forward_backward(optim, __func__, high_level, shuffle, "weights_after_forward_backward", subtest_ok, ntest, npass);
    }
    {
        int64_t ndata;
        ggml_opt_result_ndata(cd.result, &ndata);
        bool subtest_ok = ndata == 6;

        double loss;
        double loss_unc;
        ggml_opt_result_loss(cd.result, &loss, &loss_unc);
        subtest_ok = subtest_ok && almost_equal(loss, 18.0, 1e-10) && (shuffle || loss_unc == 0.0);

        double accuracy;
        double accuracy_unc;
        ggml_opt_result_accuracy(cd.result, &accuracy, &accuracy_unc);
        subtest_ok = subtest_ok && std::isnan(accuracy) && std::isnan(accuracy_unc);

        helper_after_test_forward_backward(optim, __func__, high_level, shuffle, "result_after_forward_backward", subtest_ok, ntest, npass);
    }

    helper_free_ctx_data(cd);

    return std::make_pair(npass, ntest);
}

static std::pair<int, int> test_epoch_vs_fit(
    enum ggml_opt_optimizer_type optim,
    ggml_backend_sched_t backend_sched, ggml_backend_t backend) {
    int ntest = 0;
    int npass = 0;

    float weights_epoch;
    float weights_fit;

    {
        struct helper_ctx_data cd = helper_get_ctx_data(optim, backend_sched, backend, /*init_opt_ctx =*/ true);
        ggml_opt_dataset_t dataset = cd.dataset_unsupervised;

        ggml_opt_dataset_shuffle(cd.opt_ctx, dataset, -1);
        ggml_opt_epoch(cd.opt_ctx, dataset, cd.result, nullptr, ndata, nullptr, nullptr);
        // leaked.

        ggml_backend_tensor_get(cd.weights, &weights_epoch, 0, ggml_nbytes(cd.weights));
        helper_free_ctx_data(cd);
    }
    {
        struct helper_ctx_data cd = helper_get_ctx_data(optim, backend_sched, backend, /*init_opt_ctx =*/ false);
        ggml_opt_dataset_t dataset = cd.dataset_unsupervised;

        ggml_opt_fit(backend_sched, cd.ctx_compute, cd.inputs, cd.outputs, dataset, GGML_OPT_LOSS_TYPE_SUM,
                     optim, ggml_opt_get_default_optimizer_params, 1, 1, 0.0f, true);

        ggml_backend_tensor_get(cd.weights, &weights_fit, 0, ggml_nbytes(cd.weights));
        helper_free_ctx_data(cd);
    }

    const bool subtest_ok = weights_epoch == weights_fit;

    print_ok(__func__, subtest_ok, npass, ntest);

    return std::make_pair(npass, ntest);
}

static void helper_after_test_idata_split(
        enum ggml_opt_optimizer_type optim,
        const char * func, const bool high_level, const int epoch,
        const std::string subtest, const bool subtest_ok, int & ntest, int & npass) {
    std::string options = ", epoch=";
    options += std::to_string(epoch);
    helper_after_test(optim, func, high_level, options, subtest, subtest_ok, ntest, npass);
}

static std::pair<int, int> test_idata_split(
    enum ggml_opt_optimizer_type optim,
    ggml_backend_sched_t backend_sched, ggml_backend_t backend, const bool high_level) {
    int ntest = 0;
    int npass = 0;

    struct helper_ctx_data cd = helper_get_ctx_data(optim, backend_sched, backend, /*init_opt_ctx =*/ true, /*optimizer_defaults =*/ false);
    struct ggml_tensor * loss = ggml_opt_loss(cd.opt_ctx);
    const int idata_split = ndata * 2/3;

    std::vector<float> loss_history(ndata);
    for (int64_t idata = 0; idata < ndata; ++idata) {
        loss_history[idata] = NAN;
    }

    bool const adamw = optim == GGML_OPT_OPTIMIZER_TYPE_ADAMW;
    for (int epoch = 1; epoch <= 4; ++epoch) {
        if (high_level) {
            ggml_opt_epoch(cd.opt_ctx, cd.dataset_unsupervised, cd.result, cd.result2, idata_split, nullptr, nullptr);
        } else {
            int idata = 0;
            for (; idata < idata_split; ++idata) {
                const float idataf = idata;
                ggml_opt_alloc(cd.opt_ctx, /*backward =*/ true);
                ggml_backend_tensor_set(cd.inputs, &idataf, 0, ggml_nbytes(cd.inputs));
                ggml_opt_eval(cd.opt_ctx, cd.result);
                ggml_backend_tensor_get(loss, loss_history.data() + idata, 0, sizeof(float));
            }
            for (; idata < ndata; ++idata) {
                const float idataf = idata;
                ggml_opt_alloc(cd.opt_ctx, /*backward =*/ false);
                ggml_backend_tensor_set(cd.inputs, &idataf, 0, ggml_nbytes(cd.inputs));
                ggml_opt_eval(cd.opt_ctx, cd.result2);
                ggml_backend_tensor_get(loss, loss_history.data() + idata, 0, sizeof(float));
            }
        }

        if (adamw) {
            float weights;
            ggml_backend_tensor_get(cd.weights, &weights, 0, sizeof(float));
            const bool subtest_ok = almost_equal(weights, ndata/2 - epoch*idata_split, 1e-10);
            helper_after_test_idata_split(optim, __func__, high_level, epoch, "weights", subtest_ok, ntest, npass);
        }
        if (adamw) {
            constexpr double atol = 1e-10;

            int64_t ndata_result;
            ggml_opt_result_ndata(cd.result, &ndata_result);
            bool subtest_ok = ndata_result == idata_split;

            double loss;
            double loss_unc;
            ggml_opt_result_loss(cd.result, &loss, &loss_unc);
            subtest_ok = subtest_ok && almost_equal(loss, 28.0 - epoch*16.0, atol) && almost_equal(loss_unc, 0.0, atol);

            double accuracy;
            double accuracy_unc;
            ggml_opt_result_accuracy(cd.result, &accuracy, &accuracy_unc);
            subtest_ok = subtest_ok && std::isnan(accuracy) && std::isnan(accuracy_unc);

            helper_after_test_idata_split(optim, __func__, high_level, epoch, "results_backward", subtest_ok, ntest, npass);
        }
        if (adamw) {
            constexpr double atol = 1e-10;

            int64_t ndata_result;
            ggml_opt_result_ndata(cd.result2, &ndata_result);
            bool subtest_ok = ndata_result == ndata - idata_split;

            double loss;
            double loss_unc;
            ggml_opt_result_loss(cd.result2, &loss, &loss_unc);
            subtest_ok = subtest_ok && almost_equal(loss, 15.0 - epoch*8, atol) && almost_equal(loss_unc, sqrt(0.5), atol);

            double accuracy;
            double accuracy_unc;
            ggml_opt_result_accuracy(cd.result2, &accuracy, &accuracy_unc);
            subtest_ok = subtest_ok && std::isnan(accuracy) && std::isnan(accuracy_unc);

            helper_after_test_idata_split(optim, __func__, high_level, epoch, "results_forward", subtest_ok, ntest, npass);
        }

        ggml_opt_result_reset(cd.result);
        ggml_opt_result_reset(cd.result2);
    }

    helper_free_ctx_data(cd);

    return std::make_pair(npass, ntest);
}

static void helper_after_test_gradient_accumulation(
        enum ggml_opt_optimizer_type optim,
        const char * func, const int nbatch_physical, const enum ggml_opt_loss_type loss_type, const int epoch,
        const std::string subtest, const bool subtest_ok, int & ntest, int & npass) {
    std::string options = ", nbatch_physical=";
    options += std::to_string(nbatch_physical);
    options += ", loss_type=";
    options += loss_type == GGML_OPT_LOSS_TYPE_MEAN ? "mean" : "sum";
    options += ", epoch=";
    options += std::to_string(epoch);
    helper_after_test(optim, func, false, options, subtest, subtest_ok, ntest, npass);
}

static std::pair<int, int> test_gradient_accumulation(
        enum ggml_opt_optimizer_type optim,
        ggml_backend_sched_t backend_sched, ggml_backend_t backend, const int32_t nbatch_physical, const enum ggml_opt_loss_type loss_type) {
    int ntest = 0;
    int npass = 0;

    struct helper_ctx_data cd = helper_get_ctx_data(
        optim,
        backend_sched, backend, /*init_opt_ctx =*/ true, /*optimizer_defaults =*/ false, /*nbatch_logical =*/ 6, nbatch_physical, loss_type);

    std::vector<float> grad_history(ndata);
    for (int64_t idata = 0; idata < ndata; ++idata) {
        grad_history[idata] = NAN;
    }

    bool const adamw = optim == GGML_OPT_OPTIMIZER_TYPE_ADAMW;
    if (adamw)
    for (int epoch = 1; epoch <= 4; ++epoch) {
        if (nbatch_physical == 1) {
            for (int idata = 0; idata < ndata; ++idata) {
                const float idataf = idata;
                ggml_opt_alloc(cd.opt_ctx, /*backward =*/ true);
                ggml_backend_tensor_set(cd.inputs, &idataf, 0, 1*sizeof(float));
                ggml_opt_eval(cd.opt_ctx, cd.result);
                ggml_backend_tensor_get(ggml_opt_grad_acc(cd.opt_ctx, cd.weights), grad_history.data() + idata, 0, 1*sizeof(float));
            }
        } else if (nbatch_physical == 2) {
            for (int idata = 0; idata < ndata; idata += 2) {
                const float idataf[2] = {float(idata + 0), float(idata + 1)};
                ggml_opt_alloc(cd.opt_ctx, /*backward =*/ true);
                ggml_backend_tensor_set(cd.inputs, idataf, 0, 2*sizeof(float));
                ggml_opt_eval(cd.opt_ctx, cd.result);

                grad_history[idata + 0] = 0.0f;
                ggml_backend_tensor_get(ggml_opt_grad_acc(cd.opt_ctx, cd.weights), grad_history.data() + idata + 1, 0, 1*sizeof(float));
            }
        } else {
            GGML_ASSERT(false);
        }

        {
            GGML_ASSERT(ndata == 6);
            constexpr double atol = 1e-6;
            bool subtest_ok = true;
            if (loss_type == GGML_OPT_LOSS_TYPE_SUM) {
                if (nbatch_physical == 1) {
                    subtest_ok = subtest_ok && almost_equal(grad_history[0], 1.0, atol);
                    subtest_ok = subtest_ok && almost_equal(grad_history[2], 3.0, atol);
                    subtest_ok = subtest_ok && almost_equal(grad_history[4], 5.0, atol);
                } else {
                    subtest_ok = subtest_ok && almost_equal(grad_history[0], 0.0, atol);
                    subtest_ok = subtest_ok && almost_equal(grad_history[2], 0.0, atol);
                    subtest_ok = subtest_ok && almost_equal(grad_history[4], 0.0, atol);
                }
                subtest_ok = subtest_ok && almost_equal(grad_history[1], 2.0, atol);
                subtest_ok = subtest_ok && almost_equal(grad_history[3], 4.0, atol);
                subtest_ok = subtest_ok && almost_equal(grad_history[5], 6.0, atol);
            } else if (loss_type == GGML_OPT_LOSS_TYPE_MEAN) {
                if (nbatch_physical == 1) {
                    subtest_ok = subtest_ok && almost_equal(grad_history[0], 1.0/ndata, atol);
                    subtest_ok = subtest_ok && almost_equal(grad_history[2], 3.0/ndata, atol);
                    subtest_ok = subtest_ok && almost_equal(grad_history[4], 5.0/ndata, atol);
                } else {
                    subtest_ok = subtest_ok && almost_equal(grad_history[0], 0.0/ndata, atol);
                    subtest_ok = subtest_ok && almost_equal(grad_history[2], 0.0/ndata, atol);
                    subtest_ok = subtest_ok && almost_equal(grad_history[4], 0.0/ndata, atol);
                }
                subtest_ok = subtest_ok && almost_equal(grad_history[1], 2.0/ndata, atol);
                subtest_ok = subtest_ok && almost_equal(grad_history[3], 4.0/ndata, atol);
                subtest_ok = subtest_ok && almost_equal(grad_history[5], 6.0/ndata, atol);
            } else {
                GGML_ASSERT(false);
            }
            helper_after_test_gradient_accumulation(optim, __func__, nbatch_physical, loss_type, epoch, "grads", subtest_ok, ntest, npass);
        }
        bool const adamw = optim == GGML_OPT_OPTIMIZER_TYPE_ADAMW;
        if (adamw) {
            constexpr double atol = 1e-6;
            float weights;
            ggml_backend_tensor_get(cd.weights, &weights, 0, sizeof(float));
            const bool subtest_ok = almost_equal(weights, (ndata/2) - epoch, atol);
            helper_after_test_gradient_accumulation(optim, __func__, nbatch_physical, loss_type, epoch, "weights", subtest_ok, ntest, npass);
        }
        {
            constexpr double atol = 1e-6;
            int64_t ndata_result;
            ggml_opt_result_ndata(cd.result, &ndata_result);
            bool subtest_ok = almost_equal(ndata_result, ndata/nbatch_physical, atol);

            double loss;
            ggml_opt_result_loss(cd.result, &loss, /*loss_unc =*/ nullptr);
            if (loss_type == GGML_OPT_LOSS_TYPE_SUM) {
                subtest_ok = subtest_ok && almost_equal(loss, (39.0 - epoch*6.0), atol);
            } else if (loss_type == GGML_OPT_LOSS_TYPE_MEAN) {
                subtest_ok = subtest_ok && almost_equal(loss, (39.0 - epoch*6.0) / ndata, atol);
            } else {
                GGML_ASSERT(false);
            }

            double accuracy;
            double accuracy_unc;
            ggml_opt_result_accuracy(cd.result, &accuracy, &accuracy_unc);
            subtest_ok = subtest_ok && std::isnan(accuracy) && std::isnan(accuracy_unc);

            helper_after_test_gradient_accumulation(optim, __func__, nbatch_physical, loss_type, epoch, "results", subtest_ok, ntest, npass);
        }

        ggml_opt_result_reset(cd.result);
    }

    helper_free_ctx_data(cd);

    return std::make_pair(npass, ntest);
}

float constexpr g_sgd_lr = 1e-4f;

int constexpr g_sgd_epochs = 900;

static ggml_opt_optimizer_params helper_get_regression_opt_pars(void * userdata) {
    int64_t epoch = *(int64_t*)userdata;
    ggml_opt_optimizer_params result = ggml_opt_get_default_optimizer_params(nullptr);
    result.adamw.alpha = 0.1f;
    result.sgd.alpha = g_sgd_lr * std::pow(.99, 1000 * (double)epoch / g_sgd_epochs);
    result.sgd.wd = 1e-10;
    return result;
}

static std::pair<int, int> test_regression(
        enum ggml_opt_optimizer_type optim,
        ggml_backend_sched_t backend_sched, ggml_backend_t backend) {
    int ntest = 0;
    int npass = 0;

    // Test for simple regression with f(x) = a*x + b

    constexpr int64_t ndata_regression = 201;
    constexpr float a_true = 1.2f;
    constexpr float b_true = 3.4f;

    std::mt19937 gen(12345);
    std::normal_distribution<float> nd{0.0f, 0.1f};

    ggml_opt_dataset_t dataset = ggml_opt_dataset_init(
        GGML_TYPE_F32, GGML_TYPE_F32, 1, 1, ndata_regression, ndata_regression);

    float * data   = ggml_get_data_f32(ggml_opt_dataset_data(  dataset));
    float * labels = ggml_get_data_f32(ggml_opt_dataset_labels(dataset));

    constexpr float x_min = -100.0f;
    constexpr float x_max =  100.0f;

    for (int64_t idata = 0; idata < ndata_regression; ++idata) {
        const float x = x_min + (x_max - x_min) * idata/(ndata_regression-1);
        const float y = a_true*x + b_true + nd(gen);

        data[idata]   = x;
        labels[idata] = y;
    }

    struct ggml_context * ctx_static;
    struct ggml_context * ctx_compute;
    {
        struct ggml_init_params params = {
            /*.mem_size   =*/ 3*ggml_tensor_overhead(),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ctx_static = ggml_init(params);
    }
    {
        struct ggml_init_params params = {
            /*.mem_size   =*/ GGML_DEFAULT_GRAPH_SIZE*ggml_tensor_overhead() + 3*ggml_graph_overhead(),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ctx_compute = ggml_init(params);
    }

    // The first dimension is the dimension of the datapoints, the second dimension is the number of datapoints.
    struct ggml_tensor * x = ggml_new_tensor_2d(ctx_static, GGML_TYPE_F32, 1, ndata_regression);
    ggml_set_name(x, "x");

    struct ggml_tensor * a = ggml_new_tensor_1d(ctx_static, GGML_TYPE_F32, 1);
    ggml_set_name(a, "a");
    ggml_set_param(a);

    struct ggml_tensor * b = ggml_new_tensor_1d(ctx_static, GGML_TYPE_F32, 1);
    ggml_set_name(b, "b");
    ggml_set_param(b);

    struct ggml_tensor * f = ggml_add(ctx_compute, ggml_mul(ctx_compute, x, a), b);
    ggml_set_name(f, "f");

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx_static, backend);
    const float a0 = 1.0f;
    const float b0 = 3.0f;
    ggml_backend_tensor_set(a, &a0, 0, sizeof(float));
    ggml_backend_tensor_set(b, &b0, 0, sizeof(float));

    bool const adamw = optim == GGML_OPT_OPTIMIZER_TYPE_ADAMW;
    int64_t const n_epoch = adamw ? 100 : g_sgd_epochs;
    ggml_opt_fit(backend_sched, ctx_compute, x, f, dataset, GGML_OPT_LOSS_TYPE_MEAN_SQUARED_ERROR, optim,
                 helper_get_regression_opt_pars, n_epoch, ndata_regression, 0.0f, true);

    {
        float a_fit;
        ggml_backend_tensor_get(a, &a_fit, 0, sizeof(float));
        float b_fit;
        ggml_backend_tensor_get(b, &b_fit, 0, sizeof(float));
        float tol = adamw ? 1e-2 : 5e-2;
        const bool aok = almost_equal(a_fit, a_true, tol);
        const bool bok = almost_equal(b_fit, b_true, tol);
        const bool subtest_ok = aok && bok;
        print_ok(__func__, adamw ? subtest_ok : true, npass, ntest, "subtest=weights");
    }

    ggml_backend_buffer_free(buf);
    ggml_free(ctx_static);
    ggml_opt_dataset_free(dataset);

    return std::make_pair(npass, ntest);
}

static std::pair<int, int> test_checkpoint(
        ggml_opt_optimizer_type optim, ggml_backend_sched_t sched, ggml_backend_t backend) {
    int npass = 0, ntest = 0;
    auto cd = helper_get_ctx_data(optim, sched, backend);
    const auto step = [&]() {
        GGML_ASSERT(ggml_opt_alloc(cd.opt_ctx, true));
        GGML_ASSERT(!ggml_opt_can_prepare_restore(cd.opt_ctx));
        GGML_ASSERT(!ggml_opt_alloc_for_restore(cd.opt_ctx));
        const float input = 2.0f;
        ggml_backend_tensor_set(cd.inputs, &input, 0, sizeof(input));
        ggml_opt_eval(cd.opt_ctx, cd.result);
    };
    const auto snapshot = [&]() {
        FILE * copy = tmpfile();
        GGML_ASSERT(copy);
        GGML_ASSERT(ggml_opt_state_write(cd.opt_ctx, copy, "test-model/adapter/policy"));
        rewind(copy);
        std::vector<uint8_t> bytes;
        for (int ch; (ch = fgetc(copy)) != EOF;) { bytes.push_back((uint8_t) ch); }
        fclose(copy);
        return bytes;
    };
    step();
    FILE * state = tmpfile();
    GGML_ASSERT(state);
    float saved_weight;
    ggml_backend_tensor_get(cd.weights, &saved_weight, 0, sizeof(saved_weight));
    const bool written = ggml_opt_state_write(cd.opt_ctx, state, "test-model/adapter/policy");
    print_ok(__func__, written, npass, ntest, "write");
    step();
    float expected;
    ggml_backend_tensor_get(cd.weights, &expected, 0, sizeof(expected));
    const auto before_refusal = snapshot();
    rewind(state);
    print_ok(__func__, !ggml_opt_state_read(cd.opt_ctx, state, "other-binding"), npass, ntest, "binding-refusal");
    print_ok(__func__, snapshot() == before_refusal, npass, ntest, "binding-refusal-unchanged");
    rewind(state);
    ggml_opt_state * staged = ggml_opt_state_prepare(cd.opt_ctx, state, "test-model/adapter/policy");
    print_ok(__func__, staged && snapshot() == before_refusal, npass, ntest, "prepare-no-mutation");
    const bool restored = ggml_opt_state_apply(cd.opt_ctx, staged);
    ggml_opt_state_free(staged);
    print_ok(__func__, restored, npass, ntest, "restore");
    ggml_backend_tensor_set(cd.weights, &saved_weight, 0, sizeof(saved_weight));
    step();
    float actual;
    ggml_backend_tensor_get(cd.weights, &actual, 0, sizeof(actual));
    print_ok(__func__, almost_equal(expected, actual, 1e-7), npass, ntest, "next-step-equivalence");
    helper_free_ctx_data(cd);
    cd = helper_get_ctx_data(optim, sched, backend);
    float fresh_weight;
    ggml_backend_tensor_get(cd.weights, &fresh_weight, 0, sizeof(fresh_weight));
    print_ok(__func__, ggml_opt_alloc_for_restore(cd.opt_ctx), npass, ntest, "fresh-allocation-only");
    ggml_backend_tensor_get(cd.weights, &actual, 0, sizeof(actual));
    print_ok(__func__, actual == fresh_weight, npass, ntest, "allocation-keeps-weights");
    print_ok(__func__, !ggml_opt_alloc(cd.opt_ctx, true), npass, ntest, "restore-required-before-evaluation");
    FILE * uninitialized = tmpfile();
    GGML_ASSERT(uninitialized);
    print_ok(__func__, !ggml_opt_state_write(cd.opt_ctx, uninitialized, "test-model/adapter/policy"), npass, ntest, "uninitialized-export-refusal");
    fclose(uninitialized);
    rewind(state);
    print_ok(__func__, ggml_opt_state_read(cd.opt_ctx, state, "test-model/adapter/policy"), npass, ntest, "fresh-context-restore");
    ggml_backend_tensor_set(cd.weights, &saved_weight, 0, sizeof(saved_weight));
    step();
    ggml_backend_tensor_get(cd.weights, &actual, 0, sizeof(actual));
    print_ok(__func__, almost_equal(expected, actual, 1e-7), npass, ntest, "fresh-context-next-step");
    print_ok(__func__, !ggml_opt_alloc_for_restore(cd.opt_ctx), npass, ntest, "used-allocation-refusal");
    GGML_ASSERT(ggml_opt_alloc(cd.opt_ctx, true));
    rewind(state);
    print_ok(__func__, !ggml_opt_state_read(cd.opt_ctx, state, "test-model/adapter/policy"), npass, ntest, "prepared-eval-refusal");
    print_ok(__func__, !ggml_opt_state_write(cd.opt_ctx, state, "test-model/adapter/policy"), npass, ntest, "prepared-export-refusal");
    ggml_opt_eval(cd.opt_ctx, cd.result);
    using set_abort_fn = void (*)(ggml_backend_t, ggml_abort_callback, void *);
    const auto reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));
    const auto set_abort = reinterpret_cast<set_abort_fn>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_abort_callback"));
    if (set_abort) {
        set_abort(backend, [](void *) { return true; }, nullptr);
        step();
        set_abort(backend, nullptr, nullptr);
        FILE * failed = tmpfile();
        GGML_ASSERT(failed);
        print_ok(__func__, !ggml_opt_state_write(cd.opt_ctx, failed, "test-model/adapter/policy"), npass, ntest, "failed-backend-refusal");
        step();
        print_ok(__func__, !ggml_opt_state_write(cd.opt_ctx, failed, "test-model/adapter/policy"), npass, ntest, "failure-stays-untrusted");
        rewind(state);
        print_ok(__func__, ggml_opt_state_read(cd.opt_ctx, state, "test-model/adapter/policy"), npass, ntest, "known-state-recovery");
        fclose(failed);
    }
    // The serving owner supplies a dynamic graph; allocation-only must disarm that path too.
    const auto make_dynamic = [&]() {
        auto data = helper_get_ctx_data(optim, sched, backend, false);
        auto params = data.opt_params;
        params.ctx_compute = nullptr;
        params.inputs = nullptr;
        params.outputs = nullptr;
        data.opt_ctx = ggml_opt_init(params);
        return data;
    };
    auto dynamic = make_dynamic();
    ggml_cgraph * dynamic_graph = nullptr;
    const auto prepare_dynamic = [&]() {
        // Dynamic serving owns fresh graph metadata per ubatch. Release the previous graph
        // before rebuilding; parameters and optimizer state live in their separate owners.
        ggml_backend_sched_reset(sched);
        ggml_reset(dynamic.ctx_compute);
        dynamic.outputs = ggml_scale(dynamic.ctx_compute, ggml_add(dynamic.ctx_compute, dynamic.inputs, dynamic.weights), 1.0f);
        ggml_set_name(dynamic.outputs, "outputs");
        dynamic_graph = ggml_new_graph_custom(dynamic.ctx_compute, GGML_DEFAULT_GRAPH_SIZE, true);
        ggml_build_forward_expand(dynamic_graph, dynamic.outputs);
        ggml_opt_prepare_alloc(dynamic.opt_ctx, dynamic.ctx_compute, dynamic_graph, dynamic.inputs, dynamic.outputs);
    };
    const auto step_dynamic = [&]() {
        prepare_dynamic();
        GGML_ASSERT(ggml_opt_alloc(dynamic.opt_ctx, true));
        const float input = 2.0f;
        ggml_backend_tensor_set(dynamic.inputs, &input, 0, sizeof(input));
        ggml_opt_eval(dynamic.opt_ctx, dynamic.result);
    };
    float initial_weight, first_expected;
    ggml_backend_tensor_get(dynamic.weights, &initial_weight, 0, sizeof(initial_weight));
    step_dynamic();
    ggml_backend_tensor_get(dynamic.weights, &first_expected, 0, sizeof(first_expected));
    FILE * dynamic_state = tmpfile();
    GGML_ASSERT(dynamic_state && ggml_opt_state_write(dynamic.opt_ctx, dynamic_state, "dynamic-policy"));
    float dynamic_saved, dynamic_expected;
    ggml_backend_tensor_get(dynamic.weights, &dynamic_saved, 0, sizeof(dynamic_saved));
    step_dynamic();
    ggml_backend_tensor_get(dynamic.weights, &dynamic_expected, 0, sizeof(dynamic_expected));
    helper_free_ctx_data(dynamic);
    dynamic = make_dynamic();
    dynamic_graph = nullptr;
    prepare_dynamic();
    ggml_backend_tensor_get(dynamic.weights, &fresh_weight, 0, sizeof(fresh_weight));
    print_ok(__func__, ggml_opt_alloc_for_restore(dynamic.opt_ctx), npass, ntest, "dynamic-allocation-only");
    ggml_backend_tensor_get(dynamic.weights, &actual, 0, sizeof(actual));
    print_ok(__func__, actual == fresh_weight, npass, ntest, "dynamic-allocation-keeps-weights");
    rewind(state);
    print_ok(__func__, !ggml_opt_state_read(dynamic.opt_ctx, state, "test-model/adapter/policy"), npass, ntest, "static-dynamic-layout-refusal");
    print_ok(__func__, !ggml_opt_alloc(dynamic.opt_ctx, true), npass, ntest, "failed-import-stays-pending");
    rewind(dynamic_state);
    print_ok(__func__, ggml_opt_state_read(dynamic.opt_ctx, dynamic_state, "dynamic-policy"), npass, ntest, "dynamic-fresh-restore");
    ggml_backend_tensor_set(dynamic.weights, &dynamic_saved, 0, sizeof(dynamic_saved));
    step_dynamic();
    ggml_backend_tensor_get(dynamic.weights, &actual, 0, sizeof(actual));
    print_ok(__func__, almost_equal(dynamic_expected, actual, 1e-7), npass, ntest, "dynamic-restored-next-step");
    helper_free_ctx_data(dynamic);
    dynamic = make_dynamic();
    dynamic_graph = nullptr;
    prepare_dynamic();
    GGML_ASSERT(ggml_opt_alloc_for_restore(dynamic.opt_ctx));
    FILE * initial = tmpfile();
    GGML_ASSERT(initial);
    print_ok(__func__, ggml_opt_state_write_initial(dynamic.opt_ctx, initial, "dynamic-policy"), npass, ntest, "initial-generation-export");
    ggml_backend_tensor_get(dynamic.weights, &actual, 0, sizeof(actual));
    print_ok(__func__, actual == initial_weight, npass, ntest, "initial-generation-keeps-weights");
    print_ok(__func__, !ggml_opt_alloc(dynamic.opt_ctx, true), npass, ntest, "initial-export-keeps-import-gate");
    FILE * not_completed = tmpfile();
    GGML_ASSERT(not_completed);
    print_ok(__func__, !ggml_opt_state_write(dynamic.opt_ctx, not_completed, "dynamic-policy"), npass, ntest, "initial-not-completed");
    rewind(initial);
    gguf_context * initial_meta = gguf_init_from_file_ptr(initial, {true, nullptr});
    GGML_ASSERT(initial_meta);
    FILE * bad_boundary = tmpfile();
    GGML_ASSERT(bad_boundary);
    gguf_set_val_str(initial_meta, "optimizer.state.boundary", "completed");
    GGML_ASSERT(gguf_write_to_file_ptr(initial_meta, bad_boundary, true));
    rewind(bad_boundary);
    print_ok(__func__, !ggml_opt_state_read(dynamic.opt_ctx, bad_boundary, "dynamic-policy"), npass, ntest, "initial-wrong-boundary-refused");
    fclose(bad_boundary);
    gguf_free(initial_meta);
    print_ok(__func__, !ggml_opt_alloc(dynamic.opt_ctx, true), npass, ntest, "invalid-initial-import-stays-pending");
    rewind(initial);
    print_ok(__func__, ggml_opt_state_read(dynamic.opt_ctx, initial, "dynamic-policy"), npass, ntest, "initial-generation-restore");
    print_ok(__func__, !ggml_opt_state_write(dynamic.opt_ctx, not_completed, "dynamic-policy"), npass, ntest, "restored-initial-not-completed");
    step_dynamic();
    ggml_backend_tensor_get(dynamic.weights, &actual, 0, sizeof(actual));
    print_ok(__func__, almost_equal(first_expected, actual, 1e-7), npass, ntest, "initial-generation-first-update-equivalent");
    print_ok(__func__, !ggml_opt_state_write_initial(dynamic.opt_ctx, not_completed, "dynamic-policy"), npass, ntest, "used-initial-export-refused");
    // Simulate an update whose enclosing generation commit failed: restore BOTH initial
    // parameter weights and optimizer state, then retry that same first real sample.
    rewind(initial);
    print_ok(__func__, ggml_opt_state_read(dynamic.opt_ctx, initial, "dynamic-policy"), npass, ntest, "first-commit-rollback");
    ggml_backend_tensor_set(dynamic.weights, &initial_weight, 0, sizeof(initial_weight));
    step_dynamic();
    ggml_backend_tensor_get(dynamic.weights, &actual, 0, sizeof(actual));
    print_ok(__func__, almost_equal(first_expected, actual, 1e-7), npass, ntest, "initial-rollback-next-update-equivalent");
    fclose(initial);
    fclose(not_completed);
    helper_free_ctx_data(dynamic);
    fclose(dynamic_state);
    fclose(state);
    FILE * corrupt = tmpfile();
    GGML_ASSERT(corrupt);
    fputs("broken checkpoint", corrupt);
    rewind(corrupt);
    const auto before_corrupt = snapshot();
    print_ok(__func__, !ggml_opt_state_read(cd.opt_ctx, corrupt, "test-model/adapter/policy"), npass, ntest, "corrupt-refusal");
    print_ok(__func__, snapshot() == before_corrupt, npass, ntest, "corrupt-refusal-unchanged");
    fclose(corrupt);
    FILE * truncated = tmpfile();
    GGML_ASSERT(truncated && !before_corrupt.empty());
    GGML_ASSERT(fwrite(before_corrupt.data(), 1, before_corrupt.size() - 1, truncated) == before_corrupt.size() - 1);
    rewind(truncated);
    print_ok(__func__, !ggml_opt_state_read(cd.opt_ctx, truncated, "test-model/adapter/policy"), npass, ntest, "truncated-refusal");
    print_ok(__func__, snapshot() == before_corrupt, npass, ntest, "truncated-refusal-unchanged");
    fclose(truncated);
    helper_free_ctx_data(cd);
    auto partial = helper_get_ctx_data(optim, sched, backend, true, true, 3, 1);
    GGML_ASSERT(ggml_opt_alloc(partial.opt_ctx, true));
    const float input = 2.0f;
    ggml_backend_tensor_set(partial.inputs, &input, 0, sizeof(input));
    ggml_opt_eval(partial.opt_ctx, partial.result);
    FILE * unfinished = tmpfile();
    GGML_ASSERT(unfinished);
    print_ok(__func__, !ggml_opt_state_write(partial.opt_ctx, unfinished, "test-model/adapter/policy"), npass, ntest, "partial-refusal");
    ggml_opt_set_next_step(partial.opt_ctx, true, 1.0f, nullptr);
    GGML_ASSERT(ggml_opt_alloc(partial.opt_ctx, true));
    ggml_opt_eval(partial.opt_ctx, partial.result);
    print_ok(__func__, ggml_opt_state_write(partial.opt_ctx, unfinished, "manual-period-three"), npass, ntest, "manual-boundary");
    ggml_backend_tensor_get(partial.weights, &saved_weight, 0, sizeof(saved_weight));
    GGML_ASSERT(ggml_opt_alloc(partial.opt_ctx, true));
    ggml_opt_eval(partial.opt_ctx, partial.result);
    ggml_backend_tensor_get(partial.weights, &expected, 0, sizeof(expected));
    helper_free_ctx_data(partial);
    partial = helper_get_ctx_data(optim, sched, backend, true, true, 3, 1);
    print_ok(__func__, ggml_opt_alloc_for_restore(partial.opt_ctx), npass, ntest, "manual-fresh-allocation-only");
    rewind(unfinished);
    print_ok(__func__, ggml_opt_state_read(partial.opt_ctx, unfinished, "manual-period-three"), npass, ntest, "manual-fresh-restore");
    ggml_backend_tensor_set(partial.weights, &saved_weight, 0, sizeof(saved_weight));
    ggml_backend_tensor_set(partial.inputs, &input, 0, sizeof(input));
    GGML_ASSERT(ggml_opt_alloc(partial.opt_ctx, true));
    ggml_opt_eval(partial.opt_ctx, partial.result);
    ggml_backend_tensor_get(partial.weights, &actual, 0, sizeof(actual));
    print_ok(__func__, almost_equal(expected, actual, 1e-7), npass, ntest, "manual-next-step-equivalence");
    fclose(unfinished);
    helper_free_ctx_data(partial);
    return {npass, ntest};
}

static std::pair<int, int> test_backend(
    ggml_backend_sched_t backend_sched, ggml_backend_t backend, enum ggml_opt_optimizer_type optim) {
    int npass = 0;
    int ntest = 0;

    {
        const auto partial = test_checkpoint(optim, backend_sched, backend);
        npass += partial.first;
        ntest += partial.second;
    }

    for (bool shuffle : {false, true}) {
        std::pair<int, int> partial = test_dataset(optim, backend_sched, backend, shuffle);
        npass += partial.first;
        ntest += partial.second;
    }
    {
        std::pair<int, int> partial = test_grad(optim, backend_sched, backend);
        npass += partial.first;
        ntest += partial.second;
    }
    for (bool high_level : {false, true}){
        for (bool shuffle : {false, true}) {
            if (!high_level && shuffle) {
                continue;
            }

            std::pair<int, int> partial = test_forward_backward(optim, backend_sched, backend, high_level, shuffle);
            npass += partial.first;
            ntest += partial.second;
        }
    }
    {
      std::pair<int, int> partial = test_epoch_vs_fit(optim, backend_sched, backend);
        npass += partial.first;
        ntest += partial.second;
    }
    for (bool high_level : {false, true}){
        std::pair<int, int> partial = test_idata_split(optim, backend_sched, backend, high_level);
        npass += partial.first;
        ntest += partial.second;
    }
    bool const adamw = optim == GGML_OPT_OPTIMIZER_TYPE_ADAMW;
    if (adamw) {
        for (int32_t nbatch_physical : { 2, 1 }) {
            for (enum ggml_opt_loss_type loss_type : { GGML_OPT_LOSS_TYPE_SUM, GGML_OPT_LOSS_TYPE_MEAN }) {
                std::pair<int, int> partial =
                    test_gradient_accumulation(optim, backend_sched, backend, nbatch_physical, loss_type);
                npass += partial.first;
                ntest += partial.second;
            }
        }
    }
    {
        std::pair<int, int> partial = test_regression(optim, backend_sched, backend);
        npass += partial.first;
        ntest += partial.second;
    }

    return std::make_pair(npass, ntest);
}


int main(void) {
    ggml_log_set(nullptr, nullptr);
    ggml_backend_load_all();
    const size_t dev_count = ggml_backend_dev_count();
    printf("Testing %zu devices\n\n", dev_count);
    size_t n_ok = 0;

    std::vector<ggml_backend_dev_t> devs;
    std::vector<ggml_backend_t>     backends;

    for (size_t i = 0; i < dev_count; ++i) {
        devs.push_back(ggml_backend_dev_get(i));

        ggml_backend_t backend = ggml_backend_dev_init(devs[i], NULL);
        GGML_ASSERT(backend != NULL);

        auto * reg = ggml_backend_dev_backend_reg(devs[i]);
        auto ggml_backend_set_n_threads_fn = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
        if (ggml_backend_set_n_threads_fn) {
            ggml_backend_set_n_threads_fn(backend, std::thread::hardware_concurrency() / 2);
        }
        backends.push_back(backend);
    }

    size_t n_total = 0;
    for (enum ggml_opt_optimizer_type optim : { GGML_OPT_OPTIMIZER_TYPE_ADAMW, GGML_OPT_OPTIMIZER_TYPE_SGD }) {
        for (size_t i = 0; i < dev_count; ++i) {
            // Put the backend to be tested in front so that it's prioritized:
            std::vector<ggml_backend_t> backends_modded = { backends[i] };
            backends_modded.insert(backends_modded.end(), backends.begin(), backends.end());

            ggml_backend_sched_t backend_sched = ggml_backend_sched_new(
                backends_modded.data(), nullptr, backends_modded.size(), GGML_DEFAULT_GRAPH_SIZE, false, true);

            char const* devname = ggml_backend_dev_name(devs[i]);
            printf("Backend %zu/%zu: %s\n", i + 1, dev_count, devname);
            printf("  Device description: %s\n", ggml_backend_dev_description(devs[i]));
            size_t free, total;  // NOLINT
            ggml_backend_dev_memory(devs[i], &free, &total);
            printf("  Device memory: %zu MB (%zu MB free)\n", total / 1024 / 1024, free / 1024 / 1024);
            printf("\n");

            bool skip;
            {
                struct ggml_init_params params = {
                    /*.mem_size   =*/ 6*ggml_tensor_overhead(),
                    /*.mem_buffer =*/ nullptr,
                    /*.no_alloc   =*/ true,
                };
                ggml_context * ctx = ggml_init(params);
                ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
                ggml_set_param(a);
                ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
                ggml_tensor * c = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
                ggml_tensor * d = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);

                ggml_tensor * t = nullptr;
                switch (optim) {
                    case GGML_OPT_OPTIMIZER_TYPE_ADAMW: {
                        ggml_tensor * p = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 7);
                        t = ggml_opt_step_adamw(ctx, a, b, c, d, p);
                    } break;
                    case GGML_OPT_OPTIMIZER_TYPE_SGD: {
                        ggml_tensor * p = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2);
                        t = ggml_opt_step_sgd(ctx, a, b, p);
                    } break;
                    case GGML_OPT_OPTIMIZER_TYPE_COUNT: {
                        GGML_ABORT("fatal error");
                    }
                }
                skip = !ggml_backend_supports_op(backends[i], t);
                ggml_free(ctx);
            }

            std::pair<int, int> result;
            if (!skip) {
                result = test_backend(backend_sched, backends[i], optim);
                printf("  %d/%d tests passed\n", result.first, result.second);
            }

            printf("  Backend %s %s: ", ggml_backend_name(backends[i]), ggml_opt_optimizer_name(optim));
            if (skip) {
                printf("\033[0;33mSKIPPED\033[0m\n");
                n_ok++;
            } else if (result.first == result.second) {
                printf("\033[1;32mOK\033[0m\n");
                n_ok++;
            } else {
                printf("\033[1;31mFAIL\033[0m\n");
            }
            ++n_total;
            printf("\n");
            ggml_backend_sched_free(backend_sched);
        }
    }

    for (ggml_backend_t backend : backends) {
        ggml_backend_free(backend);
    }

    printf("%zu/%zu backend*optimizer passed\n", n_ok, n_total);
    bool ok = n_ok == n_total;
    print_ok(ok);
    return ok ? 0 : 1;
}
