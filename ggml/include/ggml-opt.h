// This file contains functionality for training models using GGML.
// It is not strictly needed vs. just vanilla GGML but it provides a more high-level interface for common needs such as datasets.
// At the bottom of this file especially there are relatively high-level functions that are suitable use or adaptation in user code.
//
// Module maintainer: Johannes Gäßler (@JohannesGaessler, johannesg@5d6.de)

#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <stdint.h>
#include <stdio.h>

#ifdef  __cplusplus
extern "C" {
#endif

    struct ggml_opt_dataset;
    struct ggml_opt_context;
    struct ggml_opt_result;

    typedef struct ggml_opt_dataset * ggml_opt_dataset_t;
    typedef struct ggml_opt_context * ggml_opt_context_t;
    typedef struct ggml_opt_result  * ggml_opt_result_t;

    // ====== Loss ======

    // built-in loss types, i.e. the built-in quantities minimized by the optimizer
    // custom loss types can be defined via mean or sum which simply reduce the outputs for all datapoints to a single value
    enum ggml_opt_loss_type {
        GGML_OPT_LOSS_TYPE_MEAN,
        GGML_OPT_LOSS_TYPE_SUM,
        GGML_OPT_LOSS_TYPE_CROSS_ENTROPY,
        GGML_OPT_LOSS_TYPE_MEAN_SQUARED_ERROR,
        // GGML_OPT_LOSS_TYPE_CROSS_ENTROPY with one I32 target per output row (-1: no loss) in place of a one-hot
        // row as wide as the outputs: what a language model's training step needs (its outputs are
        // vocabulary-wide logits)
        GGML_OPT_LOSS_TYPE_CROSS_ENTROPY_TARGETS,
    };

    // ====== Dataset ======

    GGML_API ggml_opt_dataset_t ggml_opt_dataset_init(
            enum ggml_type type_data,    // the type for the internal data tensor
            enum ggml_type type_label,   // the type for the internal labels tensor
            int64_t        ne_datapoint, // number of elements per datapoint
            int64_t        ne_label,     // number of elements per label
            int64_t        ndata,        // total number of datapoints/labels
            int64_t        ndata_shard); // number of datapoints/labels per shard (unit at which the dataset is shuffled/copied)
    GGML_API void ggml_opt_dataset_free(ggml_opt_dataset_t dataset);

    // get underlying tensors that store the data
    GGML_API int64_t              ggml_opt_dataset_ndata (ggml_opt_dataset_t dataset);
    GGML_API struct ggml_tensor * ggml_opt_dataset_data  (ggml_opt_dataset_t dataset); // shape = [ne_datapoint, ndata]
    GGML_API struct ggml_tensor * ggml_opt_dataset_labels(ggml_opt_dataset_t dataset); // shape = [nd_label,     ndata]

    // shuffle idata first datapoints from dataset with RNG from opt_ctx, shuffle all datapoints if idata is negative
    GGML_API void ggml_opt_dataset_shuffle(ggml_opt_context_t opt_ctx, ggml_opt_dataset_t dataset, int64_t idata);

    // get batch at position ibatch from dataset and copy the data to data_batch and labels_batch
    GGML_API void ggml_opt_dataset_get_batch(
            ggml_opt_dataset_t   dataset,
            struct ggml_tensor * data_batch,   // shape = [ne_datapoint, ndata_batch]
            struct ggml_tensor * labels_batch, // shape = [ne_label,     ndata_batch]
            int64_t              ibatch);
    GGML_API void ggml_opt_dataset_get_batch_host(
            ggml_opt_dataset_t   dataset,
            void               * data_batch,
            size_t               nb_data_batch,
            void               * labels_batch,
            int64_t              ibatch);

    // ====== Model / Context ======

    enum ggml_opt_build_type {
        GGML_OPT_BUILD_TYPE_FORWARD = 10,
        GGML_OPT_BUILD_TYPE_GRAD    = 20,
        GGML_OPT_BUILD_TYPE_OPT     = 30,
    };

    enum ggml_opt_optimizer_type {
        GGML_OPT_OPTIMIZER_TYPE_ADAMW,
        GGML_OPT_OPTIMIZER_TYPE_SGD,

        GGML_OPT_OPTIMIZER_TYPE_COUNT
    };

    // parameters that control which optimizer is used and how said optimizer tries to find the minimal loss
    struct ggml_opt_optimizer_params {
        struct {
            float alpha; // learning rate
            float beta1; // first AdamW momentum
            float beta2; // second AdamW momentum
            float eps;   // epsilon for numerical stability
            float wd;    // weight decay - 0.0f to disable
        } adamw;
        struct {
            float alpha; // learning rate
            float wd;    // weight decay
        } sgd;
    };

    // callback to calculate optimizer parameters prior to a backward pass
    // userdata can be used to pass arbitrary data
    typedef struct ggml_opt_optimizer_params (*ggml_opt_get_optimizer_params)(void * userdata);

    // returns the default optimizer params (constant, hard-coded values)
    // userdata is not used
    GGML_API struct ggml_opt_optimizer_params ggml_opt_get_default_optimizer_params(void * userdata);

    // casts userdata to ggml_opt_optimizer_params and returns it
    GGML_API struct ggml_opt_optimizer_params ggml_opt_get_constant_optimizer_params(void * userdata);

    // parameters for initializing a new optimization context
    struct ggml_opt_params {
        ggml_backend_sched_t backend_sched; // defines which backends are used to construct the compute graphs

        // by default the forward graph needs to be reconstructed for each eval
        // if ctx_compute, inputs, and outputs are set the graphs are instead allocated statically
        struct ggml_context * ctx_compute;
        struct ggml_tensor  * inputs;
        struct ggml_tensor  * outputs;

        enum ggml_opt_loss_type  loss_type;
        enum ggml_opt_build_type build_type;

        int32_t opt_period; // after how many gradient accumulation steps an optimizer step should be done

        ggml_opt_get_optimizer_params get_opt_pars;    // callback for calculating optimizer parameters
        void *                        get_opt_pars_ud; // userdata for calculating optimizer parameters

        // only GGML_OPT_OPTIMIZER_TYPE_ADAMW needs m, v momenta per parameter tensor
        enum ggml_opt_optimizer_type optimizer;

        // Recompute instead of keep (gradient checkpointing): forward nodes whose name starts
        // with this prefix are kept for the backward pass, and every other forward intermediate
        // the backward pass reads is recomputed from the nearest of them, just before it is
        // needed. NULL = keep everything (the default). One layer's intermediates are then
        // alive at a time instead of every layer's, for about one extra forward pass.
        const char * checkpoint_prefix;
    };

    // get parameters for an optimization context with defaults set where possible
    // parameters for which no sensible defaults exist are supplied as arguments to this function
    GGML_API struct ggml_opt_params ggml_opt_default_params(
            ggml_backend_sched_t    backend_sched,
            enum ggml_opt_loss_type loss_type);

    GGML_API ggml_opt_context_t ggml_opt_init(struct ggml_opt_params params);
    GGML_API void ggml_opt_free(ggml_opt_context_t opt_ctx);

    // set gradients to zero, initialize loss, and optionally reset the optimizer
    GGML_API void ggml_opt_reset(ggml_opt_context_t opt_ctx, bool optimizer);

    GGML_API bool ggml_opt_static_graphs(ggml_opt_context_t opt_ctx); // whether the graphs are allocated_statically

    // get underlying tensors that store data
    // if not using static graphs these pointers become invalid with the next call to ggml_opt_alloc
    GGML_API struct ggml_tensor * ggml_opt_inputs(  ggml_opt_context_t opt_ctx); // forward graph input tensor
    GGML_API struct ggml_tensor * ggml_opt_outputs( ggml_opt_context_t opt_ctx); // forward graph output tensor
    GGML_API struct ggml_tensor * ggml_opt_labels(  ggml_opt_context_t opt_ctx);
    // the per-row weights of a weighted cross-entropy step (F32, one per label), or NULL when the
    // graph prepared last is unweighted (ggml_opt_set_next_label_weights)
    GGML_API struct ggml_tensor * ggml_opt_label_weights(ggml_opt_context_t opt_ctx);
    // whether the next graph's cross-entropy weighs each row (by ggml_opt_label_weights); an
    // unweighted step keeps the unweighted loss, whose kernels every backend has
    GGML_API void ggml_opt_set_next_label_weights(ggml_opt_context_t opt_ctx, bool weighted); // labels to compare outputs against
    GGML_API struct ggml_tensor * ggml_opt_loss(    ggml_opt_context_t opt_ctx); // scalar tensor that contains the loss
    GGML_API struct ggml_tensor * ggml_opt_pred(    ggml_opt_context_t opt_ctx); // predictions made by outputs
    GGML_API struct ggml_tensor * ggml_opt_ncorrect(ggml_opt_context_t opt_ctx); // number of matching predictions between outputs and labels

    // get the gradient accumulator for a node from the forward graph
    GGML_API struct ggml_tensor * ggml_opt_grad_acc(ggml_opt_context_t opt_ctx, struct ggml_tensor * node);

    GGML_API enum ggml_opt_optimizer_type ggml_opt_context_optimizer_type(ggml_opt_context_t); //TODO consistent naming scheme

    GGML_API const char * ggml_opt_optimizer_name(enum ggml_opt_optimizer_type);

    // Checkpoint optimizer state at a successful update boundary. Parameters are saved separately.
    // binding identifies the caller's model, adapter topology and optimizer policy; it must match exactly.
    // Mutable callback/schedule state must be saved by the caller in the same generation, not inferred from a callback address.
    // Import requires allocated optimizer tensors and no prepared evaluation; it validates all input before changing state.
    // The caller owns file integrity, atomic publication and serialization with compute.
    GGML_API bool ggml_opt_state_write(ggml_opt_context_t opt_ctx, FILE * file, const char * binding);
    GGML_API bool ggml_opt_state_read (ggml_opt_context_t opt_ctx, FILE * file, const char * binding);

    // ====== Optimization Result ======

    GGML_API ggml_opt_result_t ggml_opt_result_init(void);
    GGML_API void ggml_opt_result_free(ggml_opt_result_t result);
    GGML_API void ggml_opt_result_reset(ggml_opt_result_t result);

    // get data from result, uncertainties are optional and can be ignored by passing NULL
    GGML_API void ggml_opt_result_ndata(   ggml_opt_result_t result, int64_t * ndata);                  // writes 1 value, number of datapoints
    GGML_API void ggml_opt_result_loss(    ggml_opt_result_t result, double  * loss,     double * unc); // writes 1 value
    GGML_API void ggml_opt_result_pred(    ggml_opt_result_t result, int32_t * pred);                   // writes ndata values
    GGML_API void ggml_opt_result_accuracy(ggml_opt_result_t result, double  * accuracy, double * unc); // writes 1 value

    // ====== Computation ======

    // if not using static graphs, this function must be called prior to ggml_opt_alloc
    GGML_API void ggml_opt_prepare_alloc(
        ggml_opt_context_t    opt_ctx,
        struct ggml_context * ctx_compute,
        struct ggml_cgraph  * gf,
        struct ggml_tensor  * inputs,
        struct ggml_tensor  * outputs);

    // The same, for an `outputs` that is a row-wise function of `hidden` (a language model's
    // head: logits = W x, with its adapters and scales; MUL_MAT, MUL, ADD and SCALE nodes): that
    // head is taken out of gf and computed for `rows` rows of hidden at a time, and the loss is
    // the mean over every row (GGML_OPT_LOSS_TYPE_CROSS_ENTROPY_TARGETS only). Under recompute
    // (checkpoint_prefix) only one slice's outputs and their gradient exist at a time, never the
    // [n_outputs x rows] tensor and its gradient. A head with any other node is not sliced.
    // Returns the number of slices (1: not sliced; accuracy is reported only then).
    GGML_API int ggml_opt_prepare_alloc_head(
        ggml_opt_context_t    opt_ctx,
        struct ggml_context * ctx_compute,
        struct ggml_cgraph  * gf,
        struct ggml_tensor  * inputs,
        struct ggml_tensor  * hidden,
        struct ggml_tensor  * outputs,
        int64_t               rows);

    // Project the existing row-wise head before consumers such as sampling are built.
    // Row indices address hidden's token rows. A null public_rows requests no public logits.
    // Returns zero without changing gf when the head cannot be replayed.
    GGML_API int ggml_opt_project_head(
        struct ggml_context * ctx_compute,
        struct ggml_cgraph  * gf,
        struct ggml_tensor  * hidden,
        struct ggml_tensor  * outputs,
        struct ggml_tensor  * shadow_rows,
        struct ggml_tensor  * public_rows,
        struct ggml_tensor  * label_rows,
        struct ggml_tensor  * public_shadow_rows,
        struct ggml_tensor ** slice_shadow_rows,
        int64_t               rows,
        struct ggml_tensor ** public_outputs,
        struct ggml_tensor ** slices,
        size_t                slices_capacity);

    GGML_API void ggml_opt_prepare_alloc_slices(
        ggml_opt_context_t    opt_ctx,
        struct ggml_context * ctx_compute,
        struct ggml_cgraph  * gf,
        struct ggml_tensor  * inputs,
        struct ggml_tensor ** slices,
        size_t                n_slices);

    // allocate the next graph for evaluation, either forward or forward + backward
    // must be called exactly once prior to calling ggml_opt_eval
    // returns false when the graph could not be allocated (device memory): the graph is then NOT
    // ready and must not be evaluated
    GGML_API bool ggml_opt_alloc(ggml_opt_context_t opt_ctx, bool backward);

    // Why the last ggml_opt_alloc returned false, in words a caller can put in front of a person
    // ("" when it did not refuse): a node the device backend cannot run, or the memory gate.
    GGML_API const char * ggml_opt_refusal(ggml_opt_context_t opt_ctx);

    // Caps what a graph may add on each non-CPU device, over and above the device's own free
    // figure (0 = no cap). For callers that know physical memory better than the driver does: on
    // Windows (WDDM) a CUDA device reports nearly the whole card free beside other processes, and
    // an allocation past physical VRAM lands in host memory.
    GGML_API void ggml_opt_set_alloc_budget(ggml_opt_context_t opt_ctx, size_t bytes);

    // The largest graph (bytes, on a non-CPU device) the allocation preflight has measured: what a
    // run of this shape needs, measured by the run itself.
    GGML_API size_t ggml_opt_peak_graph_bytes(ggml_opt_context_t opt_ctx);

    // do forward pass, increment result if not NULL, do backward pass if allocated
    GGML_API void ggml_opt_eval(ggml_opt_context_t opt_ctx, ggml_opt_result_t result);

    // A caller-driven optimizer period, for graphs built per step (the training walk's reverse
    // pass: one graph per chunk, ONE optimizer step per window). Applies to the next
    // ggml_opt_alloc + ggml_opt_eval only:
    //   period_end: true = this graph runs the optimizer step, false = it only accumulates
    //   loss_scale: the weight of this graph's loss in the period's total (replaces 1/opt_period)
    //   extra_loss: NULL, or a scalar F32 node of the forward graph added to the loss the backward
    //               differentiates (a surrogate carrying a later graph's gradient into this one);
    //               the loss the result reports stays the unweighted loss of the outputs
    // A period starts from zero gradients after the step that ended the previous one.
    GGML_API void ggml_opt_set_next_step(
        ggml_opt_context_t   opt_ctx,
        bool                 period_end,
        float                loss_scale,
        struct ggml_tensor * extra_loss);

    // After ggml_opt_eval and until the next ggml_opt_alloc: the gradient the backward computed
    // for a GRAD leaf (ggml_set_grad) of the evaluated graph, or NULL when the graph has none.
    GGML_API struct ggml_tensor * ggml_opt_leaf_grad(ggml_opt_context_t opt_ctx, struct ggml_tensor * leaf);

    // For the graph last given to ggml_opt_prepare_alloc: its backward ADDS the gradient on `tensor`
    // (a node of that graph, flagged GRAD by this call) into `acc`, an F32 tensor of the same shape
    // in a buffer the caller owns and zeroes, instead of keeping it in the compute buffer for
    // ggml_opt_leaf_grad. Nothing gradient-sized outlives its consumer on the device: the
    // accumulator may live in host memory while the graph runs on a GPU. The optimizer never
    // zeroes it, except at its own first allocation, which precedes any accumulation.
    GGML_API void ggml_opt_accumulate_grad(ggml_opt_context_t opt_ctx, struct ggml_tensor * tensor, struct ggml_tensor * acc);

    // ############################################################################
    // ## The high-level functions start here. They do not depend on any private ##
    // ## functions or structs and can be copied to and adapted for user code.   ##
    // ############################################################################

    // ====== Intended Usage ======
    //
    // 1. Select the appropriate loss for your problem.
    // 2. Create a dataset and set the data for the "data" tensor. Also set the "labels" tensor if your loss needs them.
    //    Setting the shard size to 1 will be fine, it's the granularity with which data is shuffled/loaded (bigger values are faster).
    // 3. Create a GGML graph for your model with no_alloc == true. Use two separate contexts for the tensors.
    //    The first context should contain the model parameters and inputs and be allocated statically in user code.
    //    The second context should contain all other tensors and will be (re)allocated automatically.
    //    Due to this automated allocation the data of the second context is not defined when accessed in user code.
    //    Note that the second dimension of the inputs/outputs are interpreted as the number of datapoints in those tensors.
    // 4. Call ggml_opt_fit. If you need more control you can use ggml_opt_epoch instead.

    // signature for a callback while evaluating opt_ctx on dataset, called after an evaluation
    typedef void (*ggml_opt_epoch_callback)(
            bool               train,       // true after training evaluation, false after validation evaluation
            ggml_opt_context_t opt_ctx,
            ggml_opt_dataset_t dataset,
            ggml_opt_result_t  result,      // result associated with the dataset subsection
            int64_t            ibatch,      // number of batches that have been evaluated so far
            int64_t            ibatch_max,  // total number of batches in this dataset subsection
            int64_t            t_start_us); // time at which the evaluation on the dataset subsection was started

    // do training on front of dataset, do evaluation only on back of dataset
    GGML_API void ggml_opt_epoch(
            ggml_opt_context_t      opt_ctx,
            ggml_opt_dataset_t      dataset,
            ggml_opt_result_t       result_train,   // result to increment during training, ignored if NULL
            ggml_opt_result_t       result_eval,    // result to increment during evaluation, ignored if NULL
            int64_t                 idata_split,    // data index at which to split training and evaluation
            ggml_opt_epoch_callback callback_train,
            ggml_opt_epoch_callback callback_eval);

    // callback that prints a progress bar on stderr
    GGML_API void ggml_opt_epoch_callback_progress_bar(
            bool               train,
            ggml_opt_context_t opt_ctx,
            ggml_opt_dataset_t dataset,
            ggml_opt_result_t  result,
            int64_t            ibatch,
            int64_t            ibatch_max,
            int64_t            t_start_us);

    // fit model defined by inputs and outputs to dataset
    GGML_API void ggml_opt_fit(
            ggml_backend_sched_t            backend_sched,  // backend scheduler for constructing the compute graphs
            struct ggml_context           * ctx_compute,    // context with temporarily allocated tensors to calculate the outputs
            struct ggml_tensor            * inputs,         // input tensor with shape [ne_datapoint, ndata_batch]
            struct ggml_tensor            * outputs,        // output tensor, must have shape [ne_label, ndata_batch] if labels are used
            ggml_opt_dataset_t              dataset,        // dataset with data and optionally also labels
            enum ggml_opt_loss_type         loss_type,      // loss to minimize
            enum ggml_opt_optimizer_type    optimizer,      // sgd or adamw
            ggml_opt_get_optimizer_params   get_opt_pars,   // callback to get optimizer params, userdata is pointer to epoch (of type int64_t)
            int64_t                         nepoch,         // how many times the dataset should be iterated over
            int64_t                         nbatch_logical, // datapoints optimizer step, must be a multiple of ndata_batch in inputs/outputs
            float                           val_split,      // fraction of the dataset to use for validation, must be in [0.0f, 1.0f)
            bool                            silent);        // whether or not info prints to stderr should be suppressed


#ifdef  __cplusplus
}
#endif
