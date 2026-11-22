#pragma once

#include "llama-arch.h"
#include "llama-batch.h"
#include "llama-hparams.h"
#include "llama-adapter.h"

#include <cstdint>
#include <cstdlib>
#include <vector>
#include <memory>
#include <set>
#include <functional>
#include <map>
#include <array>

struct ggml_cgraph;
struct ggml_context;
struct ggml_tensor;

struct llama_cparams;
struct llama_layer;
struct llama_model;
struct llama_prec_policy;
struct llama_hadamard;

class llama_moe_cache;

struct llama_memory_context_i;

class llama_kv_cache_context;
class llama_kv_cache_dsa_context;
class llama_kv_cache_dsa_iswa_context;
class llama_kv_cache_msa_context;
class llama_kv_cache_dsv4_raw_context;
class llama_kv_cache_dsv4_context;
class llama_kv_cache_iswa_context;
class llama_memory_recurrent_context;
class llama_memory_hybrid_context;
class llama_memory_hybrid_iswa_context;

// certain models (typically multi-modal) can produce different types of graphs
enum llm_graph_type {
    LLM_GRAPH_TYPE_DEFAULT,
    LLM_GRAPH_TYPE_ENCODER,
    LLM_GRAPH_TYPE_DECODER,
    LLM_GRAPH_TYPE_DECODER_MTP,
};

enum llm_fused_op {
    LLM_FUSED_OP_FLASH_ATTN,
    LLM_FUSED_OP_FLASH_ATTN_KV_ROWS,
    LLM_FUSED_OP_DSV4_HC_PRE,
    LLM_FUSED_OP_DSV4_HC_COMB,
    LLM_FUSED_OP_DSV4_HC_POST,
    LLM_FUSED_OP_TOP_K_MASK,
};

enum llm_ffn_op_type : int {
    LLM_FFN_NONE = 0,           // sentinel: unset; archs must assign before use
    LLM_FFN_SILU,
    LLM_FFN_GELU,
    LLM_FFN_GELU_ERF,
    LLM_FFN_RELU,
    LLM_FFN_RELU_SQR,
    LLM_FFN_SWIGLU,
    LLM_FFN_GEGLU,
    LLM_FFN_GEGLU_ERF,
    LLM_FFN_REGLU,
    LLM_FFN_SWIGLU_OAI_MOE,
    LLM_FFN_SITU,           // kimi-k3
};

enum llm_ffn_gate_type {
    LLM_FFN_SEQ,
    LLM_FFN_PAR, // ffn_gate is parallel to ffn_up
};

enum llm_norm_type {
    LLM_NORM,
    LLM_NORM_RMS,
    LLM_NORM_GROUP,
};

// TODO: tmp - need something better to pass the data from the encoder to the decoder
struct llama_cross {
    // the output embeddings from the encoder as a ggml tensor
    // TODO: this needs more work to be correct, for now copy the embeddings data to host memory
    //       ref: https://github.com/ggml-org/llama.cpp/pull/11213#discussion_r1969892524
    //ggml_tensor * t_embd = nullptr;

    int64_t n_embd = 0;
    int64_t n_enc  = 0;

    // embeddings data copied to host memory (tmp)
    std::vector<float> v_embd;

    // needed to construct the cross-attention mask in the decoder
    std::vector<std::set<llama_seq_id>> seq_ids_enc;
};

struct llm_graph_params;

//
// llm_graph_input
//

class llm_graph_input_i {
public:
    llm_graph_input_i() {
        const char * LLAMA_GRAPH_INPUT_DEBUG = getenv("LLAMA_GRAPH_INPUT_DEBUG");
        debug = LLAMA_GRAPH_INPUT_DEBUG ? atoi(LLAMA_GRAPH_INPUT_DEBUG) : 0;
    }

    virtual ~llm_graph_input_i() = default;

    virtual void set_input(const llama_ubatch * ubatch) = 0;

    // return true if the resulting input tensors using the provided graph parameters would be
    //   the same as the previous input tensors that we have currently stored in the object
    virtual bool can_reuse(const llm_graph_params & params) {
        // returning false here by default will prevent from reusing the graph if the check
        //   for the input type has not been implemented yet
        GGML_UNUSED(params);
        return false;
    }
protected:
    // env: LLAMA_GRAPH_INPUT_DEBUG
    int debug = 0;
};

using llm_graph_input_ptr = std::unique_ptr<llm_graph_input_i>;

class llm_graph_input_embd : public llm_graph_input_i {
public:
    llm_graph_input_embd(int64_t n_embd) : n_embd(n_embd) {}
    virtual ~llm_graph_input_embd() = default;

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override;

    ggml_tensor * tokens       = nullptr; // I32 [n_batch]
    ggml_tensor * embd         = nullptr; // F32 [n_embd, n_batch]
    ggml_tensor * mixed_tokens = nullptr; // I32 [n_tok_rows], mixed path: ids of the token rows
    ggml_tensor * mixed_slots  = nullptr; // I64 [n_tok_rows], mixed path: batch index of the token rows
    ggml_tensor * mixed_embd   = nullptr; // F32 [n_embd, n_batch], mixed path: embd rows, token rows are overwritten
    ggml_tensor * scale_rows   = nullptr; // F32 [1, n_batch], per-row scale: scale_tok for token rows, 1 for embd rows
    ggml_tensor * embd_rows    = nullptr; // I32 [n_batch], embd row path: rows of ctx_other's layer input device buffer

    float scale_tok = 1.0f;

    const int64_t n_embd = 0;
};

// similar to llm_graph_input_embd but with an additional hidden state input, fed from ubatch.embd_state
class llm_graph_input_embd_h : public llm_graph_input_i {
public:
    llm_graph_input_embd_h(int64_t n_embd, int64_t n_embd_state) : n_embd(n_embd), n_embd_state(n_embd_state) {}
    virtual ~llm_graph_input_embd_h() = default;

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override;

    ggml_tensor * tokens = nullptr; // I32 [n_batch]
    ggml_tensor * embd   = nullptr; // F32 [n_embd, n_batch]
    ggml_tensor * h      = nullptr; // F32 [n_embd_state, n_batch], the host input or the rows gathered with h_rows
    ggml_tensor * h_rows = nullptr; // I32 [n_batch], rows of the h_nextn device buffer of ctx_state_src, nullptr on the host path

    const int64_t n_embd       = 0;
    const int64_t n_embd_state = 0;
};

// original batch index of each ubatch token, used as set_rows destination rows
class llm_graph_input_batch_idxs : public llm_graph_input_i {
public:
    llm_graph_input_batch_idxs() = default;
    virtual ~llm_graph_input_batch_idxs() = default;

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override;

    ggml_tensor * idxs = nullptr; // I64 [n_batch]
};

class llm_graph_input_pos : public llm_graph_input_i {
public:
    llm_graph_input_pos(uint32_t n_pos_per_embd, llama_rope_section_order rope_section_order = LLAMA_ROPE_SECTION_ORDER_TYXZ)
        : n_pos_per_embd(n_pos_per_embd), rope_section_order(rope_section_order) {}
    virtual ~llm_graph_input_pos() = default;

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override;

    ggml_tensor * pos = nullptr; // I32 [n_batch]

    const uint32_t n_pos_per_embd = 1;
    const llama_rope_section_order rope_section_order = LLAMA_ROPE_SECTION_ORDER_TYXZ;
};

// temperature tuning, used by llama4
class llm_graph_input_attn_temp : public llm_graph_input_i {
public:
    llm_graph_input_attn_temp(uint32_t n_attn_temp_floor_scale, float f_attn_temp_scale, float f_attn_temp_offset, bool attn_temp_pow = false)
        : n_attn_temp_floor_scale(n_attn_temp_floor_scale), f_attn_temp_scale(f_attn_temp_scale), f_attn_temp_offset(f_attn_temp_offset), attn_temp_pow(attn_temp_pow) {}
    virtual ~llm_graph_input_attn_temp() = default;

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override;

    ggml_tensor * attn_scale = nullptr; // F32 [n_batch]

    const uint32_t n_attn_temp_floor_scale;
    const float    f_attn_temp_scale;
    const float    f_attn_temp_offset;
    const bool     attn_temp_pow;
};

class llm_graph_input_pos_bucket : public llm_graph_input_i {
public:
    llm_graph_input_pos_bucket(const llama_hparams & hparams) : hparams(hparams) {}
    virtual ~llm_graph_input_pos_bucket() = default;

    void set_input(const llama_ubatch * ubatch) override;

    ggml_tensor * pos_bucket = nullptr; // I32 [n_batch, n_batch]

    const llama_hparams hparams;
};

class llm_graph_input_pos_bucket_kv : public llm_graph_input_i {
public:
    llm_graph_input_pos_bucket_kv(
            const llama_hparams & hparams,
            const llama_kv_cache_context * mctx) : hparams(hparams), mctx(mctx) {}
    virtual ~llm_graph_input_pos_bucket_kv() = default;

    void set_input(const llama_ubatch * ubatch) override;

    ggml_tensor * pos_bucket = nullptr; // I32 [n_kv, n_batch]

    const llama_hparams hparams;

    const llama_kv_cache_context * mctx;
};

class llm_graph_input_out_ids : public llm_graph_input_i {
public:
    llm_graph_input_out_ids(
            const llama_hparams & hparams,
            const llama_cparams & cparams,
            uint32_t n_outputs) : hparams(hparams), cparams(cparams), n_outputs(n_outputs) {}
    virtual ~llm_graph_input_out_ids() = default;

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override;

    ggml_tensor * out_ids; // I32 [n_outputs]

    const llama_hparams hparams;
    const llama_cparams cparams;

    const uint32_t n_outputs;
};

// random expert ids for the MoE GEMMs, for benchmarks only (env: LLAMA_MOE_RANDOM_ROUTING)
class llm_graph_input_moe_ids : public llm_graph_input_i {
public:
    llm_graph_input_moe_ids(int64_t n_expert) : n_expert(n_expert) {}
    virtual ~llm_graph_input_moe_ids() = default;

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override;

    ggml_tensor * ids = nullptr; // I32 [n_expert_used_max, n_tokens]

    const int64_t n_expert;
};

class llm_graph_input_mean : public llm_graph_input_i {
public:
    llm_graph_input_mean(const llama_cparams & cparams) : cparams(cparams) {}
    virtual ~llm_graph_input_mean() = default;

    void set_input(const llama_ubatch * ubatch) override;

    ggml_tensor * mean; // F32 [n_batch, n_batch]

    const llama_cparams cparams;
};

class llm_graph_input_cls : public llm_graph_input_i {
public:
    llm_graph_input_cls(const llama_cparams & cparams, const llm_arch arch) : cparams(cparams), arch(arch) {}
    virtual ~llm_graph_input_cls() = default;

    void set_input(const llama_ubatch * ubatch) override;

    ggml_tensor * cls; // I32 [n_batch]

    const llama_cparams cparams;
    const llm_arch arch;
};

// Continuum: a training ubatch's row maps (the labelled rows, the shadow adapter's rows), set from the ubatch
class llm_graph_input_training_rows : public llm_graph_input_i {
public:
    void set_input(const llama_ubatch * ubatch) override;
    ggml_tensor * public_rows = nullptr;
    ggml_tensor * label_rows = nullptr;
    ggml_tensor * shadow_rows = nullptr;
    ggml_tensor * public_shadow_rows = nullptr;
    std::vector<ggml_tensor *> slice_shadow_rows;
};

class llm_graph_input_rs : public llm_graph_input_i {
public:
    llm_graph_input_rs(const llama_memory_recurrent_context * mctx) : mctx(mctx) {}
    virtual ~llm_graph_input_rs() = default;

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override;

    ggml_tensor * s_copy;  // I32 [n_rs]

    // views of s_copy, computed once per graph
    // and shared across layers which use build_rs
    ggml_tensor * s_copy_main; // I32 [n_seqs]
    ggml_tensor * s_copy_tail; // I32 [n_rs - 1]

    // Continuum: I64 write destinations, Qwen3.5 serving without rollback (llama-graph-train.cpp)
    ggml_tensor * s_write = nullptr;
    ggml_tensor * s_write_main = nullptr;
    ggml_tensor * s_write_extra = nullptr;
    void set_write(const llama_memory_recurrent_context * rs);

    const llama_memory_recurrent_context * mctx;

    // used in view offsets, need to match for valid graph reuse
    uint32_t head;
    int32_t rs_z;
};

class llm_graph_input_cross_embd : public llm_graph_input_i {
public:
    llm_graph_input_cross_embd(
            const llama_cross * cross) : cross(cross) {}
    virtual ~llm_graph_input_cross_embd() = default;

    void set_input(const llama_ubatch * ubatch) override;

    ggml_tensor * cross_embd; // F32 [n_embd, n_outputs_enc]

    const llama_cross * cross;
};

// KQ mask and the optional per-sequence cell rows of a unified cache (see ggml_flash_attn_ext_rows)
// the mask has two shapes depending on whether `rows` is used:
//   rows != nullptr: [n_kv_rows, n_batch/n_slices, 1, n_slices]
//   rows == nullptr: [n_kv,      n_batch/n_stream, 1, n_stream]
struct llm_attn_mask {
    ggml_tensor * raw  = nullptr;
    ggml_tensor * rows = nullptr; // I32 [n_kv_rows, n_slices]
    uint32_t      n_kv = 0;

    // fill every token of a sequence with the mask row of its first token (dspark draft blocks)
    bool block_uniform = false;
};

class llm_graph_input_attn_no_cache : public llm_graph_input_i {
public:
    llm_graph_input_attn_no_cache(const llama_hparams & hparams, const llama_cparams & cparams) :
        hparams(hparams),
        cparams(cparams) {
    }
    ~llm_graph_input_attn_no_cache() = default;

    void set_input(const llama_ubatch * ubatch) override;

    // n_tokens == n_batch, n_kv == n_tokens padded to 256
    llm_attn_mask self_attn_mask;
    llm_attn_mask self_attn_mask_swa;

    const llama_hparams hparams;
    const llama_cparams cparams;
};

class llm_graph_input_attn_kv : public llm_graph_input_i {
public:
    llm_graph_input_attn_kv(
            const llama_hparams & hparams,
            const llama_cparams & cparams,
            const llama_kv_cache_context * mctx) :
        hparams(hparams),
        cparams(cparams),
        mctx(mctx) {
    }
    ~llm_graph_input_attn_kv() = default;

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override;

    ggml_tensor * get_k_idxs() const { return self_k_idxs; }
    ggml_tensor * get_v_idxs() const { return self_v_idxs; }

    ggml_tensor * self_k_idxs = nullptr; // I64 [n_batch]
    ggml_tensor * self_v_idxs = nullptr; // I64 [n_batch] or [n_batch*n_embd_v_gqa]

    llm_attn_mask self_attn_mask;

    // note: assumes v_rot^2 == I
    std::vector<ggml_tensor *> self_k_rot;
    std::vector<ggml_tensor *> self_v_rot;

    // note: these have to be copies because in order to be able to reuse a graph, its inputs
    //       need to carry these parameters with them. otherwise, they can point to freed
    //       llm_graph_params from a previous batch, causing stack-use-after-return
    const llama_hparams hparams;
    const llama_cparams cparams;

    const llama_kv_cache_context * mctx;
};

// V-less input for the KV cache
// ref: https://github.com/ggml-org/llama.cpp/pull/19067
class llm_graph_input_attn_k : public llm_graph_input_i {
public:
    llm_graph_input_attn_k(
            const llama_hparams & hparams,
            const llama_cparams & cparams,
            const llama_kv_cache_context * mctx) :
        hparams(hparams),
        cparams(cparams),
        mctx(mctx) {
    }
    ~llm_graph_input_attn_k() = default;

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override;

    // like can_reuse, but does not re-bind mctx
    bool can_reuse_impl(const llm_graph_params & params);

    ggml_tensor * get_k_idxs() const { return self_k_idxs; }

    ggml_tensor * self_k_idxs = nullptr; // I64 [n_batch]

    llm_attn_mask self_attn_mask;

    const llama_hparams hparams;
    const llama_cparams cparams;

    const llama_kv_cache_context * mctx;
};

class llm_graph_input_attn_k_dsa : public llm_graph_input_i {
public:
    llm_graph_input_attn_k_dsa(
            const llama_hparams & hparams,
            const llama_cparams & cparams,
            const llama_kv_cache_dsa_context * mctx) :
        hparams(hparams),
        cparams(cparams),
        mctx(mctx) {
    }
    ~llm_graph_input_attn_k_dsa() = default;

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override;

    // like can_reuse, but does not re-bind mctx
    bool can_reuse_impl(const llm_graph_params & params);

    ggml_tensor * get_k_idxs_mla() const { return self_k_idxs_mla; }
    ggml_tensor * get_k_idxs_lid() const { return self_k_idxs_lid; }

    ggml_tensor * self_k_idxs_mla = nullptr; // I64 [n_batch]
    ggml_tensor * self_k_idxs_lid = nullptr; // I64 [n_batch]

    llm_attn_mask self_attn_mask_mla;
    llm_attn_mask self_attn_mask_lid;

    std::vector<ggml_tensor *> self_k_rot_lid;

    const llama_hparams hparams;
    const llama_cparams cparams;

    const llama_kv_cache_dsa_context * mctx;
};

// DSA input (full-attention layers + indexer) with K-only input for the SWA layers
class llm_graph_input_attn_k_dsa_iswa : public llm_graph_input_i {
public:
    llm_graph_input_attn_k_dsa_iswa(
            std::unique_ptr<llm_graph_input_attn_k_dsa> inp_dsa,
            std::unique_ptr<llm_graph_input_attn_k>     inp_swa,
            const llama_kv_cache_dsa_iswa_context *     mctx) :
        inp_dsa(std::move(inp_dsa)),
        inp_swa(std::move(inp_swa)),
        mctx(mctx) {
    }
    ~llm_graph_input_attn_k_dsa_iswa() = default;

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override;

    llm_graph_input_attn_k_dsa * get_dsa() const { return inp_dsa.get(); }
    llm_graph_input_attn_k     * get_swa() const { return inp_swa.get(); }

    std::unique_ptr<llm_graph_input_attn_k_dsa> inp_dsa;
    std::unique_ptr<llm_graph_input_attn_k>     inp_swa;

    const llama_kv_cache_dsa_iswa_context * mctx;
};

// standard K/V attention input against the base cache, plus destination indices for the indexer key cache
class llm_graph_input_attn_kv_msa : public llm_graph_input_attn_kv {
public:
    llm_graph_input_attn_kv_msa(
            const llama_hparams & hparams,
            const llama_cparams & cparams,
            const llama_kv_cache_msa_context * mctx);
    ~llm_graph_input_attn_kv_msa() = default;

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override;

    ggml_tensor * get_k_idxs_idx() const { return self_k_idxs_idx; }

    ggml_tensor * self_k_idxs_idx = nullptr; // I64 [n_batch]

    const llama_kv_cache_msa_context * mctx_msa;
};

class llm_graph_input_attn_kv_iswa : public llm_graph_input_i {
public:
    llm_graph_input_attn_kv_iswa(
            const llama_hparams & hparams,
            const llama_cparams & cparams,
            const llama_kv_cache_iswa_context * mctx) :
        hparams(hparams),
        cparams(cparams),
        mctx(mctx) {
    }
    ~llm_graph_input_attn_kv_iswa() = default;

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override;

    ggml_tensor * get_k_idxs()     const { return self_k_idxs; }
    ggml_tensor * get_v_idxs()     const { return self_v_idxs; }
    ggml_tensor * get_k_idxs_swa() const { return self_k_idxs_swa; }
    ggml_tensor * get_v_idxs_swa() const { return self_v_idxs_swa; }

    ggml_tensor * self_k_idxs     = nullptr; // I64 [n_batch]
    ggml_tensor * self_v_idxs     = nullptr; // I64 [n_batch] or [n_batch*n_embd_v_gqa]
    ggml_tensor * self_k_idxs_swa = nullptr; // I64 [n_batch]
    ggml_tensor * self_v_idxs_swa = nullptr; // I64 [n_batch] or [n_batch*n_embd_v_gqa]

    llm_attn_mask self_attn_mask;
    llm_attn_mask self_attn_mask_swa;

    std::vector<ggml_tensor *> self_k_rot;
    std::vector<ggml_tensor *> self_v_rot;

    std::vector<ggml_tensor *> self_k_rot_swa;
    std::vector<ggml_tensor *> self_v_rot_swa;

    const llama_hparams hparams;
    const llama_cparams cparams;

    const llama_kv_cache_iswa_context * mctx;
};

class llm_graph_input_attn_k_iswa : public llm_graph_input_i {
public:
    llm_graph_input_attn_k_iswa(
            const llama_hparams & hparams,
            const llama_cparams & cparams,
            const llama_kv_cache_iswa_context * mctx) :
        hparams(hparams),
        cparams(cparams),
        mctx(mctx) {
    }
    ~llm_graph_input_attn_k_iswa() = default;

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override;

    ggml_tensor * get_k_idxs()     const { return self_k_idxs; }
    ggml_tensor * get_k_idxs_swa() const { return self_k_idxs_swa; }

    ggml_tensor * self_k_idxs     = nullptr; // I64 [n_batch]
    ggml_tensor * self_k_idxs_swa = nullptr; // I64 [n_batch]

    llm_attn_mask self_attn_mask;
    llm_attn_mask self_attn_mask_swa;

    std::vector<ggml_tensor *> self_k_rot;
    std::vector<ggml_tensor *> self_k_rot_swa;

    const llama_hparams hparams;
    const llama_cparams cparams;

    const llama_kv_cache_iswa_context * mctx;
};

// DSV4 raw graph inputs are SWA-only, but their mask may be stream-shaped
// so raw K can be concatenated with DSV4 compressed K in one attention op.
class llm_graph_input_dsv4_raw {
public:
    llm_graph_input_dsv4_raw(
            const llama_cparams & cparams,
            const llama_kv_cache_dsv4_raw_context * mctx) :
        cparams(cparams),
        mctx(mctx) {
    }

    void set_input(const llama_ubatch * ubatch);

    ggml_tensor * get_k_idxs() const { return self_k_idxs; }

    ggml_tensor * self_k_idxs = nullptr; // I64 [n_batch]

    llm_attn_mask self_attn_mask;

    std::vector<ggml_tensor *> self_k_rot;

    const llama_cparams cparams;

    const llama_kv_cache_dsv4_raw_context * mctx;
};

class llm_graph_input_dsv4 : public llm_graph_input_i {
public:
    struct comp_input {
        ggml_tensor * state_pos        = nullptr; // I32 [n_state]
        ggml_tensor * state_persist_src_idxs = nullptr; // I32 [n_state_persist]
        ggml_tensor * state_persist_dst_idxs = nullptr; // I32 [n_state_persist]
        ggml_tensor * state_restore_src_idxs = nullptr; // I32 [n_state_restore]
        ggml_tensor * state_restore_dst_idxs = nullptr; // I32 [n_state_restore]
        ggml_tensor * state_snapshot_src_idxs = nullptr; // I32 [n_state_snapshot]
        ggml_tensor * state_snapshot_dst_idxs = nullptr; // I32 [n_state_snapshot]
        ggml_tensor * state_read_idxs  = nullptr; // I32 [ratio*n_state_write]
        ggml_tensor * state_write_idxs = nullptr; // I64 [n_state_write]
        ggml_tensor * state_write_pos  = nullptr; // I32 [n_state_write]

        ggml_tensor * kq_mask    = nullptr; // F32 [n_kv, n_batch/n_stream, 1, n_stream]

        std::vector<ggml_tensor *> k_rot;
    };

    llm_graph_input_dsv4(
            const llama_cparams & cparams,
            std::unique_ptr<llm_graph_input_dsv4_raw> inp_raw,
            const llama_kv_cache_dsv4_context * mctx) :
        inp_raw(std::move(inp_raw)),
        cparams(cparams),
        mctx(mctx) {
    }
    ~llm_graph_input_dsv4() = default;

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override;

    llm_graph_input_dsv4_raw * get_raw() const { return inp_raw.get(); }
    const comp_input & get_csa() const { return inp_csa; }
    const comp_input & get_hca() const { return inp_hca; }
    const comp_input & get_lid() const { return inp_lid; }

    std::unique_ptr<llm_graph_input_dsv4_raw> inp_raw;

    comp_input inp_csa;
    comp_input inp_hca;
    comp_input inp_lid;

    const llama_cparams cparams;

    const llama_kv_cache_dsv4_context * mctx;
};

class llm_graph_input_attn_cross : public llm_graph_input_i {
public:
    llm_graph_input_attn_cross(const llama_cross * cross) : cross(cross) {}
    ~llm_graph_input_attn_cross() = default;

    void set_input(const llama_ubatch * ubatch) override;


    llm_attn_mask cross_attn_mask;

    const llama_cross * cross = nullptr;
};

class llm_graph_input_mem_hybrid : public llm_graph_input_i {
public:
    llm_graph_input_mem_hybrid(
            const llama_cparams & cparams,
            std::unique_ptr<llm_graph_input_attn_kv> inp_attn,
            std::unique_ptr<llm_graph_input_rs>      inp_rs,
            const llama_memory_hybrid_context *      mctx) :
        inp_attn(std::move(inp_attn)),
        inp_rs(std::move(inp_rs)),
        cparams(cparams),
        mctx(mctx) { }
    virtual ~llm_graph_input_mem_hybrid() = default;

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override;

    std::unique_ptr<llm_graph_input_attn_kv> inp_attn;
    std::unique_ptr<llm_graph_input_rs>      inp_rs;

    llm_graph_input_attn_kv * get_attn() const { return inp_attn.get(); }
    llm_graph_input_rs      * get_recr() const { return inp_rs.get(); }

    const llama_cparams cparams;

    const llama_memory_hybrid_context * mctx;
};

class llm_graph_input_mem_hybrid_k : public llm_graph_input_i {
public:
    llm_graph_input_mem_hybrid_k(
            const llama_cparams & cparams,
            std::unique_ptr<llm_graph_input_attn_k> inp_attn,
            std::unique_ptr<llm_graph_input_rs>      inp_rs,
            const llama_memory_hybrid_context *      mctx) :
        inp_attn(std::move(inp_attn)),
        inp_rs(std::move(inp_rs)),
        cparams(cparams),
        mctx(mctx) { }
    virtual ~llm_graph_input_mem_hybrid_k() = default;

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override;

    std::unique_ptr<llm_graph_input_attn_k> inp_attn;
    std::unique_ptr<llm_graph_input_rs>      inp_rs;

    llm_graph_input_attn_k * get_attn() const { return inp_attn.get(); }
    llm_graph_input_rs      * get_recr() const { return inp_rs.get(); }

    const llama_cparams cparams;

    const llama_memory_hybrid_context * mctx;
};

class llm_graph_input_mem_hybrid_iswa : public llm_graph_input_i {
public:
    llm_graph_input_mem_hybrid_iswa(
            const llama_cparams & cparams,
            std::unique_ptr<llm_graph_input_attn_kv_iswa> inp_attn,
            std::unique_ptr<llm_graph_input_rs>          inp_rs,
            const llama_memory_hybrid_iswa_context *     mctx) :
        inp_attn(std::move(inp_attn)),
        inp_rs(std::move(inp_rs)),
        cparams(cparams),
        mctx(mctx) { }
    virtual ~llm_graph_input_mem_hybrid_iswa() = default;

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override;

    std::unique_ptr<llm_graph_input_attn_kv_iswa> inp_attn;
    std::unique_ptr<llm_graph_input_rs>          inp_rs;

    llm_graph_input_attn_kv_iswa * get_attn() const { return inp_attn.get(); }
    llm_graph_input_rs           * get_recr() const { return inp_rs.get(); }

    const llama_cparams cparams;

    const llama_memory_hybrid_iswa_context * mctx;
};

// hybrid iswa memory with a per-head attention bias that depends on the distance from the query to the cell (Inkling)
// the KQ masks are always dense (rows == nullptr) and are filled from the same cells as the distances, so that the bias
// gathered by distance lines up with the mask
class llm_graph_input_mem_hybrid_iswa_rel : public llm_graph_input_i {
public:
    llm_graph_input_mem_hybrid_iswa_rel(
            const llama_cparams & cparams,
            std::unique_ptr<llm_graph_input_attn_kv_iswa> inp_attn,
            std::unique_ptr<llm_graph_input_rs>          inp_rs,
            const llama_memory_hybrid_iswa_context *     mctx,
            uint32_t extent,
            uint32_t extent_swa) :
        inp_attn(std::move(inp_attn)),
        inp_rs(std::move(inp_rs)),
        extent(extent),
        extent_swa(extent_swa),
        cparams(cparams),
        mctx(mctx) { }
    virtual ~llm_graph_input_mem_hybrid_iswa_rel() = default;

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override;

    std::unique_ptr<llm_graph_input_attn_kv_iswa> inp_attn;
    std::unique_ptr<llm_graph_input_rs>          inp_rs;

    llm_graph_input_attn_kv_iswa * get_attn() const { return inp_attn.get(); }
    llm_graph_input_rs           * get_recr() const { return inp_rs.get(); }

    // distance from the token to the cell, extent for the empty, future and more distant cells
    ggml_tensor * self_rel_pos     = nullptr; // I32 [n_kv,     n_batch]
    ggml_tensor * self_rel_pos_swa = nullptr; // I32 [n_kv_swa, n_batch]

    const uint32_t extent;
    const uint32_t extent_swa;

    const llama_cparams cparams;

    const llama_memory_hybrid_iswa_context * mctx;
};

class llm_graph_input_sampling : public llm_graph_input_i {
public:
    llm_graph_input_sampling(std::map<llama_seq_id, llama_sampler *> samplers) :
        samplers(std::move(samplers)) { }
    virtual ~llm_graph_input_sampling() = default;

    void set_input(const llama_ubatch * ubatch) override;
    bool can_reuse(const llm_graph_params & params) override;

    std::map<llama_seq_id, llama_sampler *> samplers;
};

//
// llm_graph_result
//

// these objects deliver the result from the graph build process back to the llama_context
// note that the input tensors created for the graph are referenced here - the goal is to be able to populate their
//   specific data, by calling the set_inputs() method
// along with the input tensors, the object also provides commonly used outputs tensors, such as logits, embeddings, etc.
//   these are used by the llama_context to extact the relevant data, based on the compute parameters

// callback that allows us to apply custom logic to each tensor (e.g. ggml-alloc, offloading, etc.)
using llm_graph_cb = std::function<void(const llama_ubatch & ubatch, ggml_tensor * cur, const char * name, int il)>;

class llm_graph_result;

struct llm_graph_params {
    const llama_model * model = nullptr;

    llama_cparams cparams;

    llama_ubatch ubatch; // note: intentionally make a copy

    llm_graph_type gtype;

    ggml_backend_sched_t sched;
    ggml_backend_t backend_cpu;

    const llama_adapter_cvec     * cvec;
    const llama_adapter_loras    * loras;
    const llama_memory_context_i * mctx;
    const llama_cross            * cross;
    const llama_moe_cache        * moe_cache;

    ggml_tensor * layer_inp_dev = nullptr; // own layer input device buffer (write)
    ggml_tensor * layer_inp_src = nullptr; // ctx_other's layer input device buffer (read)
    ggml_tensor * h_nextn_dev   = nullptr; // own h_nextn device buffer (write)
    ggml_tensor * h_nextn_src   = nullptr; // ctx_state_src's h_nextn device buffer (read)

    std::map<llama_seq_id, llama_sampler *> samplers;

    static bool samplers_equal(
          const std::map<llama_seq_id, llama_sampler *> & lhs,
          const std::map<llama_seq_id, llama_sampler *> & rhs) {
        if (lhs.size() != rhs.size()) {
            return false;
        }
        for (const auto & [seq_id, sampler] : lhs) {
            auto it = rhs.find(seq_id);
            if (it == rhs.end() || it->second != sampler) {
                return false;
            }
        }
        return true;
    }

    uint32_t n_outputs;

    llm_graph_cb cb;

    llm_graph_result * res;

    // Continuum (set after upstream's initializer, in llama_context::graph_params): THE SHADOW, the attached adapter
    // being trained (llama_opt_init), the layer that switches on and off. It applies only to the ubatch's labelled
    // rows (n_shadow_rows of them); a ubatch with no label never applies it, so serving is the stable set's graph
    const llama_adapter_lora * shadow = nullptr;
    uint32_t                   n_shadow_rows = 0;
    // the physical attention stream of each of the ubatch's unique sequences (seq_id_unq order): the graph's K/V
    // views are built at these streams, so a graph is reused only when they match
    std::vector<uint32_t> streams = {};

    // return true if the "other" params would result in a graph with the same topology as with the current params
    //   having the same topology allows us to reuse the graph in some cases
    bool allow_reuse(const llm_graph_params & other) const {
        // first check the ubatch
        bool can_reuse_ubatch =
            ubatch.equal_seqs() == other.ubatch.equal_seqs() &&
            ubatch.n_tokens     == other.ubatch.n_tokens &&
            ubatch.n_seq_tokens == other.ubatch.n_seq_tokens &&
            ubatch.n_seqs       == other.ubatch.n_seqs &&
            ubatch.n_seqs_unq   == other.ubatch.n_seqs_unq &&
            ubatch.is_mixed()   == other.ubatch.is_mixed() &&
            (
                (!ubatch.token && !other.ubatch.token) ||
                (!ubatch.embd  && !other.ubatch.embd)  ||
                (ubatch.token && other.ubatch.token && ubatch.embd && other.ubatch.embd)
            ) &&
            (!ubatch.embd_state == !other.ubatch.embd_state) &&
            (!ubatch.embd_row   == !other.ubatch.embd_row);

        // when we split the batch using "equal_seqs" we have to verify that the participating sequences are the same
        //   the reason is because the set of attention streams would be different for different sequences
        if (can_reuse_ubatch && ubatch.equal_seqs()) {
            if (!ubatch.data) {
                // if the old ubatch does not own it's data, then we cannot guarantee that it is still alive, and
                //   therefore we cannot perform the sequence id check. normally should never happen
                can_reuse_ubatch = false;
            } else {
                for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
                    can_reuse_ubatch &= ubatch.seq_id_unq[s] == other.ubatch.seq_id_unq[s];
                }
            }
        }

        // Continuum: the graph's K/V views are built at these physical streams, whatever the split
        can_reuse_ubatch = can_reuse_ubatch && streams == other.streams;

        if (!can_reuse_ubatch) {
            return false;
        }

        if (n_outputs != other.n_outputs) {
            return false;
        }

        if (!samplers_equal(samplers, other.samplers)) {
            return false;
        }

        if (samplers.size() > 0) {
            if (!ubatch.data || !other.ubatch.data) {
                return false;
            }

            // check that the outputs are the same for all samplers
            for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
                if (ubatch.output[i]    != other.ubatch.output[i] ||
                    ubatch.seq_id[i][0] != other.ubatch.seq_id[i][0]) {
                    return false;
                }
            }
        }

        // TODO: https://github.com/ggml-org/llama.cpp/pull/24340#discussion_r3448035248
        if (cparams.nextn_layer_offset != other.cparams.nextn_layer_offset) {
            return false;
        }

        return
            cparams.embeddings              == other.cparams.embeddings              &&
            cparams.embeddings_nextn        == other.cparams.embeddings_nextn        &&
            cparams.embeddings_nextn_masked == other.cparams.embeddings_nextn_masked &&
            cparams.causal_attn             == other.cparams.causal_attn             &&
            layer_inp_dev == other.layer_inp_dev &&
            layer_inp_src == other.layer_inp_src &&
            h_nextn_dev   == other.h_nextn_dev   &&
            h_nextn_src   == other.h_nextn_src   &&
            model == other.model &&
            gtype == other.gtype &&
            cvec  == other.cvec  &&
            loras == other.loras &&
            cparams.training == other.cparams.training && // Continuum: a training graph and its shadow
            shadow == other.shadow &&
            n_shadow_rows == other.n_shadow_rows &&
            cross == other.cross;
    }
};

struct llm_graph_fused_node {
    llm_fused_op op;
    ggml_tensor * tensor;
    int il;
};

class llm_graph_result {
public:
    llm_graph_result(int64_t max_nodes);

    virtual ~llm_graph_result() = default;

    ggml_tensor * get_inp_tokens()  const { return t_inp_tokens; }
    ggml_tensor * get_logits()      const { return t_logits; }
    ggml_tensor * get_embd()        const { return t_embd; }
    ggml_tensor * get_embd_pooled() const { return t_embd_pooled; }
    ggml_tensor * get_h_nextn()     const { return t_h_nextn; }

    ggml_tensor * get_layer_inp(int il) const { return t_layer_inp[il]; }

    ggml_cgraph  * get_gf()  const { return gf; }
    ggml_context * get_ctx() const { return ctx_compute.get(); }

    int64_t get_max_nodes() const;

    void set_max_nodes(int64_t max_nodes);

    void reset();

    void set_inputs(const llama_ubatch * ubatch);
    void set_outputs(const llm_graph_params & params);

    // try to update the existing graph result using the new graph parameters in order to reuse it
    // this can only be done if we determine that the resulting graph using the new graph parameters
    //   would be identical to the existing graph. in that case, we simply have to update the memory
    //   contexts of the input tensors of the graph and we can reuse it for another computation
    // return true if the graph was updated and can be reused
    bool can_reuse(const llm_graph_params & params);

    llm_graph_input_i * add_input(llm_graph_input_ptr input);

    void add_fused_node(llm_graph_fused_node result);

    // Hadamard-transformed activations, keyed by (input, rotation): folded weights that read the same activation share one transform
    ggml_tensor * get_hdmd_input(const ggml_tensor * cur, const ggml_tensor * rot) const;
    void          set_hdmd_input(const ggml_tensor * cur, const ggml_tensor * rot, ggml_tensor * res);

    const std::vector<llm_graph_fused_node> & get_fused_nodes() const { return fused_nodes; }

    void set_params(const llm_graph_params & params);

    // important graph nodes
    ggml_tensor * t_inp_tokens  = nullptr;
    ggml_tensor * t_inp_embd    = nullptr; // [n_embd_inp, n_tokens]
    ggml_tensor * t_logits      = nullptr;
    std::vector<ggml_tensor *> t_loss_slices; // Continuum: the training loss, per slice

    // Continuum, THE EXACT WALK: per attention layer of a reverse-pass chunk graph, the tensors the walk
    // fills before the step and reads after it (llama_context::opt_epoch_iter)
    struct walk_layer {
        int           il;
        uint32_t      n_past;
        uint32_t      g0;           // the first position whose K/V gradient is kept
        ggml_tensor * tk = nullptr; // the cached K/V of [grad_from, n_past) as F32, tapped: their
        ggml_tensor * tv = nullptr; //   gradient adds into ak / av (ggml_opt_accumulate_grad)
        ggml_tensor * ak = nullptr; // the accumulator's rows [grad_from, n_past), shaped like tk / tv
        ggml_tensor * av = nullptr;
    };
    std::vector<walk_layer> t_walk;
    // the recurrent states of a reverse-pass chunk graph, one per state tensor of the memory
    // (a layer's conv state r_l and its recurrent state s_l): the chain runs through them too
    struct walk_state {
        ggml_tensor * cache = nullptr; // the memory's state tensor this entry belongs to
        ggml_tensor * tap   = nullptr; // the state ENTERING the chunk, tapped: its gradient adds into the
                                       //   walk's host accumulator for this state (ggml_opt_accumulate_grad)
        ggml_tensor * gs    = nullptr; // input: the gradient later chunks put on the state it LEAVES
    };
    std::vector<walk_state> t_walk_state;
    walk_state & walk_state_of(ggml_tensor * cache) {
        for (auto & w : t_walk_state) {
            if (w.cache == cache) {
                return w;
            }
        }
        t_walk_state.push_back({ cache, nullptr, nullptr });
        return t_walk_state.back();
    }
    ggml_tensor * t_walk_surrogate = nullptr; // sum over layers of <k_cur, gk> + <v_cur, gv>
    ggml_tensor * t_embd        = nullptr;
    ggml_tensor * t_embd_pooled = nullptr;
    ggml_tensor * t_h_nextn     = nullptr; // [n_embd, n_outputs] hidden state before final output norm

    std::vector<ggml_tensor *> t_layer_inp;

    std::vector<ggml_tensor *> t_sampled;
    std::vector<ggml_tensor *> t_sampled_probs;
    std::vector<ggml_tensor *> t_sampled_logits;
    std::vector<ggml_tensor *> t_candidates;

    std::vector<ggml_tensor *> t_sampling_all;

    // [0] is the LLAMA_MOE_RANDOM_ROUTING input, the rest are views of it
    std::vector<ggml_tensor *> t_moe_ids;

    std::vector<llm_graph_input_ptr> inputs;
    std::vector<llm_graph_fused_node> fused_nodes;

    std::map<std::pair<const ggml_tensor *, const ggml_tensor *>, ggml_tensor *> hdmd_inputs;

    ggml_context_ptr ctx_compute;

    // memory buffers used to evaluate the model
    std::unique_ptr<uint8_t[]> buf_compute_meta;
    size_t buf_compute_meta_size = 0;

    ggml_cgraph * gf;

private:
    int64_t max_nodes;

    // keep a copy of the previous graph parameters
    // we will use this to determine whether the graph can be reused by comparing them with the new parameters
    // note: these are updated after constructing the new graph
    llm_graph_params params;

    // env: LLAMA_GRAPH_RESULT_DEBUG
    int debug = 0;
};

using llm_graph_result_ptr = std::unique_ptr<llm_graph_result>;

//
// llm_graph_context
//

// used in build_rs to properly order writes and avoid unnecessary copies
using llm_graph_get_rows_fn = std::function<ggml_tensor * (ggml_context *, ggml_tensor * states, ggml_tensor * ids)>;

struct llm_graph_qkv {
    ggml_tensor * q; // [n_embd_head, n_head,    n_tokens]
    ggml_tensor * k; // [n_embd_head, n_head_kv, n_tokens]
    ggml_tensor * v; // [n_embd_head, n_head_kv, n_tokens]
};

struct llm_graph_amax_src1 {
    ggml_tensor * res;     // cur rescaled
    ggml_tensor * scale_s; // scale to undo on the matmul result, or nullptr
};

struct llm_graph_context {
    const llama_model & model;

    const llama_hparams & hparams;
    const llama_cparams & cparams;
    const llama_ubatch  & ubatch;

    const llama_prec_policy & prec_policy;

    const int64_t n_embd;
    const int64_t n_layer;
    const int64_t n_layer_nextn;
    const int64_t n_rot;
    const int64_t n_ctx;       // user-specified context size (can be different from n_ctx_train)
    const int64_t n_head;
    const int64_t n_head_kv;
    const int64_t n_embd_head_k;
    const int64_t n_embd_k_gqa;
    const int64_t n_embd_head_v;
    const int64_t n_embd_v_gqa;
    const int64_t n_expert;
    const int64_t n_expert_used;

    const float freq_base;
    const float freq_scale;
    const float ext_factor;
    const float attn_factor;
    const float beta_fast;
    const float beta_slow;
    const float norm_eps;
    const float norm_rms_eps;

    const int64_t n_tokens;
    const int64_t n_outputs;
    const int32_t n_ctx_orig; // yarn

    const enum llama_pooling_type pooling_type;
    const enum llama_rope_type    rope_type;

    ggml_backend_sched_t sched;

    ggml_backend_t backend_cpu; // TODO: needed by build_attn_mha, figure out a way to remove?

    const llama_adapter_cvec     * cvec;
    const llama_adapter_loras    * loras;
    const llama_memory_context_i * mctx;
    const llama_cross            * cross;
    const llama_moe_cache        * moe_cache;

    // Continuum (llama-graph-train.cpp)
    const llama_adapter_lora     * shadow;
    const uint32_t                 n_shadow_rows;
    mutable ggml_tensor * inp_train_rows = nullptr; // build_inp_train_rows, once per graph
    mutable llm_graph_input_training_rows * inp_training_rows = nullptr;
    // the in-place training attention's K/V grad sinks (tap k, chunk k, tap v, chunk v), set by
    // build_attn_train before build_attn_mha and taken by the FLASH_ATTN_EXT it builds
    mutable std::array<ggml_tensor *, 4> walk_grad_sinks = {};

    const llama_hadamard * hdmd;

    ggml_tensor * layer_inp_dev;
    ggml_tensor * layer_inp_src;
    ggml_tensor * h_nextn_dev;
    ggml_tensor * h_nextn_src;

    std::map<llama_seq_id, llama_sampler *> samplers;

    const llm_graph_cb & cb_func;

    llm_graph_result * res;

    ggml_context * ctx0 = nullptr;
    ggml_cgraph  * gf   = nullptr;

    llm_graph_context(const llm_graph_params & params);
    virtual ~llm_graph_context() = default;

    void cb(ggml_tensor * cur, const char * name, int il) const;

    // true when the last layer must be narrowed to the output rows before the nextn hidden state is captured
    bool crop_before_nextn(const ggml_tensor * inp_out_ids) const {
        return inp_out_ids != nullptr && (!cparams.embeddings_nextn || cparams.embeddings_nextn_masked);
    }

    // true when the nextn hidden state must be narrowed to the output rows after it is captured
    bool crop_after_nextn(const ggml_tensor * inp_out_ids) const {
        return inp_out_ids != nullptr && cparams.embeddings_nextn && !cparams.embeddings_nextn_masked;
    }

    //
    // common
    //

    ggml_tensor * build_cvec(
             ggml_tensor * cur,
                     int   il) const;

    // apply the activation-side transform of a Hadamard-folded weight, if any
    ggml_tensor * build_hadamard_input(
              ggml_tensor * w,
              ggml_tensor * cur) const;

    // do mat_mul, while optionally apply lora and per-tensor scale
    // w_s defaults to the weight scale of w
    ggml_tensor * build_lora_mm(
              ggml_tensor * w,
              ggml_tensor * cur,
              ggml_tensor * w_s = nullptr) const;

    // do mat_mul_id, while optionally apply lora and per-expert scale
    // if slots is set, the experts are selected by slots instead of ids, from the MoE cache if cached (see llama_moe_cache::build_lookup)
    ggml_tensor * build_lora_mm_id(
              ggml_tensor * w,   // ggml_tensor * as
              ggml_tensor * cur, // ggml_tensor * b
              ggml_tensor * ids,
              ggml_tensor * slots  = nullptr,
                     bool   cached = true) const;

    // rescale `cur` (the src1 of a matmul) by a power of two, computed from its maximum absolute
    // keep both individual values and Q8 activation block sums within F16 range
    llm_graph_amax_src1 build_amax_src1_f16(
              ggml_tensor * cur) const;

    ggml_tensor * build_norm(
             ggml_tensor * cur,
             ggml_tensor * mw,
             ggml_tensor * mb,
           llm_norm_type   type,
                     int   il) const;


    // compute Q, K, V projections with optional bias and reshape
    // supports both fused wqkv and separate wq/wk/wv paths
    llm_graph_qkv build_qkv(
        const llama_layer & layer,
              ggml_tensor * cur,
                  int64_t   n_embd_head,
                  int64_t   n_head,
                  int64_t   n_head_kv,
                      int   il) const;

    // Set reshape to false to return contiguous projections before clamp/reshape.
    llm_graph_qkv build_qkv(
        const llama_layer & layer,
              ggml_tensor * cur,
                  int64_t   n_embd_head_q,
                  int64_t   n_head_q,
                  int64_t   n_embd_head_k,
                  int64_t   n_head_k,
                  int64_t   n_embd_head_v,
                  int64_t   n_head_v,
                      int   il,
                     bool   reshape = true) const;

    ggml_tensor * build_ffn(
             ggml_tensor * cur,
             ggml_tensor * up,
             ggml_tensor * up_b,
             ggml_tensor * up_s,
             ggml_tensor * gate,
             ggml_tensor * gate_b,
             ggml_tensor * gate_s,
             ggml_tensor * down,
             ggml_tensor * down_b,
             ggml_tensor * down_s,
             ggml_tensor * act_scales,
         llm_ffn_op_type   type_op,
       llm_ffn_gate_type   type_gate,
                     int   il) const;

    // build MoE FFN without bias tensors
    ggml_tensor * build_moe_ffn(
             ggml_tensor * cur,
             ggml_tensor * gate_inp,
             ggml_tensor * up_exps,
             ggml_tensor * gate_exps,
             ggml_tensor * down_exps,
             ggml_tensor * exp_probs_b,
                 int64_t   n_expert,
                 int64_t   n_expert_used,
         llm_ffn_op_type   type_op,
                    bool   norm_w,
                   float   w_scale,
            llama_expert_gating_func_type gating_op,
                     int   il,
             ggml_tensor * probs_in = nullptr,
             ggml_tensor * gate_up_exps = nullptr,
             ggml_tensor * selected_experts_in = nullptr,
             ggml_tensor * post_exp_norm = nullptr,
                   float   post_exp_norm_eps = 1e-5f) const;

    ggml_tensor * build_moe_ffn(
             ggml_tensor * cur,
             ggml_tensor * gate_inp,
             ggml_tensor * gate_inp_b,
             ggml_tensor * up_exps,
             ggml_tensor * up_exps_b,
             ggml_tensor * gate_exps,
             ggml_tensor * gate_exps_b,
             ggml_tensor * down_exps,
             ggml_tensor * down_exps_b,
             ggml_tensor * exp_probs_b,
                 int64_t   n_expert,
                 int64_t   n_expert_used,
         llm_ffn_op_type   type_op,
                    bool   norm_w,
                   float   w_scale,
            llama_expert_gating_func_type gating_op,
                     int   il,
             ggml_tensor * probs_in = nullptr,
             ggml_tensor * gate_up_exps = nullptr,
             ggml_tensor * gate_up_exps_b = nullptr,
             ggml_tensor * selected_experts_in = nullptr,
             ggml_tensor * post_exp_norm = nullptr,
                   float   post_exp_norm_eps = 1e-5f) const;

    //
    // inputs
    //

    // tok_scale: applied to token rows only
    ggml_tensor * build_inp_embd(ggml_tensor * tok_embd, float tok_scale = 1.0f) const;

    // the state input of an MTP graph: rows of the h_nextn device buffer of ctx_state_src (see llama_set_ctx_state_src()),
    // or uploaded from ubatch.embd_state
    ggml_tensor * build_inp_embd_state(llm_graph_input_embd_h * inp, int64_t n_embd_state) const;

    ggml_tensor * build_inp_pos() const;

    // applies the RoPE of layer il, cur is returned as is if the layer has none
    ggml_tensor * build_rope(ggml_tensor * cur, ggml_tensor * pos, int il) const;

    // same, for the lightning indexer
    ggml_tensor * build_rope_indexer(ggml_tensor * cur, ggml_tensor * pos, int il) const;
    ggml_tensor * build_inp_attn_scale() const;
    ggml_tensor * build_inp_out_ids() const;
    ggml_tensor * build_inp_mean() const;
    ggml_tensor * build_inp_cls() const;
    ggml_tensor * build_inp_moe_ids(int64_t n_used, int64_t n_rows) const;

    ggml_tensor * build_inp_cross_embd() const;
    ggml_tensor * build_inp_pos_bucket_enc() const;
    ggml_tensor * build_inp_pos_bucket_dec() const;
    ggml_tensor * build_pos_bias(ggml_tensor * pos_bucket, ggml_tensor * attn_rel_b) const;

    //
    // attention
    //

    ggml_tensor * build_attn_mha(
            ggml_tensor * q,       // [n_embd_head_q, n_head_q, n_tokens]
            ggml_tensor * k,       // [n_embd_head_k, n_head_k, n_tokens]
            ggml_tensor * v,       // [n_embd_head_v, n_head_v, n_tokens] (v_trans = false)
            ggml_tensor * kq_b,
            ggml_tensor * kq_mask,
            ggml_tensor * kv_rows, // [n_kv, n_stream], k and v are shared by the streams
            ggml_tensor * sinks,   // [n_head_q]
            ggml_tensor * v_mla,   // [n_embd_head_v_mla, n_embd_head_v, n_head_v]
                int64_t   n_kv_max,
                  float   kq_scale,
                    int   il) const;

    llm_graph_input_attn_no_cache * build_attn_inp_no_cache() const;

    ggml_tensor * build_attn(
            llm_graph_input_attn_no_cache * inp,
            ggml_tensor * wo,
            ggml_tensor * wo_b,
            ggml_tensor * wo_s,
            ggml_tensor * q_cur, // [n_embd_head_q, n_head_q, n_tokens]
            ggml_tensor * k_cur, // [n_embd_head_k, n_head_k, n_tokens]
            ggml_tensor * v_cur, // [n_embd_head_v, n_head_v, n_tokens]
            ggml_tensor * kq_b,
            ggml_tensor * sinks, // [n_head_q]
            ggml_tensor * v_mla, // [n_embd_head_v_mla, n_embd_head_v, n_head_v]
                  float   kq_scale,
                    int   il) const;

    llm_graph_input_attn_kv * build_attn_inp_kv() const;

    ggml_tensor * build_attn(
            llm_graph_input_attn_kv * inp,
            ggml_tensor * wo,
            ggml_tensor * wo_b,
            ggml_tensor * wo_s,
            ggml_tensor * q_cur, // [n_embd_head_q, n_head_q, n_tokens]
            ggml_tensor * k_cur, // [n_embd_head_k, n_head_k, n_tokens]
            ggml_tensor * v_cur, // [n_embd_head_v, n_head_v, n_tokens]
            ggml_tensor * kq_b,
            ggml_tensor * sinks, // [n_head_q]
            ggml_tensor * v_mla, // [n_embd_head_v_mla, n_embd_head_v, n_head_v] // TODO: remove
                  float   kq_scale,
                    int   il) const;

    llm_graph_input_attn_k  * build_attn_inp_k() const;

    ggml_tensor * build_attn(
            llm_graph_input_attn_k * inp,
            ggml_tensor * wo,
            ggml_tensor * wo_b,
            ggml_tensor * wo_s,
            ggml_tensor * q_cur, // [n_embd_head_q, n_head_q, n_tokens]
            ggml_tensor * k_cur, // [n_embd_head_k, n_head_k, n_tokens]
            ggml_tensor * v_cur, // [n_embd_head_v, n_head_v, n_tokens]
            ggml_tensor * kq_b,
            ggml_tensor * sinks, // [n_head_q]
            ggml_tensor * v_mla, // [n_embd_head_v_mla, n_embd_head_v, n_head_v]
                  float   kq_scale,
                    int   il) const;

    llm_graph_input_attn_k_dsa * build_attn_inp_k_dsa() const;

    llm_graph_input_attn_k_dsa_iswa * build_attn_inp_k_dsa_iswa() const;

    llm_graph_input_attn_kv_msa * build_attn_inp_kv_msa(bool msa_enabled) const;

    ggml_tensor * build_attn(
            llm_graph_input_attn_k_dsa * inp,
            ggml_tensor * wo,
            ggml_tensor * wo_b,
            ggml_tensor * wo_s,
            ggml_tensor * q_cur, // [n_embd_head_q, n_head_q, n_tokens]
            ggml_tensor * k_cur, // [n_embd_head_k, n_head_k, n_tokens]
            ggml_tensor * v_cur, // [n_embd_head_v, n_head_v, n_tokens]
            ggml_tensor * kq_b,
            ggml_tensor * sinks, // [n_head_q]
            ggml_tensor * v_mla, // [n_embd_head_v_mla, n_embd_head_v, n_head_v]
            ggml_tensor * top_k, // [n_indexer_top_k, n_tokens]
                  float   kq_scale,
                    int   il) const;

    // ggml_top_k_mask with top_k viewed in the row layout of kq_mask, see ggml.h
    ggml_tensor * build_fused_top_k_mask(
            ggml_tensor * kq_mask, // [n_kv, n_tokens/n_stream, 1, n_stream]
            ggml_tensor * top_k,   // [n_top, n_tokens]
                    int   il) const;

    // kq_mask with only the cells named by top_k left visible, for sparse attention
    ggml_tensor * build_top_k_mask(
            ggml_tensor * kq_mask, // [n_kv, n_tokens/n_stream, 1, n_stream]
            ggml_tensor * top_k,   // [n_top, n_tokens]
                    int   il) const;

    llm_graph_input_attn_kv_iswa * build_attn_inp_kv_iswa() const;

    llm_graph_input_dsv4 * build_inp_dsv4() const;

    // note: if k_cur or v_cur are not provided, they will not be stored in the memory
    ggml_tensor * build_attn(
            llm_graph_input_attn_kv_iswa * inp,
            ggml_tensor * wo,
            ggml_tensor * wo_b,
            ggml_tensor * wo_s,
            ggml_tensor * q_cur, // [n_embd_head_q, n_head_q, n_tokens]
            ggml_tensor * k_cur, // [n_embd_head_k, n_head_k, n_tokens] optional
            ggml_tensor * v_cur, // [n_embd_head_v, n_head_v, n_tokens] optional
            ggml_tensor * kq_b,
            ggml_tensor * sinks, // [n_head_q]
            ggml_tensor * v_mla, // [n_embd_head_v_mla, n_embd_head_v, n_head_v]
                  float   kq_scale,
                    int   il) const;

    llm_graph_input_attn_k_iswa * build_attn_inp_k_iswa() const;

    // note: if k_cur is not provided, it will not be stored in the memory
    // note: the K cache is used as V (MLA-style attention)
    ggml_tensor * build_attn(
            llm_graph_input_attn_k_iswa * inp,
            ggml_tensor * wo,
            ggml_tensor * wo_b,
            ggml_tensor * wo_s,
            ggml_tensor * q_cur, // [n_embd_head_q, n_head_q, n_tokens]
            ggml_tensor * k_cur, // [n_embd_head_k, n_head_k, n_tokens] optional
            ggml_tensor * v_cur, // [n_embd_head_v, n_head_v, n_tokens] optional
            ggml_tensor * kq_b,
            ggml_tensor * sinks, // [n_head_q]
            ggml_tensor * v_mla, // [n_embd_head_v_mla, n_embd_head_v, n_head_v]
                  float   kq_scale,
                    int   il) const;

    llm_graph_input_attn_cross * build_attn_inp_cross() const;

    ggml_tensor * build_attn(
            llm_graph_input_attn_cross * inp,
            ggml_tensor * wo,
            ggml_tensor * wo_b,
            ggml_tensor * wo_s,
            ggml_tensor * q_cur, // [n_embd_head_q, n_head_q, n_tokens]
            ggml_tensor * k_cur, // [n_embd_head_k, n_head_k, n_tokens]
            ggml_tensor * v_cur, // [n_embd_head_v, n_head_v, n_tokens]
            ggml_tensor * kq_b,
            ggml_tensor * sinks, // [n_head_q]
            ggml_tensor * v_mla, // [n_embd_head_v_mla, n_embd_head_v, n_head_v]
                  float   kq_scale,
                    int   il) const;

    //
    // recurrent
    //

    // TODO: move this implementation to llama_memory_recurrent.
    //       this is analogous to llama_kv_cache::cpy_k / cpy_v
    //       when moving, avoid passing `ggml_cgraph` - only pass `ggml_context`. would likely need to split the
    //         implementation in 2 separate methods. the goal is to avoid calling `ggml_build_forward_expand` in
    //         `llama_memory_recurrent`
    ggml_tensor * build_rs(
            ggml_tensor * s,
            ggml_tensor * state_copy,
            ggml_tensor * state_copy_main,
                int32_t   state_size,
                int32_t   n_seqs,
               uint32_t   n_rs,
               uint32_t   rs_head,
               uint32_t   rs_size,
                int32_t   rs_zero,
            const llm_graph_get_rows_fn & get_state_rows = nullptr,
            ggml_tensor * state_write_extra = nullptr) const; // Continuum

    llm_graph_input_rs * build_rs_inp() const;

    ggml_tensor * build_rs(
            llm_graph_input_rs * inp,
            ggml_tensor * s,
                int32_t   state_size,
                int32_t   n_seqs,
            const llm_graph_get_rows_fn & get_state_rows = nullptr) const;

    // store the conv history of the ubatch cells, the last n_cols columns of conv_x {n_cols + n_seq_tokens, channels, n_seqs},
    // at byte offset offs of each row of s, and with n_rs_seq > 0 once per rollback slot of this ubatch
    void build_conv_state_store(
            llm_graph_input_rs * inp,
            ggml_tensor * s,
            ggml_tensor * conv_x,
                int64_t   n_cols,
                 size_t   offs) const;

    // Continuum: what a training graph adds to the serving graph (llama-graph-train.cpp)

    // A training chunk's cache rows: I32 [n_tokens], its positions (in the walk a position
    // is its cell). One per graph, shared by every attention layer.
    ggml_tensor * build_inp_train_rows() const;

    // whether an attached adapter's term is in this graph: every one but the shadow always, the
    // shadow only when the ubatch has labelled rows
    bool lora_on(const llama_adapter_lora * adapter) const;
    // base plus the shadow's term, on the labelled rows only (Cormac, 2026-10-14): those rows of
    // base and of the term are gathered and added, and written back into a copy of base; every
    // other row is base, byte for byte. A multiplied 0/1 mask was not: an untrained layer's inf or
    // NaN times zero is NaN, and x + 0 turns a -0 into +0, on a serving row co-batched with
    // training.
    ggml_tensor * shadow_add(ggml_tensor * base, ggml_tensor * term) const;

    // Whether this layer's attention runs through FLASH_ATTN_EXT: the one decision
    // build_attn_mha makes and a training graph reads to attend to the cache in place.
    bool uses_flash_attn(ggml_tensor * kq_b, ggml_tensor * sinks, ggml_tensor * v_mla) const;

    // the K/V a training graph attends to (her cache, in place or as constants before the chunk); false when the
    // graph is not training, leaving serving's path to the caller; kq_mask leaves trimmed to the attended cells
    bool build_attn_train(
            const llama_kv_cache_context * mctx_cur,
            llm_graph_input_attn_kv * inp,
            ggml_tensor * k_cur,
            ggml_tensor * v_cur,
            ggml_tensor * kq_b,
            ggml_tensor * sinks,
            ggml_tensor * v_mla,
                    int   il,
            ggml_tensor *& kq_mask,
            ggml_tensor *& k,
            ggml_tensor *& v) const;
    // the walk keeps its chunk's F32 K/V where flash attention would cast them to F16
    bool walk_keeps_f32() const;
    // a training graph's quantized K/V read as F16 through a node of its own
    void train_fa_kv(ggml_tensor *& k, ggml_tensor *& v) const;
    // a training graph's flash attention node: its backward tile and the walk's grad sinks
    void train_fa_node(ggml_tensor * cur) const;

    // THE ONE RULE for how many recurrent states a graph keeps per sequence, read by every family's
    // state write (delta-net, mamba, lfm2, bailing) and by the ops that produce snapshots. Serving with
    // rollback slots keeps K = n_rs_seq + 1 (group g = the state g tokens back, so a draft can be undone);
    // a context without them, and a TRAINING graph, keep one: the live state, group 0, which is the same
    // bytes in both layouts. So training reads and writes the state serving reads, in the serving
    // context; the undo groups it leaves stale are rewritten by the next serving decode, and the walk
    // refuses a sequence with an undo pending (llama_memory_recurrent::rollback_pending).
    int64_t rs_keep() const;
    // the exact walk's tap on the state entering the chunk
    void train_rs_tap(ggml_tensor * s, ggml_tensor * output_states) const;
    // the extra states written by index when the graph carries write rows, else at the head's offset
    void build_rs_extra(
            ggml_tensor * s,
            ggml_tensor * states,
            ggml_tensor * states_extra,
                int32_t   state_size,
                int32_t   n_seqs,
               uint32_t   n_rs,
               uint32_t   rs_head,
            ggml_tensor * state_write_extra) const;
    // one sequence's last conv state written to its cell, and offered to the walk's exit surrogate
    void build_conv_state_write(
            llm_graph_input_rs * inp,
            ggml_tensor * s,
            ggml_tensor * last,
            ggml_tensor * cell,
                 size_t   offs) const;
    // Qwen3.5's recurrent write rows, so its serving graph never depends on the cache head
    void build_rs_write_rows(llm_graph_input_rs * inp_rs, const llama_memory_recurrent_context * rs) const;

    // THE EXACT WALK: the state a chunk leaves (`exit`, what is written back to `cache`) meets the
    // gradient later chunks put on it, through the surrogate <exit, gs>. A no-op outside the walk.
    void build_walk_state_exit(ggml_tensor * cache, ggml_tensor * exit) const;
    // rows [row, row + n) of a layer's walk accumulator, shaped like `like`'s K / V span
    ggml_tensor * walk_acc_rows(ggml_tensor * acc, const ggml_tensor * like, int64_t row) const;
    // <k_cur, G_k> + <v_cur, G_v>: what later chunks of the run put on this chunk's own K/V
    // (accumulator rows from `row`), carried into it through the loss
    void build_walk_surrogate(int il, ggml_tensor * k_cur, ggml_tensor * v_cur, int64_t row) const;

    bool build_training_head() const;
    llm_graph_input_training_rows * training_rows() const;

    // an embeddings-only graph keeps the pooled output and every cache write and drops the logits branch
    // (llama-graph-embeddings.cpp)
    void prune_to_embeddings(const llm_graph_params & params) const;

    ggml_tensor * build_rwkv_token_shift_load(
        llm_graph_input_rs * inp,
        const llama_ubatch & ubatch,
                       int   il) const;

    ggml_tensor * build_rwkv_token_shift_store(
             ggml_tensor * token_shift,
      const llama_ubatch & ubatch,
                     int   il) const;
    //
    // hybrid
    //

    llm_graph_input_mem_hybrid * build_inp_mem_hybrid() const;
    llm_graph_input_mem_hybrid_k * build_inp_mem_hybrid_k() const;

    llm_graph_input_mem_hybrid_iswa * build_inp_mem_hybrid_iswa() const;

    // extent, extent_swa: the distance that the bias tables of the full and the SWA layers stop at
    llm_graph_input_mem_hybrid_iswa_rel * build_inp_mem_hybrid_iswa_rel(
            uint32_t extent,
            uint32_t extent_swa) const;

    // kq_b is added to the KQ mask, after the KQ scale
    ggml_tensor * build_attn(
            llm_graph_input_mem_hybrid_iswa_rel * inp,
            ggml_tensor * wo,
            ggml_tensor * wo_b,
            ggml_tensor * wo_s,
            ggml_tensor * q_cur, // [n_embd_head_q, n_head_q, n_tokens]
            ggml_tensor * k_cur, // [n_embd_head_k, n_head_k, n_tokens]
            ggml_tensor * v_cur, // [n_embd_head_v, n_head_v, n_tokens]
            ggml_tensor * kq_b,  // [n_kv, n_tokens/n_stream, n_head_q, n_stream]
            ggml_tensor * sinks, // [n_head_q]
                  float   kq_scale,
                    int   il) const;

    //
    // pooling
    //

    void build_pooling(
            ggml_tensor * cls,
            ggml_tensor * cls_b,
            ggml_tensor * cls_out,
            ggml_tensor * cls_out_b,
            ggml_tensor * cls_norm) const;

    //
    // logits
    //

    void build_logits() const;

    //
    // sampling (backend sampling)
    //

    void build_sampling() const;

    // write enabled layer inputs into layer_inp_dev at their batch rows
    void build_layer_inp_dev() const;

    // write the unmasked h_nextn rows into h_nextn_dev at their batch rows
    void build_h_nextn_dev() const;

    //
    // dense (out)
    //

    void build_dense_out(
            ggml_tensor * dense_2,
            ggml_tensor * dense_2_b,
            ggml_tensor * dense_3) const;
};

// TODO: better name
int32_t llama_relative_position_bucket(llama_pos x, llama_pos y, uint64_t n_buckets, bool bidirectional);
