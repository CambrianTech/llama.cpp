#pragma once

#include "ggml.h"

//
// cache line
//

#if defined(__cpp_lib_hardware_interference_size)
#define CACHE_LINE_SIZE std::hardware_destructive_interference_size
#else
#if defined(__POWER9_VECTOR__)
#define CACHE_LINE_SIZE 128
#elif defined(__VXE__) || defined(__VXE2__)
#define CACHE_LINE_SIZE 256
#else
#define CACHE_LINE_SIZE 64
#endif
#endif

static const size_t CACHE_LINE_SIZE_F32 = CACHE_LINE_SIZE/sizeof(float);

// Work buffer size for im2col operations in CONV2D
#define GGML_IM2COL_WORK_SIZE (16 * 1024 * 1024)

#ifdef __cplusplus
extern "C" {
#endif

void ggml_compute_forward_dup(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_add(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_add_id(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_add1(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_acc(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_sum(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_sum_rows(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_cumsum(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_mean(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_argmax(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_count_equal(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_repeat(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_repeat_back(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_concat(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_silu_back(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_norm(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_rms_norm(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_rms_norm_mul_fused(const struct ggml_compute_params * params, struct ggml_tensor * dst_rms_norm, struct ggml_tensor * dst_mul);
void ggml_compute_forward_rms_norm_back(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_group_norm(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_l2_norm(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_out_prod(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_mul_mat_id_back(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_scale(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_set(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_cpy(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_cont(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_get_rows(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_get_rows_back(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_set_rows(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_diag(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_diag_mask_inf(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_diag_mask_zero(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_soft_max(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_soft_max_ext_back(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_rope(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_rope_back(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_clamp(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_conv_transpose_1d(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_im2col(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_im2col_back_f32(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_im2col_3d(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_col2im_1d(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_conv_2d(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_conv_3d(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_conv_transpose_2d(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_conv_2d_dw(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_pool_1d(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_pool_2d(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_pool_2d_back(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_upscale(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_pad(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_pad_reflect_1d(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_roll(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_arange(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_timestep_embedding(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_argsort(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_top_k(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_leaky_relu(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_tri(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_fill(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_flash_attn_ext(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_flash_attn_back(const struct ggml_compute_params * params, struct ggml_tensor * dst);
// OUT_PROD with a quantized src0: BN dst rows per task share one dequantization of each KT-row tile
// of src0. Its per-thread scratch, in floats, for a row of ne0; the work planner reserves this.
#define GGML_OUT_PROD_BN 32
#define GGML_OUT_PROD_KT 32
static inline int64_t ggml_out_prod_cpu_wsize(int64_t ne0) {
    return GGML_OUT_PROD_KT*ne0 + GGML_OUT_PROD_BN*GGML_OUT_PROD_KT;
}

// ggml_compute_forward_mul_mat_id_back's work buffer, in bytes: the routing grouped by expert
// with a counting sort (an offset per expert, plus one, then every (slot, token) pair once),
// and each task's dequantized tile of GGML_OUT_PROD_KT expert rows.
static inline size_t ggml_mul_mat_id_back_cpu_wsize(int64_t cols, int64_t n_expert, int64_t n_slots, int n_tasks) {
    return (size_t) (n_expert + 1)*sizeof(int64_t) + (size_t) n_slots*2*sizeof(int32_t) +
           (size_t) n_tasks*GGML_OUT_PROD_KT*cols*sizeof(float) + 2*CACHE_LINE_SIZE;
}
// FLASH_ATTN_BACK's blocks: BR query rows (pass 1) and BC KV positions (pass 2) share one conversion of
// each K and V row. Its per-thread scratch, in floats; the work-size planner reserves this per task.
#define GGML_FA_BACK_BR 8
#define GGML_FA_BACK_BC 16
static inline int64_t ggml_flash_attn_back_cpu_wsize(int64_t D, int64_t Dv, int64_t M) {
    const int64_t pass1 = GGML_FA_BACK_BR*(M + D + Dv) + D + Dv;
    const int64_t pass2 = GGML_FA_BACK_BC*(D + Dv) + D + Dv;
    return pass1 > pass2 ? pass1 : pass2;
}
void ggml_compute_forward_ssm_conv(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_ssm_scan(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_win_part(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_win_unpart(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_unary(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_glu(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_get_rel_pos(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_add_rel_pos(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_rwkv_wkv6(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_rwkv_wkv7(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_solve_tri(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_gla(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_gated_delta_net(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_gated_delta_net_back(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_lightning_indexer(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_dsv4_hc_comb(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_dsv4_hc_pre(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_dsv4_hc_post(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_map_custom1(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_map_custom2(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_map_custom3(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_custom(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_cross_entropy_loss(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_cross_entropy_loss_back(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_opt_step_adamw(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_mul_mat(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_fwht(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_opt_step_sgd(const struct ggml_compute_params * params, struct ggml_tensor * dst);
#ifdef __cplusplus
}
#endif
