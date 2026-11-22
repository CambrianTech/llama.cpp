static void ggml_compute_forward_batched_get_rows_back_f32(
        const ggml_compute_params * params,
              ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    if (params->ith != 0) {
        return;
    }

    GGML_ASSERT(ggml_is_contiguous(dst));

    // ggml_compute_forward_dup_same_cont(params, opt0, dst);

    memset(dst->data, 0, ggml_nbytes(dst));

    const int nc = src0->ne[0];

    GGML_ASSERT( dst->ne[0] == nc);
    GGML_ASSERT(src0->nb[0] == sizeof(float));

    // batch for batch (ggml_get_rows_back): index k of batch (i2, i3) adds its gradient row
    // into that batch's row r of dst
    for (int64_t i3 = 0; i3 < src1->ne[2]; ++i3) {
        for (int64_t i2 = 0; i2 < src1->ne[1]; ++i2) {
            for (int64_t k = 0; k < src1->ne[0]; ++k) {
                const int32_t r = *(const int32_t *) ((const char *) src1->data + k*src1->nb[0] + i2*src1->nb[1] + i3*src1->nb[2]);
                GGML_ASSERT(r >= 0 && r < dst->ne[1]);

                float * d = (float *) ((char *) dst->data + r*dst->nb[1] + i2*dst->nb[2] + i3*dst->nb[3]);
                ggml_vec_add_f32(nc, d, d, (const float *) ((const char *) src0->data + k*src0->nb[1] + i2*src0->nb[2] + i3*src0->nb[3]));
            }
        }
    }
}
