// FLASH_ATTN_EXT's gradients through either backward (the native ggml_flash_attn_back or the
// composed tiles), against finite differences: GQA, a causal mask, q, k and v all trained
struct test_flash_attn_ext_grad : public test_case {
    const int64_t hs, nh, nhkv, kv, nb;
    const bool    native;

    std::string vars() override {
        return VARS_TO_STR6(hs, nh, nhkv, kv, nb, native);
    }

    test_flash_attn_ext_grad(int64_t hs = 16, int64_t nh = 4, int64_t nhkv = 2, int64_t kv = 24, int64_t nb = 8, bool native = true)
        : hs(hs), nh(nh), nhkv(nhkv), kv(kv), nb(nb), native(native) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hs, nb, nh);
        ggml_tensor * k = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hs, kv, nhkv);
        ggml_tensor * v = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hs, kv, nhkv);
        ggml_set_param(q); ggml_set_name(q, "q");
        ggml_set_param(k); ggml_set_name(k, "k");
        ggml_set_param(v); ggml_set_name(v, "v");
        ggml_tensor * m = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, kv, nb);
        ggml_set_name(m, "m");
        ggml_tensor * out = ggml_flash_attn_ext(ctx, q, k, v, m, 1.0f/sqrtf((float) hs), 0.0f, 0.0f);
        ggml_prec_set_acc(out, GGML_PREC_F32);
        ggml_flash_attn_ext_set_back_tile(out, native ? GGML_FA_BACK_NATIVE : 0);
        // weighted by a fixed random tensor: a plain sum of attention outputs nearly cancels the
        // query gradient (each row's weights sum to one), and the finite difference drowns in it
        ggml_tensor * w = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, out->ne[0], out->ne[1], out->ne[2]);
        ggml_set_name(w, "w");
        out = ggml_mul(ctx, out, w);
        ggml_set_name(out, "out");
        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            if (strcmp(t->name, "m") == 0) {
                // causal: query row i sees KV positions j <= kv - nb + i
                std::vector<ggml_fp16_t> m(ggml_nelements(t));
                for (int64_t i = 0; i < t->ne[1]; ++i) {
                    for (int64_t j = 0; j < t->ne[0]; ++j) {
                        m[i*t->ne[0] + j] = ggml_fp32_to_fp16(j <= kv - nb + i ? 0.0f : -INFINITY);
                    }
                }
                ggml_backend_tensor_set(t, m.data(), 0, ggml_nbytes(t));
            } else {
                init_tensor_uniform(t);
            }
        }
    }

    bool grad_precise() override {
        return true;
    }
};

// GGML_OP_FLASH_ATTN_BACK itself: each backend's kernel against the CPU reference, at the cache
// types training reads (F16, F32, q8_0) and a GQA group
struct test_flash_attn_back : public test_case {
    const int64_t   hs, nh, nhkv, kv, nb;
    const ggml_type type_kv;
    const int64_t   kv0; // the first key row with a dK/dV (a frozen prefix before it)

    std::string vars() override {
        return VARS_TO_STR7(hs, nh, nhkv, kv, nb, type_kv, kv0);
    }

    test_flash_attn_back(int64_t hs = 128, int64_t nh = 8, int64_t nhkv = 2, int64_t kv = 512, int64_t nb = 64, ggml_type type_kv = GGML_TYPE_F16,
            int64_t kv0 = 0)
        : hs(hs), nh(nh), nhkv(nhkv), kv(kv), nb(nb), type_kv(type_kv), kv0(kv0) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hs, nb, nh);
        ggml_tensor * k = ggml_new_tensor_3d(ctx, type_kv, hs, kv, nhkv);
        ggml_tensor * v = ggml_new_tensor_3d(ctx, type_kv, hs, kv, nhkv);
        ggml_tensor * m = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, kv, nb);
        ggml_tensor * d = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hs, nh, nb);
        ggml_set_name(q, "q"); ggml_set_name(k, "k"); ggml_set_name(v, "v"); ggml_set_name(m, "m"); ggml_set_name(d, "d");
        ggml_tensor * out = ggml_flash_attn_back(ctx, q, k, v, m, d, 1.0f/sqrtf((float) hs), kv0);
        ggml_set_name(out, "out");
        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            if (strcmp(t->name, "m") == 0) {
                init_tensor_kq_mask(t);
            } else {
                init_tensor_uniform(t);
            }
        }
    }

    double max_nmse_err() override {
        return 1e-5; // the kernel and the reference sum in different orders over the window
    }

    // the five products of the ideal backward (S, dP, dV, dQ, dK), so perf reports TFLOPS against
    // the forward's two
    uint64_t op_flops(ggml_tensor * t) override {
        GGML_UNUSED(t);
        return 2ull*nb*nh*(3*kv*hs + 2*(kv - kv0)*hs); // pass 1 over every key, pass 2 from kv0
    }
};
