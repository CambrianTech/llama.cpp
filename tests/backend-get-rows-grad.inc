// GET_ROWS's gradient on batched rows (a MoE gathers its router weights per token: [1, n_expert,
// n_tokens] by [n_used, n_tokens]), checked by finite differences. Every row of every batch is
// gathered (r >= m) and the output is weighted by a fixed random tensor: the suite's metric,
// (a-b)/(a+b) averaged, is NaN on any element whose gradient is 0 on both sides, and a NaN mean
// never exceeds the threshold, so a row nothing gathers would pass the case whatever the kernel did.
struct test_get_rows_grad : public test_case {
    const int n; // cols
    const int m; // rows
    const int r; // rows to get, >= m
    const int be1; // batches

    std::string vars() override {
        return VARS_TO_STR4(n, m, r, be1);
    }

    std::string op_desc(ggml_tensor * t) override {
        GGML_UNUSED(t);
        return "GET_ROWS";
    }

    test_get_rows_grad(int n = 1, int m = 6, int r = 9, int be1 = 7) : n(n), m(m), r(r), be1(be1) {
        GGML_ASSERT(r >= m);
    }

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * in = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n, m, be1);
        ggml_set_name(in, "in");
        ggml_set_param(in);

        ggml_tensor * rows = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, r, be1);
        ggml_set_name(rows, "rows");

        ggml_tensor * w = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n, r, be1);
        ggml_set_name(w, "w");

        ggml_tensor * out = ggml_mul(ctx, ggml_get_rows(ctx, in, rows), w);
        ggml_set_name(out, "out");

        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        std::random_device rd;
        std::default_random_engine rng(rd());
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            if (t->type == GGML_TYPE_I32) {
                for (int b = 0; b < be1; b++) {
                    std::vector<int32_t> data(r);
                    for (int k = 0; k < r; k++) {
                        data[k] = k % m; // every row at least once
                    }
                    std::shuffle(data.begin(), data.end(), rng);
                    ggml_backend_tensor_set(t, data.data(), b*t->nb[1], r*sizeof(int32_t));
                }
            } else {
                init_tensor_uniform(t);
            }
        }
    }
};
