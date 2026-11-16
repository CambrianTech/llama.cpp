// what this catches: a checkpoint as a PARKED SEQUENCE (CONTINUUM-ENGINE 4.7.11), by id, never
// serialized. The live sequence is copied to a parked id with llama_memory_seq_cp (attention KV cells
// shared by sequence id, the recurrent state shared and copied on the device only when the live
// sequence advances), the live sequence moves on through different tokens, then is restored from the
// parked id; its next-token logits must equal a cold run's over the same prefix. Also times park and
// restore against today's path (llama_state_seq_get_data serializing the sequence to host bytes).
//
// usage: test-seq-park -m <model.gguf>  (a hybrid or recurrent model exercises the recurrent path)
#include "arg.h"
#include "common.h"
#include "llama.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

static int64_t now_us() {
    return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static llama_context * make_ctx(const common_params & params, llama_model * model) {
    auto cparams       = common_context_params_to_llama(params);
    cparams.n_seq_max  = 2;    // seq 0 live, seq 1 parked
    cparams.n_ctx      = 1024;
    cparams.n_batch    = 512;
    cparams.n_ubatch   = 512;
    cparams.kv_unified = true; // one stream: a parked id shares the live sequence's cells
    return llama_init_from_model(model, cparams);
}

static bool decode(llama_context * ctx, const std::vector<llama_token> & toks, llama_pos p0, llama_seq_id seq) {
    llama_batch batch = llama_batch_init((int32_t) toks.size(), 0, 1);
    for (size_t i = 0; i < toks.size(); ++i) {
        common_batch_add(batch, toks[i], p0 + (llama_pos) i, { seq }, i + 1 == toks.size());
    }
    const bool ok = llama_decode(ctx, batch) == 0;
    llama_batch_free(batch);
    return ok;
}

static std::vector<float> last_logits(llama_context * ctx, int n_vocab) {
    const float * l = llama_get_logits_ith(ctx, -1);
    return std::vector<float>(l, l + n_vocab);
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    common_params params;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }
    common_init();
    llama_backend_init();
    llama_model * model = llama_model_load_from_file(params.model.path.c_str(), common_model_params_to_llama(params));
    if (model == nullptr) {
        std::fprintf(stderr, "FAILED: model did not load\n");
        return 1;
    }
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    std::printf("model: recurrent=%d hybrid=%d\n", (int) llama_model_is_recurrent(model), (int) llama_model_is_hybrid(model));

    constexpr int N = 300, M = 120; // the prefix to keep; tokens the live sequence advances through
    std::vector<llama_token> prefix, detour;
    for (int i = 0; i < N; ++i) prefix.push_back((llama_token) (1000 + (i * 7919) % 20000));
    for (int i = 0; i < M; ++i) detour.push_back((llama_token) (2000 + (i * 104729) % 20000));
    const llama_token next = 4242;

    // reference: a cold run over prefix + next
    std::vector<float> ref;
    {
        llama_context * ctx = make_ctx(params, model);
        std::vector<llama_token> all = prefix;
        all.push_back(next);
        if (!decode(ctx, all, 0, 0)) { std::fprintf(stderr, "FAILED: reference decode\n"); return 1; }
        ref = last_logits(ctx, n_vocab);
        llama_free(ctx);
    }

    llama_context * ctx = make_ctx(params, model);
    llama_memory_t mem = llama_get_memory(ctx);
    if (!decode(ctx, prefix, 0, 0)) { std::fprintf(stderr, "FAILED: prefix decode\n"); return 1; }
    llama_synchronize(ctx);

    // today's checkpoint: serialize the live sequence to host bytes
    const int64_t t_ser0 = now_us();
    std::vector<uint8_t> blob(llama_state_seq_get_size(ctx, 0));
    const size_t written = llama_state_seq_get_data(ctx, blob.data(), blob.size(), 0);
    const int64_t t_ser = now_us() - t_ser0;

    // the parked checkpoint: a sequence id, no bytes moved
    const int64_t t_park0 = now_us();
    llama_memory_seq_cp(mem, 0, 1, -1, -1);
    const int64_t t_park = now_us() - t_park0;

    // the live sequence moves on through other tokens (this is where the recurrent state is copied on
    // the device, copy-on-write, inside the step's own graph)
    if (!decode(ctx, detour, N, 0)) { std::fprintf(stderr, "FAILED: detour decode\n"); return 1; }
    llama_synchronize(ctx);

    // restore: drop the live sequence, take the parked one's cells and state by id
    const int64_t t_rest0 = now_us();
    if (!llama_memory_seq_rm(mem, 0, -1, -1)) { std::fprintf(stderr, "FAILED: seq_rm\n"); return 1; }
    llama_memory_seq_cp(mem, 1, 0, -1, -1);
    const int64_t t_rest = now_us() - t_rest0;

    if (!decode(ctx, { next }, N, 0)) { std::fprintf(stderr, "FAILED: restored decode\n"); return 1; }
    const std::vector<float> got = last_logits(ctx, n_vocab);

    // the server's form: a checkpoint holds only the PARTIAL state (recurrent / SWA), attention KV is
    // trimmed by position. Park only the recurrent cell (an empty attention range [0, 0) copies no
    // KV cells), advance, then restore the recurrent cell by id and trim attention past the prefix.
    // start from a clean prefix: the recurrent state cannot be trimmed back past `next`
    llama_memory_seq_rm(mem, 1, -1, -1);
    llama_memory_seq_rm(mem, 0, -1, -1);
    if (!decode(ctx, prefix, 0, 0)) { std::fprintf(stderr, "FAILED: second prefix decode\n"); return 1; }
    llama_synchronize(ctx);
    const int64_t t_ppark0 = now_us();
    llama_memory_seq_cp(mem, 0, 1, 0, 0);
    const int64_t t_ppark = now_us() - t_ppark0;
    const llama_pos parked_kv_max = llama_memory_seq_pos_max(mem, 1);
    if (!decode(ctx, detour, N, 0)) { std::fprintf(stderr, "FAILED: second detour\n"); return 1; }
    llama_synchronize(ctx);
    const int64_t t_prest0 = now_us();
    llama_memory_seq_cp(mem, 1, 0, 0, 0);                       // the recurrent cell, by id
    const bool trimmed = llama_memory_seq_rm(mem, 0, N, -1);    // attention past the prefix
    const int64_t t_prest = now_us() - t_prest0;
    if (!trimmed) { std::fprintf(stderr, "FAILED: attention trim after the partial restore\n"); return 1; }
    if (!decode(ctx, { next }, N, 0)) { std::fprintf(stderr, "FAILED: partial restored decode\n"); return 1; }
    const std::vector<float> got_partial = last_logits(ctx, n_vocab);
    float max_diff_partial = 0.0f;
    int   argmax_partial   = 0;
    for (int i = 0; i < n_vocab; ++i) {
        max_diff_partial = std::max(max_diff_partial, std::fabs(ref[i] - got_partial[i]));
        if (got_partial[i] > got_partial[argmax_partial]) argmax_partial = i;
    }
    // today's partial checkpoint, for the same comparison (the size of the state it would copy)
    const int64_t t_pser0 = now_us();
    std::vector<uint8_t> pblob(llama_state_seq_get_size_ext(ctx, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY));
    const size_t pwritten = llama_state_seq_get_data_ext(ctx, pblob.data(), pblob.size(), 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    const int64_t t_pser = now_us() - t_pser0;

    float max_diff = 0.0f;
    int   argmax_ref = 0, argmax_got = 0;
    for (int i = 0; i < n_vocab; ++i) {
        max_diff = std::max(max_diff, std::fabs(ref[i] - got[i]));
        if (ref[i] > ref[argmax_ref]) argmax_ref = i;
        if (got[i] > got[argmax_got]) argmax_got = i;
    }
    std::printf("serialized checkpoint: %zu bytes in %lld us (host copy of the sequence)\n", written, (long long) t_ser);
    std::printf("parked checkpoint: 0 bytes moved, park %lld us, restore %lld us\n", (long long) t_park, (long long) t_rest);
    std::printf("restored vs cold logits: max |diff| = %.6g, argmax %d vs %d\n", max_diff, argmax_got, argmax_ref);
    std::printf("partial (server form): serialized %zu bytes in %lld us; parked 0 bytes (attention cells held by the parked id: pos_max %d), park %lld us, restore %lld us\n",
                pwritten, (long long) t_pser, (int) parked_kv_max, (long long) t_ppark, (long long) t_prest);
    std::printf("partial restored vs cold logits: max |diff| = %.6g, argmax %d vs %d\n", max_diff_partial, argmax_partial, argmax_ref);

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    // the same backend, the same tokens, the same positions: the restored state must be the cold one
    if (argmax_partial != argmax_ref || max_diff_partial > 1e-2f) {
        std::fprintf(stderr, "FAILED: the partial (recurrent-only) restore does not reproduce the cold run\n");
        return 1;
    }
    if (argmax_got != argmax_ref || max_diff > 1e-2f) {
        std::fprintf(stderr, "FAILED: the restored sequence does not reproduce the cold run\n");
        return 1;
    }
    std::printf("test-seq-park: ok\n");
    return 0;
}
