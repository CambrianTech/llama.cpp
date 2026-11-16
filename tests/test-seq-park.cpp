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
#if defined(__APPLE__)
#include <mach/mach.h>
#endif
#include <cstdlib>
#include <vector>

static int64_t now_us() {
    return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static llama_context * make_ctx(const common_params & params, llama_model * model) {
    auto cparams       = common_context_params_to_llama(params);
    cparams.n_seq_max  = 2;    // seq 0 live, seq 1 parked
    cparams.n_ctx      = 8192;
    cparams.n_batch    = 4096;
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

    // the prefix to keep (TEST_SEQ_PARK_N overrides, to measure how a checkpoint grows with context)
    const int N = std::getenv("TEST_SEQ_PARK_N") ? std::atoi(std::getenv("TEST_SEQ_PARK_N")) : 300;
    constexpr int M = 120; // tokens the live sequence advances through
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

    // ---- the revised design: the target's partial state as a DEVICE SNAPSHOT keyed by snapshot id, on a
    // context with ONE sequence (no parked ids, no extra rollback rows), plus the fault replays from
    // Codex's review of 7ca9509f0
    {
        auto cparams       = common_context_params_to_llama(params);
        cparams.n_seq_max  = 1;
        cparams.n_ctx      = 8192;
        cparams.n_batch    = 4096;
        cparams.n_ubatch   = 512;
        llama_context * sctx = llama_init_from_model(model, cparams);
        llama_memory_t  smem = llama_get_memory(sctx);
        const auto P = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;
        auto require = [](bool ok, const char * what) { if (!ok) { std::fprintf(stderr, "FAILED: %s\n", what); std::exit(1); } };
        auto restored_matches_cold = [&](uint64_t id, const char * what) {
            require(llama_state_seq_snapshot_load(sctx, 0, id, P) == LLAMA_SNAPSHOT_OK, what);
            require(llama_memory_seq_rm(smem, 0, N, -1), "trim attention after the restore");
            require(decode(sctx, { next }, N, 0), "decode after the restore");
            const std::vector<float> got = last_logits(sctx, n_vocab);
            float d = 0.0f; int am = 0;
            for (int i = 0; i < n_vocab; ++i) { d = std::max(d, std::fabs(ref[i] - got[i])); if (got[i] > got[am]) am = i; }
            require(am == argmax_ref && d <= 1e-2f, what);
            // back to the clean prefix for the next case: the recurrent state cannot be trimmed past `next`,
            // so restore the snapshot again (the mechanism under test) and trim attention
            require(llama_state_seq_snapshot_load(sctx, 0, id, P) == LLAMA_SNAPSHOT_OK, "re-restore to the prefix");
            require(llama_memory_seq_rm(smem, 0, N, -1), "trim back to the prefix");
            return d;
        };
        auto host_state = [&]() { // test-only witness of the live state: its serialized bytes
            std::vector<uint8_t> v(llama_state_seq_get_size_ext(sctx, 0, P));
            llama_state_seq_get_data_ext(sctx, v.data(), v.size(), 0, P);
            return v;
        };

        require(decode(sctx, prefix, 0, 0), "snapshot prefix decode");
        llama_synchronize(sctx); // time the snapshot, not the prefill still in flight
        size_t quoted = 0, bytes = 0;
        require(llama_state_seq_snapshot_bytes(sctx, 0, P, &quoted) == LLAMA_SNAPSHOT_OK && quoted > 0, "size quote before save");
        const int64_t t_s0 = now_us();
        require(llama_state_seq_snapshot_save(sctx, 0, 42, P, &bytes) == LLAMA_SNAPSHOT_OK, "save");
        const int64_t t_save = now_us() - t_s0;
        require(bytes == quoted, "the physical quote equals what the save holds (the pool reserves exactly this)");

        // 1. a failed REPLACEMENT keeps the prior snapshot loadable
        require(decode(sctx, detour, N, 0), "detour");
        llama_state_seq_snapshot_test_fail_next_allocs(1);
        size_t b2 = 0;
        require(llama_state_seq_snapshot_save(sctx, 0, 42, P, &b2) == LLAMA_SNAPSHOT_ALLOC_FAILED, "a failed allocation is reported typed");
        // 2. an incompatible load is refused BEFORE the live sequence changes
        const auto before = host_state();
        require(llama_state_seq_snapshot_load(sctx, 0, 42, LLAMA_STATE_SEQ_FLAGS_NONE) == LLAMA_SNAPSHOT_INCOMPATIBLE, "flags mismatch refused");
        require(llama_state_seq_snapshot_load(sctx, 0, 999, P) == LLAMA_SNAPSHOT_ABSENT, "an unknown id is absent");
        require(host_state() == before, "a refused load leaves the live sequence byte-identical");
        llama_synchronize(sctx);
        const int64_t t_l0 = now_us();
        require(llama_state_seq_snapshot_load(sctx, 0, 42, P) == LLAMA_SNAPSHOT_OK, "timed restore");
        const int64_t t_load = now_us() - t_l0; // the restore alone (it synchronizes on entry; nothing is in flight)
        const float d1 = restored_matches_cold(42, "the prior snapshot survived the failed replacement");

        // 2b. a failure AFTER the read began (metadata mutated, copy not done) quarantines the target: it is
        //     cleared whole, never half-restored; the snapshot itself is intact and restores it again
        llama_state_seq_snapshot_test_fail_next_load_commits(1);
        require(llama_state_seq_snapshot_load(sctx, 0, 42, P) == LLAMA_SNAPSHOT_COPY_FAILED, "a post-read failure is typed");
        require(llama_memory_seq_pos_max(smem, 0) == -1, "the quarantined target is empty, never half-restored");
        require(llama_state_seq_snapshot_load(sctx, 0, 42, P) == LLAMA_SNAPSHOT_OK, "an empty target restores from the intact snapshot");
        // the recurrent row is back; attention for the prefix was cleared with the quarantine, so re-prefill it
        // (attention is trimmed/re-built by position, as the server does) and check the next token
        require(llama_memory_seq_rm(smem, 0, -1, -1), "clear for the prefix re-build");
        require(decode(sctx, prefix, 0, 0), "re-prefill after quarantine");
        llama_synchronize(sctx);
        require(llama_state_seq_snapshot_save(sctx, 0, 42, P, &bytes) == LLAMA_SNAPSHOT_OK, "re-save after quarantine");

        // 3. save / free with backend work still in flight (no explicit synchronize by the caller)
        require(decode(sctx, detour, N, 0), "detour 2");               // submitted, not synchronized
        size_t b3 = 0;
        require(llama_state_seq_snapshot_save(sctx, 0, 70, P, &b3) == LLAMA_SNAPSHOT_OK, "save with work in flight");
        require(decode(sctx, { next }, N + M, 0), "more work in flight");
        require(llama_state_seq_snapshot_free(sctx, 70) == LLAMA_SNAPSHOT_OK, "free with work in flight");
        require(llama_state_seq_snapshot_free(sctx, 70) == LLAMA_SNAPSHOT_ABSENT, "a second free is absent, typed");

        // 4. allocate / recycle cycles: physical footprint must not grow
        auto footprint = []() -> int64_t {
#if defined(__APPLE__)
            task_vm_info_data_t info; mach_msg_type_number_t n = TASK_VM_INFO_COUNT;
            return task_info(mach_task_self(), TASK_VM_INFO, (task_info_t) &info, &n) == KERN_SUCCESS ? (int64_t) info.phys_footprint : -1;
#else
            return -1;
#endif
        };
        int64_t peak = 0;
        const int64_t base = footprint();
        for (int cycle = 0; cycle < 200; ++cycle) {
            size_t b4 = 0;
            require(llama_state_seq_snapshot_save(sctx, 0, 100 + (uint64_t) (cycle % 4), P, &b4) == LLAMA_SNAPSHOT_OK, "cycle save");
            if (cycle % 4 == 3) {
                for (uint64_t id = 100; id < 104; ++id) llama_state_seq_snapshot_free(sctx, id);
            }
            peak = std::max(peak, footprint());
        }
        const int64_t after = footprint();
        std::printf("device snapshot (n_seq_max 1): %zu device bytes (quoted %zu), save %lld us, restore %lld us, logits max |diff| %.6g\n",
                    bytes, quoted, (long long) t_save, (long long) t_load, d1);
        std::printf("fault replays ok: failed replacement kept the prior, incompatible/absent refused with the live state byte-identical, save/free with work in flight\n");
        std::printf("200 save/free cycles of 4 snapshots: footprint base %.1f MiB, peak %.1f MiB, after %.1f MiB\n",
                    base / 1048576.0, peak / 1048576.0, after / 1048576.0);
        require(after - base < (int64_t) bytes, "no growth across cycles beyond one snapshot of slack");
        llama_state_seq_snapshot_free(sctx, 42);
        llama_free(sctx);
    }

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
