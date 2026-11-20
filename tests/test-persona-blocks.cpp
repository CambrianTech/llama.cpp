// what this catches: moving a persona's state between the memory and host memory (llama_context::persona_save /
// persona_load, CONTINUUM-ENGINE 4.14) changing her state, or a refused move changing anything. Every move must leave
// her next tokens' logits byte-identical to decoding in place: out of one stream and into another bound to her own
// sequence id, the mirror direction, a same-stream round trip, and two live personas exchanging streams. Refusals
// (stream holding state, blocks not prepared, layout mismatch, sequence holding state) must be typed and change
// nothing. Runs on a plain KV model and on a hybrid (attention + recurrent, e.g. a tiny random qwen35moe from
// `test-llama-archs -a qwen35moe -o DIR`); the same interface covers both.
//
// Measure mode (-n N, a prompt of N tokens): only the timed move runs. It reports what recomputing her costs (the
// prefill of her N tokens) against the fenced save and load of her native state (bytes, ms, GB/s), and still requires
// her next tokens to be byte-identical after the move.

#include "arg.h"
#include "common.h"
#include "llama.h"

#include "../src/llama-context.h"
#include "../src/llama-batch.h"
#include "../src/llama-kv-cache.h"
#include "../src/llama-model.h"

#include <algorithm>
#include <atomic>
#include <new>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

// allocation fault injection: while armed, every allocation in this process throws std::bad_alloc
static std::atomic<bool> g_fail_alloc { false };

static void * checked_alloc(std::size_t n) {
    if (g_fail_alloc.load(std::memory_order_relaxed)) {
        throw std::bad_alloc();
    }
    if (void * p = std::malloc(n ? n : 1)) {
        return p;
    }
    throw std::bad_alloc();
}
void * operator new(std::size_t n) { return checked_alloc(n); }
void * operator new[](std::size_t n) { return checked_alloc(n); }
void operator delete(void * p) noexcept { std::free(p); }
void operator delete[](void * p) noexcept { std::free(p); }
void operator delete(void * p, std::size_t) noexcept { std::free(p); }
void operator delete[](void * p, std::size_t) noexcept { std::free(p); }

static void require(bool condition, const char * message) {
    if (!condition) {
        fprintf(stderr, "FAILED: %s\n", message);
        std::exit(1);
    }
}

static void require_move(llama_persona_move got, llama_persona_move want, const char * what) {
    if (got != want) {
        fprintf(stderr, "FAILED: %s: got %s, want %s\n", what, llama_persona_move_name(got), llama_persona_move_name(want));
        std::exit(1);
    }
}

static void require_identical(const char * what, const std::vector<float> & want, const std::vector<float> & got) {
    float max_diff = 0.0f;
    for (size_t i = 0; i < std::min(want.size(), got.size()); ++i) {
        max_diff = std::max(max_diff, std::fabs(want[i] - got[i]));
    }
    const bool identical = want.size() == got.size() && std::memcmp(want.data(), got.data(), want.size() * sizeof(float)) == 0;
    fprintf(stderr, "%s: byte-identical=%d max|diff|=%g\n", what, identical ? 1 : 0, max_diff);
    require(identical, what);
}

// the owners in this test: each persona's blocks are saved and loaded under her identity
static const llama_persona_identity A { 0xA, 1 };
static const llama_persona_identity B { 0xB, 1 };

struct fixture {
    llama_context * ctx     = nullptr;
    int             n_vocab = 0;
    int             n_prompt = 24;

    std::vector<float> decode_one(llama_seq_id seq, llama_token token, llama_pos pos) {
        llama_batch b = llama_batch_init(1, 0, 1);
        common_batch_add(b, token, pos, { seq }, true);
        require(llama_decode(ctx, b) == 0, "decode a token");
        const float * l = llama_get_logits_ith(ctx, 0);
        std::vector<float> out(l, l + n_vocab);
        llama_batch_free(b);
        return out;
    }

    // her prompt (salt makes two personas' prompts differ), decoded in batches of n_batch
    int n_batch = 2048;
    void prompt(llama_seq_id seq, int salt) {
        llama_batch b = llama_batch_init(n_batch, 0, 1);
        for (int i0 = 0; i0 < n_prompt; i0 += n_batch) {
            common_batch_clear(b);
            for (int i = i0; i < std::min(n_prompt, i0 + n_batch); ++i) {
                common_batch_add(b, 1 + (i * 7 + salt * 13) % (n_vocab - 1), i, { seq }, false);
            }
            require(llama_decode(ctx, b) == 0, "decode a prompt");
        }
        llama_batch_free(b);
    }

    // her next two tokens decoded in place from a clear memory: the reference
    std::vector<std::vector<float>> reference(llama_seq_id seq, int salt) {
        llama_memory_clear(llama_get_memory(ctx), true);
        prompt(seq, salt);
        std::vector<std::vector<float>> out;
        out.push_back(decode_one(seq, 3 % n_vocab, n_prompt));
        out.push_back(decode_one(seq, 5 % n_vocab, n_prompt + 1));
        return out;
    }

    void require_continues(const char * what, llama_seq_id seq, const std::vector<std::vector<float>> & ref) {
        char label[160];
        snprintf(label, sizeof(label), "%s: first token", what);
        require_identical(label, ref[0], decode_one(seq, 3 % n_vocab, n_prompt));
        snprintf(label, sizeof(label), "%s: second token", what);
        require_identical(label, ref[1], decode_one(seq, 5 % n_vocab, n_prompt + 1));
    }

    llama_pos pos_max(llama_seq_id seq) const { return llama_memory_seq_pos_max(llama_get_memory(ctx), seq); }
};

// Exercise placement and receipt ownership directly. Four metadata cells suffice to wrap
// the sliding window, independent of the architecture and weights of the fixture model.
static void test_checkpoint_after_swa_overwrite(const llama_model & model) {
    llama_kv_cache cache(model, model.hparams, GGML_TYPE_F16, GGML_TYPE_F16, false, false, true,
                         4, 1, 1, 2, LLAMA_SWA_TYPE_STANDARD, nullptr, [](int) { return false; }, nullptr, nullptr);
    llama_seq_id seq = 0;
    llama_seq_id * ids = &seq;
    llama_pos pos = 0;
    int32_t count = 1;
    llama_ubatch batch{};
    batch.n_seq_tokens = batch.n_seqs = batch.n_tokens = batch.n_seqs_unq = batch.n_pos = batch.b_equal_seqs = 1;
    batch.pos = &pos;
    batch.n_seq_id = &count;
    batch.seq_id = &ids;
    batch.seq_id_unq = &seq;
    auto plan = [&] {
        auto slots = cache.prepare({batch});
        require(slots.size() == 1, "one SWA slot planned");
        return slots.front();
    };
    auto take = [&](llama_persona_checkpoint_blocks & mark) {
        require_move(cache.persona_check_checkpoint(seq, mark), llama_persona_move::done, "prepare SWA checkpoint");
        cache.persona_commit_checkpoint(seq, mark);
        mark.taken = true;
    };
    for (; pos < 4; ++pos) {
        cache.apply_ubatch(plan(), batch);
    }
    llama_persona_checkpoint_blocks before{}, after{};
    take(before);
    const auto slot = plan();
    require_move(cache.persona_check_rewind(seq, before), llama_persona_move::done, "planning preserves old KV");
    cache.apply_ubatch(slot, batch);
    take(after);
    require_move(cache.persona_check_rewind(seq, before), llama_persona_move::checkpoint_stale, "overwritten KV revoked");
    require_move(cache.persona_check_rewind(seq, after), llama_persona_move::done, "new sliding window retained");
}

int main(int argc, char ** argv) {
    common_params params;
    params.kv_unified      = false; // one attention stream per sequence
    params.n_parallel      = 2;
    params.n_ctx           = 512;
    params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED; // V rows contiguous
    // --test-rs-seq N: the speculative configuration serving uses (draft-mtp, n_max N), which gives the context N
    // rollback snapshot groups (common_params_speculative::need_n_rs_seq); stripped before the common parse
    std::vector<char *> args(argv, argv + argc);
    for (size_t i = 1; i + 1 < args.size(); ++i) {
        if (std::strcmp(args[i], "--test-rs-seq") == 0) {
            params.speculative.types       = { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
            params.speculative.draft.n_max = std::atoi(args[i + 1]);
            args.erase(args.begin() + i, args.begin() + i + 2);
            break;
        }
    }
    common_init();
    if (!common_params_parse((int) args.size(), args.data(), params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }
    ggml_backend_load_all();
    common_init_result_ptr init = common_init_from_params(params);
    fixture f;
    f.ctx = init->context();
    require(f.ctx != nullptr, "context");
    f.n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(f.ctx)));
    llama_context & ctx = *f.ctx;

    f.n_batch = (int) llama_n_batch(f.ctx);

    llama_persona_blocks a, b;
    require_move(ctx.persona_prepare(a), llama_persona_move::done, "prepare persona a");
    require_move(ctx.persona_prepare(b), llama_persona_move::done, "prepare persona b");

    if (params.n_predict > 0) {
        // measure mode: what recomputing her costs (prefill) vs her native state moved out and back. Recurrent state
        // cannot be rolled back, so the in-place reference comes from a second, clean run.
        f.n_prompt = params.n_predict;
        llama_memory_clear(llama_get_memory(f.ctx), true);
        const int64_t t0 = ggml_time_us();
        f.prompt(0, 1);
        llama_synchronize(f.ctx);
        const int64_t t_prefill = ggml_time_us() - t0;

        const int64_t t1 = ggml_time_us();
        require_move(ctx.persona_save(0, a, A), llama_persona_move::done, "her state out");
        const int64_t t_save = ggml_time_us() - t1;
        size_t bytes = 0;
        for (const auto & v : a.kv.k) { bytes += v.size(); }
        for (const auto & v : a.kv.v) { bytes += v.size(); }
        for (const auto & v : a.rs.r) { bytes += v.size(); }
        for (const auto & v : a.rs.s) { bytes += v.size(); }
        const uint32_t other = 1 - llama_get_memory(f.ctx)->seq_stream(0);
        const int64_t t2 = ggml_time_us();
        require_move(ctx.persona_load(0, other, a, A), llama_persona_move::done, "her state into the other stream");
        const int64_t t_load = ggml_time_us() - t2;
        std::vector<std::vector<float>> moved;
        moved.push_back(f.decode_one(0, 3 % f.n_vocab, f.n_prompt));
        moved.push_back(f.decode_one(0, 5 % f.n_vocab, f.n_prompt + 1));

        llama_memory_clear(llama_get_memory(f.ctx), true);
        f.prompt(0, 1);
        f.require_continues("her next tokens after the timed move", 0, moved);

        const double gb = bytes / 1e9;
        fprintf(stderr, "measure: n_prompt=%d state=%.1f MB | recompute (prefill) %.1f ms (%.0f tok/s) | "
                        "save %.2f ms (%.2f GB/s) | load %.2f ms (%.2f GB/s) | load is %.0fx faster than recompute\n",
                f.n_prompt, bytes / 1e6, t_prefill / 1e3, f.n_prompt / (t_prefill / 1e6),
                t_save / 1e3, gb / (t_save / 1e6), t_load / 1e3, gb / (t_load / 1e6), (double) t_prefill / t_load);
        return 0;
    }

    test_checkpoint_after_swa_overwrite(*llama_get_model(f.ctx));

    const auto ref_a0 = f.reference(0, 1); // persona a on sequence 0
    const auto ref_b1 = f.reference(1, 2); // persona b on sequence 1

    // 1. out of stream 0, into stream 1 bound to her own sequence 0
    llama_memory_clear(llama_get_memory(f.ctx), true);
    f.prompt(0, 1);
    require_move(ctx.persona_save(0, a, A), llama_persona_move::done, "a out");
    require(f.pos_max(0) < 0, "nothing of a is left in the memory");
    require(a.seq == 0, "her blocks carry her sequence");
    // what this catches: the native load trusting its caller. a's blocks under b's identity are refused before
    // anything is written, on the native path itself, not only the public seam.
    require_move(ctx.persona_load(0, 1, a, B), llama_persona_move::wrong_owner, "a's blocks refused under b's identity");
    require(f.pos_max(1) < 0, "the refused load wrote nothing");
    require_move(ctx.persona_load(0, 1, a, A), llama_persona_move::done, "a into stream 1");
    require(f.pos_max(0) == f.n_prompt - 1, "a's sequence holds her prompt");
    f.require_continues("a moved to stream 1", 0, ref_a0);

    // 1b. what this catches: a mind bound only back as the sequence she was saved from, so two streams cannot serve
    // any number of minds. Saved as sequence 0, she binds as sequence 1: her cells answer to 1, nothing is left on 0,
    // and she continues byte-identically.
    llama_memory_clear(llama_get_memory(f.ctx), true);
    f.prompt(0, 1);
    require_move(ctx.persona_save(0, a, A), llama_persona_move::done, "a out (to rebind as another sequence)");
    require_move(ctx.persona_load(1, 1, a, A), llama_persona_move::done, "a bound as sequence 1");
    require(f.pos_max(1) == f.n_prompt - 1 && f.pos_max(0) < 0, "her cells answer to sequence 1 only");
    f.require_continues("a continues as sequence 1", 1, ref_a0);

    // 2. the mirror: b's sequence 1 out of its stream and into the other
    llama_memory_clear(llama_get_memory(f.ctx), true);
    f.prompt(1, 2);
    const uint32_t b_stream = llama_get_memory(f.ctx)->seq_stream(1);
    require_move(ctx.persona_save(1, b, B), llama_persona_move::done, "b out");
    require_move(ctx.persona_load(1, 1 - b_stream, b, B), llama_persona_move::done, "b into the other stream");
    f.require_continues("b moved to the other stream", 1, ref_b1);

    // 3. same-stream round trip
    llama_memory_clear(llama_get_memory(f.ctx), true);
    f.prompt(0, 1);
    const uint32_t a_stream = llama_get_memory(f.ctx)->seq_stream(0);
    require_move(ctx.persona_save(0, a, A), llama_persona_move::done, "a out (round trip)");
    require_move(ctx.persona_load(0, a_stream, a, A), llama_persona_move::done, "a back into her own stream");
    f.require_continues("a same-stream round trip", 0, ref_a0);

    // 4. refusals are typed and change nothing
    {
        llama_memory_clear(llama_get_memory(f.ctx), true);
        f.prompt(0, 1);
        f.prompt(1, 2);
        llama_persona_blocks unprepared;
        require(ctx.persona_save(0, unprepared, A) != llama_persona_move::done, "an unprepared save is refused");
        require(f.pos_max(0) == f.n_prompt - 1, "a refused save leaves her in place");

        require_move(ctx.persona_save(0, a, A), llama_persona_move::done, "a out (refusal setup)");
        const uint32_t occupied = llama_get_memory(f.ctx)->seq_stream(1);
        require_move(ctx.persona_load(0, occupied, a, A), llama_persona_move::stream_not_empty, "load over b's stream");
        require(f.pos_max(1) == f.n_prompt - 1, "b untouched by the refused load");

        llama_persona_blocks bad = a; // the same persona with one layer's rows missing
        bad.kv.k.pop_back();
        require_move(ctx.persona_load(0, 1 - occupied, bad, A), llama_persona_move::layout_mismatch, "a layout mismatch");
        require(f.pos_max(0) < 0, "a refused load writes nothing");
        if (a.rs.has_state) { // hybrid: a rollback index past the snapshot groups is refused before any write
            llama_persona_blocks far = a;
            far.rs.rs_idx = 1u << 20;
            require_move(ctx.persona_load(0, 1 - occupied, far, A), llama_persona_move::invalid_rollback, "a rollback out of bounds");
            require(f.pos_max(0) < 0, "a refused hybrid load writes neither half");
        }

        require_move(ctx.persona_load(0, 1 - occupied, a, A), llama_persona_move::done, "a back after the refusals");
        f.require_continues("a after the refusals", 0, ref_a0);
        f.require_continues("b untouched by a's refusals", 1, ref_b1);
    }

    // 5. two live personas exchange streams: the map stays a permutation and each keeps her own state
    {
        llama_memory_clear(llama_get_memory(f.ctx), true);
        f.prompt(0, 1);
        f.prompt(1, 2);
        const uint32_t sa = llama_get_memory(f.ctx)->seq_stream(0);
        const uint32_t sb = llama_get_memory(f.ctx)->seq_stream(1);
        require(sa != sb, "two personas, two streams");
        require_move(ctx.persona_save(0, a, A), llama_persona_move::done, "a out (exchange)");
        require_move(ctx.persona_save(1, b, B), llama_persona_move::done, "b out (exchange)");
        require_move(ctx.persona_load(0, sb, a, A), llama_persona_move::done, "a into b's stream");
        require_move(ctx.persona_load(1, sa, b, B), llama_persona_move::done, "b into a's stream");
        require(llama_get_memory(f.ctx)->seq_stream(0) == sb && llama_get_memory(f.ctx)->seq_stream(1) == sa,
                "the streams are exchanged");
        f.require_continues("a after the exchange", 0, ref_a0);
        f.require_continues("b after the exchange", 1, ref_b1);
    }

    // 7. allocation faults: a commit allocates nothing (it runs with every allocation failing), and an allocation
    //    failure in the reserve phase leaves her whole and continuing exactly
    {
        llama_memory_t mem = llama_get_memory(f.ctx);
        llama_memory_clear(mem, true);
        f.prompt(0, 1);
        llama_synchronize(f.ctx);
        llama_persona_blocks c;
        require_move(ctx.persona_prepare(c), llama_persona_move::done, "prepare persona c");
        require_move(mem->persona_check_save(0, c), llama_persona_move::done, "check (reserve) c's save");
        g_fail_alloc = true;
        mem->persona_commit_save(0, c); // must not allocate: any allocation throws and aborts the test
        g_fail_alloc = false;
        require(f.pos_max(0) < 0, "c's commit took her out");
        llama_persona_slot slot;
        require_move(mem->persona_check_load(0, 1 - mem->seq_stream(0), c, slot), llama_persona_move::done, "check c's load");
        g_fail_alloc = true;
        mem->persona_commit_load(0, slot, c);
        g_fail_alloc = false;
        f.require_continues("c after allocation-free commits", 0, ref_a0);

        llama_memory_clear(mem, true);
        f.prompt(0, 1);
        llama_synchronize(f.ctx);
        llama_persona_blocks d;
        require_move(ctx.persona_prepare(d), llama_persona_move::done, "prepare persona d");
        bool reserve_failed = false;
        g_fail_alloc = true;
        try {
            (void) mem->persona_check_save(0, d);
        } catch (const std::bad_alloc &) {
            reserve_failed = true;
        }
        g_fail_alloc = false;
        require(reserve_failed, "the reserve phase allocates (and here fails)");
        require(f.pos_max(0) == f.n_prompt - 1, "a failed reserve leaves her in the memory");
        f.require_continues("her after a failed reserve", 0, ref_a0);
    }

    // 6. with rollback snapshot groups (n_rs_seq > 0, e.g. --spec-type draft-mtp --spec-draft-n-max 4): her state is
    //    moved while a rollback is pending (two drafted tokens, one taken back) and she continues exactly as in place
    if (llama_n_rs_seq(f.ctx) > 0) {
        auto drafted_then_rolled_back = [&]() {
            llama_memory_clear(llama_get_memory(f.ctx), true);
            f.prompt(0, 1);
            f.decode_one(0, 3 % f.n_vocab, f.n_prompt);
            f.decode_one(0, 5 % f.n_vocab, f.n_prompt + 1);
            require(llama_memory_seq_rm(llama_get_memory(f.ctx), 0, f.n_prompt + 1, -1), "take the second draft back");
        };
        drafted_then_rolled_back();
        const std::vector<float> in_place = f.decode_one(0, 7 % f.n_vocab, f.n_prompt + 1);

        drafted_then_rolled_back();
        llama_persona_blocks r;
        require_move(ctx.persona_prepare(r), llama_persona_move::done, "prepare persona r");
        require_move(ctx.persona_save(0, r, A), llama_persona_move::done, "r out with a rollback pending");
        require(!r.rs.has_state || r.rs.rs_idx > 0, "her pending rollback travels with her");
        require_move(ctx.persona_load(0, 1 - llama_get_memory(f.ctx)->seq_stream(0), r, A), llama_persona_move::done, "r in");
        require_identical("continues after a move with a rollback pending", in_place,
                          f.decode_one(0, 7 % f.n_vocab, f.n_prompt + 1));
        fprintf(stderr, "rollback groups exercised: n_rs_seq=%u, her rs_idx=%u\n", llama_n_rs_seq(f.ctx), r.rs.rs_idx);
    }

    // the public seam's owner (Codex on #147 at e046): her blocks move only under the owner of her token history.
    {
        auto st = [](int32_t s) { return (llama_persona_move) s; };
        llama_memory_clear(llama_get_memory(f.ctx), true);
        f.prompt(0, 1);
        llama_persona * p = llama_persona_init(f.ctx, 0xA11CE, 7);
        require(p != nullptr, "a persona for this memory, owned from the start");
        uint64_t who = 0, when = 0;
        require(!llama_persona_owner(p, &who, &when), "no stamp before a completed save");
        require_move(st(llama_persona_grow_tokens(p, f.n_prompt, 1)), llama_persona_move::done, "her history grows with her");
        require_move(st(llama_persona_shrink_tokens(p)), llama_persona_move::done, "and shrinks as she cools, dropping nothing");
        require_move(st(llama_persona_save(f.ctx, 0, p)), llama_persona_move::done, "save under her owner");
        require(llama_persona_owner(p, &who, &when) && who == 0xA11CE && when == 7, "her blocks are stamped with her owner");
        require_move(st(llama_persona_bind(f.ctx, 0, p)), llama_persona_move::done, "bound into a stream the memory chose");
        f.require_continues("continues after an owned move through the public seam", 0, ref_a0);

        // every stream occupied: a bind places nobody and changes nothing
        llama_persona * q = llama_persona_init(f.ctx, 0xB0B, 1);
        require(q != nullptr, "a second persona");
        llama_memory_clear(llama_get_memory(f.ctx), true);
        f.prompt(0, 1);
        f.prompt(1, 2);
        require_move(st(llama_persona_bind(f.ctx, 1, q)), llama_persona_move::no_empty_stream, "no empty stream to bind into");
        f.require_continues("both residents untouched by the refused bind", 0, ref_a0);
        llama_persona_free(q);
        llama_persona_free(p);
    }

    // 8. what this catches (CONTINUUM-ENGINE 4.15): a mind re-prefilling her whole prompt every act because the engine
    //    could not go back to a stable boundary (the M5, 2026-11-18: reused 0 of 11k-30k tokens on every act). A
    //    checkpoint taken at her prompt's end, after she decoded past it, rewinds her so a DIFFERENT continuation at
    //    that position is byte-identical to decoding it in place; the same checkpoint rewinds twice (it is not consumed);
    //    a rewind of a checkpoint never taken, or one ahead of what she holds, is refused and changes nothing.
    {
        auto st = [](int32_t s) { return (llama_persona_move) s; };
        const llama_token other = 9 % f.n_vocab;
        // the reference: her prompt, then a token she never decoded in the checkpointed run
        llama_memory_clear(llama_get_memory(f.ctx), true);
        f.prompt(0, 1);
        const std::vector<float> in_place = f.decode_one(0, other, f.n_prompt);

        llama_persona_checkpoint * cp = llama_persona_checkpoint_init();
        require(cp != nullptr, "a checkpoint");
        llama_memory_clear(llama_get_memory(f.ctx), true);
        f.prompt(0, 1);
        require_move(st(llama_persona_checkpoint_rewind(f.ctx, 0, 0xA, 1, cp)), llama_persona_move::no_checkpoint, "a rewind of nothing taken");
        require(f.pos_max(0) == f.n_prompt - 1, "a refused rewind leaves her as she was");
        require_move(st(llama_persona_checkpoint_take(f.ctx, 0, 0xA, 1, cp)), llama_persona_move::done, "checkpoint at her prompt's end");
        require(llama_persona_checkpoint_pos(cp) == f.n_prompt - 1, "it holds through her last prompt position");
        require(f.pos_max(0) == f.n_prompt - 1, "taking a checkpoint leaves her bound and whole");
        for (int round = 0; round < 2; ++round) {
            f.decode_one(0, 3 % f.n_vocab, f.n_prompt);     // she goes on past the boundary...
            f.decode_one(0, 5 % f.n_vocab, f.n_prompt + 1);
            require_move(st(llama_persona_checkpoint_rewind(f.ctx, 0, 0xA, 1, cp)), llama_persona_move::done, "rewind to the boundary");
            require(f.pos_max(0) == f.n_prompt - 1, "what she held past the boundary is cut");
            char label[96];
            snprintf(label, sizeof(label), "a different continuation after rewind %d", round + 1);
            require_identical(label, in_place, f.decode_one(0, other, f.n_prompt)); // ...and continues as if she never had
            require_move(st(llama_persona_checkpoint_rewind(f.ctx, 0, 0xA, 1, cp)), llama_persona_move::done, "rewind again before the next round");
        }
        // what this catches (Codex on #153): A's rows spliced onto B's KV when B reuses A's slot at the same layout and
        // position. B in the slot A was checkpointed in: refused as stale under A's owner, as wrong owner under B's,
        // and B continues exactly as in place.
        const auto ref_b = f.reference(0, 2);
        llama_memory_clear(llama_get_memory(f.ctx), true);
        f.prompt(0, 1);
        require_move(st(llama_persona_checkpoint_take(f.ctx, 0, 0xA, 1, cp)), llama_persona_move::done, "A's checkpoint");
        llama_memory_clear(llama_get_memory(f.ctx), true);
        f.prompt(0, 2); // B, same slot, same length
        require_move(st(llama_persona_checkpoint_rewind(f.ctx, 0, 0xA, 1, cp)), llama_persona_move::checkpoint_stale, "A's rows never reach B's cells");
        require_move(st(llama_persona_checkpoint_rewind(f.ctx, 0, 0xB, 1, cp)), llama_persona_move::wrong_owner, "nor under B's owner");
        f.require_continues("B untouched by the refused rewinds", 0, ref_b);

        // what this catches: a checkpoint lost when she moves between streams (four minds rotate through two). Taken
        // on her stream, she is saved and bound into the other one, and the rewind there continues byte-identically.
        llama_memory_clear(llama_get_memory(f.ctx), true);
        f.prompt(0, 1);
        require_move(st(llama_persona_checkpoint_take(f.ctx, 0, 0xA, 1, cp)), llama_persona_move::done, "A's checkpoint before her move");
        f.decode_one(0, 3 % f.n_vocab, f.n_prompt);
        require_move(ctx.persona_save(0, a, A), llama_persona_move::done, "A out");
        require_move(ctx.persona_load(0, 1 - llama_get_memory(f.ctx)->seq_stream(0), a, A), llama_persona_move::done, "A into the other stream");
        require_move(st(llama_persona_checkpoint_rewind(f.ctx, 0, 0xA, 1, cp)), llama_persona_move::done, "her checkpoint follows her");
        require_identical("a different continuation after her move and rewind", in_place, f.decode_one(0, other, f.n_prompt));

        // what this catches (Codex on #153 at 12a98779b): a newer take erasing the cut receipt an older checkpoint
        // needed. Take A, cut below it and refill a different suffix to the same length, take B: A is stale (by its
        // cut receipt), B still rewinds.
        // A hybrid memory cuts its recurrent half only by a checkpoint rewind (a plain seq_rm mid-sequence is refused),
        // so the cut below A is a rewind to an earlier checkpoint taken at half her prompt.
        {
            llama_persona_checkpoint * newer = llama_persona_checkpoint_init();
            llama_persona_checkpoint * early = llama_persona_checkpoint_init();
            require(newer != nullptr && early != nullptr, "two more checkpoints");
            llama_memory_clear(llama_get_memory(f.ctx), true);
            const int full = f.n_prompt;
            const llama_pos half = full / 2;
            f.n_prompt = half;
            f.prompt(0, 1);
            f.n_prompt = full;
            require_move(st(llama_persona_checkpoint_take(f.ctx, 0, 0xA, 1, early)), llama_persona_move::done, "checkpoint at half");
            for (llama_pos p = half; p < full; ++p) {
                f.decode_one(0, 7 % f.n_vocab, p);
            }
            require_move(st(llama_persona_checkpoint_take(f.ctx, 0, 0xA, 1, cp)), llama_persona_move::done, "checkpoint A");
            require_move(st(llama_persona_checkpoint_rewind(f.ctx, 0, 0xA, 1, early)), llama_persona_move::done, "the head survives a later checkpoint on the unchanged prefix");
            llama_persona_checkpoint_free(early);
            for (llama_pos p = half; p < full; ++p) {
                f.decode_one(0, 11 % f.n_vocab, p); // a different suffix, to the same length
            }
            require_move(st(llama_persona_checkpoint_take(f.ctx, 0, 0xA, 1, newer)), llama_persona_move::done, "checkpoint B");
            require_move(st(llama_persona_checkpoint_rewind(f.ctx, 0, 0xA, 1, cp)), llama_persona_move::checkpoint_stale, "A no longer matches what she holds");
            require_move(st(llama_persona_checkpoint_rewind(f.ctx, 0, 0xA, 1, newer)), llama_persona_move::done, "B, the newest, still rewinds");
            llama_persona_checkpoint_free(newer);
        }

        // ahead: she holds less than the checkpoint does
        llama_memory_clear(llama_get_memory(f.ctx), true);
        const int full = f.n_prompt;
        f.n_prompt = full / 2;
        f.prompt(0, 1);
        require_move(st(llama_persona_checkpoint_rewind(f.ctx, 0, 0xA, 1, cp)), llama_persona_move::checkpoint_ahead, "a rewind never moves forward");
        require(f.pos_max(0) == f.n_prompt - 1, "a refused forward rewind leaves her as she was");
        f.n_prompt = full;
        fprintf(stderr, "checkpoint: %zu bytes of recurrent rows (attention KV stays in place)\n", llama_persona_checkpoint_bytes(cp));
        llama_persona_checkpoint_free(cp);
    }

    // 9. what this catches (CONTINUUM-ENGINE 4.16, restore is the norm): a restart that drops a mind. Her saved blocks
    // written to a file and read into a fresh persona bind and continue byte-identically; a file read under another
    // owner or another epoch of her is refused and leaves that persona as it was (it still binds nothing).
    {
        auto st = [](int32_t s) { return (llama_persona_move) s; };
        const std::string path = (std::filesystem::temp_directory_path() / "test-persona-blocks-a.cpb").string();
        llama_persona * saved = llama_persona_init(f.ctx, 0xA, 1);
        require(saved != nullptr, "a persona to save");
        llama_memory_clear(llama_get_memory(f.ctx), true);
        f.prompt(0, 1);
        require_move(st(llama_persona_save(f.ctx, 0, saved)), llama_persona_move::done, "a saved");
        require_move(st(llama_persona_write_file(saved, path.c_str())), llama_persona_move::done, "a written to her file");
        // a second save to her fixed path replaces the first (Windows CRT rename refuses an existing destination)
        require_move(st(llama_persona_write_file(saved, path.c_str())), llama_persona_move::done, "a written again over her file");
        llama_persona_free(saved);

        llama_persona * other = llama_persona_init(f.ctx, 0xB, 1);
        llama_persona * stale = llama_persona_init(f.ctx, 0xA, 2);
        llama_persona * back  = llama_persona_init(f.ctx, 0xA, 1);
        require(other != nullptr && stale != nullptr && back != nullptr, "fresh personas to read into");
        require_move(st(llama_persona_read_file(other, path.c_str())), llama_persona_move::file_mismatch, "another owner's file is refused");
        require_move(st(llama_persona_read_file(stale, path.c_str())), llama_persona_move::file_mismatch, "another epoch of her is refused");
        require_move(st(llama_persona_read_file(back, path.c_str())), llama_persona_move::done, "a read back from her file");
        // a truncated file is refused before anything is sized from it, and leaves the blocks she holds as they were
        const std::string cut = path + ".cut";
        std::filesystem::copy_file(path, cut, std::filesystem::copy_options::overwrite_existing);
        std::filesystem::resize_file(cut, std::filesystem::file_size(cut) / 2);
        const auto refused = st(llama_persona_read_file(back, cut.c_str()));
        require(refused == llama_persona_move::io_error || refused == llama_persona_move::file_mismatch, "a truncated file is refused");
        // a length the file claims beyond this memory's geometry (here, its first K layer's, at byte 52: magic, version,
        // owner, epoch, seq, row id, recurrent layout, row and layer counts) is refused before any buffer is sized from it
        std::filesystem::copy_file(path, cut, std::filesystem::copy_options::overwrite_existing);
        {
            std::fstream patch(cut, std::ios::in | std::ios::out | std::ios::binary);
            const uint64_t huge = UINT64_C(1) << 62;
            patch.seekp(52);
            patch.write(reinterpret_cast<const char *>(&huge), sizeof(huge));
        }
        require_move(st(llama_persona_read_file(back, cut.c_str())), llama_persona_move::file_mismatch, "a length past the geometry is refused");
        std::remove(cut.c_str());
        llama_memory_clear(llama_get_memory(f.ctx), true);
        require_move(st(llama_persona_bind(f.ctx, 0, back)), llama_persona_move::done, "a bound from her file");
        f.require_continues("a continues after a restart from her file", 0, ref_a0);
        llama_persona_free(other);
        llama_persona_free(stale);
        llama_persona_free(back);
        std::remove(path.c_str());
    }

    fprintf(stderr, "test-persona-blocks: ok (%zu attention layers, %zu recurrent layers)\n", a.kv.k.size(), a.rs.r.size());
    return 0;
}
