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

#include <algorithm>
#include <atomic>
#include <new>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
        require_move(ctx.persona_save(0, a), llama_persona_move::done, "her state out");
        const int64_t t_save = ggml_time_us() - t1;
        size_t bytes = 0;
        for (const auto & v : a.kv.k) { bytes += v.size(); }
        for (const auto & v : a.kv.v) { bytes += v.size(); }
        for (const auto & v : a.rs.r) { bytes += v.size(); }
        for (const auto & v : a.rs.s) { bytes += v.size(); }
        const uint32_t other = 1 - llama_get_memory(f.ctx)->seq_stream(0);
        const int64_t t2 = ggml_time_us();
        require_move(ctx.persona_load(0, other, a), llama_persona_move::done, "her state into the other stream");
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

    const auto ref_a0 = f.reference(0, 1); // persona a on sequence 0
    const auto ref_b1 = f.reference(1, 2); // persona b on sequence 1

    // 1. out of stream 0, into stream 1 bound to her own sequence 0
    llama_memory_clear(llama_get_memory(f.ctx), true);
    f.prompt(0, 1);
    require_move(ctx.persona_save(0, a), llama_persona_move::done, "a out");
    require(f.pos_max(0) < 0, "nothing of a is left in the memory");
    require(a.seq == 0, "her blocks carry her sequence");
    require_move(ctx.persona_load(1, 1, a), llama_persona_move::wrong_persona, "a refused as sequence 1 (never retagged)");
    require(f.pos_max(1) < 0, "the refused load wrote nothing");
    require_move(ctx.persona_load(0, 1, a), llama_persona_move::done, "a into stream 1");
    require(f.pos_max(0) == f.n_prompt - 1, "a's sequence holds her prompt");
    f.require_continues("a moved to stream 1", 0, ref_a0);

    // 2. the mirror: b's sequence 1 out of its stream and into the other
    llama_memory_clear(llama_get_memory(f.ctx), true);
    f.prompt(1, 2);
    const uint32_t b_stream = llama_get_memory(f.ctx)->seq_stream(1);
    require_move(ctx.persona_save(1, b), llama_persona_move::done, "b out");
    require_move(ctx.persona_load(1, 1 - b_stream, b), llama_persona_move::done, "b into the other stream");
    f.require_continues("b moved to the other stream", 1, ref_b1);

    // 3. same-stream round trip
    llama_memory_clear(llama_get_memory(f.ctx), true);
    f.prompt(0, 1);
    const uint32_t a_stream = llama_get_memory(f.ctx)->seq_stream(0);
    require_move(ctx.persona_save(0, a), llama_persona_move::done, "a out (round trip)");
    require_move(ctx.persona_load(0, a_stream, a), llama_persona_move::done, "a back into her own stream");
    f.require_continues("a same-stream round trip", 0, ref_a0);

    // 4. refusals are typed and change nothing
    {
        llama_memory_clear(llama_get_memory(f.ctx), true);
        f.prompt(0, 1);
        f.prompt(1, 2);
        llama_persona_blocks unprepared;
        require(ctx.persona_save(0, unprepared) != llama_persona_move::done, "an unprepared save is refused");
        require(f.pos_max(0) == f.n_prompt - 1, "a refused save leaves her in place");

        require_move(ctx.persona_save(0, a), llama_persona_move::done, "a out (refusal setup)");
        const uint32_t occupied = llama_get_memory(f.ctx)->seq_stream(1);
        require_move(ctx.persona_load(0, occupied, a), llama_persona_move::stream_not_empty, "load over b's stream");
        require(f.pos_max(1) == f.n_prompt - 1, "b untouched by the refused load");

        llama_persona_blocks bad = a; // the same persona with one layer's rows missing
        bad.kv.k.pop_back();
        require_move(ctx.persona_load(0, 1 - occupied, bad), llama_persona_move::layout_mismatch, "a layout mismatch");
        require(f.pos_max(0) < 0, "a refused load writes nothing");
        if (a.rs.has_state) { // hybrid: a rollback index past the snapshot groups is refused before any write
            llama_persona_blocks far = a;
            far.rs.rs_idx = 1u << 20;
            require_move(ctx.persona_load(0, 1 - occupied, far), llama_persona_move::invalid_rollback, "a rollback out of bounds");
            require(f.pos_max(0) < 0, "a refused hybrid load writes neither half");
        }

        require_move(ctx.persona_load(0, 1 - occupied, a), llama_persona_move::done, "a back after the refusals");
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
        require_move(ctx.persona_save(0, a), llama_persona_move::done, "a out (exchange)");
        require_move(ctx.persona_save(1, b), llama_persona_move::done, "b out (exchange)");
        require_move(ctx.persona_load(0, sb, a), llama_persona_move::done, "a into b's stream");
        require_move(ctx.persona_load(1, sa, b), llama_persona_move::done, "b into a's stream");
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
        require_move(ctx.persona_save(0, r), llama_persona_move::done, "r out with a rollback pending");
        require(!r.rs.has_state || r.rs.rs_idx > 0, "her pending rollback travels with her");
        require_move(ctx.persona_load(0, 1 - llama_get_memory(f.ctx)->seq_stream(0), r), llama_persona_move::done, "r in");
        require_identical("continues after a move with a rollback pending", in_place,
                          f.decode_one(0, 7 % f.n_vocab, f.n_prompt + 1));
        fprintf(stderr, "rollback groups exercised: n_rs_seq=%u, her rs_idx=%u\n", llama_n_rs_seq(f.ctx), r.rs.rs_idx);
    }

    fprintf(stderr, "test-persona-blocks: ok (%zu attention layers, %zu recurrent layers)\n", a.kv.k.size(), a.rs.r.size());
    return 0;
}
