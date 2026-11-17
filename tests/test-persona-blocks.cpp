// what this catches: moving a persona's state between the memory and host memory (llama_context::persona_save /
// persona_load, CONTINUUM-ENGINE 4.14) changing her state, or a refused move changing anything. Every move must leave
// her next tokens' logits byte-identical to decoding in place: out of one stream and into another bound to her own
// sequence id, the mirror direction, a same-stream round trip, and two live personas exchanging streams. Refusals
// (stream holding state, blocks not prepared, layout mismatch, sequence holding state) must be typed and change
// nothing. Runs on a plain KV model and on a hybrid (attention + recurrent, e.g. a tiny random qwen35moe from
// `test-llama-archs -a qwen35moe -o DIR`); the same interface covers both.

#include "arg.h"
#include "common.h"
#include "llama.h"

#include "../src/llama-context.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

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

    // her prompt (salt makes two personas' prompts differ)
    void prompt(llama_seq_id seq, int salt) {
        llama_batch b = llama_batch_init(n_prompt, 0, 1);
        for (int i = 0; i < n_prompt; ++i) {
            common_batch_add(b, 1 + (i * 7 + salt * 13) % (n_vocab - 1), i, { seq }, false);
        }
        require(llama_decode(ctx, b) == 0, "decode a prompt");
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
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }
    ggml_backend_load_all();
    common_init_result_ptr init = common_init_from_params(params);
    fixture f;
    f.ctx = init->context();
    require(f.ctx != nullptr, "context");
    f.n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(f.ctx)));
    llama_context & ctx = *f.ctx;

    llama_persona_blocks a, b;
    require_move(ctx.persona_prepare(a), llama_persona_move::done, "prepare persona a");
    require_move(ctx.persona_prepare(b), llama_persona_move::done, "prepare persona b");

    const auto ref_a0 = f.reference(0, 1); // persona a on sequence 0
    const auto ref_b1 = f.reference(1, 2); // persona b on sequence 1

    // 1. out of stream 0, into stream 1 bound to her own sequence 0
    llama_memory_clear(llama_get_memory(f.ctx), true);
    f.prompt(0, 1);
    require_move(ctx.persona_save(0, a), llama_persona_move::done, "a out");
    require(f.pos_max(0) < 0, "nothing of a is left in the memory");
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

    fprintf(stderr, "test-persona-blocks: ok (%zu attention layers, %zu recurrent layers)\n", a.kv.k.size(), a.rs.r.size());
    return 0;
}
