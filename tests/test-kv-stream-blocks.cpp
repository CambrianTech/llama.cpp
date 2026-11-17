// what this catches: moving a persona's KV between a stream and host memory (llama_kv_cache::stream_save /
// stream_load / stream_bind, CONTINUUM-ENGINE 4.14) changing her state: after a raw save out of stream 0, a raw load
// into stream 1 and binding her own sequence id to stream 1 (no metadata retagged), her next token's logits must be
// byte-identical to decoding it in place. Also: the stream is left empty by a save,
// and a load into a stream that holds state is refused. No cell walk, serialization or recompute is involved; if any
// row or metadata were lost or misplaced, the logits would differ.

#include "arg.h"
#include "common.h"
#include "llama.h"

#include "../src/llama-kv-cache.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <vector>

static void require(bool condition, const char * message) {
    if (!condition) {
        fprintf(stderr, "FAILED: %s\n", message);
        std::exit(1);
    }
}

static llama_kv_cache * g_kv = nullptr;

// the completion fence: llama_decode queues device work and returns; a raw copy of a stream reads or writes device
// memory, so the context's queued compute must have completed first (the same fence llama_state_* takes)
static bool fenced(llama_context * ctx) {
    llama_synchronize(ctx);
    return true;
}

// her stream's cell metadata as the decode left it: used count, extent, and each used cell's (index, pos, has her seq)
static void dump_cells(const char * what, llama_seq_id seq) {
    const auto & cells = g_kv->get_cells(seq);
    fprintf(stderr, "%s: seq %d used=%u used_max_p1=%u cells:", what, seq, cells.get_used(), cells.used_max_p1());
    for (uint32_t i = 0; i < cells.size(); ++i) {
        if (!cells.is_empty(i) && (i < 2 || i + 3 >= cells.used_max_p1())) {
            fprintf(stderr, " [%u p%d s%d]", i, cells.pos_get(i), cells.seq_has(i, seq) ? 1 : 0);
        }
    }
    fprintf(stderr, "\n");
}

static std::vector<float> next_logits(llama_context * ctx, llama_seq_id seq, llama_token token, llama_pos pos) {
    llama_batch b = llama_batch_init(1, 0, 1);
    common_batch_add(b, token, pos, { seq }, true);
    require(llama_decode(ctx, b) == 0, "decode the next token");
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx)));
    const float * l = llama_get_logits_ith(ctx, 0);
    std::vector<float> out(l, l + n_vocab);
    llama_batch_free(b);
    return out;
}

// the largest absolute difference and whether the top token agrees (diagnostic when bytes differ)
static void report(const char * what, const std::vector<float> & a, const std::vector<float> & b) {
    float max_diff = 0.0f;
    size_t top_a = 0, top_b = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        max_diff = std::max(max_diff, std::fabs(a[i] - b[i]));
        top_a = a[i] > a[top_a] ? i : top_a;
        top_b = b[i] > b[top_b] ? i : top_b;
    }
    const bool same = std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
    fprintf(stderr, "%s: byte-identical=%d max|diff|=%g top=%zu/%zu\n", what, same ? 1 : 0, max_diff, top_a, top_b);
}

int main(int argc, char ** argv) {
    common_params params;
    params.kv_unified      = false; // one stream per sequence: a persona's KV is one contiguous slice per layer
    params.n_parallel      = 2;
    params.n_ctx           = 512;
    params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED; // V rows contiguous (not transposed)
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }
    ggml_backend_load_all();
    common_init_result_ptr init = common_init_from_params(params);
    llama_context * ctx = init->context();
    require(ctx != nullptr, "context");
    auto * kv = dynamic_cast<llama_kv_cache *>(llama_get_memory(ctx));
    require(kv != nullptr, "a plain KV cache (the test model is not hybrid)");
    g_kv = kv;

    // her prompt on sequence 0 (stream 0)
    const int n_prompt = 40;
    llama_batch batch = llama_batch_init(n_prompt, 0, 1);
    for (int i = 0; i < n_prompt; ++i) {
        common_batch_add(batch, 1 + (i * 7) % 300, i, { 0 }, false);
    }
    require(llama_decode(ctx, batch) == 0, "decode her prompt");
    llama_batch_free(batch);

    // the reference: her next token decoded in place, then taken back so stream 0 holds exactly her prompt again
    const llama_token next = 42;
    const std::vector<float> reference = next_logits(ctx, 0, next, n_prompt);
    dump_cells("after reference decode", 0);
    require(llama_memory_seq_rm(llama_get_memory(ctx), 0, n_prompt, -1), "take the reference token back");

    // control: the same prompt decoded natively on sequence 1 (stream 1); does stream identity alone change logits?
    {
        llama_batch b1 = llama_batch_init(n_prompt, 0, 1);
        for (int i = 0; i < n_prompt; ++i) {
            common_batch_add(b1, 1 + (i * 7) % 300, i, { 1 }, false);
        }
        require(llama_decode(ctx, b1) == 0, "decode her prompt natively on stream 1");
        llama_batch_free(b1);
        const std::vector<float> native1 = next_logits(ctx, 1, next, n_prompt);
        report("native on stream 1 vs stream 0", reference, native1);
        require(llama_memory_seq_rm(llama_get_memory(ctx), 1, -1, -1), "clear the stream 1 control");
    }

    // determinism: the same token decoded in place again
    {
        const std::vector<float> again = next_logits(ctx, 0, next, n_prompt);
        report("in place twice", reference, again);
        require(llama_memory_seq_rm(llama_get_memory(ctx), 0, n_prompt, -1), "take it back again");
    }
    // the copies alone: a round trip through host memory into the SAME stream, no bind
    {
        llama_kv_stream_blocks rt;
        require(fenced(ctx) && kv->stream_save(0, rt) == llama_kv_stream_move::done, "round-trip save");
        require(fenced(ctx) && kv->stream_load(0, rt) == llama_kv_stream_move::done, "round-trip load");
        const std::vector<float> same_stream = next_logits(ctx, 0, next, n_prompt);
        report("same stream round trip", reference, same_stream);
        require(llama_memory_seq_rm(llama_get_memory(ctx), 0, n_prompt, -1), "take it back after the round trip");
    }

    // out: raw rows + her metadata swapped out; stream 0 left empty
    llama_kv_stream_blocks blocks;
    require(fenced(ctx) && kv->stream_save(0, blocks) == llama_kv_stream_move::done, "save stream 0");
    require(blocks.n_rows == (uint32_t) n_prompt, "she uses exactly her prompt's rows");
    require(llama_memory_seq_pos_max(llama_get_memory(ctx), 0) < 0, "stream 0 is empty after the save");

    // a load into a stream that holds state is refused, and changes nothing
    {
        llama_batch b = llama_batch_init(1, 0, 1);
        common_batch_add(b, 5, 0, { 1 }, false);
        require(llama_decode(ctx, b) == 0, "occupy stream 1");
        llama_batch_free(b);
        require(fenced(ctx) && kv->stream_load(1, blocks) == llama_kv_stream_move::stream_not_empty, "never loaded over state");
        require(llama_memory_seq_rm(llama_get_memory(ctx), 1, -1, -1), "clear stream 1");
    }

    // in: raw rows + her metadata swapped into physical stream 1; she keeps her own sequence id 0, bound to stream 1
    require(fenced(ctx) && kv->stream_load(1, blocks) == llama_kv_stream_move::done, "load into stream 1");
    require(kv->stream_bind(0, 1) == llama_kv_stream_move::done, "bind her sequence to stream 1");
    {
        // her rows as they now sit in stream 1, read back raw, against the bytes that were saved
        llama_kv_stream_blocks back;
        require(fenced(ctx) && kv->stream_save(1, back) == llama_kv_stream_move::done, "read stream 1 back");
        bool rows_equal = back.n_rows == blocks.n_rows;
        for (size_t i = 0; rows_equal && i < back.k.size(); ++i) {
            rows_equal = back.k[i] == blocks.k[i] && back.v[i] == blocks.v[i];
        }
        fprintf(stderr, "stream 1 rows after load equal the saved bytes: %d (layers %zu)\n", rows_equal ? 1 : 0, back.k.size());
        require(fenced(ctx) && kv->stream_load(1, back) == llama_kv_stream_move::done, "put them back");
    }
    require(llama_memory_seq_pos_max(llama_get_memory(ctx), 0) == n_prompt - 1, "her sequence holds her prompt");
    require(llama_memory_seq_pos_max(llama_get_memory(ctx), 1) < 0, "sequence 1 now maps to the empty stream 0");
    dump_cells("moved, before decode", 0);
    const std::vector<float> moved = next_logits(ctx, 0, next, n_prompt);
    dump_cells("moved, after decode", 0);
    require(moved.size() == reference.size(), "same vocabulary");
    report("moved to stream 1", reference, moved);
    const bool moved_exact = std::memcmp(moved.data(), reference.data(), moved.size() * sizeof(float)) == 0;

    // bisection variants, each from a cleared cache with the identity mapping
    auto reset = [&]() {
        llama_memory_clear(llama_get_memory(ctx), true);
        require(kv->stream_bind(0, 0) == llama_kv_stream_move::done, "identity bind 0");
        require(kv->stream_bind(1, 1) == llama_kv_stream_move::done, "identity bind 1");
    };
    auto prompt_on = [&](llama_seq_id seq) {
        llama_batch b = llama_batch_init(n_prompt, 0, 1);
        for (int i = 0; i < n_prompt; ++i) {
            common_batch_add(b, 1 + (i * 7) % 300, i, { seq }, false);
        }
        require(llama_decode(ctx, b) == 0, "decode the prompt");
        llama_batch_free(b);
    };
    {   // X1: move to stream 1 + bind, without occupying stream 1 first
        reset();
        prompt_on(0);
        llama_kv_stream_blocks x;
        require(fenced(ctx) && kv->stream_save(0, x) == llama_kv_stream_move::done, "X1 save");
        require(fenced(ctx) && kv->stream_load(1, x) == llama_kv_stream_move::done, "X1 load");
        require(kv->stream_bind(0, 1) == llama_kv_stream_move::done, "X1 bind");
        report("X1 move+bind, no occupy", reference, next_logits(ctx, 0, next, n_prompt));
    }
    {   // X3: mirror: native on seq 1 / stream 1, moved to stream 0, seq 1 bound to stream 0
        reset();
        prompt_on(1);
        llama_kv_stream_blocks x;
        require(fenced(ctx) && kv->stream_save(1, x) == llama_kv_stream_move::done, "X3 save");
        require(fenced(ctx) && kv->stream_load(0, x) == llama_kv_stream_move::done, "X3 load");
        require(kv->stream_bind(1, 0) == llama_kv_stream_move::done, "X3 bind");
        report("X3 mirror stream1->stream0", reference, next_logits(ctx, 1, next, n_prompt));
    }
    {   // X4: in place, two tokens; then moved, two tokens: does the second token differ too?
        reset();
        prompt_on(0);
        next_logits(ctx, 0, next, n_prompt);
        const std::vector<float> in_place_2 = next_logits(ctx, 0, next + 1, n_prompt + 1);
        reset();
        prompt_on(0);
        llama_kv_stream_blocks x;
        require(fenced(ctx) && kv->stream_save(0, x) == llama_kv_stream_move::done, "X4 save");
        require(fenced(ctx) && kv->stream_load(1, x) == llama_kv_stream_move::done, "X4 load");
        require(kv->stream_bind(0, 1) == llama_kv_stream_move::done, "X4 bind");
        report("X4 first token after move", reference, next_logits(ctx, 0, next, n_prompt));
        report("X4 second token after move", in_place_2, next_logits(ctx, 0, next + 1, n_prompt + 1));
    }
    require(moved_exact, "her next-token logits are byte-identical after the move");

    fprintf(stderr, "test-kv-stream-blocks: ok (%u rows moved raw)\n", (unsigned) n_prompt);
    return 0;
}
