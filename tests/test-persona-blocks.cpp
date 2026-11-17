// what this catches: moving a persona's WHOLE state on a hybrid model (attention stream + recurrent rows,
// llama_memory_hybrid::persona_save / persona_load, CONTINUUM-ENGINE 4.14) changing her state: after her state goes
// out of stream 0 and her recurrent cell to host memory and back into stream 1 bound to her own sequence id, her next
// two tokens' logits must be byte-identical to decoding them in place. Nothing is recomputed; a lost or misplaced row
// or record changes the logits. The model argument is a hybrid (e.g. a tiny random qwen35moe from
// `test-llama-archs -a qwen35moe -o DIR`).

#include "arg.h"
#include "common.h"
#include "llama.h"

#include "../src/llama-memory-hybrid.h"

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

static bool same(const char * what, const std::vector<float> & a, const std::vector<float> & b) {
    float max_diff = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        max_diff = std::max(max_diff, std::fabs(a[i] - b[i]));
    }
    const bool identical = a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
    fprintf(stderr, "%s: byte-identical=%d max|diff|=%g\n", what, identical ? 1 : 0, max_diff);
    return identical;
}

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
    llama_context * ctx = init->context();
    require(ctx != nullptr, "context");
    auto * mem = dynamic_cast<llama_memory_hybrid *>(llama_get_memory(ctx));
    require(mem != nullptr, "a hybrid memory (attention + recurrent)");
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx)));

    const int n_prompt = 24;
    auto prompt_on = [&](llama_seq_id seq) {
        llama_batch b = llama_batch_init(n_prompt, 0, 1);
        for (int i = 0; i < n_prompt; ++i) {
            common_batch_add(b, 1 + (i * 7) % (n_vocab - 1), i, { seq }, false);
        }
        require(llama_decode(ctx, b) == 0, "decode her prompt");
        llama_batch_free(b);
    };
    const llama_token t1 = 3 % n_vocab;
    const llama_token t2 = 5 % n_vocab;

    // the reference: her prompt, then her next two tokens decoded in place
    prompt_on(0);
    const std::vector<float> ref1 = next_logits(ctx, 0, t1, n_prompt);
    const std::vector<float> ref2 = next_logits(ctx, 0, t2, n_prompt + 1);

    // the same prompt again from a clear memory, then her whole state moved out and into stream 1, bound to her id
    llama_memory_clear(llama_get_memory(ctx), true);
    prompt_on(0);
    llama_synchronize(ctx); // the completion fence before raw copies
    llama_persona_blocks blocks;
    require(mem->persona_save(0, blocks), "her state out");
    require(blocks.rs.has_state, "she has recurrent state");
    require(llama_memory_seq_pos_max(llama_get_memory(ctx), 0) < 0, "nothing of her is left in the pool");
    require(mem->persona_load(0, 1, blocks), "her state into stream 1 and a free recurrent cell");
    require(llama_memory_seq_pos_max(llama_get_memory(ctx), 0) == n_prompt - 1, "her sequence holds her prompt again");

    const bool ok1 = same("first token after the move", ref1, next_logits(ctx, 0, t1, n_prompt));
    const bool ok2 = same("second token after the move", ref2, next_logits(ctx, 0, t2, n_prompt + 1));
    require(ok1 && ok2, "her logits are byte-identical after her whole state moved");

    fprintf(stderr, "test-persona-blocks: ok (%zu attention layers, %zu recurrent layers, moved raw)\n",
            blocks.kv.k.size(), blocks.rs.r.size());
    return 0;
}
