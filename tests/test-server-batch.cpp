// what this catches: removing one slot's tokens from a frame (server_batch::remove_slot, CONTINUUM-ENGINE 4.11.10:
// the slot decode could not place is preempted, the others decode) leaving any index into the batch wrong: a later
// slot's sampled index or speculative verification indexes still pointing at the old positions, an embedding row
// left behind its token, a removed token's output flag carried over, or the batch decoded from its stale rendering.
// The middle slot is removed with speculative groups on both sides of it. Model-free.
#include "../tools/server/server-batch.h"

#include <cstdlib>
#include <iostream>
#include <vector>

static void require(bool condition, const char * message) {
    if (!condition) {
        std::cerr << "FAILED: " << message << '\n';
        std::exit(1);
    }
}

// a slot's indexes into the batch, as server_slot holds them
struct slot_indexes {
    int32_t              id;
    int32_t              i_batch;
    std::vector<int32_t> spec_i_batch;
};

// a generating slot with speculation: its sampled token then its drafts, every one verified (output)
static void add_group(server_batch & b, slot_indexes & s, llama_pos pos, int n_draft) {
    s.i_batch = b.size();
    for (int k = 0; k <= n_draft; ++k) {
        s.spec_i_batch.push_back(b.size());
        require(b.add(s.id, 100 * s.id + k, pos + k, true, false), "room in the batch");
    }
}

static void remap(slot_indexes & s, const std::vector<int32_t> & moved) {
    s.i_batch = moved[s.i_batch];
    for (auto & i : s.spec_i_batch) {
        i = moved[i];
    }
}

static void the_middle_slot_leaves_with_speculative_groups_on_both_sides() {
    server_batch b;
    b.init(16, 0);
    slot_indexes a{ 0, -1, {} };
    slot_indexes mid{ 1, -1, {} };
    slot_indexes c{ 2, -1, {} };
    add_group(b, a, 50, 2);   // tokens 0..2
    add_group(b, mid, 70, 3); // tokens 3..6
    add_group(b, c, 90, 1);   // tokens 7..8
    b.render();
    require(b.batch.n_tokens == 9, "rendered before the failed decode");

    const std::vector<int32_t> moved = b.remove_slot(mid.id);
    require(moved.size() == 9, "one entry per old token");
    for (int i = 3; i <= 6; ++i) {
        require(moved[i] == -1, "the removed slot's tokens map nowhere");
    }
    require(moved[0] == 0 && moved[2] == 2 && moved[7] == 3 && moved[8] == 4, "earlier tokens stay, later ones shift");
    remap(a, moved);
    remap(c, moved);
    require(a.i_batch == 0 && a.spec_i_batch == std::vector<int32_t>({ 0, 1, 2 }), "the earlier group is unchanged");
    require(c.i_batch == 3 && c.spec_i_batch == std::vector<int32_t>({ 3, 4 }), "the later group follows its tokens");

    require(!b.batch_rendered, "the stale rendering is gone");
    b.render();
    require(b.batch.n_tokens == 5, "only the kept tokens are decoded");
    for (int i = 0; i < 5; ++i) {
        require(b.batch.seq_id[i][0] != mid.id, "no token of the removed slot");
        require(b.batch.logits[i] == 1, "every kept verification token is still an output");
    }
    require(b.batch.token[3] == 200 && b.batch.pos[3] == 90, "the later slot's first token, at its own position");
    require(b.batch.token[2] == 2 && b.batch.pos[2] == 52, "the earlier slot's last draft, untouched");
}

static void embedding_rows_move_with_their_tokens() {
    const int32_t n_embd = 3;
    server_batch b;
    b.init(8, n_embd);
    for (int32_t slot = 0; slot < 3; ++slot) {
        for (int k = 0; k < 2; ++k) {
            const float v = (float) (10 * slot + k);
            require(b.add(slot, std::vector<float>{ v, v + 0.1f, v + 0.2f }, k, k == 1, true), "room in the batch");
        }
    }
    b.render();
    b.remove_slot(1);
    require(b.size() == 4 && b.embd.size() == 4 * (size_t) n_embd, "two tokens and their rows removed");
    require(b.embd[2 * n_embd] == 20.0f && b.embd[2 * n_embd + 2] == 20.2f, "slot 2's first row moved up whole");
    require(b.embd[n_embd] == 1.0f, "slot 0's rows untouched");
    b.render();
    require(b.batch.embd == b.embd.data() && b.batch.n_tokens == 4, "rendered from the kept rows");
    b.clear();
    require(b.batch.token != nullptr, "the token array is restored after an embedding batch");
}

int main() {
    the_middle_slot_leaves_with_speculative_groups_on_both_sides();
    embedding_rows_move_with_their_tokens();
    std::cout << "test-server-batch: ok\n";
    return 0;
}
