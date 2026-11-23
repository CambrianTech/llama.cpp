// Continuum: a training row's label through upstream's llama_batch_ext. A labelled llama_batch reaches decode through
// llama_batch_compat, so each token carries its label and weight into the ext batch, and the allocator binds them to
// its internal batch, where the label validation and the ubatch split read them. Kept beside llama-batch.cpp so
// upstream's batch merges without touching it.
#include "llama-batch.h"

// copy row i's label (and its weight in the loss) from a public batch into its ext token
void llama_batch_label_copy(llama_batch_ext::token & t, const llama_batch & in, int32_t i) {
    t.label        = in.label ? in.label[i] : -1;
    t.label_weight = in.label_weight ? in.label_weight[i] : 1.0f;
}

// bind the ext batch's labels to the internal batch: nullptr when every row is unlabelled (-1), so inference stays
// label-free; any other value is bound and validated (a negative label other than -1 is refused, never dropped)
void llama_batch_allocr::bind_labels(const llama_batch_ext & in) {
    const int32_t n_tok = (int32_t) in.tokens.size();
    bool any = false;
    for (int32_t i = 0; i < n_tok && !any; ++i) {
        any = in.tokens[i].label != -1; // a label other than "none" is bound, so an invalid one is refused, never dropped
    }
    label_vec.clear();
    label_weight_vec.clear();
    if (!any) {
        batch.label        = nullptr;
        batch.label_weight = nullptr;
        return;
    }
    label_vec.resize(n_tok);
    label_weight_vec.resize(n_tok);
    for (int32_t i = 0; i < n_tok; ++i) {
        label_vec[i]        = in.tokens[i].label;
        label_weight_vec[i] = in.tokens[i].label_weight;
    }
    batch.label        = label_vec.data();
    batch.label_weight = label_weight_vec.data();
}
