// Continuum: an embeddings-only graph (an embedding service on the serving model) keeps the pooled output and every
// cache write, and drops the logits branch, kept beside llama-model.cpp so upstream's graph build merges untouched.
#include "llama-graph.h"

#include <algorithm>
#include <stdexcept>
#include <unordered_set>
#include <vector>

void llm_graph_context::prune_to_embeddings(const llm_graph_params & params) const {
    if (params.cparams.training || params.cparams.embeddings_nextn || !params.samplers.empty() ||
            std::any_of(params.cparams.embeddings_layer_inp.begin(), params.cparams.embeddings_layer_inp.end(), [](bool enabled) { return enabled; })) {
        throw std::runtime_error("embeddings_only cannot train, sample, or extract auxiliary outputs");
    }
    GGML_ASSERT(res->t_embd_pooled != nullptr);
    // Cache writes are explicit graph roots, not dependencies of cache reads.
    // Preserve those roots and their original ordering when pruning logits.
    const auto ancestors = [](ggml_tensor * root) {
        std::unordered_set<ggml_tensor *> seen;
        std::vector<ggml_tensor *> pending { root };
        while (!pending.empty()) {
            auto * node = pending.back();
            pending.pop_back();
            if (!node || !seen.insert(node).second) {
                continue;
            }
            for (auto * src : node->src) {
                pending.push_back(src);
            }
        }
        return seen;
    };
    const auto logits = ancestors(res->t_logits);
    const auto pooled = ancestors(res->t_embd_pooled);
    std::vector<ggml_tensor *> roots;
    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
        auto * node = ggml_graph_node(gf, i);
        if (!logits.count(node) || pooled.count(node)) {
            roots.push_back(node);
        }
    }
    res->t_logits = nullptr;
    ggml_graph_clear(gf);
    for (auto * root : roots) {
        ggml_build_forward_expand(gf, root);
    }
}
