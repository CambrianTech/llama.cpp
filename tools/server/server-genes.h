#pragma once

// Continuum fork: genes (LoRA adapters) paged between device memory and host RAM.
//
// A mind runs only the genes her turn is near (continuum genome::selection); the rest of the
// genes the node has loaded need not hold device memory. The core's working-set owner decides,
// by temperature and within one memory budget, which genes stay on the device; these verbs are
// the mechanism: `page_out` copies an adapter's tensors to host RAM and frees its device buffers,
// `page_in` re-allocates them and copies back, `unload` (POST /lora-adapters/unload) drops it.
// An adapter keeps its id and identity throughout, so a slot's adapter list, its KV and a saved
// page computed under it stay valid. A task that applies a host-resident adapter pages it in
// first and counts the miss, so the engine never computes on freed memory.
// Design: continuum docs/architecture/SERVING-ENGINE.md (the `genes` module, §3 one budget).

#include "server-pages.h"

#include "common.h"

#include <cstdint>
#include <functional>
#include <string>
#include <utility>
#include <vector>

// Where an adapter's tensors are, after a move.
struct server_gene_move {
    int32_t     id = -1;
    std::string path;
    bool        on_device = true;
    uint64_t    n_bytes   = 0;   // the adapter's bytes where it is now
    uint64_t    n_moved   = 0;   // bytes this call copied; 0 when it was already there
    double      t_ms      = 0.0;

    json to_json() const;
};

// The result of SERVER_TASK_TYPE_PAGE_LORA.
struct server_task_result_gene_move : server_task_result {
    server_gene_move move;
    virtual json to_json() override { return move.to_json(); }
};

// Moves adapter `id` of `loras` to the device (`to_device`) or to host RAM. Refuses with
// server_page_error: an unknown or retired id (INVALID_REQUEST); to host, the adapter the learning
// optimizer holds or one active by default (INVALID_REQUEST), or one a processing slot applies
// (UNAVAILABLE: retry when it finishes; `applied_by_processing_slot` returns that slot's id or -1);
// a failed allocation (SERVER), the adapter then left where it was. The caller synchronizes the
// contexts that ran the adapter before paging it out.
server_gene_move server_gene_page(const std::vector<common_adapter_lora_info> & loras, int32_t id, bool to_device,
                                  int32_t learn_lora_id, const std::function<int(int32_t)> & applied_by_processing_slot);

// Pages in every adapter `applied` runs (scale != 0) that is in host RAM: a task is about to run
// it. Returns what moved; empty on the fast path. Throws server_page_error on a failed allocation.
std::vector<server_gene_move> server_gene_page_in_applied(const std::vector<common_adapter_lora_info> & applied);

// (device bytes, host bytes) of every loaded adapter.
std::pair<uint64_t, uint64_t> server_gene_bytes(const std::vector<common_adapter_lora_info> & loras);
