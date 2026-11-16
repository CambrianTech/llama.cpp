#pragma once

#include "server-page-adapters.h"
#include "server-page-io.h"
#include "server-task.h"

#include <atomic>
#include <list>
#include <memory>
#include <string>
#include <vector>

// Continuum fork: one KV page switch in flight between the task loop and its file IO worker
// (server-page-io.h). The loop fills the half it owns (a save's copied state; a restore's slot and
// occupancy), the worker the other (the file written; the page read), and the completion task
// carries it back to the loop, which alone touches the slot.
struct server_page_job {
    bool        is_save = true;
    int         id_slot = -1;
    std::string filename;
    std::string filepath;

    // the slot's occupancy when the switch was asked (server_slot::n_occupancy): a restore whose
    // slot has since been given another task or erased is refused, never applied over it
    uint64_t occupancy = 0;

    // save: what the loop copied out of the slot (the page body, its tokens, its sidecars)
    size_t               n_tokens = 0; // the slot's prompt tokens (the receipt's n_saved)
    std::vector<uint8_t> body;
    std::vector<int32_t> tokens;
    std::vector<uint8_t> ckpt_sidecar;
    std::vector<uint8_t> lora_sidecar;

    // restore: what the worker read (the adapter identity is checked on the loop, against the
    // adapters loaded when it applies)
    std::vector<server_page_adapter>    adapters;
    std::list<common_prompt_checkpoint> checkpoints;
    bool                                checkpoints_ok = false;
    std::string                         ckpt_path;

    int id_task = -1; // the switch's HTTP waiter

    // the host bytes this switch holds, admitted against the one page IO budget (server-page-io.h)
    server_page_reservation reservation;
    uint64_t                bytes    = 0; // what it was admitted for
    double                  admit_ms = 0; // asked to admitted

    // a restore whose slot was given another task or erased while it was buffered: its bytes are
    // returned at once and its read stops (server-context-page.inc page_drop_restore)
    std::atomic<bool> dropped{false};
    std::atomic<bool> io_done{false}; // the worker is done with the buffers; the loop owns them
    bool              answered = false; // its waiter already has its receipt

    std::string error;       // the IO's failure; empty when it succeeded
    size_t      n_bytes = 0; // the page file's bytes, written or read
    double      copy_ms = 0; // the loop's share: the state copied out or in
    double      io_ms   = 0; // the worker's share: the files written or read

    std::shared_ptr<server_restore_progress> progress;

    // A switch dropped before it applied (its waiter left and its completion was cancelled, or the
    // server stopped) never holds its slot's restore progress open.
    ~server_page_job() {
        if (progress && !progress->terminal()) {
            progress->phase.store(server_restore_progress::cancelled, std::memory_order_relaxed);
        }
    }
};
