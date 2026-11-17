#pragma once

// Continuum fork (CONTINUUM-ENGINE 4.12.1): where a turn's results go. One verb: deliver a typed result. What a result
// says is built by turn_results.h; how it reaches its reader is an adapter's: responder_http (the OpenAI-compatible
// HTTP server's result queue, for external drivers) and, with contract slice 2, responder_frames (our core).

#include "../server-task.h"

struct responder {
    virtual ~responder() = default;
    virtual void deliver(server_task_result_ptr result) = 0;
};
