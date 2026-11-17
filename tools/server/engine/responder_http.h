#pragma once

// Continuum fork (CONTINUUM-ENGINE 4.12.1): the responder for the HTTP server: results go to its result queue, where a
// route's response reader waits for them by task id.

#include "responder.h"

#include "../server-queue.h"

class responder_http final : public responder {
public:
    explicit responder_http(server_response & queue) : queue_(queue) {}
    void deliver(server_task_result_ptr result) override;

private:
    server_response & queue_;
};
