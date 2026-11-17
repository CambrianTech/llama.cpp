// Continuum fork (CONTINUUM-ENGINE 4.12.1): see responder_http.h.

#include "responder_http.h"

void responder_http::deliver(server_task_result_ptr result) {
    queue_.send(std::move(result));
}
