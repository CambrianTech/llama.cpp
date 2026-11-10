#pragma once

#include <cstdint>

// Continuum fork: the serving contexts' geometry, what a live reconfigure changes
// (server-reconfigure.h). Its own header so a task can carry one (server-task.h).
struct server_geometry {
    int32_t n_ctx = 0; // tokens the target context allocates, every sequence together (-c)
    int32_t n_seq = 0; // sequences, one per slot (-np)

    bool operator==(const server_geometry & other) const { return n_ctx == other.n_ctx && n_seq == other.n_seq; }
    bool operator!=(const server_geometry & other) const { return !(*this == other); }
};
