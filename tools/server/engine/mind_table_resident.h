#pragma once

// Continuum fork (CONTINUUM-ENGINE 4.12.1): the #139 resident-minds table, built behind mind_table. The class itself
// is private to mind_table_resident.cpp; the engine holds only the interface.

#include "mind_table.h"

#include <memory>
#include <string>
#include <vector>

class server_page_io_pool;

// the table, or why the configuration cannot have one (typed refusal, never a half-built table)
struct mind_table_build {
    std::unique_ptr<mind_table> table;
    std::string                 refusal; // non-empty when table is null
};

mind_table_build make_resident_minds(const mind_table_config & cfg, memory_pool & pool, server_page_io_pool & page_io,
                                     std::vector<server_slot> & slots);
