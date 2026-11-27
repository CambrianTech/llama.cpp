#pragma once

// Continuum fork (CONTINUUM-ENGINE 4.14): one persona, the object behind the public `llama_persona *`: her blocks (her
// execution state out of the memory pool) and her token history. Its moves are in llama-context.cpp; its files (her
// blocks on disk, for a restart that demotes instead of dropping her, 4.16) are in llama-persona-file.cpp.

#include "llama-persona-blocks.h"
#include "llama-persona-tokens.h"

#include <memory>

struct llama_persona {
    llama_persona_blocks                  blocks;
    std::unique_ptr<llama_persona_tokens> tokens; // her token history, beside her blocks (llama-persona-tokens.h)
};

// a checkpoint of a bound mind (llama_persona_checkpoint_take): her recurrent rows and where she was, never her KV; its
// file form beside her record (llama-persona-file.cpp) carries it across a restart
struct llama_persona_checkpoint {
    llama_persona_checkpoint_blocks blocks;
};
