# Model-free scheduling validator: compile the same traversal used by server-context.
# Model-free: placed checkpoints keep her head through a full store (CONTINUUM-ENGINE 4.7.3, llama.cpp#121).
add_executable(test-server-checkpoint-policy test-server-checkpoint-policy.cpp)
target_compile_features(test-server-checkpoint-policy PRIVATE cxx_std_17)
add_test(NAME test-server-checkpoint-policy COMMAND test-server-checkpoint-policy)
set_tests_properties(test-server-checkpoint-policy PROPERTIES LABELS "main")

add_executable(test-server-media-range test-server-media-range.cpp)
target_compile_features(test-server-media-range PRIVATE cxx_std_17)
add_test(NAME test-server-media-range COMMAND test-server-media-range)
set_tests_properties(test-server-media-range PROPERTIES LABELS "main")

add_executable(test-server-prefill test-server-prefill.cpp)
target_compile_features(test-server-prefill PRIVATE cxx_std_17)
add_test(NAME test-server-prefill COMMAND test-server-prefill)
set_tests_properties(test-server-prefill PROPERTIES LABELS "main")

# Model-free scheduling validator: which slot's adapter set holds the next batch.
add_executable(test-server-adapter-turn test-server-adapter-turn.cpp)
target_compile_features(test-server-adapter-turn PRIVATE cxx_std_17)
add_test(NAME test-server-adapter-turn COMMAND test-server-adapter-turn)
set_tests_properties(test-server-adapter-turn PROPERTIES LABELS "main")

# Model-free validator: whether a saved page's KV belongs to the loaded adapters.
add_executable(test-server-page-adapters test-server-page-adapters.cpp)
target_compile_features(test-server-page-adapters PRIVATE cxx_std_17)
add_test(NAME test-server-page-adapters COMMAND test-server-page-adapters)
set_tests_properties(test-server-page-adapters PROPERTIES LABELS "main")

# Model-free validator: KV page file IO off the task loop (a worker pool, one switch per file at a time).
add_executable(test-server-page-io test-server-page-io.cpp)
target_compile_features(test-server-page-io PRIVATE cxx_std_17)
find_package(Threads REQUIRED)
target_link_libraries(test-server-page-io PRIVATE Threads::Threads)
add_test(NAME test-server-page-io COMMAND test-server-page-io)
set_tests_properties(test-server-page-io PROPERTIES LABELS "main")

# Model-free learning-budget validator: the ledger every learning step debits.
add_executable(test-server-learn-ledger test-server-learn-ledger.cpp)
target_compile_features(test-server-learn-ledger PRIVATE cxx_std_17)
add_test(NAME test-server-learn-ledger COMMAND test-server-learn-ledger)
set_tests_properties(test-server-learn-ledger PROPERTIES LABELS "main")

llama_build_and_test(test-server-view-end.cpp)

# llama_kv_cells' derived indices held natively answer exactly as the node containers they replace (CONTINUUM-ENGINE 4.14)
llama_build_and_test(test-kv-cell-index.cpp)
# a persona's token history appends only a request's new ids, in one validated transaction (CONTINUUM-ENGINE 4.14)
llama_build_and_test(test-persona-tokens.cpp)

# GGML_TENSOR_FLAG_GRAD: a leaf whose gradient the backward computes, never optimized (the walk reads the cached prefix K/V gradient)
llama_build_and_test(test-grad-leaf.cpp)

# ggml-opt with per-step graphs (the llama training path) starts each optimizer period from zero accumulators
llama_build_and_test(test-opt-dynamic-accum.cpp)
llama_build_and_test(test-sched-measure.cpp)
if (NOT WIN32 OR NOT BUILD_SHARED_LIBS)
    llama_build_and_test(test-walk-plan.cpp)
    # the exact walk trains on the window's whole gradient whatever the chunking: run on a pure-attention AND a
    # hybrid model (-m), on a GPU backend as well as the CPU (a hybrid passed on the CPU while Metal tore its state)
    llama_build(test-walk-exact.cpp)
    llama_build(test-graph-slots.cpp)
    llama_build_and_test(test-server-tokens-range.cpp)
    target_include_directories(test-server-tokens-range PRIVATE ${PROJECT_SOURCE_DIR}/tools/server ${PROJECT_SOURCE_DIR}/tools/mtmd)
    target_link_libraries(test-server-tokens-range PRIVATE server-context)
    # a persona's state moved raw between the memory and host memory is byte-identical, and a refused move changes
    # nothing (CONTINUUM-ENGINE 4.14): on a plain KV model, on the generated hybrid (qwen35moe: attention + recurrent),
    # and on the hybrid with the rollback groups serving's draft-mtp configuration creates
    llama_build(test-persona-blocks.cpp)
    llama_test(
        test-persona-blocks
        NAME test-persona-blocks-hybrid
        LABEL main
        ARGS -m "${MODEL_DIR}/qwen35moe-moe.gguf"
    )
    set_tests_properties(test-persona-blocks-hybrid PROPERTIES FIXTURES_REQUIRED generate-models)
    llama_test(
        test-persona-blocks
        NAME test-persona-blocks-rollback
        LABEL main
        ARGS -m "${MODEL_DIR}/qwen35moe-moe.gguf" --test-rs-seq 4
    )
    set_tests_properties(test-persona-blocks-rollback PROPERTIES FIXTURES_REQUIRED generate-models)

endif()
# the tiled FLASH_ATTN_EXT backward gives plain attention's gradients on every backend (no model needed)
llama_build_and_test(test-flash-attn-back.cpp)
llama_build_and_test(test-moe-gather.cpp)
llama_build_and_test(test-dup-quantized.cpp)
llama_build_and_test(test-state-restore-streams.cpp LABEL "model" ARGS -m "${MODEL_DEST}")
set_tests_properties(test-state-restore-streams PROPERTIES FIXTURES_REQUIRED test-download-model)

if (NOT WIN32 OR NOT BUILD_SHARED_LIBS)
    # test-persona-blocks (built above with the internal-API tests) on the downloaded plain KV model
    llama_test(test-persona-blocks LABEL "model" ARGS -m "${MODEL_DEST}")
    set_tests_properties(test-persona-blocks PROPERTIES FIXTURES_REQUIRED test-download-model)
endif()

add_executable(test-moe-residency test-moe-residency.cpp)
target_compile_features(test-moe-residency PRIVATE cxx_std_17)
target_include_directories(test-moe-residency PRIVATE ${PROJECT_SOURCE_DIR}/ggml/src ${PROJECT_SOURCE_DIR}/ggml/include)
add_test(NAME test-moe-residency COMMAND test-moe-residency)
