include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})

# server-context containing the core server logic, used by llama-server and CLI

set(TARGET server-context)

add_library(${TARGET} STATIC
    server-chat.cpp
    server-chat.h
    server-task.cpp
    server-task.h
    server-queue.cpp
    server-queue.h
    server-common.cpp
    server-common.h
    server-context.cpp
    server-train.cpp
    server-context.h
    server-stream.cpp
    server-stream.h
    server-tools.cpp
    server-tools.h
    server-mcp.cpp
    server-mcp.h
    server-schema.cpp
    server-schema.h
    server-continuum.cpp
    server-continuum.h
    ${CMAKE_CURRENT_BINARY_DIR}/server-continuum-schema.c
)

# the contract's schema identity (server-continuum.h): the first 16 bytes of the header's SHA-256, as the core's
# build.rs computes it; reconfigured whenever the header changes
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/server-continuum.h)
file(SHA256 ${CMAKE_CURRENT_SOURCE_DIR}/server-continuum.h CTM_SCHEMA_SHA256)
string(SUBSTRING "${CTM_SCHEMA_SHA256}" 0 32 CTM_SCHEMA_HEX)
string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1, " CTM_SCHEMA_BYTES "${CTM_SCHEMA_HEX}")
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/server-continuum-schema.c.in ${CMAKE_CURRENT_BINARY_DIR}/server-continuum-schema.c @ONLY)

if (BUILD_SHARED_LIBS)
    set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

target_include_directories(${TARGET} PRIVATE ../mtmd)
target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(${TARGET} PUBLIC llama-common mtmd ${CMAKE_THREAD_LIBS_INIT})

# the faults tools/server/tests inject (server-test-faults.h); only the server test build sets it
option(LLAMA_SERVER_TEST_FAULTS "llama-server: compile in the test fault injection (tests only, never a release)" OFF)
if (LLAMA_SERVER_TEST_FAULTS)
    target_compile_definitions(${TARGET} PRIVATE LLAMA_SERVER_TEST_FAULTS)
endif()

# llama-server-impl: server logic, reusable by app

set(TARGET llama-server-impl)

add_library(${TARGET}
    server.cpp
    server-http.cpp
    server-http.h
    server-models.cpp
    server-models.h
)
set_target_properties(${TARGET} PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)

target_include_directories(${TARGET} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(${TARGET} PRIVATE ../mtmd ${CMAKE_SOURCE_DIR})
target_link_libraries(${TARGET} PUBLIC server-context llama-ui cpp-httplib ${CMAKE_THREAD_LIBS_INIT})

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} LIBRARY)
endif()

# llama-server executable

set(TARGET llama-server)

add_executable(${TARGET} main.cpp)
install(TARGETS ${TARGET} RUNTIME)

target_link_libraries(${TARGET} PRIVATE llama-server-impl)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

# the engine edge of the core<->engine contract, model-free (tests/test-server-continuum.cpp); built here because it
# needs the generated schema identity beside server-continuum.cpp
if (LLAMA_BUILD_TESTS)
    add_executable(test-server-continuum ${CMAKE_SOURCE_DIR}/tests/test-server-continuum.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/server-continuum.cpp ${CMAKE_CURRENT_BINARY_DIR}/server-continuum-schema.c)
    target_include_directories(test-server-continuum PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_compile_features(test-server-continuum PRIVATE cxx_std_17)
    add_test(NAME test-server-continuum COMMAND test-server-continuum)
    set_tests_properties(test-server-continuum PROPERTIES LABELS "main")
endif()
