add_executable(llama-fit-device fit-device.cpp)
target_link_libraries(llama-fit-device PRIVATE llama ggml)
target_include_directories(llama-fit-device PRIVATE ${CMAKE_SOURCE_DIR}/ggml/include)
target_compile_features(llama-fit-device PRIVATE cxx_std_17)
