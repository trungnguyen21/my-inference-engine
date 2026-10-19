cmake_minimum_required(VERSION 3.16)
project(my_own_inference_engine LANGUAGES CXX)

include(FetchContent)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Let editors use the same include paths and compiler flags as the build.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(ENGINE_OPENMP "Enable OpenMP parallel CPU kernels" ON)
if(ENGINE_OPENMP)
    # Homebrew's keg-only libomp is outside Apple Clang's default search paths.
    # OpenMP_ROOT also accepts a custom prefix supplied with -DOpenMP_ROOT=...
    if(APPLE)
        list(APPEND OpenMP_ROOT
            "/opt/homebrew/opt/libomp"
            "/usr/local/opt/libomp"
        )
    endif()
    find_package(OpenMP REQUIRED COMPONENTS CXX)
    message(STATUS "Engine OpenMP: enabled")
else()
    message(STATUS "Engine OpenMP: disabled (serial CPU kernels)")
endif()

# llama.cpp is used only for its GGUF-aware tokenizer API. The inference
# executable continues to load weights and run model math through this project.
set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(GGML_BLAS OFF CACHE BOOL "" FORCE)
set(GGML_METAL OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
    llama_cpp
    GIT_REPOSITORY https://github.com/ggml-org/llama.cpp.git
    GIT_TAG a7b94df2c616bc1f62a73b964b4a71cb0dcc488e
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(llama_cpp)

add_executable(inference
    src/main.cpp
    src/app.cpp
    src/engine.cpp
    src/architecture.cpp
    src/qwen2_architecture.cpp
    src/sampler.cpp
    src/transformer_ops.cpp
    src/tensor.cpp
    src/quantization.cpp
    src/quantized_matvec.cpp
    src/tokenizer.cpp
    src/gguf_reader.cpp
    src/metrics.cpp
    src/terminal.cpp
)

target_include_directories(inference PRIVATE include)
target_link_libraries(inference PRIVATE llama)

if(APPLE OR CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_compile_definitions(inference PRIVATE _POSIX_C_SOURCE=200809L)
endif()

if(MSVC)
    target_compile_options(inference PRIVATE /W4)
else()
    target_compile_options(inference PRIVATE -Wall -Wextra -Wpedantic)
endif()

include(CTest)

if(BUILD_TESTING)
    add_executable(tensor_reference_test
        tests/tensor_reference_test.cpp
        src/tensor.cpp
    )

    target_include_directories(tensor_reference_test PRIVATE include)

    # Keep assert active for this test executable, including Release builds.
    if(MSVC)
        target_compile_options(tensor_reference_test PRIVATE /W4 /UNDEBUG)
    else()
        target_compile_options(tensor_reference_test PRIVATE -Wall -Wextra -Wpedantic -UNDEBUG)
    endif()

    add_test(
        NAME tensor_reference_test
        COMMAND tensor_reference_test
    )

    add_executable(gguf_f32_view_test
        tests/gguf_f32_view_test.cpp
        src/gguf_reader.cpp
    )

    target_include_directories(gguf_f32_view_test PRIVATE include src)

    if(APPLE OR CMAKE_SYSTEM_NAME STREQUAL "Linux")
        target_compile_definitions(gguf_f32_view_test PRIVATE _POSIX_C_SOURCE=200809L)
    endif()

    # Keep assert active for this test executable, including Release builds.
    if(MSVC)
        target_compile_options(gguf_f32_view_test PRIVATE /W4 /UNDEBUG)
    else()
        target_compile_options(gguf_f32_view_test PRIVATE -Wall -Wextra -Wpedantic -UNDEBUG)
    endif()

    add_test(
        NAME gguf_f32_view_test
        COMMAND gguf_f32_view_test
    )

    add_executable(gguf_float_metadata_test
        tests/gguf_float_metadata_test.cpp
        src/gguf_reader.cpp
    )

    target_include_directories(gguf_float_metadata_test PRIVATE include src)

    if(APPLE OR CMAKE_SYSTEM_NAME STREQUAL "Linux")
        target_compile_definitions(gguf_float_metadata_test PRIVATE _POSIX_C_SOURCE=200809L)
    endif()

    # Keep assert active for this test executable, including Release builds.
    if(MSVC)
        target_compile_options(gguf_float_metadata_test PRIVATE /W4 /UNDEBUG)
    else()
        target_compile_options(gguf_float_metadata_test PRIVATE -Wall -Wextra -Wpedantic -UNDEBUG)
    endif()

    add_test(
        NAME gguf_float_metadata_test
        COMMAND gguf_float_metadata_test
    )

    add_executable(q8_0_reference_test
        tests/q8_0_reference_test.cpp
        src/quantization.cpp
    )

    target_include_directories(q8_0_reference_test PRIVATE include)

    # Keep assert active for this test executable, including Release builds.
    if(MSVC)
        target_compile_options(q8_0_reference_test PRIVATE /W4 /UNDEBUG)
    else()
        target_compile_options(q8_0_reference_test PRIVATE -Wall -Wextra -Wpedantic -UNDEBUG)
    endif()

    add_test(
        NAME q8_0_reference_test
        COMMAND q8_0_reference_test
    )

    add_executable(q5_0_reference_test
        tests/q5_0_reference_test.cpp
        src/quantization.cpp
    )

    target_include_directories(q5_0_reference_test PRIVATE include)

    # Keep assert active for this test executable, including Release builds.
    if(MSVC)
        target_compile_options(q5_0_reference_test PRIVATE /W4 /UNDEBUG)
    else()
        target_compile_options(q5_0_reference_test PRIVATE -Wall -Wextra -Wpedantic -UNDEBUG)
    endif()

    add_test(
        NAME q5_0_reference_test
        COMMAND q5_0_reference_test
    )

    add_executable(q4_k_reference_test
        tests/q4_k_reference_test.cpp
        src/quantization.cpp
    )

    target_include_directories(q4_k_reference_test PRIVATE include)

    # Keep assert active for this test executable, including Release builds.
    if(MSVC)
        target_compile_options(q4_k_reference_test PRIVATE /W4 /UNDEBUG)
    else()
        target_compile_options(q4_k_reference_test PRIVATE -Wall -Wextra -Wpedantic -UNDEBUG)
    endif()

    add_test(
        NAME q4_k_reference_test
        COMMAND q4_k_reference_test
    )

    add_executable(q6_k_reference_test
        tests/q6_k_reference_test.cpp
        src/quantization.cpp
    )

    target_include_directories(q6_k_reference_test PRIVATE include)

    # Keep assert active for this test executable, including Release builds.
    if(MSVC)
        target_compile_options(q6_k_reference_test PRIVATE /W4 /UNDEBUG)
    else()
        target_compile_options(q6_k_reference_test PRIVATE -Wall -Wextra -Wpedantic -UNDEBUG)
    endif()

    add_test(
        NAME q6_k_reference_test
        COMMAND q6_k_reference_test
    )

    add_executable(gguf_quantized_view_test
        tests/gguf_quantized_view_test.cpp
        src/gguf_reader.cpp
    )

    target_include_directories(gguf_quantized_view_test PRIVATE include src)

    if(APPLE OR CMAKE_SYSTEM_NAME STREQUAL "Linux")
        target_compile_definitions(gguf_quantized_view_test PRIVATE _POSIX_C_SOURCE=200809L)
    endif()

    # Keep assert active for this test executable, including Release builds.
    if(MSVC)
        target_compile_options(gguf_quantized_view_test PRIVATE /W4 /UNDEBUG)
    else()
        target_compile_options(gguf_quantized_view_test PRIVATE -Wall -Wextra -Wpedantic -UNDEBUG)
    endif()

    add_test(
        NAME gguf_quantized_view_test
        COMMAND gguf_quantized_view_test
    )

    add_executable(quantized_matvec_reference_test
        tests/quantized_matvec_reference_test.cpp
        src/quantized_matvec.cpp
        src/quantization.cpp
    )

    target_include_directories(quantized_matvec_reference_test PRIVATE include src)

    # Keep assert active for this test executable, including Release builds.
    if(MSVC)
        target_compile_options(quantized_matvec_reference_test PRIVATE /W4 /UNDEBUG)
    else()
        target_compile_options(quantized_matvec_reference_test PRIVATE -Wall -Wextra -Wpedantic -UNDEBUG)
    endif()

    add_test(
        NAME quantized_matvec_reference_test
        COMMAND quantized_matvec_reference_test
    )

    add_executable(quantized_matvec_failure_test
        tests/quantized_matvec_failure_test.cpp
        src/quantized_matvec.cpp
    )
    target_include_directories(quantized_matvec_failure_test PRIVATE include src)
    if(MSVC)
        target_compile_options(quantized_matvec_failure_test PRIVATE /W4 /UNDEBUG)
    else()
        target_compile_options(quantized_matvec_failure_test PRIVATE -Wall -Wextra -Wpedantic -UNDEBUG)
    endif()
    add_test(NAME quantized_matvec_failure_test COMMAND quantized_matvec_failure_test)

    # This reference test covers the completed local numerical helpers used by
    # the Milestone 06 transformer forward path.
    add_executable(transformer_ops_reference_test
        tests/transformer_ops_reference_test.cpp
        src/transformer_ops.cpp
        src/tensor.cpp
    )

    target_include_directories(transformer_ops_reference_test PRIVATE include src)

    if(MSVC)
        target_compile_options(transformer_ops_reference_test PRIVATE /W4 /UNDEBUG)
    else()
        target_compile_options(transformer_ops_reference_test PRIVATE -Wall -Wextra -Wpedantic -UNDEBUG)
    endif()

    add_test(
        NAME transformer_ops_reference_test
        COMMAND transformer_ops_reference_test
    )

    # Milestone 06 contracts cover state ownership and the forward schedule.
    add_executable(inference_state_contract_test
        tests/inference_state_contract_test.cpp
        src/engine.cpp
        src/architecture.cpp
        src/qwen2_architecture.cpp
        src/tensor.cpp
        src/transformer_ops.cpp
        src/quantization.cpp
        src/quantized_matvec.cpp
        src/sampler.cpp
        src/tokenizer.cpp
        src/gguf_reader.cpp
    )

    target_include_directories(inference_state_contract_test PRIVATE include src)
    target_link_libraries(inference_state_contract_test PRIVATE llama)

    if(APPLE OR CMAKE_SYSTEM_NAME STREQUAL "Linux")
        target_compile_definitions(inference_state_contract_test PRIVATE _POSIX_C_SOURCE=200809L)
    endif()

    if(MSVC)
        target_compile_options(inference_state_contract_test PRIVATE /W4 /UNDEBUG)
    else()
        target_compile_options(inference_state_contract_test PRIVATE -Wall -Wextra -Wpedantic -UNDEBUG)
    endif()

    add_test(
        NAME inference_state_contract_test
        COMMAND inference_state_contract_test
    )

    add_executable(forward_reference_test
        tests/forward_reference_test.cpp
        src/engine.cpp
        src/architecture.cpp
        src/qwen2_architecture.cpp
        src/sampler.cpp
        src/transformer_ops.cpp
        src/tensor.cpp
        src/tokenizer.cpp
        src/gguf_reader.cpp
        src/quantization.cpp
        src/quantized_matvec.cpp
    )

    target_include_directories(forward_reference_test PRIVATE include src)
    target_link_libraries(forward_reference_test PRIVATE llama)

    if(APPLE OR CMAKE_SYSTEM_NAME STREQUAL "Linux")
        target_compile_definitions(forward_reference_test PRIVATE _POSIX_C_SOURCE=200809L)
    endif()

    if(MSVC)
        target_compile_options(forward_reference_test PRIVATE /W4 /UNDEBUG)
    else()
        target_compile_options(forward_reference_test PRIVATE -Wall -Wextra -Wpedantic -UNDEBUG)
    endif()

    add_test(
        NAME forward_reference_test
        COMMAND forward_reference_test
    )

    # Focused packed-weight, mixed-forward, and prompt-format regressions.
    foreach(checkpoint IN ITEMS qwen2_weight_reference_test forward_quantized_reference_test prompt_format_reference_test)
        add_executable(${checkpoint}
            tests/${checkpoint}.cpp
            src/qwen2_architecture.cpp
            src/gguf_reader.cpp
            src/tensor.cpp
            src/transformer_ops.cpp
            src/quantization.cpp
            src/quantized_matvec.cpp
        )
        target_include_directories(${checkpoint} PRIVATE include src)
        if(APPLE OR CMAKE_SYSTEM_NAME STREQUAL "Linux")
            target_compile_definitions(${checkpoint} PRIVATE _POSIX_C_SOURCE=200809L)
        endif()
        if(MSVC)
            target_compile_options(${checkpoint} PRIVATE /W4 /UNDEBUG)
        else()
            target_compile_options(${checkpoint} PRIVATE -Wall -Wextra -Wpedantic -UNDEBUG)
        endif()
    endforeach()

    target_sources(prompt_format_reference_test PRIVATE
        src/engine.cpp src/architecture.cpp src/tokenizer.cpp)
    target_link_libraries(prompt_format_reference_test PRIVATE llama)

    add_test(NAME qwen2_embedding_reference_test COMMAND qwen2_weight_reference_test embedding)
    add_test(NAME qwen2_matvec_reference_test COMMAND qwen2_weight_reference_test matvec)
    add_test(NAME forward_quantized_reference_test COMMAND forward_quantized_reference_test)
    set_tests_properties(
        qwen2_embedding_reference_test qwen2_matvec_reference_test forward_quantized_reference_test
        PROPERTIES LABELS "milestone06-quantized"
    )

    add_executable(sampler_greedy_test tests/sampler_greedy_test.cpp src/sampler.cpp)
    add_executable(app_generation_contract_test tests/app_generation_contract_test.cpp src/app.cpp)
    add_executable(operation_profile_report_test
        tests/operation_profile_report_test.cpp src/terminal.cpp src/metrics.cpp)
    foreach(checkpoint IN ITEMS sampler_greedy_test app_generation_contract_test operation_profile_report_test)
        target_include_directories(${checkpoint} PRIVATE include src)
        if(MSVC)
            target_compile_options(${checkpoint} PRIVATE /W4 /UNDEBUG)
        else()
            target_compile_options(${checkpoint} PRIVATE -Wall -Wextra -Wpedantic -UNDEBUG)
        endif()
    endforeach()

    # Greedy selection and exact prompt-format contracts.
    add_test(NAME sampler_greedy_test COMMAND sampler_greedy_test)
    add_test(NAME prompt_format_reference_test COMMAND prompt_format_reference_test)
    set_tests_properties(sampler_greedy_test prompt_format_reference_test
        PROPERTIES LABELS "milestone07-exercises")
    # These checks cover completed wiring and the existing app, using mocks.
    add_test(NAME sampler_validation_test COMMAND sampler_greedy_test validation)
    add_test(NAME prompt_dispatch_test COMMAND prompt_format_reference_test dispatch)
    add_test(NAME app_generation_contract_test COMMAND app_generation_contract_test)
    add_test(NAME operation_profile_report_test COMMAND operation_profile_report_test)
    set_tests_properties(sampler_validation_test prompt_dispatch_test app_generation_contract_test
        PROPERTIES LABELS "milestone07-infrastructure")

    # All automated checks use small synthetic fixtures or scripted backends.
    # Keep the few cross-component checks selectable separately from unit tests.
    set(engine_integration_tests
        inference_state_contract_test
        forward_reference_test
        forward_quantized_reference_test
        app_generation_contract_test
    )
    get_property(engine_tests DIRECTORY PROPERTY TESTS)
    foreach(engine_test IN LISTS engine_tests)
        if(engine_test IN_LIST engine_integration_tests)
            set_property(TEST ${engine_test} APPEND PROPERTY LABELS integration)
        else()
            set_property(TEST ${engine_test} APPEND PROPERTY LABELS unit)
        endif()
        set_tests_properties(${engine_test} PROPERTIES TIMEOUT 30)
    endforeach()
endif()

# Match the executable and all checks that compile F32 or packed matvec kernels.
# The imported target supplies compiler flags, headers, and runtime linkage.
if(ENGINE_OPENMP)
    foreach(engine_target IN ITEMS
        inference
        tensor_reference_test
        quantized_matvec_reference_test
        quantized_matvec_failure_test
        transformer_ops_reference_test
        inference_state_contract_test
        forward_reference_test
        qwen2_weight_reference_test
        forward_quantized_reference_test
        prompt_format_reference_test
    )
        if(TARGET ${engine_target})
            target_link_libraries(${engine_target} PRIVATE OpenMP::OpenMP_CXX)
        endif()
    endforeach()
endif()
