# Actual executable + actual local weights. No throughput or exact-word gate:
# this milestone requires visible text, not a particular sentence or speed.
if(NOT EXISTS "${INFERENCE_EXECUTABLE}" OR NOT EXISTS "${MODEL_PATH}")
    message(FATAL_ERROR "Supply a built inference executable and ENGINE_TEST_MODEL_PATH.")
endif()

execute_process(
    COMMAND "${INFERENCE_EXECUTABLE}"
        --model "${MODEL_PATH}"
        --prompt "What is the capital of France?"
        --max-tokens 16 --temperature 0 --top-p 1
    RESULT_VARIABLE result
    OUTPUT_VARIABLE output
    ERROR_VARIABLE diagnostics
)
if(NOT "${result}" STREQUAL "0")
    message(FATAL_ERROR "Generation exited ${result}.\n${diagnostics}\n${output}")
endif()

string(ASCII 27 escape)
string(REGEX REPLACE "${escape}\\[[0-9;]*m" "" output "${output}")
string(FIND "${output}" "Output: " begin)
string(FIND "${output}" "\n\nGeneration complete" end)
if(begin LESS 0 OR end LESS 0 OR end LESS begin)
    message(FATAL_ERROR "Missing generated-output section or completion report.\n${output}")
endif()
math(EXPR begin "${begin} + 8")
math(EXPR length "${end} - ${begin}")
string(SUBSTRING "${output}" ${begin} ${length} generated)
string(STRIP "${generated}" trimmed)
if(trimmed STREQUAL "")
    message(FATAL_ERROR "The real model produced no visible text (or only whitespace).")
endif()
foreach(marker IN ITEMS "<|im_start|>" "<|im_end|>" "<|endoftext|>")
    string(FIND "${generated}" "${marker}" found)
    if(NOT found EQUAL -1)
        message(FATAL_ERROR "A protocol token appeared in generated text: ${marker}")
    endif()
endforeach()
string(REGEX MATCH "Generated tokens: ([0-9]+)" token_report "${output}")
if(NOT token_report OR CMAKE_MATCH_1 LESS 1 OR CMAKE_MATCH_1 GREATER 16)
    message(FATAL_ERROR "Invalid generated-token count.\n${output}")
endif()
string(REGEX MATCH "Elapsed time: [0-9.]+ s" elapsed "${output}")
string(REGEX MATCH "Throughput: [0-9.]+ tokens/s" throughput "${output}")
if(NOT elapsed OR NOT throughput)
    message(FATAL_ERROR "Missing elapsed-time or throughput report.\n${output}")
endif()
message(STATUS "Real model generated visible text: ${generated}")
