/* Standalone synthetic throughput benchmark, not a correctness test or model profiler. */
#include "quantized_matvec.h"
#include "quantization.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>
#include <stdint.h>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

struct Format {
    const char *name;
    uint32_t type;
    size_t elements;
    size_t bytes;
};

static bool positive_int(const char *text, int *value)
{
    if (text == nullptr || *text == '\0') return false;
    char *end = nullptr;
    errno = 0;
    long parsed = strtol(text, &end, 10);
    if (errno != 0 || *end != '\0' || parsed <= 0 || parsed > INT_MAX) return false;
    *value = (int)parsed;
    return true;
}

static void fill_blocks(std::vector<unsigned char> &data, const Format &format)
{
    /* Reproducible nonzero payload; valid, finite F16 factors in every block. */
    uint32_t rng = 0x12345678u;
    for (size_t i = 0; i < data.size(); ++i) {
        rng = rng * 1664525u + 1013904223u;
        data[i] = (unsigned char)(rng >> 24);
    }
    for (size_t offset = 0; offset < data.size(); offset += format.bytes) {
        size_t scale_offset = offset + (format.type == ENGINE_GGML_TYPE_Q6_K ? format.bytes - 2 : 0);
        data[scale_offset] = 0x00;
        data[scale_offset + 1] = 0x38; /* F16 0.5; Q6_K stores its scale at the end */
        if (format.type == ENGINE_GGML_TYPE_Q4_K) {
            data[offset + 2] = 0x00;
            data[offset + 3] = 0x38; /* F16 0.5 minimum factor */
        }
    }
}

static bool benchmark(const Format &format, int rows, int columns, int repetitions)
{
    const size_t blocks = (size_t)rows * (size_t)columns / format.elements;
    std::vector<unsigned char> data(blocks * format.bytes);
    std::vector<float> input((size_t)columns);
    std::vector<float> output((size_t)rows);
    fill_blocks(data, format);
    for (int i = 0; i < columns; ++i) input[(size_t)i] = (float)((i % 17) - 8) / 16.0f;

    GgufQuantizedTensorView view = {};
    view.data = data.data();
    view.ggml_type = format.type;
    view.dimension_count = 2;
    view.dimensions[0] = (uint64_t)columns;
    view.dimensions[1] = (uint64_t)rows;
    view.element_count = (uint64_t)rows * (uint64_t)columns;
    view.block_elements = format.elements;
    view.block_bytes = format.bytes;
    view.block_count = blocks;

    for (int i = 0; i < 2; ++i) {
        if (!quantized_tensor_matvec_f32(&view, input.data(), columns, output.data(), rows)) return false;
    }
    std::vector<double> milliseconds;
    milliseconds.reserve((size_t)repetitions);
    for (int i = 0; i < repetitions; ++i) {
        auto start = std::chrono::steady_clock::now();
        bool ok = quantized_tensor_matvec_f32(&view, input.data(), columns, output.data(), rows) != 0;
        auto end = std::chrono::steady_clock::now();
        if (!ok) return false;
        milliseconds.push_back(std::chrono::duration<double, std::milli>(end - start).count());
    }
    double checksum = 0.0;
    for (float value : output) checksum += value;
    if (!std::isfinite(checksum)) return false;
    std::sort(milliseconds.begin(), milliseconds.end());
    double median = (milliseconds[(size_t)(repetitions - 1) / 2] +
                     milliseconds[(size_t)repetitions / 2]) / 2.0;
    printf("%-6s %12.3f %17.2f %14.3f %14.3f\n", format.name, median,
           (double)rows * columns / (median * 1000.0),
           (double)data.size() / (1024.0 * 1024.0), checksum);
    return true;
}

int main(int argc, char **argv)
{
    int rows = 1024, columns = 1024, repetitions = 10;
    if (argc > 4 || (argc > 1 && !positive_int(argv[1], &rows)) ||
        (argc > 2 && !positive_int(argv[2], &columns)) ||
        (argc > 3 && !positive_int(argv[3], &repetitions)) ||
        columns % 256 != 0 || repetitions > 10000 ||
        (uint64_t)rows * (uint64_t)columns > 128ull * 1024ull * 1024ull) {
        fprintf(stderr, "Usage: %s [rows=1024] [columns=1024 (multiple of 256)] [repetitions=10]\n", argv[0]);
        return 1;
    }
    const Format formats[] = {
        {"Q5_0", ENGINE_GGML_TYPE_Q5_0, Q5_0_BLOCK_ELEMENTS, Q5_0_BLOCK_BYTES},
        {"Q8_0", ENGINE_GGML_TYPE_Q8_0, Q8_0_BLOCK_ELEMENTS, Q8_0_BLOCK_BYTES},
        {"Q4_K", ENGINE_GGML_TYPE_Q4_K, Q4_K_BLOCK_ELEMENTS, Q4_K_BLOCK_BYTES},
        {"Q6_K", ENGINE_GGML_TYPE_Q6_K, Q6_K_BLOCK_ELEMENTS, Q6_K_BLOCK_BYTES}
    };
#ifdef _OPENMP
    printf("OpenMP threads: %d\n", omp_get_max_threads());
#else
    printf("OpenMP: disabled\n");
#endif
    printf("Synthetic packed matvec: %d rows x %d columns, %d timed calls per format\n",
           rows, columns, repetitions);
    printf("%-6s %12s %17s %14s %14s\n", "Format", "median ms", "Mweights/s", "matrix MiB", "checksum");
    for (const Format &format : formats) {
        if (!benchmark(format, rows, columns, repetitions)) {
            fprintf(stderr, "Benchmark failed for %s\n", format.name);
            return 1;
        }
    }
    return 0;
}
