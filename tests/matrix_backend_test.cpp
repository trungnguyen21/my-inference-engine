#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "matrix_backend.h"
#include "quantization.h"
#include "quantized_matvec.h"
#include "tensor.h"

static void setup_q8_view(GgufQuantizedTensorView *view, const Q8_0Block *blocks, uint64_t rows)
{
    memset(view, 0, sizeof(*view));
    view->data = (const unsigned char *)blocks;
    view->ggml_type = ENGINE_GGML_TYPE_Q8_0;
    view->dimension_count = 2;
    view->dimensions[0] = Q8_0_BLOCK_ELEMENTS;
    view->dimensions[1] = rows;
    view->element_count = Q8_0_BLOCK_ELEMENTS * rows;
    view->block_elements = Q8_0_BLOCK_ELEMENTS;
    view->block_bytes = Q8_0_BLOCK_BYTES;
    view->block_count = rows;
}

static void test_descriptors(void)
{
    const MatrixBackend *cpu = matrix_backend_cpu();
    const MatrixBackend *gpu = matrix_backend_gpu();
    assert(cpu != nullptr && cpu->available && cpu->type == MATRIX_BACKEND_CPU);
    assert(gpu != nullptr && !gpu->available && gpu->type == MATRIX_BACKEND_GPU);
    assert(matrix_backend_get(MATRIX_BACKEND_CPU) == cpu);
    assert(matrix_backend_get(MATRIX_BACKEND_GPU) == gpu);
    assert(matrix_backend_get((MatrixBackendType)99) == nullptr);
    assert(matrix_backend_default() == cpu);
}

static void test_cpu_f32_matches_kernels(void)
{
    const MatrixBackend *cpu = matrix_backend_cpu();
    const float matrix[] = {1, 2, 3, 4, 5, 6};
    const float inputs[] = {1, 0, -1, 2, 1, 0};
    float expected[4] = {}, actual[4] = {};

    assert(tensor_matvec_f32(matrix, 2, 3, inputs, expected) == 1);
    assert(matrix_backend_matvec_f32(cpu, matrix, 2, 3, inputs, actual) == 1);
    assert(memcmp(expected, actual, 2 * sizeof(float)) == 0);

    memset(expected, 0, sizeof(expected));
    memset(actual, 0, sizeof(actual));
    assert(tensor_matmul_f32(matrix, 2, 3, inputs, 2, expected) == 1);
    assert(matrix_backend_matmul_f32(cpu, matrix, 2, 3, inputs, 2, actual) == 1);
    assert(memcmp(expected, actual, sizeof(expected)) == 0);

    /* The kernel's own argument validation is preserved through the API. */
    assert(matrix_backend_matvec_f32(cpu, nullptr, 2, 3, inputs, actual) == 0);
    assert(matrix_backend_matmul_f32(cpu, matrix, 2, 3, inputs, 0, actual) == 0);
}

static void test_cpu_quantized_matches_kernels(void)
{
    const MatrixBackend *cpu = matrix_backend_cpu();
    Q8_0Block blocks[2] = {};
    GgufQuantizedTensorView view;
    float inputs[2 * Q8_0_BLOCK_ELEMENTS] = {};
    float expected[4] = {}, actual[4] = {};

    blocks[0].scale_f16_bits = blocks[1].scale_f16_bits = 0x3800u;
    blocks[0].quantized_values[0] = 2;
    blocks[0].quantized_values[1] = -4;
    blocks[1].quantized_values[0] = 6;
    blocks[1].quantized_values[1] = 2;
    inputs[0] = 1;
    inputs[1] = 2;
    inputs[Q8_0_BLOCK_ELEMENTS] = -1;
    inputs[Q8_0_BLOCK_ELEMENTS + 1] = 3;
    setup_q8_view(&view, blocks, 2);

    assert(quantized_tensor_matvec_f32(&view, inputs, 32, expected, 2) == 1);
    assert(matrix_backend_matvec_quantized(cpu, &view, inputs, 32, actual, 2) == 1);
    assert(memcmp(expected, actual, 2 * sizeof(float)) == 0);

    memset(expected, 0, sizeof(expected));
    memset(actual, 0, sizeof(actual));
    assert(quantized_tensor_matmul_f32(&view, inputs, 32, 2, expected, 2) == 1);
    assert(matrix_backend_matmul_quantized(cpu, &view, inputs, 32, 2, actual, 2) == 1);
    assert(memcmp(expected, actual, sizeof(expected)) == 0);

    assert(matrix_backend_matvec_quantized(cpu, nullptr, inputs, 32, actual, 2) == 0);
}

static void test_gpu_stub_fails_without_touching_output(void)
{
    const MatrixBackend *gpu = matrix_backend_gpu();
    const float matrix[] = {1, 2, 3, 4};
    const float input[] = {1, 1};
    float output[2] = {-7.0f, -7.0f};
    Q8_0Block block = {};
    GgufQuantizedTensorView view;
    setup_q8_view(&view, &block, 1);
    float packed_input[Q8_0_BLOCK_ELEMENTS] = {};
    float packed_output[1] = {-7.0f};

    assert(gpu->matvec_f32 == nullptr && gpu->matmul_f32 == nullptr);
    assert(gpu->matvec_quantized == nullptr && gpu->matmul_quantized == nullptr);
    assert(matrix_backend_matvec_f32(gpu, matrix, 2, 2, input, output) == 0);
    assert(matrix_backend_matmul_f32(gpu, matrix, 2, 2, input, 1, output) == 0);
    assert(output[0] == -7.0f && output[1] == -7.0f);
    assert(matrix_backend_matvec_quantized(gpu, &view, packed_input, 32,
                                           packed_output, 1) == 0);
    assert(matrix_backend_matmul_quantized(gpu, &view, packed_input, 32, 1,
                                           packed_output, 1) == 0);
    assert(packed_output[0] == -7.0f);
}

static void test_null_backend_is_rejected(void)
{
    const float matrix[] = {1, 2, 3, 4};
    const float input[] = {1, 1};
    float output[2] = {};
    assert(matrix_backend_matvec_f32(nullptr, matrix, 2, 2, input, output) == 0);
    assert(matrix_backend_matmul_f32(nullptr, matrix, 2, 2, input, 1, output) == 0);
    assert(matrix_backend_matvec_quantized(nullptr, nullptr, input, 2, output, 2) == 0);
    assert(matrix_backend_matmul_quantized(nullptr, nullptr, input, 2, 1, output, 2) == 0);
}

int main(void)
{
    printf("Running matrix backend tests...\n");
    test_descriptors();
    test_cpu_f32_matches_kernels();
    test_cpu_quantized_matches_kernels();
    test_gpu_stub_fails_without_touching_output();
    test_null_backend_is_rejected();
    printf("All matrix backend tests passed.\n");
    return 0;
}
