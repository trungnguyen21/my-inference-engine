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

static void test_weight_constructors(void)
{
    const float data[6] = {};
    MatrixWeight weight;
    Q8_0Block block = {};
    GgufQuantizedTensorView view;
    setup_q8_view(&view, &block, 1);

    assert(matrix_weight_dense(&weight, data, 2, 3) == 1);
    assert(weight.kind == MATRIX_WEIGHT_DENSE_F32 && weight.rows == 2 && weight.columns == 3);
    assert(weight.dense == data && weight.packed == nullptr);
    assert(matrix_weight_dense(&weight, nullptr, 2, 3) == 0 && weight.dense == nullptr);
    assert(matrix_weight_dense(&weight, data, 0, 3) == 0);
    assert(matrix_weight_dense(&weight, data, 2, -1) == 0);
    assert(matrix_weight_dense(nullptr, data, 2, 3) == 0);

    assert(matrix_weight_packed(&weight, &view) == 1);
    assert(weight.kind == MATRIX_WEIGHT_PACKED && weight.packed == &view);
    assert(weight.columns == Q8_0_BLOCK_ELEMENTS && weight.rows == 1 && weight.dense == nullptr);
    assert(matrix_weight_packed(&weight, nullptr) == 0 && weight.packed == nullptr);
    view.dimension_count = 1;
    assert(matrix_weight_packed(&weight, &view) == 0);
    view.dimension_count = 2;
    view.dimensions[1] = 0;
    assert(matrix_weight_packed(&weight, &view) == 0);
    view.dimensions[1] = (uint64_t)1 << 40; /* does not fit in int */
    assert(matrix_weight_packed(&weight, &view) == 0);
    view.dimensions[1] = 1;
    view.data = nullptr;
    assert(matrix_weight_packed(&weight, &view) == 0);
    assert(matrix_weight_packed(nullptr, &view) == 0);
}

static void test_cpu_dense_matches_kernels(void)
{
    const MatrixBackend *cpu = matrix_backend_cpu();
    const float matrix[] = {1, 2, 3, 4, 5, 6};
    const float inputs[] = {1, 0, -1, 2, 1, 0};
    float expected[4] = {}, actual[4] = {};
    MatrixWeight weight;
    assert(matrix_weight_dense(&weight, matrix, 2, 3) == 1);

    assert(matrix_matvec_dense_f32(matrix, 2, 3, inputs, expected) == 1);
    assert(matrix_backend_matvec(cpu, &weight, inputs, actual) == 1);
    assert(memcmp(expected, actual, 2 * sizeof(float)) == 0);

    memset(expected, 0, sizeof(expected));
    memset(actual, 0, sizeof(actual));
    assert(matrix_matmul_dense_f32(matrix, 2, 3, inputs, 2, expected) == 1);
    assert(matrix_backend_matmul(cpu, &weight, inputs, 2, actual) == 1);
    assert(memcmp(expected, actual, sizeof(expected)) == 0);

    assert(matrix_backend_matvec(cpu, &weight, nullptr, actual) == 0);
    assert(matrix_backend_matvec(cpu, &weight, inputs, nullptr) == 0);
    assert(matrix_backend_matmul(cpu, &weight, inputs, 0, actual) == 0);
    assert(matrix_backend_matmul(cpu, nullptr, inputs, 1, actual) == 0);
}

static void test_cpu_packed_matches_kernels(void)
{
    const MatrixBackend *cpu = matrix_backend_cpu();
    Q8_0Block blocks[2] = {};
    GgufQuantizedTensorView view;
    float inputs[2 * Q8_0_BLOCK_ELEMENTS] = {};
    float expected[4] = {}, actual[4] = {};
    MatrixWeight weight;

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
    assert(matrix_weight_packed(&weight, &view) == 1);

    assert(matrix_matvec_packed_f32(&view, inputs, 32, expected, 2) == 1);
    assert(matrix_backend_matvec(cpu, &weight, inputs, actual) == 1);
    assert(memcmp(expected, actual, 2 * sizeof(float)) == 0);

    memset(expected, 0, sizeof(expected));
    memset(actual, 0, sizeof(actual));
    assert(matrix_matmul_packed_f32(&view, inputs, 32, 2, expected, 2) == 1);
    assert(matrix_backend_matmul(cpu, &weight, inputs, 2, actual) == 1);
    assert(memcmp(expected, actual, sizeof(expected)) == 0);

    /* An inconsistent descriptor is rejected before reaching the kernel. */
    weight.rows = 3;
    assert(matrix_backend_matvec(cpu, &weight, inputs, actual) == 0);
}

static void test_overlapping_buffers_are_rejected(void)
{
    const MatrixBackend *cpu = matrix_backend_cpu();
    const float matrix[] = {1, 2, 3, 4};
    float buffer[8] = {1, 1, 0, 0, 0, 0, 0, 0};
    float sentinel[2] = {-7.0f, -7.0f};
    MatrixWeight weight;
    assert(matrix_weight_dense(&weight, matrix, 2, 2) == 1);

    assert(matrix_backend_matvec(cpu, &weight, buffer, buffer) == 0);      /* identical */
    assert(matrix_backend_matvec(cpu, &weight, buffer, buffer + 1) == 0);  /* partial */
    assert(matrix_backend_matmul(cpu, &weight, buffer, 2, buffer + 3) == 0);
    assert(matrix_backend_matvec(cpu, &weight, buffer, buffer + 2) == 1);  /* adjacent is fine */

    /* Output may not alias a dense weight either. */
    float writable[4] = {1, 2, 3, 4};
    float input[2] = {1, 1};
    MatrixWeight aliased;
    assert(matrix_weight_dense(&aliased, writable, 2, 2) == 1);
    assert(matrix_backend_matvec(cpu, &aliased, input, writable) == 0);
    assert(sentinel[0] == -7.0f && writable[0] == 1.0f);
}

static void test_gpu_stub_fails_without_touching_output(void)
{
    const MatrixBackend *gpu = matrix_backend_gpu();
    const float matrix[] = {1, 2, 3, 4};
    const float input[] = {1, 1};
    float output[2] = {-7.0f, -7.0f};
    Q8_0Block block = {};
    GgufQuantizedTensorView view;
    float packed_input[Q8_0_BLOCK_ELEMENTS] = {};
    float packed_output[1] = {-7.0f};
    MatrixWeight dense, packed;
    setup_q8_view(&view, &block, 1);
    assert(matrix_weight_dense(&dense, matrix, 2, 2) == 1);
    assert(matrix_weight_packed(&packed, &view) == 1);

    /* Both operations exist as stubs, but the descriptor stays unavailable. */
    assert(gpu->matvec != nullptr && gpu->matmul != nullptr);
    assert(!gpu->available);

    /* Calling the stubs directly (bypassing the availability guard) also
     * reports failure and leaves the outputs alone. */
    assert(gpu->matvec(&dense, input, output) == 0);
    assert(gpu->matmul(&dense, input, 1, output) == 0);
    assert(gpu->matvec(&packed, packed_input, packed_output) == 0);
    assert(gpu->matmul(&packed, packed_input, 1, packed_output) == 0);
    assert(output[0] == -7.0f && output[1] == -7.0f && packed_output[0] == -7.0f);

    /* The checked wrappers refuse it for both weight kinds. */
    assert(matrix_backend_matvec(gpu, &dense, input, output) == 0);
    assert(matrix_backend_matmul(gpu, &dense, input, 1, output) == 0);
    assert(matrix_backend_matvec(gpu, &packed, packed_input, packed_output) == 0);
    assert(matrix_backend_matmul(gpu, &packed, packed_input, 1, packed_output) == 0);
    assert(output[0] == -7.0f && output[1] == -7.0f && packed_output[0] == -7.0f);
}

static void test_null_backend_is_rejected(void)
{
    const float matrix[] = {1, 2, 3, 4};
    const float input[] = {1, 1};
    float output[2] = {};
    MatrixWeight weight;
    assert(matrix_weight_dense(&weight, matrix, 2, 2) == 1);
    assert(matrix_backend_matvec(nullptr, &weight, input, output) == 0);
    assert(matrix_backend_matmul(nullptr, &weight, input, 1, output) == 0);
}

int main(void)
{
    printf("Running matrix backend tests...\n");
    test_descriptors();
    test_weight_constructors();
    test_cpu_dense_matches_kernels();
    test_cpu_packed_matches_kernels();
    test_overlapping_buffers_are_rejected();
    test_gpu_stub_fails_without_touching_output();
    test_null_backend_is_rejected();
    printf("All matrix backend tests passed.\n");
    return 0;
}
