#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "quantization.h"
#include "quantized_matvec.h"
#include "qwen2_quantized_test_utils.h"

static const float EPSILON = 0.00001f;

static int floats_nearly_equal(float actual, float expected)
{
    return fabsf(actual - expected) <= EPSILON;
}

static void setup_matrix_view(
    GgufQuantizedTensorView *view,
    const void *data,
    uint32_t ggml_type,
    uint64_t columns,
    uint64_t rows,
    uint64_t block_elements,
    uint64_t block_bytes
)
{
    memset(view, 0, sizeof(*view));
    view->data = (const unsigned char *)data;
    view->ggml_type = ggml_type;
    view->dimension_count = 2;
    view->dimensions[0] = columns;
    view->dimensions[1] = rows;
    view->element_count = columns * rows;
    view->block_elements = block_elements;
    view->block_bytes = block_bytes;
    view->block_count = view->element_count / block_elements;
}

static void test_q8_0_two_row_matrix(void)
{
    Q8_0Block blocks[2];
    GgufQuantizedTensorView view;
    float input[Q8_0_BLOCK_ELEMENTS];
    float output[2];

    memset(blocks, 0, sizeof(blocks));
    memset(input, 0, sizeof(input));
    blocks[0].scale_f16_bits = 0x3800u;
    blocks[0].quantized_values[0] = 2;
    blocks[0].quantized_values[1] = -4;
    blocks[1].scale_f16_bits = 0x3800u;
    blocks[1].quantized_values[0] = 6;
    blocks[1].quantized_values[1] = 2;
    input[0] = 1.0f;
    input[1] = 2.0f;

    setup_matrix_view(
        &view,
        blocks,
        ENGINE_GGML_TYPE_Q8_0,
        Q8_0_BLOCK_ELEMENTS,
        2u,
        Q8_0_BLOCK_ELEMENTS,
        Q8_0_BLOCK_BYTES
    );

    assert(quantized_tensor_matvec_f32(&view, input, 32, output, 2) == 1);
    assert(floats_nearly_equal(output[0], -3.0f));
    assert(floats_nearly_equal(output[1], 5.0f));
}

static void test_q5_0_single_row_matrix(void)
{
    Q5_0Block block;
    GgufQuantizedTensorView view;
    float input[Q5_0_BLOCK_ELEMENTS];
    float output[1];

    memset(&block, 0, sizeof(block));
    memset(input, 0, sizeof(input));
    block.scale_f16_bits = 0x3800u;
    input[0] = 1.0f;
    setup_matrix_view(
        &view,
        &block,
        ENGINE_GGML_TYPE_Q5_0,
        Q5_0_BLOCK_ELEMENTS,
        1u,
        Q5_0_BLOCK_ELEMENTS,
        Q5_0_BLOCK_BYTES
    );

    assert(quantized_tensor_matvec_f32(&view, input, 32, output, 1) == 1);
    assert(floats_nearly_equal(output[0], -8.0f));
}

static void test_q4_k_single_row_matrix(void)
{
    Q4_KBlock block;
    GgufQuantizedTensorView view;
    float input[Q4_K_BLOCK_ELEMENTS];
    float output[1];

    memset(&block, 0, sizeof(block));
    memset(input, 0, sizeof(input));
    block.scale_f16_bits = 0x3800u;
    block.min_f16_bits = 0x3400u;
    block.scale_and_min[0] = 1u;
    block.scale_and_min[4] = 1u;
    input[0] = 1.0f;
    setup_matrix_view(
        &view,
        &block,
        ENGINE_GGML_TYPE_Q4_K,
        Q4_K_BLOCK_ELEMENTS,
        1u,
        Q4_K_BLOCK_ELEMENTS,
        Q4_K_BLOCK_BYTES
    );

    assert(quantized_tensor_matvec_f32(&view, input, 256, output, 1) == 1);
    assert(floats_nearly_equal(output[0], -0.25f));
}

static void test_q6_k_single_row_matrix(void)
{
    Q6_KBlock block;
    GgufQuantizedTensorView view;
    float input[Q6_K_BLOCK_ELEMENTS];
    float output[1];

    memset(&block, 0, sizeof(block));
    memset(input, 0, sizeof(input));
    block.scale_f16_bits = 0x3800u;
    block.scales[0] = 1;
    input[0] = 1.0f;
    setup_matrix_view(
        &view,
        &block,
        ENGINE_GGML_TYPE_Q6_K,
        Q6_K_BLOCK_ELEMENTS,
        1u,
        Q6_K_BLOCK_ELEMENTS,
        Q6_K_BLOCK_BYTES
    );

    assert(quantized_tensor_matvec_f32(&view, input, 256, output, 1) == 1);
    assert(floats_nearly_equal(output[0], -16.0f));
}

static void test_multirow_multiblock_matrix(uint32_t type)
{
    const int rows = 17; /* Uneven row partitions with four workers. */
    const int blocks_per_row = 3;
    const int block_elements = fixture_block_elements(type);
    const int block_bytes = fixture_block_bytes(type);
    const int columns = blocks_per_row * block_elements;
    std::vector<unsigned char> storage(1 + rows * blocks_per_row * block_bytes);
    std::vector<float> input(columns), expected(rows, 0.0f);
    std::vector<float> serial(rows + 2, -999.0f), parallel(rows + 2, -999.0f);
    for (int col = 0; col < columns; ++col) input[col] = (float)(col % 13 - 6) / 8.0f;
    for (int row = 0; row < rows; ++row) {
        for (int block = 0; block < blocks_per_row; ++block) {
            int index = row * blocks_per_row + block;
            float weight = fixture_write_block(type, index, 0,
                                                storage.data() + 1 + index * block_bytes);
            for (int col = 0; col < block_elements; ++col) {
                expected[row] += weight * input[block * block_elements + col];
            }
        }
    }
    GgufQuantizedTensorView view;
    setup_matrix_view(&view, storage.data() + 1, type, columns, rows,
                      block_elements, block_bytes);
#ifdef _OPENMP
    int previous_threads = omp_get_max_threads();
    int previous_dynamic = omp_get_dynamic();
    omp_set_dynamic(0);
    omp_set_num_threads(1);
#endif
    assert(quantized_tensor_matvec_f32(&view, input.data(), columns,
                                      serial.data() + 1, rows) == 1);
#ifdef _OPENMP
    omp_set_num_threads(4);
#endif
    assert(quantized_tensor_matvec_f32(&view, input.data(), columns,
                                      parallel.data() + 1, rows) == 1);
    assert(memcmp(serial.data(), parallel.data(), serial.size() * sizeof(float)) == 0);
    assert(serial.front() == -999.0f && serial.back() == -999.0f);
    for (int row = 0; row < rows; ++row) {
        assert(floats_nearly_equal(parallel[row + 1], expected[row]));
    }
#ifdef _OPENMP
    omp_set_num_threads(previous_threads);
    omp_set_dynamic(previous_dynamic);
#endif
}

static void test_invalid_arguments(void)
{
    Q8_0Block block;
    GgufQuantizedTensorView view;
    float input[Q8_0_BLOCK_ELEMENTS];
    float output[1];

    memset(&block, 0, sizeof(block));
    memset(input, 0, sizeof(input));
    setup_matrix_view(
        &view,
        &block,
        ENGINE_GGML_TYPE_Q8_0,
        Q8_0_BLOCK_ELEMENTS,
        1u,
        Q8_0_BLOCK_ELEMENTS,
        Q8_0_BLOCK_BYTES
    );

    assert(quantized_tensor_matvec_f32(nullptr, input, 32, output, 1) == 0);
    assert(quantized_tensor_matvec_f32(&view, nullptr, 32, output, 1) == 0);
    assert(quantized_tensor_matvec_f32(&view, input, 32, nullptr, 1) == 0);
    assert(quantized_tensor_matvec_f32(&view, input, 31, output, 1) == 0);
    assert(quantized_tensor_matvec_f32(&view, input, 32, output, 2) == 0);

    view.dimension_count = 1;
    assert(quantized_tensor_matvec_f32(&view, input, 32, output, 1) == 0);

    view.dimension_count = 2;
    view.ggml_type = 99u;
    assert(quantized_tensor_matvec_f32(&view, input, 32, output, 1) == 0);

    view.ggml_type = ENGINE_GGML_TYPE_Q8_0;
    view.block_count = 0u;
    assert(quantized_tensor_matvec_f32(&view, input, 32, output, 1) == 0);
}

int main(void)
{
    printf("Running quantized matrix-vector reference tests...\n");

    test_q8_0_two_row_matrix();
    test_q5_0_single_row_matrix();
    test_q4_k_single_row_matrix();
    test_q6_k_single_row_matrix();
    test_multirow_multiblock_matrix(ENGINE_GGML_TYPE_Q5_0);
    test_multirow_multiblock_matrix(ENGINE_GGML_TYPE_Q8_0);
    test_multirow_multiblock_matrix(ENGINE_GGML_TYPE_Q4_K);
    test_multirow_multiblock_matrix(ENGINE_GGML_TYPE_Q6_K);
    test_invalid_arguments();

    printf("All quantized matrix-vector reference tests passed.\n");
    return 0;
}
