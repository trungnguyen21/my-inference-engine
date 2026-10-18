/* Inject block-dot failures to verify they propagate across worker rows. Real
 * block-dot math is covered by quantized_matvec_reference_test. */
#include "quantized_matvec.h"
#include "quantization.h"

#include <assert.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

static int block_result(uint16_t marker, float *result)
{
    if (marker == 0xffffu) return 0;
    *result = 1.0f;
    return 1;
}

int q5_0_dot_f32(const Q5_0Block *block, const float *, float *result)
{
    return block_result(block->scale_f16_bits, result);
}
int q8_0_dot_f32(const Q8_0Block *block, const float *, float *result)
{
    return block_result(block->scale_f16_bits, result);
}
int q4_k_dot_f32(const Q4_KBlock *block, const float *, float *result)
{
    return block_result(block->scale_f16_bits, result);
}
int q6_k_dot_f32(const Q6_KBlock *block, const float *, float *result)
{
    return block_result(block->scale_f16_bits, result);
}

static void test_failures()
{
    const struct {
        uint32_t type;
        int elements, bytes;
        size_t scale_offset;
    } formats[] = {
        {ENGINE_GGML_TYPE_Q5_0, Q5_0_BLOCK_ELEMENTS, Q5_0_BLOCK_BYTES,
         offsetof(Q5_0Block, scale_f16_bits)},
        {ENGINE_GGML_TYPE_Q8_0, Q8_0_BLOCK_ELEMENTS, Q8_0_BLOCK_BYTES,
         offsetof(Q8_0Block, scale_f16_bits)},
        {ENGINE_GGML_TYPE_Q4_K, Q4_K_BLOCK_ELEMENTS, Q4_K_BLOCK_BYTES,
         offsetof(Q4_KBlock, scale_f16_bits)},
        {ENGINE_GGML_TYPE_Q6_K, Q6_K_BLOCK_ELEMENTS, Q6_K_BLOCK_BYTES,
         offsetof(Q6_KBlock, scale_f16_bits)}
    };
    const int rows = 17, blocks_per_row = 2;
    const int failed_rows[] = {-1, 0, 7, 16};
    for (const auto &format : formats) {
        int columns = blocks_per_row * format.elements;
        for (int failed_row : failed_rows) {
            std::vector<unsigned char> storage(1 + rows * blocks_per_row * format.bytes);
            std::vector<float> input(columns, 1.0f), output(rows, -99.0f);
            if (failed_row >= 0) {
                /* Fail the second block so a partial sum must not be published. */
                size_t block = (size_t)failed_row * blocks_per_row + 1;
                uint16_t marker = 0xffffu;
                memcpy(storage.data() + 1 + block * format.bytes + format.scale_offset,
                       &marker, sizeof(marker));
            }
            GgufQuantizedTensorView view = {};
            view.data = storage.data() + 1;
            view.ggml_type = format.type;
            view.dimension_count = 2;
            view.dimensions[0] = columns;
            view.dimensions[1] = rows;
            view.element_count = (uint64_t)columns * rows;
            view.block_elements = format.elements;
            view.block_bytes = format.bytes;
            view.block_count = rows * blocks_per_row;
            assert(quantized_tensor_matvec_f32(&view, input.data(), columns,
                                              output.data(), rows) == (failed_row < 0));
            for (int row = 0; row < rows; ++row) {
                assert(output[row] == (row == failed_row ? -99.0f : 2.0f));
            }
        }
    }
}

int main()
{
#ifdef _OPENMP
    omp_set_dynamic(0);
    omp_set_num_threads(1);
#endif
    test_failures();
#ifdef _OPENMP
    omp_set_num_threads(4);
    test_failures();
#endif
    puts("Packed matvec worker failures propagate with one and four threads.");
    return 0;
}
