#include "quantized_matvec.h"
#include "quantization.h"

#include <limits.h>
#include <stdint.h>
#include <string.h>

/* One worker owns this row's accumulator, block copies, and output. Keeping
 * early returns in a helper lets a block-dot failure leave the worker normally. */
static int quantized_row_dot_f32(
    const GgufQuantizedTensorView *matrix,
    const float *input,
    uint64_t row,
    uint64_t blocks_per_row,
    float *result
)
{
    float row_sum = 0.0f;

    for (uint64_t block_index = 0; block_index < blocks_per_row; block_index++) {
        uint64_t tensor_block_index = row * blocks_per_row + block_index;
        const unsigned char *block_address = matrix->data + (tensor_block_index * matrix->block_bytes);

        switch (matrix->ggml_type) {
            case ENGINE_GGML_TYPE_Q5_0: {
                Q5_0Block block;
                float block_res = 0.0f;
                const float *input_slice = input + matrix->block_elements * block_index;

                memcpy(&block, block_address, sizeof(block));
                int fres = q5_0_dot_f32(&block, input_slice, &block_res);
                if (fres == 0) {
                    return 0;
                }
                row_sum += block_res;
                break;
            }
            case ENGINE_GGML_TYPE_Q8_0: {
                Q8_0Block block;
                float block_res = 0.0f;
                const float *input_slice = input + matrix->block_elements * block_index;

                memcpy(&block, block_address, sizeof(block));
                int fres = q8_0_dot_f32(&block, input_slice, &block_res);
                if (fres == 0) {
                    return 0;
                }
                row_sum += block_res;
                break;
            }
            case ENGINE_GGML_TYPE_Q4_K: {
                Q4_KBlock block;
                float block_res = 0.0f;
                const float *input_slice = input + matrix->block_elements * block_index;

                memcpy(&block, block_address, sizeof(block));
                int fres = q4_k_dot_f32(&block, input_slice, &block_res);
                if (fres == 0) {
                    return 0;
                }
                row_sum += block_res;
                break;
            }
            case ENGINE_GGML_TYPE_Q6_K: {
                Q6_KBlock block;
                float block_res = 0.0f;
                const float *input_slice = input + matrix->block_elements * block_index;

                memcpy(&block, block_address, sizeof(block));
                int fres = q6_k_dot_f32(&block, input_slice, &block_res);
                if (fres == 0) {
                    return 0;
                }
                row_sum += block_res;
                break;
            }
            default:
                return 0;
        }
    }

    *result = row_sum;
    return 1;
}

int quantized_tensor_matvec_f32(
    const GgufQuantizedTensorView *matrix,
    const float *input,
    int input_count,
    float *output,
    int output_count
)
{
    if ((matrix == nullptr) || (matrix->data == nullptr) || (input == nullptr) ||
        (output == nullptr) || (input_count <= 0) || (output_count <= 0)) {
        return 0;
    }

    if (matrix->dimension_count != 2) {
        return 0;
    }

    uint64_t ncols = matrix->dimensions[0];
    uint64_t nrows = matrix->dimensions[1];

    if ((ncols == 0) || (nrows == 0) || (ncols > INT_MAX) || (nrows > INT_MAX) ||
        (input_count != (int)ncols) || (output_count != (int)nrows)) {
        return 0;
    }

    uint64_t expected_block_elements;
    uint64_t expected_block_bytes;

    /* Match the view's layout fields against the selected GGML format. */
    switch (matrix->ggml_type) {
        case ENGINE_GGML_TYPE_Q5_0:
            expected_block_elements = Q5_0_BLOCK_ELEMENTS;
            expected_block_bytes = Q5_0_BLOCK_BYTES;
            break;
        case ENGINE_GGML_TYPE_Q8_0:
            expected_block_elements = Q8_0_BLOCK_ELEMENTS;
            expected_block_bytes = Q8_0_BLOCK_BYTES;
            break;
        case ENGINE_GGML_TYPE_Q4_K:
            expected_block_elements = Q4_K_BLOCK_ELEMENTS;
            expected_block_bytes = Q4_K_BLOCK_BYTES;
            break;
        case ENGINE_GGML_TYPE_Q6_K:
            expected_block_elements = Q6_K_BLOCK_ELEMENTS;
            expected_block_bytes = Q6_K_BLOCK_BYTES;
            break;
        default:
            return 0;
    }

    if ((matrix->block_elements != expected_block_elements) ||
        (matrix->block_bytes != expected_block_bytes) ||
        (ncols % expected_block_elements != 0)) {
        return 0;
    }

    const uint64_t blocks_per_row = ncols / expected_block_elements;
    if ((blocks_per_row > UINT64_MAX / nrows) ||
        (matrix->block_count != blocks_per_row * nrows) ||
        (ncols > UINT64_MAX / nrows) ||
        (matrix->element_count != ncols * nrows) ||
        (matrix->block_count > SIZE_MAX / expected_block_bytes)) {
        return 0;
    }

    int failed = 0;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) reduction(|:failed)
#endif
    for (int row = 0; row < output_count; ++row) {
        if (!quantized_row_dot_f32(matrix, input, (uint64_t)row,
                                   blocks_per_row, &output[row])) {
            failed = 1;
        }
    }
    return !failed;
}
