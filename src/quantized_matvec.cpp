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
#pragma omp parallel for schedule(dynamic, 16) reduction(|:failed)
#endif
    for (int row = 0; row < output_count; ++row) {
        if (!quantized_row_dot_f32(matrix, input, (uint64_t)row,
                                   blocks_per_row, &output[row])) {
            failed = 1;
        }
    }
    return !failed;
}

int quantized_tensor_matmul_f32(
    const GgufQuantizedTensorView *matrix, const float *inputs,
    int input_count, int batch_count, float *outputs, int output_count)
{
    if (matrix == nullptr || matrix->data == nullptr || inputs == nullptr ||
        outputs == nullptr || input_count <= 0 || output_count <= 0 || batch_count <= 0 ||
        matrix->dimension_count != 2) return 0;
    uint64_t columns = matrix->dimensions[0], rows = matrix->dimensions[1];
    if (columns == 0 || rows == 0 || columns > INT_MAX || rows > INT_MAX ||
        input_count != (int)columns || output_count != (int)rows ||
        (size_t)batch_count > SIZE_MAX / (size_t)input_count ||
        (size_t)batch_count > SIZE_MAX / (size_t)output_count) return 0;

    uint64_t block_elements = 0, block_bytes = 0;
    switch (matrix->ggml_type) {
        case ENGINE_GGML_TYPE_Q5_0: block_elements = Q5_0_BLOCK_ELEMENTS; block_bytes = Q5_0_BLOCK_BYTES; break;
        case ENGINE_GGML_TYPE_Q8_0: block_elements = Q8_0_BLOCK_ELEMENTS; block_bytes = Q8_0_BLOCK_BYTES; break;
        case ENGINE_GGML_TYPE_Q4_K: block_elements = Q4_K_BLOCK_ELEMENTS; block_bytes = Q4_K_BLOCK_BYTES; break;
        case ENGINE_GGML_TYPE_Q6_K: block_elements = Q6_K_BLOCK_ELEMENTS; block_bytes = Q6_K_BLOCK_BYTES; break;
        default: return 0;
    }
    if (matrix->block_elements != block_elements || matrix->block_bytes != block_bytes ||
        columns % block_elements != 0) return 0;
    uint64_t blocks_per_row = columns / block_elements;
    if (blocks_per_row > UINT64_MAX / rows ||
        matrix->block_count != blocks_per_row * rows ||
        columns > UINT64_MAX / rows || matrix->element_count != columns * rows ||
        matrix->block_count > SIZE_MAX / block_bytes) return 0;

    int failed = 0;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 16) reduction(|:failed)
#endif
    for (int row = 0; row < output_count; ++row) {
        for (int batch = 0; batch < batch_count; ++batch)
            outputs[(size_t)batch * output_count + row] = 0.0f;
        for (uint64_t block = 0; block < blocks_per_row; ++block) {
            uint64_t packed_index = (uint64_t)row * blocks_per_row + block;
            const unsigned char *address = matrix->data + packed_index * block_bytes;
#define ACCUMULATE_BATCH(BlockType, dot_fn) do { \
                BlockType local; memcpy(&local, address, sizeof(local)); \
                for (int batch = 0; batch < batch_count; ++batch) { \
                    const float *input = inputs + (size_t)batch * input_count + block * block_elements; \
                    float partial = 0.0f; \
                    if (!dot_fn(&local, input, &partial)) { failed = 1; break; } \
                    outputs[(size_t)batch * output_count + row] += partial; \
                } \
            } while (0)
            switch (matrix->ggml_type) {
                case ENGINE_GGML_TYPE_Q5_0: ACCUMULATE_BATCH(Q5_0Block, q5_0_dot_f32); break;
                case ENGINE_GGML_TYPE_Q8_0: ACCUMULATE_BATCH(Q8_0Block, q8_0_dot_f32); break;
                case ENGINE_GGML_TYPE_Q4_K: ACCUMULATE_BATCH(Q4_KBlock, q4_k_dot_f32); break;
                case ENGINE_GGML_TYPE_Q6_K: ACCUMULATE_BATCH(Q6_KBlock, q6_k_dot_f32); break;
                default: failed = 1; break;
            }
#undef ACCUMULATE_BATCH
            if (failed) break;
        }
    }
    return !failed;
}

#include <stdlib.h>

/* Validate the view's layout the same way the f32 paths do and report the
 * block geometry. Returns 0 on any mismatch. */
static int quantized_view_geometry(
    const GgufQuantizedTensorView *matrix,
    int input_count, int output_count,
    uint64_t *blocks_per_row_out
)
{
    if (matrix == nullptr || matrix->data == nullptr ||
        matrix->dimension_count != 2) return 0;
    uint64_t columns = matrix->dimensions[0], rows = matrix->dimensions[1];
    if (columns == 0 || rows == 0 || columns > INT_MAX || rows > INT_MAX ||
        input_count != (int)columns || output_count != (int)rows) return 0;

    uint64_t block_elements = 0, block_bytes = 0;
    switch (matrix->ggml_type) {
        case ENGINE_GGML_TYPE_Q5_0: block_elements = Q5_0_BLOCK_ELEMENTS; block_bytes = Q5_0_BLOCK_BYTES; break;
        case ENGINE_GGML_TYPE_Q8_0: block_elements = Q8_0_BLOCK_ELEMENTS; block_bytes = Q8_0_BLOCK_BYTES; break;
        case ENGINE_GGML_TYPE_Q4_K: block_elements = Q4_K_BLOCK_ELEMENTS; block_bytes = Q4_K_BLOCK_BYTES; break;
        case ENGINE_GGML_TYPE_Q6_K: block_elements = Q6_K_BLOCK_ELEMENTS; block_bytes = Q6_K_BLOCK_BYTES; break;
        default: return 0;
    }
    if (matrix->block_elements != block_elements ||
        matrix->block_bytes != block_bytes ||
        columns % block_elements != 0) return 0;
    uint64_t blocks_per_row = columns / block_elements;
    if (blocks_per_row > UINT64_MAX / rows ||
        matrix->block_count != blocks_per_row * rows ||
        columns > UINT64_MAX / rows || matrix->element_count != columns * rows ||
        matrix->block_count > SIZE_MAX / block_bytes) return 0;
    *blocks_per_row_out = blocks_per_row;
    return 1;
}

/* One row's integer dot against pre-quantized activations. */
static int quantized_row_dot_act_q8(
    const GgufQuantizedTensorView *matrix,
    const int8_t *values, const float *scales, const int32_t *sums,
    uint64_t row, uint64_t blocks_per_row, float *result
)
{
    const unsigned char *row_data =
        matrix->data + row * blocks_per_row * matrix->block_bytes;

    switch (matrix->ggml_type) {
        case ENGINE_GGML_TYPE_Q5_0:
            return q5_0_row_dot_q8(row_data, blocks_per_row, values, scales,
                                   result);
        case ENGINE_GGML_TYPE_Q8_0:
            return q8_0_row_dot_q8(row_data, blocks_per_row, values, scales,
                                   result);
        case ENGINE_GGML_TYPE_Q4_K:
            return q4_k_row_dot_q8(row_data, blocks_per_row, values, scales,
                                   sums, result);
        case ENGINE_GGML_TYPE_Q6_K:
            return q6_k_row_dot_q8(row_data, blocks_per_row, values, scales,
                                   result);
        default:
            return 0;
    }
}

int quantized_tensor_matvec_act_q8(
    const GgufQuantizedTensorView *matrix,
    const float *input,
    int input_count,
    float *output,
    int output_count
)
{
    if (input == nullptr || output == nullptr ||
        input_count <= 0 || output_count <= 0) return 0;
    uint64_t blocks_per_row = 0;
    if (!quantized_view_geometry(matrix, input_count, output_count,
                                 &blocks_per_row)) return 0;

    const int groups = input_count / ACTIVATION_Q8_GROUP;
    int8_t *values = (int8_t *)malloc((size_t)input_count);
    float *scales = (float *)malloc((size_t)groups * sizeof(float));
    int32_t *sums = (int32_t *)malloc((size_t)groups * sizeof(int32_t));
    if (values == nullptr || scales == nullptr || sums == nullptr ||
        !quantize_activations_q8(input, input_count, values, scales, sums)) {
        free(values); free(scales); free(sums);
        return quantized_tensor_matvec_f32(matrix, input, input_count,
                                           output, output_count);
    }

    int failed = 0;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 16) reduction(|:failed)
#endif
    for (int row = 0; row < output_count; ++row) {
        if (!quantized_row_dot_act_q8(matrix, values, scales, sums,
                                      (uint64_t)row, blocks_per_row,
                                      &output[row])) {
            failed = 1;
        }
    }
    free(values); free(scales); free(sums);
    return !failed;
}

int quantized_tensor_matmul_act_q8(
    const GgufQuantizedTensorView *matrix,
    const float *inputs,
    int input_count,
    int batch_count,
    float *outputs,
    int output_count
)
{
    if (inputs == nullptr || outputs == nullptr || batch_count <= 0 ||
        input_count <= 0 || output_count <= 0 ||
        (size_t)batch_count > SIZE_MAX / (size_t)input_count ||
        (size_t)batch_count > SIZE_MAX / (size_t)output_count) return 0;
    uint64_t blocks_per_row = 0;
    if (!quantized_view_geometry(matrix, input_count, output_count,
                                 &blocks_per_row)) return 0;

    const int groups = input_count / ACTIVATION_Q8_GROUP;
    int8_t *values = (int8_t *)malloc((size_t)batch_count * input_count);
    float *scales = (float *)malloc((size_t)batch_count * groups * sizeof(float));
    int32_t *sums = (int32_t *)malloc((size_t)batch_count * groups * sizeof(int32_t));
    int quantize_ok = values != nullptr && scales != nullptr && sums != nullptr;
    for (int batch = 0; quantize_ok && batch < batch_count; ++batch) {
        quantize_ok = quantize_activations_q8(
            inputs + (size_t)batch * input_count, input_count,
            values + (size_t)batch * input_count,
            scales + (size_t)batch * groups,
            sums + (size_t)batch * groups);
    }
    if (!quantize_ok) {
        free(values); free(scales); free(sums);
        return quantized_tensor_matmul_f32(matrix, inputs, input_count,
                                           batch_count, outputs, output_count);
    }

    int failed = 0;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 16) reduction(|:failed)
#endif
    for (int row = 0; row < output_count; ++row) {
        for (int batch = 0; batch < batch_count; ++batch) {
            if (!quantized_row_dot_act_q8(
                    matrix,
                    values + (size_t)batch * input_count,
                    scales + (size_t)batch * groups,
                    sums + (size_t)batch * groups,
                    (uint64_t)row, blocks_per_row,
                    &outputs[(size_t)batch * output_count + row])) {
                failed = 1;
            }
        }
    }
    free(values); free(scales); free(sums);
    return !failed;
}
