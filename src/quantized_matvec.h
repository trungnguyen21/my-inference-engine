#ifndef QUANTIZED_MATVEC_H
#define QUANTIZED_MATVEC_H

#include "gguf_reader.h"

/*
 * Multiply a packed two-dimensional GGUF tensor by an F32 input vector.
 *
 * matrix is a valid GgufQuantizedTensorView whose GGUF dimensions follow the
 * GGML convention: dimensions[0] is the number of columns in each row and
 * dimensions[1] is the number of rows. input has input_count values and
 * output has output_count values. Require input_count == dimensions[0] and
 * output_count == dimensions[1]. Both dimensions must fit in int.
 * Output must not overlap the input vector or packed matrix storage.
 *
 * Accept only Q5_0, Q8_0, Q4_K, and Q6_K views. The column count must divide
 * evenly by matrix->block_elements. Verify the view's element_count,
 * block_count, and block layout agree with its shape and type before reading.
 * For every row, walk its contiguous packed blocks, call the matching direct
 * block-dot function, and accumulate their results into output[row].
 *
 * matrix->data is a byte pointer, so copy each raw block into a local correctly
 * aligned Q5_0Block, Q8_0Block, Q4_KBlock, or Q6_KBlock with memcpy before
 * calling its dot function. Do not allocate memory or call a block decoder.
 * With OpenMP enabled, distribute independent rows across workers; each row's
 * accumulation order stays unchanged. Combine worker failures after all rows
 * finish. Output contents are unspecified on failure.
 * Return 0 for null pointers, invalid views, mismatched shapes/counts, or an
 * unsupported type or failed block dot. Otherwise return 1 after writing every
 * output value.
 */
int quantized_tensor_matvec_f32(
    const GgufQuantizedTensorView *matrix,
    const float *input,
    int input_count,
    float *output,
    int output_count
);

/* Multiply one packed matrix by token-major F32 vectors. Load each packed
 * weight block into aligned local storage and apply it across the batch. inputs is
 * [batch_count,input_count], outputs is [batch_count,output_count]. */
int quantized_tensor_matmul_f32(
    const GgufQuantizedTensorView *matrix,
    const float *inputs,
    int input_count,
    int batch_count,
    float *outputs,
    int output_count
);

#endif
