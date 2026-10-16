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
 * Return 0 for null pointers, invalid views, mismatched shapes/counts, or an
 * unsupported type. Otherwise return 1 after writing every output value.
 */
int quantized_tensor_matvec_f32(
    const GgufQuantizedTensorView *matrix,
    const float *input,
    int input_count,
    float *output,
    int output_count
);

#endif
