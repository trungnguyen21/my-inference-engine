#ifndef TENSOR_H
#define TENSOR_H

// Caller always provide a separate output buffer

// Add two vectors element by element.
// Returns 1 on success and 0 for invalid arguments.
int tensor_add_f32(
    const float *left,
    const float *right,
    int length,
    float *output
);

// Compute the dot product of two vectors.
// Returns 1 on success and 0 for invalid arguments.
int tensor_dot_f32(
    const float *left,
    const float *right,
    int length,
    float *result
);

// Multiply a row-major matrix by a vector.
// matrix has rows * columns elements; input has columns elements;
// output must have room for rows elements.
// Returns 1 on success and 0 for invalid arguments.
int tensor_matvec_f32(
    const float *matrix,
    int rows,
    int columns,
    const float *input,
    float *output
);

// Multiply one row-major matrix by a token-major batch of input vectors.
// inputs is [batch_count, columns], outputs is [batch_count, rows].
int tensor_matmul_f32(
    const float *matrix, int rows, int columns,
    const float *inputs, int batch_count, float *outputs
);

// Apply RMS normalization followed by one learned weight per element.
// epsilon must be positive. Returns 1 on success and 0 for invalid arguments.
int tensor_rms_norm_f32(
    const float *input,
    const float *weight,
    int length,
    float epsilon,
    float *output
);

// Apply the SiLU activation to every input element.
// Returns 1 on success and 0 for invalid arguments.
int tensor_silu_f32(
    const float *input,
    int length,
    float *output
);

// Multiply two vectors element by element.
// Returns 1 on success and 0 for invalid arguments.
int tensor_multiply_f32(
    const float *left,
    const float *right,
    int length,
    float *output
);

// Convert a vector of scores to probabilities using numerically stable softmax.
// Returns 1 on success and 0 for invalid arguments.
int tensor_softmax_f32(
    const float *input,
    int length,
    float *output
);

#endif
