#include <math.h>
#include "tensor.h"

int tensor_add_f32(
    const float *left,
    const float *right,
    int length,
    float *output
) {
    if ((left == nullptr) || (right == nullptr) || (output == nullptr)) {
        return 0;
    }

    if (length <= 0) {
        return 0;
    }

    for (int i = 0; i < length; i++) {
        output[i] = left[i] + right[i];
    }
    return 1;
}

int tensor_dot_f32(
    const float *left,
    const float *right,
    int length,
    float *result
) {
    if ((left == nullptr) || (right == nullptr) || (result == nullptr)) {
        return 0;
    }

    if (length <= 0) {
        return 0;
    }

    float res = 0.0f;
    for (int i = 0; i < length; i ++) {
        res += (left[i] * right[i]);
    }
    *result = res;
    return 1;
}

int tensor_matvec_f32(
    const float *matrix,
    int rows,
    int columns,
    const float *input,
    float *output
) {
    if ((matrix == nullptr) || (input == nullptr) || (output == nullptr)) {
        return 0;
    }

    if ((rows <= 0) || (columns <= 0)) {
        return 0;
    }

    for (int i = 0; i < rows; i ++) {
        float row_sum = 0.0f;
        for (int j = 0; j < columns; j++) {
            row_sum += matrix[i * columns + j] * input[j];
        }
        output[i] = row_sum;
    }
    return 1;
}

int tensor_rms_norm_f32(
    const float *input,
    const float *weight,
    int length,
    float epsilon,
    float *output
) {
    // input:  the current vector of model values, with length elements.
    // weight: one learned multiplier for each input element.
    // output: a caller-provided array where the normalized values belong.
    //
    // First, add up input[i] * input[i] for the whole vector and divide that
    // sum by length. Compute denominator = sqrtf(mean_square + epsilon).
    // Then write (input[i] / denominator) * weight[i] to output[i]. Return 0
    // for null pointers, non-positive length, or non-positive epsilon;
    // otherwise return 1.

    if ((input == nullptr) || (weight == nullptr) || (output == nullptr)) {
        return 0;
    }

    if ((length <= 0) || (epsilon <= 0)) {
        return 0;
    }

    float denom = 0.0f;
    for (int j = 0; j < length; j++) {
        denom += input[j] * input[j];
    }

    denom = sqrtf((denom / length) + epsilon);

    for (int i = 0; i < length; i++) {
        output[i] = (input[i] / denom) * weight[i];
    }
    
    return 1;
}

int tensor_silu_f32(
    const float *input,
    int length,
    float *output
) {
    // input:  a vector of values from a feed-forward projection.
    // output: a caller-provided array with one result per input value.
    //
    // For every index i, calculate input[i] / (1 + exp(-input[i])) and store
    // it in output[i]. Include <math.h> for expf. Return 0 for null pointers
    // or a non-positive length; otherwise return 1.
    if ((input == nullptr) || (output == nullptr) || (length <= 0)) {
        return 0;
    }

    for (int i = 0; i < length; i++) {
        output[i] = input[i] / (1 + expf(-input[i]));
    }
    return 1;
}

int tensor_multiply_f32(
    const float *left,
    const float *right,
    int length,
    float *output
) {
    // left and right: two input vectors with length elements each.
    // output:         a caller-provided array with length result elements.
    //
    // For every index i, multiply left[i] by right[i] and store the result in
    // output[i]. Follow the same pointer and length validation pattern used by
    // tensor_add_f32. Return 1 after writing all results.
    if ((left == nullptr) || (right == nullptr) || (output == nullptr)) {
        return 0;
    }

    if (length <= 0) {
        return 0;
    }

    for (int i = 0; i < length; i++) {
        output[i] = left[i] * right[i];
    }
    return 1;
}

int tensor_softmax_f32(
    const float *input,
    int length,
    float *output
) {
    // input:  a vector of scores. Larger scores should receive larger output
    //         probabilities.
    // output: a caller-provided array that will contain probabilities whose
    //         sum is approximately 1.0f.
    //
    // Use three passes. First find the largest input value. Next, write
    // expf(input[i] - largest_value) to output[i] while adding those values to
    // a total. Last, divide each output[i] by that total. Subtracting the
    // largest value keeps expf from overflowing for large scores. Return 0
    // for null pointers or a non-positive length; otherwise return 1.
    if ((input == nullptr) || (output == nullptr) || (length <= 0)) {
        return 0;
    }

    // find max
    float maxVal = input[0];
    for (int i = 1; i < length; i++) {
        maxVal = fmaxf(maxVal, input[i]);
    }

    float total = 0.0f;
    for (int i = 0; i < length; i++) {
        output[i] = expf(input[i] - maxVal);
        total += output[i];
    }

    for (int i = 0; i < length; i++) {
        output[i] = output[i] / total;
    }
    return 1;
}
