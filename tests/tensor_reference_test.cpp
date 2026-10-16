#include <assert.h>
#include <math.h>
#include <stdio.h>

#include "tensor.h"

static const float EPSILON = 0.00001f;

static int floats_nearly_equal(float actual, float expected)
{
    return fabsf(actual - expected) <= EPSILON;
}

static int arrays_nearly_equal(
    const float *actual,
    const float *expected,
    int length
)
{
    for (int i = 0; i < length; ++i) {
        if (!floats_nearly_equal(actual[i], expected[i])) {
            return 0;
        }
    }

    return 1;
}

static void test_dot_product_integers(void)
{
    const float left[] = {1.0f, 2.0f, 3.0f};
    const float right[] = {4.0f, 5.0f, 6.0f};
    float result = 0.0f;

    assert(tensor_dot_f32(left, right, 3, &result) == 1);
    assert(floats_nearly_equal(result, 32.0f));
}

static void test_dot_product_decimals(void)
{
    const float left[] = {0.5f, 1.5f};
    const float right[] = {2.0f, 4.0f};
    float result = 0.0f;

    assert(tensor_dot_f32(left, right, 2, &result) == 1);
    assert(floats_nearly_equal(result, 7.0f));
}

static void test_vector_add(void)
{
    const float left[] = {0.75f, 0.85f, 1.0f};
    const float right[] = {0.0f, -3.5f, 5.0f};
    const float expected[] = {0.75f, -2.65f, 6.0f};
    float output[] = {0.0f, 0.0f, 0.0f};

    assert(tensor_add_f32(left, right, 3, output) == 1);
    assert(arrays_nearly_equal(output, expected, 3));
}

static void test_matrix_vector_product(void)
{
    const float matrix[] = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f
    };
    const float input[] = {1.0f, 0.0f, -1.0f};
    const float expected[] = {-2.0f, -2.0f};
    float output[] = {0.0f, 0.0f};

    assert(tensor_matvec_f32(matrix, 2, 3, input, output) == 1);
    assert(arrays_nearly_equal(output, expected, 2));
}

static void test_rms_norm(void)
{
    const float input[] = {1.0f, 1.0f};
    const float weight[] = {2.0f, 3.0f};
    const float expected[] = {1.0f, 1.5f};
    float output[] = {0.0f, 0.0f};

    assert(tensor_rms_norm_f32(input, weight, 2, 3.0f, output) == 1);
    assert(arrays_nearly_equal(output, expected, 2));
}

static void test_silu(void)
{
    const float input[] = {0.0f, 1.0f, -1.0f};
    const float expected[] = {0.0f, 0.7310586f, -0.2689414f};
    float output[] = {0.0f, 0.0f, 0.0f};

    assert(tensor_silu_f32(input, 3, output) == 1);
    assert(arrays_nearly_equal(output, expected, 3));
}

static void test_vector_multiply(void)
{
    const float left[] = {2.0f, -3.0f, 0.5f};
    const float right[] = {4.0f, 2.0f, -6.0f};
    const float expected[] = {8.0f, -6.0f, -3.0f};
    float output[] = {0.0f, 0.0f, 0.0f};

    assert(tensor_multiply_f32(left, right, 3, output) == 1);
    assert(arrays_nearly_equal(output, expected, 3));
}

static void test_softmax(void)
{
    const float positive_scores[] = {1000.0f, 1001.0f};
    const float negative_scores[] = {-1000.0f, -999.0f};
    const float zero_scores[] = {0.0f, 0.0f};
    const float expected[] = {0.2689414f, 0.7310586f};
    const float zero_expected[] = {0.5f, 0.5f};
    float output[] = {0.0f, 0.0f};

    assert(tensor_softmax_f32(positive_scores, 2, output) == 1);
    assert(arrays_nearly_equal(output, expected, 2));
    assert(floats_nearly_equal(output[0] + output[1], 1.0f));

    assert(tensor_softmax_f32(negative_scores, 2, output) == 1);
    assert(arrays_nearly_equal(output, expected, 2));
    assert(floats_nearly_equal(output[0] + output[1], 1.0f));

    assert(tensor_softmax_f32(zero_scores, 2, output) == 1);
    assert(arrays_nearly_equal(output, zero_expected, 2));
    assert(floats_nearly_equal(output[0] + output[1], 1.0f));
}

static void test_invalid_arguments(void)
{
    const float values[] = {1.0f, 2.0f, 3.0f};
    const float matrix[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float output[] = {0.0f, 0.0f, 0.0f};
    float result = 0.0f;

    assert(tensor_add_f32(nullptr, values, 3, output) == 0);
    assert(tensor_add_f32(values, values, 0, output) == 0);
    assert(tensor_dot_f32(values, values, 3, nullptr) == 0);
    assert(tensor_dot_f32(values, values, -1, &result) == 0);
    assert(tensor_matvec_f32(nullptr, 2, 2, values, output) == 0);
    assert(tensor_matvec_f32(matrix, 0, 2, values, output) == 0);
    assert(tensor_matvec_f32(matrix, 2, -1, values, output) == 0);

    assert(tensor_rms_norm_f32(nullptr, values, 3, 0.00001f, output) == 0);
    assert(tensor_rms_norm_f32(values, nullptr, 3, 0.00001f, output) == 0);
    assert(tensor_rms_norm_f32(values, values, 3, 0.0f, output) == 0);
    assert(tensor_silu_f32(nullptr, 3, output) == 0);
    assert(tensor_silu_f32(values, 0, output) == 0);
    assert(tensor_multiply_f32(nullptr, values, 3, output) == 0);
    assert(tensor_multiply_f32(values, values, 0, output) == 0);
    assert(tensor_softmax_f32(nullptr, 3, output) == 0);
    assert(tensor_softmax_f32(values, 0, output) == 0);
}

int main(void)
{
    printf("Running tensor reference tests...\n");

    test_dot_product_integers();
    test_dot_product_decimals();
    test_vector_add();
    test_matrix_vector_product();
    test_rms_norm();
    test_silu();
    test_vector_multiply();
    test_softmax();
    test_invalid_arguments();

    printf("All tensor tests passed.\n");
    return 0;
}
