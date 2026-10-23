#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "quantization.h"

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

static void test_f16_bits_to_f32(void)
{
    float result = 0.0f;

    assert(f16_bits_to_f32(0x0000u, &result) == 1);
    assert(result == 0.0f);

    assert(f16_bits_to_f32(0x3c00u, &result) == 1);
    assert(result == 1.0f);

    assert(f16_bits_to_f32(0xc000u, &result) == 1);
    assert(result == -2.0f);

    assert(f16_bits_to_f32(0x3800u, &result) == 1);
    assert(result == 0.5f);

    assert(f16_bits_to_f32(0x0001u, &result) == 1);
    assert(fabsf(result - 0.000000059604645f) <= 0.000000000001f);

    /* All 65,536 encodings: compare finite values bit-for-bit to an
     * independent arithmetic reference, including subnormals and -0. */
    for (unsigned raw = 0; raw <= 0xffffu; ++raw) {
        unsigned exponent = (raw >> 10) & 0x1fu;
        unsigned fraction = raw & 0x3ffu;
        assert(f16_bits_to_f32((uint16_t)raw, &result) == 1);
        if (exponent == 31) {
            if (fraction) assert(isnan(result));
            else assert(isinf(result) && (!!signbit(result) == !!(raw & 0x8000u)));
            continue;
        }
        float expected = exponent == 0
            ? ldexpf((float)fraction, -24)
            : ldexpf((float)(1024u + fraction), (int)exponent - 25);
        if (raw & 0x8000u) expected = -expected;
        assert(memcmp(&result, &expected, sizeof(float)) == 0);
    }
}

static void test_q8_0_dequantize_block(void)
{
    Q8_0Block block;
    float output[Q8_0_BLOCK_ELEMENTS];
    float expected[Q8_0_BLOCK_ELEMENTS];

    memset(&block, 0, sizeof(block));
    memset(expected, 0, sizeof(expected));
    block.scale_f16_bits = 0x3800u;
    block.quantized_values[0] = -4;
    block.quantized_values[1] = 0;
    block.quantized_values[2] = 6;
    expected[0] = -2.0f;
    expected[1] = 0.0f;
    expected[2] = 3.0f;

    assert(q8_0_dequantize_block(&block, output) == 1);
    assert(arrays_nearly_equal(output, expected, Q8_0_BLOCK_ELEMENTS));
}

static void test_q8_0_dot_f32(void)
{
    Q8_0Block block;
    float input[Q8_0_BLOCK_ELEMENTS];
    float result = 0.0f;

    memset(&block, 0, sizeof(block));
    memset(input, 0, sizeof(input));
    block.scale_f16_bits = 0x3800u;
    block.quantized_values[0] = 2;
    block.quantized_values[1] = -4;
    block.quantized_values[2] = 6;
    input[0] = 1.0f;
    input[1] = 2.0f;
    input[2] = -1.0f;

    assert(q8_0_dot_f32(&block, input, &result) == 1);
    assert(floats_nearly_equal(result, -6.0f));
}

static void test_invalid_arguments(void)
{
    Q8_0Block block;
    float input[Q8_0_BLOCK_ELEMENTS];
    float output[Q8_0_BLOCK_ELEMENTS];
    float result = 0.0f;

    memset(&block, 0, sizeof(block));
    memset(input, 0, sizeof(input));

    assert(f16_bits_to_f32(0x3c00u, nullptr) == 0);
    assert(q8_0_dequantize_block(nullptr, output) == 0);
    assert(q8_0_dequantize_block(&block, nullptr) == 0);
    assert(q8_0_dot_f32(nullptr, input, &result) == 0);
    assert(q8_0_dot_f32(&block, nullptr, &result) == 0);
    assert(q8_0_dot_f32(&block, input, nullptr) == 0);
}

int main(void)
{
    printf("Running Q8_0 reference tests...\n");

    test_f16_bits_to_f32();
    test_q8_0_dequantize_block();
    test_q8_0_dot_f32();
    test_invalid_arguments();

    printf("All Q8_0 reference tests passed.\n");
    return 0;
}
