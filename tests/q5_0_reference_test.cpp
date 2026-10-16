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

static void make_reference_block(Q5_0Block *block)
{
    memset(block, 0, sizeof(*block));

    /* The scale is F16 0.5. All untouched values decode to -8.0. */
    block->scale_f16_bits = 0x3800u;

    /*
     * Each packed byte has values for positions i (low nibble) and i + 16
     * (high nibble). The four high_bits bytes exercise both bit 0 and bit 7
     * of every byte, which catches mistakes in the fifth-bit mapping.
     */
    block->packed_nibbles[0] = 0x82u;
    block->packed_nibbles[8] = 0xf1u;
    block->packed_nibbles[15] = 0x70u;
    block->high_bits[0] = 0x01u;
    block->high_bits[1] = 0x81u;
    block->high_bits[3] = 0x81u;
}

static void test_q5_0_dequantize_block(void)
{
    Q5_0Block block;
    float output[Q5_0_BLOCK_ELEMENTS];
    float expected[Q5_0_BLOCK_ELEMENTS];

    make_reference_block(&block);
    for (int i = 0; i < Q5_0_BLOCK_ELEMENTS; ++i) {
        expected[i] = -8.0f;
    }

    expected[0] = 1.0f;
    expected[8] = 0.5f;
    expected[15] = 0.0f;
    expected[16] = -4.0f;
    expected[24] = 7.5f;
    expected[31] = 3.5f;

    assert(q5_0_dequantize_block(&block, output) == 1);
    assert(arrays_nearly_equal(output, expected, Q5_0_BLOCK_ELEMENTS));
}

static void test_q5_0_dot_f32(void)
{
    Q5_0Block block;
    float input[Q5_0_BLOCK_ELEMENTS];
    float result = 99.0f;

    make_reference_block(&block);
    memset(input, 0, sizeof(input));
    input[0] = 2.0f;
    input[16] = 3.0f;
    input[24] = -2.0f;

    /* (1.0 * 2.0) + (-4.0 * 3.0) + (7.5 * -2.0) = -25.0 */
    assert(q5_0_dot_f32(&block, input, &result) == 1);
    assert(floats_nearly_equal(result, -25.0f));
}

static void test_invalid_arguments(void)
{
    Q5_0Block block;
    float input[Q5_0_BLOCK_ELEMENTS];
    float output[Q5_0_BLOCK_ELEMENTS];
    float result = 0.0f;

    memset(&block, 0, sizeof(block));
    memset(input, 0, sizeof(input));

    assert(q5_0_dequantize_block(nullptr, output) == 0);
    assert(q5_0_dequantize_block(&block, nullptr) == 0);
    assert(q5_0_dot_f32(nullptr, input, &result) == 0);
    assert(q5_0_dot_f32(&block, nullptr, &result) == 0);
    assert(q5_0_dot_f32(&block, input, nullptr) == 0);
}

int main(void)
{
    printf("Running Q5_0 reference tests...\n");

    test_q5_0_dequantize_block();
    test_q5_0_dot_f32();
    test_invalid_arguments();

    printf("All Q5_0 reference tests passed.\n");
    return 0;
}
