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

static void make_reference_block(Q4_KBlock *block)
{
    memset(block, 0, sizeof(*block));

    /* F16 0.5 and 0.25 are the shared scale and minimum factors. */
    block->scale_f16_bits = 0x3800u;
    block->min_f16_bits = 0x3400u;

    /*
     * The unpacked scale multipliers are {1, 2, 3, 4, 17, 18, 19, 20}.
     * The unpacked minimum multipliers are {5, 6, 7, 8, 33, 34, 35, 36}.
     * This uses low and high portions of every packed six-bit field.
     */
    block->scale_and_min[0] = 0x41u;
    block->scale_and_min[1] = 0x42u;
    block->scale_and_min[2] = 0x43u;
    block->scale_and_min[3] = 0x44u;
    block->scale_and_min[4] = 0x85u;
    block->scale_and_min[5] = 0x86u;
    block->scale_and_min[6] = 0x87u;
    block->scale_and_min[7] = 0x88u;
    block->scale_and_min[8] = 0x11u;
    block->scale_and_min[9] = 0x22u;
    block->scale_and_min[10] = 0x33u;
    block->scale_and_min[11] = 0x44u;

    /* Each byte drives positions g*64 and g*64 + 32 for one offset. */
    block->packed_nibbles[0] = 0x42u;
    block->packed_nibbles[32] = 0x53u;
    block->packed_nibbles[64] = 0x76u;
    block->packed_nibbles[96] = 0x98u;
}

static void test_q4_k_unpack_scale_min(void)
{
    Q4_KBlock block;
    const uint8_t expected_scales[Q4_K_SUBBLOCK_COUNT] = {
        1u, 2u, 3u, 4u, 17u, 18u, 19u, 20u
    };
    const uint8_t expected_mins[Q4_K_SUBBLOCK_COUNT] = {
        5u, 6u, 7u, 8u, 33u, 34u, 35u, 36u
    };

    make_reference_block(&block);
    for (int subblock = 0; subblock < Q4_K_SUBBLOCK_COUNT; ++subblock) {
        uint8_t scale = 0u;
        uint8_t minimum = 0u;

        assert(q4_k_unpack_scale_min(
            block.scale_and_min,
            subblock,
            &scale,
            &minimum
        ) == 1);
        assert(scale == expected_scales[subblock]);
        assert(minimum == expected_mins[subblock]);
    }
}

static void test_q4_k_dequantize_block(void)
{
    Q4_KBlock block;
    float output[Q4_K_BLOCK_ELEMENTS];
    float expected[Q4_K_BLOCK_ELEMENTS];
    const float default_values[Q4_K_SUBBLOCK_COUNT] = {
        -1.25f, -1.5f, -1.75f, -2.0f,
        -8.25f, -8.5f, -8.75f, -9.0f
    };

    make_reference_block(&block);
    for (int subblock = 0; subblock < Q4_K_SUBBLOCK_COUNT; ++subblock) {
        for (int offset = 0; offset < Q4_K_SUBBLOCK_ELEMENTS; ++offset) {
            expected[subblock * Q4_K_SUBBLOCK_ELEMENTS + offset] =
                default_values[subblock];
        }
    }

    expected[0] = -0.25f;
    expected[32] = 2.5f;
    expected[64] = 2.75f;
    expected[96] = 8.0f;
    expected[128] = 42.75f;
    expected[160] = 54.5f;
    expected[192] = 67.25f;
    expected[224] = 81.0f;

    assert(q4_k_dequantize_block(&block, output) == 1);
    assert(arrays_nearly_equal(output, expected, Q4_K_BLOCK_ELEMENTS));
}

static void test_q4_k_dot_f32(void)
{
    Q4_KBlock block;
    float input[Q4_K_BLOCK_ELEMENTS];
    float result = 99.0f;

    make_reference_block(&block);
    memset(input, 0, sizeof(input));
    input[0] = 1.0f;
    input[32] = 2.0f;
    input[64] = -1.0f;
    input[96] = 0.5f;
    input[128] = 4.0f;
    input[160] = -2.0f;
    input[192] = 8.0f;
    input[224] = -3.0f;

    /* The eight selected decoded values sum to a dot product of 363.0. */
    assert(q4_k_dot_f32(&block, input, &result) == 1);
    assert(floats_nearly_equal(result, 363.0f));
}

static void test_invalid_arguments(void)
{
    Q4_KBlock block;
    uint8_t scale_and_min[Q4_K_SCALE_AND_MIN_BYTES];
    uint8_t scale = 0u;
    uint8_t minimum = 0u;
    float input[Q4_K_BLOCK_ELEMENTS];
    float output[Q4_K_BLOCK_ELEMENTS];
    float result = 0.0f;

    memset(&block, 0, sizeof(block));
    memset(scale_and_min, 0, sizeof(scale_and_min));
    memset(input, 0, sizeof(input));

    assert(q4_k_unpack_scale_min(nullptr, 0, &scale, &minimum) == 0);
    assert(q4_k_unpack_scale_min(scale_and_min, -1, &scale, &minimum) == 0);
    assert(q4_k_unpack_scale_min(
        scale_and_min,
        Q4_K_SUBBLOCK_COUNT,
        &scale,
        &minimum
    ) == 0);
    assert(q4_k_unpack_scale_min(scale_and_min, 0, nullptr, &minimum) == 0);
    assert(q4_k_unpack_scale_min(scale_and_min, 0, &scale, nullptr) == 0);
    assert(q4_k_dequantize_block(nullptr, output) == 0);
    assert(q4_k_dequantize_block(&block, nullptr) == 0);
    assert(q4_k_dot_f32(nullptr, input, &result) == 0);
    assert(q4_k_dot_f32(&block, nullptr, &result) == 0);
    assert(q4_k_dot_f32(&block, input, nullptr) == 0);
}

int main(void)
{
    printf("Running Q4_K reference tests...\n");

    test_q4_k_unpack_scale_min();
    test_q4_k_dequantize_block();
    test_q4_k_dot_f32();
    test_invalid_arguments();

    printf("All Q4_K reference tests passed.\n");
    return 0;
}
