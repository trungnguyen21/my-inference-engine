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

static void make_reference_block(Q6_KBlock *block)
{
    memset(block, 0, sizeof(*block));

    /* F16 0.5 is the shared factor. */
    block->scale_f16_bits = 0x3800u;
    for (int i = 0; i < Q6_K_SCALE_COUNT; ++i) {
        block->scales[i] = (int8_t)(i + 1);
    }

    /* Each byte contains upper-bit fields 0, 1, 2, and 3 in that order. */
    memset(block->upper_bits, 0xe4, sizeof(block->upper_bits));

    /* Exercise all four low-nibble sources in both 128-value groups. */
    block->lower_bits[0] = 0xa5u;
    block->lower_bits[32] = 0xc3u;
    block->lower_bits[64] = 0xb6u;
    block->lower_bits[96] = 0xd7u;
}

static void test_q6_k_unpack_value(void)
{
    Q6_KBlock block;
    const int selected_indices[Q6_K_SCALE_COUNT] = {
        0, 16, 32, 48, 64, 80, 96, 112,
        128, 144, 160, 176, 192, 208, 224, 240
    };
    const int expected_values[Q6_K_SCALE_COUNT] = {
        -27, -32, -13, -16, 10, 0, 28, 16,
        -26, -32, -9, -16, 11, 0, 29, 16
    };

    make_reference_block(&block);
    for (int i = 0; i < Q6_K_SCALE_COUNT; ++i) {
        int8_t quantized_value = 0;
        int8_t scale_multiplier = 0;

        assert(q6_k_unpack_value(
            &block,
            selected_indices[i],
            &quantized_value,
            &scale_multiplier
        ) == 1);
        assert(quantized_value == expected_values[i]);
        assert(scale_multiplier == i + 1);
    }
}

static void test_q6_k_dequantize_block(void)
{
    Q6_KBlock block;
    float output[Q6_K_BLOCK_ELEMENTS];
    float expected[Q6_K_BLOCK_ELEMENTS];
    const int default_quantized_values[4] = {-32, -16, 0, 16};
    const int selected_indices[8] = {0, 32, 64, 96, 128, 160, 192, 224};
    const int selected_values[8] = {-27, -13, 10, 28, -26, -9, 11, 29};

    make_reference_block(&block);
    for (int index = 0; index < Q6_K_BLOCK_ELEMENTS; ++index) {
        int group = index / Q6_K_GROUP_ELEMENTS;
        int local = index % Q6_K_GROUP_ELEMENTS;
        int region = local / 32;
        int offset = local % 32;
        int scale_index = group * 8 + region * 2 + offset / 16;

        expected[index] = 0.5f * (scale_index + 1) *
            default_quantized_values[region];
    }
    for (int i = 0; i < 8; ++i) {
        int index = selected_indices[i];
        int group = index / Q6_K_GROUP_ELEMENTS;
        int local = index % Q6_K_GROUP_ELEMENTS;
        int region = local / 32;
        int offset = local % 32;
        int scale_index = group * 8 + region * 2 + offset / 16;

        expected[index] = 0.5f * (scale_index + 1) * selected_values[i];
    }

    assert(q6_k_dequantize_block(&block, output) == 1);
    assert(arrays_nearly_equal(output, expected, Q6_K_BLOCK_ELEMENTS));
}

static void test_q6_k_dot_f32(void)
{
    Q6_KBlock block;
    float input[Q6_K_BLOCK_ELEMENTS];
    float result = 99.0f;
    const int selected_indices[Q6_K_SCALE_COUNT] = {
        0, 16, 32, 48, 64, 80, 96, 112,
        128, 144, 160, 176, 192, 208, 224, 240
    };

    make_reference_block(&block);
    memset(input, 0, sizeof(input));
    for (int i = 0; i < Q6_K_SCALE_COUNT; ++i) {
        input[selected_indices[i]] = 1.0f;
    }

    /* The selected sixteen decoded values sum to 84.5. */
    assert(q6_k_dot_f32(&block, input, &result) == 1);
    assert(floats_nearly_equal(result, 84.5f));
}

static void test_invalid_arguments(void)
{
    Q6_KBlock block;
    int8_t quantized_value = 0;
    int8_t scale_multiplier = 0;
    float input[Q6_K_BLOCK_ELEMENTS];
    float output[Q6_K_BLOCK_ELEMENTS];
    float result = 0.0f;

    memset(&block, 0, sizeof(block));
    memset(input, 0, sizeof(input));

    assert(q6_k_unpack_value(nullptr, 0, &quantized_value, &scale_multiplier) == 0);
    assert(q6_k_unpack_value(&block, -1, &quantized_value, &scale_multiplier) == 0);
    assert(q6_k_unpack_value(
        &block,
        Q6_K_BLOCK_ELEMENTS,
        &quantized_value,
        &scale_multiplier
    ) == 0);
    assert(q6_k_unpack_value(&block, 0, nullptr, &scale_multiplier) == 0);
    assert(q6_k_unpack_value(&block, 0, &quantized_value, nullptr) == 0);
    assert(q6_k_dequantize_block(nullptr, output) == 0);
    assert(q6_k_dequantize_block(&block, nullptr) == 0);
    assert(q6_k_dot_f32(nullptr, input, &result) == 0);
    assert(q6_k_dot_f32(&block, nullptr, &result) == 0);
    assert(q6_k_dot_f32(&block, input, nullptr) == 0);
}

int main(void)
{
    printf("Running Q6_K reference tests...\n");

    test_q6_k_unpack_value();
    test_q6_k_dequantize_block();
    test_q6_k_dot_f32();
    test_invalid_arguments();

    printf("All Q6_K reference tests passed.\n");
    return 0;
}
