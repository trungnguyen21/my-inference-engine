#include <assert.h>
#include <math.h>
#include <stdio.h>

#include "transformer_ops.h"

static const float EPSILON = 0.0001f;

static int floats_nearly_equal(float actual, float expected)
{
    return fabsf(actual - expected) <= EPSILON;
}

static void test_rope_rotates_query_and_key(void)
{
    float query[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float key[] = {-2.0f, 1.0f, 4.0f, -3.0f};

    assert(transformer_apply_rope_f32(query, 1, key, 1, 4, 4, 1, 10000.0f) == 1);

    assert(floats_nearly_equal(query[0], -1.984111f));
    assert(floats_nearly_equal(query[1], 1.959901f));
    assert(floats_nearly_equal(query[2], 2.462378f));
    assert(floats_nearly_equal(query[3], 4.019799f));
    assert(floats_nearly_equal(key[0], -4.446489f));
    assert(floats_nearly_equal(key[1], 1.029850f));
    assert(floats_nearly_equal(key[2], 0.478267f));
    assert(floats_nearly_equal(key[3], -2.989850f));
}

static void test_rope_rejects_invalid_arguments(void)
{
    float query[4] = {0.0f};
    float key[4] = {0.0f};

    assert(transformer_apply_rope_f32(nullptr, 1, key, 1, 4, 4, 0, 10000.0f) == 0);
    assert(transformer_apply_rope_f32(query, 1, nullptr, 1, 4, 4, 0, 10000.0f) == 0);
    assert(transformer_apply_rope_f32(query, 1, key, 1, 4, 3, 0, 10000.0f) == 0);
    assert(transformer_apply_rope_f32(query, 1, key, 1, 4, 4, -1, 10000.0f) == 0);
    assert(transformer_apply_rope_f32(query, 1, key, 1, 4, 4, 0, 0.0f) == 0);
}

static void test_grouped_query_attention(void)
{
    const float query[] = {
        1.0f, 0.0f,
        0.0f, 1.0f
    };
    const float keys[] = {
        1.0f, 0.0f,
        0.0f, 1.0f
    };
    const float values[] = {
        10.0f, 20.0f,
        30.0f, 40.0f
    };
    float probabilities[4] = {0.0f};
    float output[4] = {0.0f};

    assert(transformer_attention_f32(
        query, 2, 1, 2, keys, values, 2, probabilities, output
    ) == 1);

    assert(floats_nearly_equal(probabilities[0], 0.669762f));
    assert(floats_nearly_equal(probabilities[1], 0.330238f));
    assert(floats_nearly_equal(probabilities[2], 0.330238f));
    assert(floats_nearly_equal(probabilities[3], 0.669762f));
    assert(floats_nearly_equal(output[0], 16.604769f));
    assert(floats_nearly_equal(output[1], 26.604769f));
    assert(floats_nearly_equal(output[2], 23.395231f));
    assert(floats_nearly_equal(output[3], 33.395233f));
}

static void test_attention_rejects_invalid_arguments(void)
{
    float query[4] = {0.0f};
    float cache[4] = {0.0f};
    float probabilities[2] = {0.0f};
    float output[4] = {0.0f};

    assert(transformer_attention_f32(
        nullptr, 2, 1, 2, cache, cache, 1, probabilities, output
    ) == 0);
    assert(transformer_attention_f32(
        query, 3, 2, 2, cache, cache, 1, probabilities, output
    ) == 0);
    assert(transformer_attention_f32(
        query, 2, 1, 0, cache, cache, 1, probabilities, output
    ) == 0);
    assert(transformer_attention_f32(
        query, 2, 1, 2, cache, cache, 0, probabilities, output
    ) == 0);
}

static void test_swiglu(void)
{
    float gate[] = {0.0f, 1.0f, -1.0f};
    const float up[] = {4.0f, 2.0f, 5.0f};

    assert(transformer_swiglu_f32(gate, up, 3) == 1);
    assert(floats_nearly_equal(gate[0], 0.0f));
    assert(floats_nearly_equal(gate[1], 1.462117f));
    assert(floats_nearly_equal(gate[2], -1.344707f));

    assert(transformer_swiglu_f32(nullptr, up, 3) == 0);
    assert(transformer_swiglu_f32(gate, nullptr, 3) == 0);
    assert(transformer_swiglu_f32(gate, up, 0) == 0);
}

int main(void)
{
    printf("Running Milestone 06 transformer-operation reference tests...\n");

    test_rope_rotates_query_and_key();
    test_rope_rejects_invalid_arguments();
    test_grouped_query_attention();
    test_attention_rejects_invalid_arguments();
    test_swiglu();

    printf("All Milestone 06 transformer-operation reference tests passed.\n");
    return 0;
}
