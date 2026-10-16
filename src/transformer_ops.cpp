#include "transformer_ops.h"
#include "tensor.h"

#include <math.h>

/*
Rotate a flat array to use RoPE

values is a flat array, but conceptually, it's a nested list:
[val | val | val | val]
 +--------+  +-------+
 head          head
 +-> we rotate each head
 */
static void rotate_array_helper(
    float *values,
    int head_count,
    int head_dimension,
    int rotary_dimension,
    int position,
    float frequency_base
) {
    if (values == nullptr) {
        return;
    }
    
    for (int head_index = 0; head_index < head_count; head_index++) {
        int head_start = head_index * head_dimension;

        for (int i = 0; i < rotary_dimension / 2; i++) {
            float x = values[head_start + i];
            float y = values[head_start + i + rotary_dimension / 2];
            float angle = (float)position / powf(frequency_base, (2.0f * (float)i / (float)rotary_dimension));
            
            float rotated_x = x * cosf(angle) - y * sinf(angle);
            float rotated_y = x * sinf(angle) + y * cosf(angle);
            values[head_start + i] = rotated_x;
            values[head_start + i + rotary_dimension / 2] = rotated_y;
        }
    }
}

int transformer_apply_rope_f32(
    float *query,
    int query_head_count,
    float *key,
    int key_head_count,
    int head_dimension,
    int rotary_dimension,
    int position,
    float frequency_base
)
{
    bool invalid_args = 
        (query == nullptr) || (key == nullptr) ||
        (query_head_count <= 0) || (key_head_count <= 0) ||
        (rotary_dimension % 2 != 0) || (frequency_base <= 0.0f) ||
        (rotary_dimension <= 0) || (head_dimension <= 0) || (rotary_dimension > head_dimension) ||
        (position < 0) || (!isfinite(frequency_base));
    
    if (invalid_args) {
        return 0;
    }

    rotate_array_helper(query, query_head_count, head_dimension, rotary_dimension, position, frequency_base);
    rotate_array_helper(key, key_head_count, head_dimension, rotary_dimension, position, frequency_base);

    return 1;
}

int transformer_attention_f32(
    const float *query,
    int query_head_count,
    int key_value_head_count,
    int head_dimension,
    const float *keys,
    const float *values,
    int token_count,
    float *attention_probabilities,
    float *output
)
{
    bool invalid_args = 
        (query == nullptr) || 
        (keys == nullptr) ||
        (values == nullptr) ||
        (output == nullptr) ||
        (attention_probabilities == nullptr) ||
        (query_head_count <= 0) ||
        (key_value_head_count <= 0) ||
        (query_head_count % key_value_head_count != 0) ||
        (head_dimension <= 0) ||
        (token_count <= 0);
    
    if (invalid_args) {
        return 0;
    }

    for (int head_index = 0; head_index < query_head_count; head_index++) {
        const float *head_start = query + head_index * head_dimension;
        int queries_per_kv_head = query_head_count / key_value_head_count;
        int kv_head_index = head_index / queries_per_kv_head;
        
        for (int token_index = 0; token_index < token_count; token_index++) {
            int key_start_index =
                (token_index * key_value_head_count + kv_head_index) * head_dimension;
            const float *key_start = keys + key_start_index;
            float score;
            int res = tensor_dot_f32(head_start, key_start, head_dimension, &score);
            if (!res) {
                return 0;
            }
            score /= sqrtf((float)head_dimension);
            int score_index = head_index * token_count + token_index;
            attention_probabilities[score_index] = score;
        }
        /* Convert only this query head's token scores into probabilities. */
        int first_score = head_index * token_count;
        float max_score = attention_probabilities[first_score];
        for (int token_index = 1; token_index < token_count; token_index++) {
            float score = attention_probabilities[first_score + token_index];
            if (score > max_score) {
                max_score = score;
            }
        }
        float probability_sum = 0.0f;
        for (int token_index = 0; token_index < token_count; token_index++) {
            int score_index = first_score + token_index;
            attention_probabilities[score_index] =
                expf(attention_probabilities[score_index] - max_score);
            probability_sum += attention_probabilities[score_index];
        }
        if (!isfinite(probability_sum) || probability_sum <= 0.0f) {
            return 0;
        }
        for (int token_index = 0; token_index < token_count; token_index++) {
            int score_index = first_score + token_index;
            attention_probabilities[score_index] /= probability_sum;
        }

        int output_start = head_index * head_dimension;

        for (int d = 0; d < head_dimension; d++) {
            output[output_start + d] = 0.0f;
        }

        for (int token_index = 0; token_index < token_count; token_index++) {
            int score_index = first_score + token_index;
            float prob = attention_probabilities[score_index];
            int value_start_index =
                (token_index * key_value_head_count + kv_head_index) * head_dimension;

            for (int dim = 0; dim < head_dimension; dim++) {
                output[output_start + dim] += prob * values[value_start_index + dim];
            }
        }
    }


    return 1;
}

static float silu(float x) {
    return x / (1.0f + expf(-x));
}

int transformer_swiglu_f32(float *gate, const float *up, int length)
{
    bool invalid_args = (gate == nullptr) || (up == nullptr) || (length <= 0);
    if (invalid_args) {
        return 0;
    }

    for (int i = 0; i < length; i++) {
        gate[i] = up[i] * silu(gate[i]);
    }
    return 1;
}
