#ifndef TRANSFORMER_OPS_H
#define TRANSFORMER_OPS_H   

/*
 * Rotate the first rotary_dimension values of every query and key head in
 * place using Qwen's split-half RoPE layout.
 *
 * query contains query_head_count consecutive heads, and key contains
 * key_head_count consecutive heads. Each head contains head_dimension F32
 * values. rotary_dimension is an even positive value no larger than
 * head_dimension. position is the zero-based token position and frequency_base
 * comes from qwen2.rope.freq_base. Values after rotary_dimension are unchanged.
 *
 * Return 1 after rotating every head. Return 0 for null pointers, invalid
 * counts or dimensions, a negative position, or a non-positive frequency base.
 * This function allocates no memory.
 */
int transformer_apply_rope_f32(
    float *query,
    int query_head_count,
    float *key,
    int key_head_count,
    int head_dimension,
    int rotary_dimension,
    int position,
    float frequency_base
);

/*
 * Calculate causal grouped-query attention for one transformer layer at one
 * token position. keys and values are temporary arrays built during the
 * current full-prefix forward pass; they are not retained between calls.
 *
 * query is query_head_count * head_dimension F32 values, arranged as
 * [query head][dimension]. keys and values each hold token_count key/value
 * vectors, arranged as [token][key/value head][dimension]. They include the
 * current token's K and V. query_head_count must be divisible by
 * key_value_head_count; each group of query heads reads the same KV head.
 *
 * For each query head, scale Q dot K by 1 / sqrt(head_dimension), write the
 * stable-softmax probabilities into attention_probabilities at
 * [query head][token], and write the weighted V sum into output at
 * [query head][dimension]. output and attention_probabilities must not alias
 * query, keys, or values. Return 1 on success and 0 for invalid arguments.
 * This function allocates no memory.
 */
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
);

/*
 * Apply Qwen's SwiGLU gate in place.
 *
 * gate and up each contain length F32 values produced by their separate FFN
 * projections. Replace gate[i] with SiLU(gate[i]) * up[i]. The caller then
 * passes gate to the FFN-down projection. Return 1 on success and 0 for null
 * pointers or a non-positive length. This function allocates no memory.
 */
int transformer_swiglu_f32(float *gate, const float *up, int length);

#endif
