#ifndef QWEN2_INTERNAL_H
#define QWEN2_INTERNAL_H

#include "engine_internal.h"

/*
 * Private state for the Qwen2 full-prefix backend. Other architecture modules
 * may define a different layout, but their first member must be
 * EngineStateHeader so generic engine destruction can dispatch safely.
 *
 * qwen2_state_init allocates and initializes every pointer. The forward path
 * reuses these buffers; it must
 * not call malloc, calloc, realloc, or free per token.
 */
typedef struct Qwen2InferenceState {
    EngineStateHeader header;
    int max_context_tokens; /* requested prompt plus generated-token capacity */
    int token_count;        /* token IDs currently stored in token_ids */
    int head_dimension;     /* model.config.dim / model.config.n_heads */
    int key_value_dimension; /* model.config.n_kv_heads * head_dimension */

    int *token_ids; /* max_context_tokens values; append one ID per forward call */

    /* One activation vector per prefix position: [position][model dim]. */
    float *activations;

    /* Per-layer workspace rebuilt while evaluating the full prefix. */
    float *keys;                    /* [position][KV head][head dimension] */
    float *values;                  /* [position][KV head][head dimension] */
    float *normalized;              /* model dim */
    float *query;                   /* n_heads * head_dimension */
    float *key;                     /* n_kv_heads * head_dimension */
    float *value;                   /* n_kv_heads * head_dimension */
    float *attention_probabilities; /* n_heads * max_context_tokens */
    float *attention_output;        /* model dim */
    float *ffn_gate;                /* hidden_dim */
    float *ffn_up;                  /* hidden_dim */
    float *matvec_output;           /* maximum of model dim, hidden_dim, vocab */
} Qwen2InferenceState;

/*
 * Decode one row of a packed embedding matrix.
 * matrix uses GGUF [columns, rows] order. Require columns == dim, a valid
 * token_id, and whole blocks per row. Support Q5_0, Q8_0, Q4_K, and Q6_K.
 * Validate the view and sizes before writing exactly dim floats into output.
 * Packed data may be unaligned: copy each block into its aligned local type.
 * Allocate no memory. Return 1 on success, or 0 with an error on failure.
 */
int qwen2_decode_quantized_embedding_row(
    const GgufQuantizedTensorView *matrix,
    int token_id,
    int dim,
    float *output,
    char *error,
    int error_capacity
);

/*
 * Project through a fully named model tensor, e.g. blk.0.attn_q.weight or
 * output.weight. input_count is columns; output_count is rows. Validate the
 * tensor shape and storage, then dispatch to the F32 or direct packed matvec.
 * Keep input/output separate and allocate no memory. Return 0 with an error
 * for invalid arguments, missing/invalid weights, or a failed operation.
 */
int qwen2_weight_matvec(
    const GgufFile *file,
    const char *name,
    const float *input,
    int input_count,
    float *output,
    int output_count,
    char *error,
    int error_capacity
);

#endif
