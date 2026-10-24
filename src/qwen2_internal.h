#ifndef QWEN2_INTERNAL_H
#define QWEN2_INTERNAL_H

#include "engine.h"
#include "engine_internal.h"

#define QWEN2_PREFILL_CHUNK_SIZE 8

/*
 * Private state for the Qwen2 full-prefix backend. Other architecture modules
 * may define a different layout, but their first member must be
 * EngineStateHeader so generic engine destruction can dispatch safely.
 *
 * qwen2_state_init allocates and initializes every pointer. The forward path
 * reuses these buffers; it must
 * not call malloc, calloc, realloc, or free per token.
 *
 * KV-cache exercise: follow docs/kv-cache-implementation-checklist.md and
 * search for TODO(kv-cache/...). The active backend is still full-prefix.
 */
typedef struct Qwen2InferenceState {
    EngineStateHeader header;
    int max_context_tokens; /* requested prompt plus generated-token capacity */
    int token_count;        /* token IDs currently stored in token_ids */
    int head_dimension;     /* model.config.dim / model.config.n_heads */
    int key_value_dimension; /* model.config.n_kv_heads * head_dimension */

    int *token_ids; /* max_context_tokens values; append one ID per forward call */

    /*
     * One activation vector per prefix position: [model dim].
     * Cached execution needs only model dim floats.
     */
    float *activations;

    /*
     * Expand with KV Cache:
     * keys and values now store computed after Q, K, V projection and RoPE
     * in this format: [layer][position][KV head][head dimension].
     */
    float *keys;                    /* [layer][position][KV head][head dimension] */
    float *values;                  /* [layer][position][KV head][head dimension] */
    float *normalized;              /* chunk * model dim */
    float *query;                   /* chunk * n_heads * head_dimension */
    float *key;                     /* chunk * n_kv_heads * head_dimension */
    float *value;                   /* chunk * n_kv_heads * head_dimension */
    float *attention_probabilities; /* n_heads * max_context_tokens */
    float *attention_output;        /* chunk * model dim */
    float *ffn_gate;                /* chunk * hidden_dim */
    float *ffn_up;                  /* chunk * hidden_dim */
    float *matvec_output;           /* chunk * maximum of model dim, hidden_dim */
} Qwen2InferenceState;

/*
 * Borrowed pointers into one layer of the FUTURE persistent KV cache.
 * keys/values start at position zero; key_slot/value_slot point at the
 * position being appended. No member owns storage or needs to be freed.
 */
typedef struct Qwen2KvCacheLayerView {
    float *keys;
    float *values;
    float *key_slot;
    float *value_slot;
    int token_count; /* active attention length, including the appended token */
} Qwen2KvCacheLayerView;

/*
 * The qwen2_kv_cache_layer_view function initializes a view of the key-value cache for a specified
 * layer and position in a model's inference state, ensuring that all parameters are valid and that
 * the model and state dimensions are consistent. It populates the provided Qwen2KvCacheLayerView structure
 * with pointers to the appropriate keys and values.
*/
int qwen2_kv_cache_layer_view(
    const Model *model,
    Qwen2InferenceState *private_state,
    int layer,
    int position,
    Qwen2KvCacheLayerView *view,
    char *error,
    int error_capacity
);

/* Full-prefix reference for tests. Use a separate state from cached execution. */
int qwen2_forward_token_reference(
    const Model *model,
    InferenceState *state,
    int token_id,
    char *error,
    int error_capacity
);

int qwen2_forward_token_cached(
    const Model *model,
    InferenceState *state,
    int token_id,
    char *error,
    int error_capacity
);

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
