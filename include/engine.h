#ifndef INFERENCE_ENGINE_H
#define INFERENCE_ENGINE_H

#include "app.h"
#include "metrics.h"

#define TOKEN_TEXT_CAPACITY 257
#define ENGINE_ERROR_CAPACITY 512

typedef struct {
    int dim; // transformer dimension
    int hidden_dim; // for ffn layers
    int n_layers; // number of layers
    int n_heads; // number of query heads
    int n_kv_heads; // number of key/value heads (can be < query heads for GQA)
    int vocab_size; // number of tokenizer entries
    int seq_len; // max sequence length
    float rms_norm_epsilon; // architecture-provided RMSNorm epsilon
    float rope_frequency_base; // architecture-provided RoPE frequency base
} ModelConfig;

typedef struct Model {
    void *implementation;
    ModelConfig config;
} Model;

typedef struct InferenceState {
    void *implementation; // private token history and temporary workspace
    float *logits;        // one score per vocabulary token after a forward call
    int logits_count;     // must equal ModelConfig.vocab_size on forward success
    OperationProfile profile; // enable after init; totals accumulate per state
} InferenceState;

/* Parse GGUF metadata/tensors and retain model/tokenizer data for later steps. */
int engine_load_model(const AppConfig *config, Model *model,
                      char *error, int error_capacity);
void engine_unload_model(Model *model);

/*
 * Format a single user message for the loaded model's instruction protocol.
 * Separate, non-overlapping input/output buffers are required. Return 1 with
 * a complete NUL-terminated prompt, or 0 with an error and output[0] cleared
 * when writable. No truncation. Empty user content is allowed. The current
 * Qwen exercise targets the local Qwen2.5-Instruct default system message;
 * arbitrary GGUF Jinja templates, tools, and conversation history are deferred.
 */
int engine_format_prompt(const Model *model, const char *user_prompt,
                         char *output, int output_capacity,
                         char *error, int error_capacity);

/* Encode text into token IDs, recognizing the instruction prompt's special tokens. */
int tokenizer_encode(const Model *model, const char *text, int *token_ids,
                     int token_capacity, int *token_count,
                     char *error, int error_capacity);
/* Write one NUL-terminated token piece, respecting text_capacity. */
int tokenizer_decode_one(const Model *model, int token_id, char *text,
                         int text_capacity, char *error, int error_capacity);
int tokenizer_is_eos(const Model *model, int token_id);

/*
 * Allocate all storage that remains live across token forwards.
 *
 * model is a successfully loaded transformer model. max_context_tokens is the total
 * number of prompt and generated tokens that the caller intends to forward; it
 * must be positive and no greater than model->config.seq_len. state is output
 * storage supplied by the caller. On success it owns a logits vector sized to
 * model->config.vocab_size, token-history storage, and reusable activation and
 * temporary K/V buffers for a full-prefix forward pass. K/V values are rebuilt
 * for each call rather than retained between calls. It starts at position zero.
 *
 * Return 1 on success. Return 0 and write a useful message for null arguments,
 * invalid model configuration, an out-of-range context, integer-overflow in an
 * allocation size, or allocation failure. Leave state safe to pass to
 * inference_state_destroy after either result. No token-forward function may
 * allocate memory after this initialization succeeds.
 */
int inference_state_init(const Model *model, InferenceState *state,
                         int max_context_tokens,
                         char *error, int error_capacity);

/*
 * Append one input token to state history, recompute the complete prefix from
 * its first token, and calculate logits for the token after the prefix. model
 * and state must be a matched successful load/init pair. token_id
 * must be in [0, model->config.vocab_size), and state must have a free context
 * position. The selected architecture backend performs embedding lookup, its
 * transformer layers, the final norm when required, and the output projection.
 * It writes
 * model->config.vocab_size finite F32 values to state->logits, sets
 * state->logits_count to that count, and advances the internal token count
 * exactly once. Earlier activations and K/V vectors are recomputed on each
 * call; they are not reused from the previous call.
 *
 * Return 1 only after all of those outputs are valid. Return 0 with an error
 * for invalid arguments, exhausted context, malformed model tensor data, or a
 * failed tensor operation. A failed call must not advance the state position.
 */
int engine_forward_token(const Model *model, InferenceState *state,
                         int token_id, char *error, int error_capacity);

/* Free all state-owned allocations and clear the caller's InferenceState. */
void inference_state_destroy(InferenceState *state);

/*
 * Select one token from the latest logits. Temperature zero uses greedy
 * selection (lowest ID on ties); top_p is ignored on this path after checking
 * its [0, 1] range. Nonzero temperature sampling is a later exercise.
 * Return 1 with a valid ID, or 0 with *token_id == -1 when writable and an error.
 * Logits must all be finite; temperature and top_p must be finite in [0, 1].
 */
int sampler_sample(const float *logits, int vocab_size,
                   float temperature, float top_p, int *token_id,
                   char *error, int error_capacity);

#endif
