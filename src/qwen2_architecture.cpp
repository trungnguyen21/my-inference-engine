#include "qwen2_architecture.h"

#include "qwen2_internal.h"
#include "quantization.h"
#include "quantized_matvec.h"
#include "operation_timer.h"
#include "tensor.h"
#include "transformer_ops.h"

#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void set_error(char *error, int capacity, const char *message)
{
    if (error != nullptr && capacity > 0) {
        snprintf(error, (size_t)capacity, "%s", message);
    }
}

static int read_config_value(
    const GgufFile *file,
    const char *key,
    int *result,
    char *error,
    int error_capacity
)
{
    uint64_t value;

    if (!gguf_get_uint(file, key, &value) || value == 0 || value > INT_MAX) {
        if (error != nullptr && error_capacity > 0) {
            snprintf(error, (size_t)error_capacity,
                     "Qwen2 GGUF metadata is missing or invalid: %s", key);
        }
        return 0;
    }
    *result = (int)value;
    return 1;
}

static int tensor_has_shape(
    const GgufTensorInfo *tensor,
    uint64_t dim0,
    uint64_t dim1
)
{
    return tensor != nullptr && tensor->dimension_count == 2 &&
           tensor->dimensions[0] == dim0 && tensor->dimensions[1] == dim1;
}

static int tensor_has_vector_shape(const GgufTensorInfo *tensor, uint64_t size)
{
    return tensor != nullptr && tensor->dimension_count == 1 &&
           tensor->dimensions[0] == size;
}

static int qwen2_configure_model(
    const GgufFile *file,
    int vocab_size,
    ModelConfig *config,
    void **architecture_data,
    char *error,
    int error_capacity
)
{
    const GgufTensorInfo *embedding;
    const GgufTensorInfo *classifier;
    const GgufTensorInfo *output_norm;
    int layer;
    double rms_norm_epsilon;
    double rope_frequency_base;
    float rms_norm_epsilon_float;
    float rope_frequency_base_float;

    if (config == nullptr || architecture_data == nullptr || file == nullptr ||
        vocab_size <= 0) {
        set_error(error, error_capacity, "Qwen2 model configuration arguments are invalid.");
        return 0;
    }
    memset(config, 0, sizeof(*config));
    *architecture_data = nullptr;

    if (!read_config_value(file, "qwen2.embedding_length", &config->dim,
                           error, error_capacity) ||
        !read_config_value(file, "qwen2.feed_forward_length", &config->hidden_dim,
                           error, error_capacity) ||
        !read_config_value(file, "qwen2.block_count", &config->n_layers,
                           error, error_capacity) ||
        !read_config_value(file, "qwen2.attention.head_count", &config->n_heads,
                           error, error_capacity) ||
        !read_config_value(file, "qwen2.attention.head_count_kv", &config->n_kv_heads,
                           error, error_capacity) ||
        !read_config_value(file, "qwen2.context_length", &config->seq_len,
                           error, error_capacity)) {
        return 0;
    }
    config->vocab_size = vocab_size;

    if (!gguf_get_float(file, "qwen2.attention.layer_norm_rms_epsilon",
                        &rms_norm_epsilon) ||
        !gguf_get_float(file, "qwen2.rope.freq_base", &rope_frequency_base)) {
        set_error(error, error_capacity,
                  "Qwen2 normalization or RoPE metadata is missing or invalid.");
        return 0;
    }
    rms_norm_epsilon_float = (float)rms_norm_epsilon;
    rope_frequency_base_float = (float)rope_frequency_base;
    if (!isfinite(rms_norm_epsilon) || rms_norm_epsilon <= 0.0 ||
        !isfinite(rope_frequency_base) || rope_frequency_base <= 0.0 ||
        !isfinite(rms_norm_epsilon_float) || rms_norm_epsilon_float <= 0.0f ||
        !isfinite(rope_frequency_base_float) || rope_frequency_base_float <= 0.0f) {
        set_error(error, error_capacity,
                  "Qwen2 normalization or RoPE metadata is out of range.");
        return 0;
    }
    config->rms_norm_epsilon = rms_norm_epsilon_float;
    config->rope_frequency_base = rope_frequency_base_float;

    if (config->dim % config->n_heads != 0 ||
        config->n_kv_heads > config->n_heads ||
        config->n_heads % config->n_kv_heads != 0) {
        set_error(error, error_capacity, "Qwen2 head dimensions in GGUF metadata are inconsistent.");
        return 0;
    }

    embedding = gguf_find_tensor(file, "token_embd.weight");
    if (!tensor_has_shape(embedding, (uint64_t)config->dim, (uint64_t)vocab_size)) {
        set_error(error, error_capacity,
                  "Qwen2 token_embd.weight shape does not match model metadata.");
        return 0;
    }
    classifier = gguf_find_tensor(file, "output.weight");
    if (!tensor_has_shape(classifier, (uint64_t)config->dim, (uint64_t)vocab_size)) {
        set_error(error, error_capacity,
                  "Qwen2 output.weight shape does not match model metadata.");
        return 0;
    }
    output_norm = gguf_find_tensor(file, "output_norm.weight");
    if (!tensor_has_vector_shape(output_norm, (uint64_t)config->dim)) {
        set_error(error, error_capacity,
                  "Qwen2 output_norm.weight is missing or has an invalid shape.");
        return 0;
    }

    {
        uint64_t dim = (uint64_t)config->dim;
        uint64_t hidden_dim = (uint64_t)config->hidden_dim;
        uint64_t kv_dim = dim / (uint64_t)config->n_heads *
                          (uint64_t)config->n_kv_heads;
        const char *suffixes[] = {
            "attn_norm.weight", "attn_q.weight", "attn_k.weight",
            "attn_v.weight", "attn_q.bias", "attn_k.bias", "attn_v.bias",
            "attn_output.weight", "ffn_norm.weight", "ffn_gate.weight",
            "ffn_up.weight", "ffn_down.weight"
        };
        for (layer = 0; layer < config->n_layers; ++layer) {
            char name[96];
            const GgufTensorInfo *tensor;
            uint64_t d0;
            uint64_t d1;
            int i;
            for (i = 0; i < (int)(sizeof(suffixes) / sizeof(suffixes[0])); ++i) {
                int valid;
                snprintf(name, sizeof(name), "blk.%d.%s", layer, suffixes[i]);
                tensor = gguf_find_tensor(file, name);
                switch (i) {
                case 0:
                case 8:
                    valid = tensor_has_vector_shape(tensor, dim);
                    break;
                case 1:
                    d0 = dim; d1 = dim;
                    valid = tensor_has_shape(tensor, d0, d1);
                    break;
                case 2:
                case 3:
                    d0 = dim; d1 = kv_dim;
                    valid = tensor_has_shape(tensor, d0, d1);
                    break;
                case 4:
                    valid = tensor_has_vector_shape(tensor, dim);
                    break;
                case 5:
                case 6:
                    valid = tensor_has_vector_shape(tensor, kv_dim);
                    break;
                case 7:
                    valid = tensor_has_shape(tensor, dim, dim);
                    break;
                case 9:
                case 10:
                    valid = tensor_has_shape(tensor, dim, hidden_dim);
                    break;
                default:
                    valid = tensor_has_shape(tensor, hidden_dim, dim);
                    break;
                }
                if (!valid) {
                    if (error != nullptr && error_capacity > 0) {
                        snprintf(error, (size_t)error_capacity,
                                 "Qwen2 tensor %s is missing or has an invalid shape.", name);
                    }
                    return 0;
                }
            }
        }
    }
    return 1;
}

static void qwen2_destroy_model_data(void *architecture_data)
{
    /* No Qwen2-only model allocation exists yet. Add cleanup here if one does. */
    (void)architecture_data;
}

static void qwen2_state_destroy(InferenceState *state);

static int qwen2_state_init(
    const Model *model,
    InferenceState *state,
    int max_context_tokens,
    char *error,
    int error_capacity
)
{
    ModelConfig config;
    Qwen2InferenceState *private_state;
    size_t context, dim, hidden, kv_dim, heads, vocab, max_projection;

    if (state == nullptr || model == nullptr) {
        set_error(error, error_capacity, "Qwen2 inference-state arguments are invalid.");
        return 0;
    }
    memset(state, 0, sizeof(*state));
    config = model->config;
    if (max_context_tokens <= 0 || config.seq_len <= 0 ||
        max_context_tokens > config.seq_len || config.dim <= 0 ||
        config.hidden_dim <= 0 || config.n_layers <= 0 || config.n_heads <= 0 ||
        config.n_kv_heads <= 0 || config.vocab_size <= 0 ||
        config.dim % config.n_heads != 0 || config.n_kv_heads > config.n_heads ||
        config.n_heads % config.n_kv_heads != 0) {
        set_error(error, error_capacity, "Qwen2 inference-state configuration is invalid.");
        return 0;
    }
    context = (size_t)max_context_tokens;
    dim = (size_t)config.dim;
    hidden = (size_t)config.hidden_dim;
    heads = (size_t)config.n_heads;
    vocab = (size_t)config.vocab_size;
    kv_dim = (size_t)(config.dim / config.n_heads) * (size_t)config.n_kv_heads;
    max_projection = dim > hidden ? dim : hidden;
    if (vocab > max_projection) max_projection = vocab;
    if (context > SIZE_MAX / dim || context * dim > SIZE_MAX / sizeof(float) ||
        context > SIZE_MAX / kv_dim || context * kv_dim > SIZE_MAX / sizeof(float) ||
        heads > SIZE_MAX / context || heads * context > SIZE_MAX / sizeof(float) ||
        max_projection > SIZE_MAX / sizeof(float)) {
        set_error(error, error_capacity, "Qwen2 inference-state allocation size overflows.");
        return 0;
    }

    private_state = (Qwen2InferenceState *)calloc(1, sizeof(Qwen2InferenceState));
    if (private_state == nullptr) {
        set_error(error, error_capacity, "Could not allocate Qwen2 inference state.");
        return 0;
    }
    private_state->header.architecture = qwen2_architecture_ops();
    state->implementation = private_state;
    private_state->max_context_tokens = max_context_tokens;
    private_state->head_dimension = config.dim / config.n_heads;
    private_state->key_value_dimension = (int)kv_dim;
#define ALLOC_FIELD(field, count, type) \
    private_state->field = (type *)calloc((count), sizeof(type)); \
    if (private_state->field == nullptr) goto allocation_failure
    ALLOC_FIELD(token_ids, context, int);
    ALLOC_FIELD(activations, context * dim, float);
    ALLOC_FIELD(keys, context * kv_dim, float);
    ALLOC_FIELD(values, context * kv_dim, float);
    ALLOC_FIELD(normalized, dim, float);
    ALLOC_FIELD(query, dim, float);
    ALLOC_FIELD(key, kv_dim, float);
    ALLOC_FIELD(value, kv_dim, float);
    ALLOC_FIELD(attention_probabilities, heads * context, float);
    ALLOC_FIELD(attention_output, dim, float);
    ALLOC_FIELD(ffn_gate, hidden, float);
    ALLOC_FIELD(ffn_up, hidden, float);
    ALLOC_FIELD(matvec_output, max_projection, float);
    state->logits = (float *)calloc(vocab, sizeof(float));
    if (state->logits == nullptr) goto allocation_failure;
    state->logits_count = config.vocab_size;
#undef ALLOC_FIELD
    if (error != nullptr && error_capacity > 0) error[0] = '\0';
    return 1;

allocation_failure:
#undef ALLOC_FIELD
    qwen2_state_destroy(state);
    set_error(error, error_capacity, "Could not allocate Qwen2 inference-state workspace.");
    return 0;
}

int qwen2_decode_quantized_embedding_row(
    const GgufQuantizedTensorView *matrix,
    int token_id,
    int dim,
    float *output,
    char *error,
    int error_capacity
)
{
    if (matrix == nullptr || matrix->data == nullptr || output == nullptr ||
        dim <= 0 || token_id < 0) {
        set_error(error, error_capacity, "Qwen2 packed embedding arguments are invalid.");
        return 0;
    }
    if (matrix->dimension_count != 2 || matrix->dimensions[0] != (uint64_t)dim ||
        matrix->dimensions[1] == 0 || (uint64_t)token_id >= matrix->dimensions[1]) {
        set_error(error, error_capacity, "Qwen2 packed embedding shape or token ID is invalid.");
        return 0;
    }

    uint64_t block_elements;
    uint64_t block_bytes;
    switch (matrix->ggml_type) {
        case ENGINE_GGML_TYPE_Q5_0:
            block_elements = Q5_0_BLOCK_ELEMENTS;
            block_bytes = Q5_0_BLOCK_BYTES;
            break;
        case ENGINE_GGML_TYPE_Q8_0:
            block_elements = Q8_0_BLOCK_ELEMENTS;
            block_bytes = Q8_0_BLOCK_BYTES;
            break;
        case ENGINE_GGML_TYPE_Q4_K:
            block_elements = Q4_K_BLOCK_ELEMENTS;
            block_bytes = Q4_K_BLOCK_BYTES;
            break;
        case ENGINE_GGML_TYPE_Q6_K:
            block_elements = Q6_K_BLOCK_ELEMENTS;
            block_bytes = Q6_K_BLOCK_BYTES;
            break;
        default:
            set_error(error, error_capacity, "Qwen2 packed embedding type is unsupported.");
            return 0;
    }
    const uint64_t columns = matrix->dimensions[0];
    const uint64_t rows = matrix->dimensions[1];
    if (matrix->block_elements != block_elements || matrix->block_bytes != block_bytes ||
        columns % block_elements != 0) {
        set_error(error, error_capacity, "Qwen2 packed embedding block layout is invalid.");
        return 0;
    }
    const uint64_t blocks_per_row = columns / block_elements;
    if (columns > UINT64_MAX / rows || matrix->element_count != columns * rows ||
        blocks_per_row > UINT64_MAX / rows ||
        matrix->block_count != blocks_per_row * rows ||
        matrix->block_count > SIZE_MAX / block_bytes ||
        columns > SIZE_MAX / sizeof(float)) {
        set_error(error, error_capacity, "Qwen2 packed embedding storage size is invalid or overflows.");
        return 0;
    }

    /* The checked tensor size bounds every block offset in this selected row.
     * Packed GGUF bytes need not have the alignment required by block structs. */
    const size_t row_offset = (size_t)((uint64_t)token_id * blocks_per_row * block_bytes);
    for (uint64_t i = 0; i < blocks_per_row; ++i) {
        const unsigned char *source = matrix->data + row_offset + (size_t)(i * block_bytes);
        float *destination = output + (size_t)(i * block_elements);
        int ok = 0;
        switch (matrix->ggml_type) {
            case ENGINE_GGML_TYPE_Q5_0: {
                Q5_0Block block;
                memcpy(&block, source, sizeof(block));
                ok = q5_0_dequantize_block(&block, destination);
                break;
            }
            case ENGINE_GGML_TYPE_Q8_0: {
                Q8_0Block block;
                memcpy(&block, source, sizeof(block));
                ok = q8_0_dequantize_block(&block, destination);
                break;
            }
            case ENGINE_GGML_TYPE_Q4_K: {
                Q4_KBlock block;
                memcpy(&block, source, sizeof(block));
                ok = q4_k_dequantize_block(&block, destination);
                break;
            }
            case ENGINE_GGML_TYPE_Q6_K: {
                Q6_KBlock block;
                memcpy(&block, source, sizeof(block));
                ok = q6_k_dequantize_block(&block, destination);
                break;
            }
        }
        if (!ok) {
            set_error(error, error_capacity, "Qwen2 packed embedding block decoding failed.");
            return 0;
        }
    }
    if (error != nullptr && error_capacity > 0) error[0] = '\0';
    return 1;
}

static int qwen2_lookup_embedding(
    const GgufFile *file,
    int token_id,
    int dim,
    float *output,
    char *error,
    int error_capacity
) {
    if (file == nullptr || output == nullptr || dim <= 0) {
        set_error(error, error_capacity, "Qwen2 embedding lookup arguments are invalid.");
        return 0;
    }

    const GgufTensorInfo *tensor = gguf_find_tensor(file, "token_embd.weight");
    if (tensor == nullptr) {
        set_error(error, error_capacity, "Qwen2 token_embd.weight is missing.");
        return 0;
    }

    if (tensor->dimension_count != 2 ||
        tensor->dimensions[0] != (uint64_t)dim) {
        set_error(error, error_capacity,
                  "Qwen2 token_embd.weight must be a matrix with dim values per row.");
        return 0;
    }
    if (token_id < 0 || (uint64_t)token_id >= tensor->dimensions[1]) {
        set_error(error, error_capacity, "Qwen2 embedding token ID is out of range.");
        return 0;
    }
    if (tensor->type != ENGINE_GGML_TYPE_F32) {
        GgufQuantizedTensorView quantized_view;
        if (!gguf_get_quantized_tensor_view(file, "token_embd.weight", &quantized_view)) {
            set_error(error, error_capacity,
                      "Qwen2 token_embd.weight quantized data is invalid or unsupported.");
            return 0;
        }

        return qwen2_decode_quantized_embedding_row(
            &quantized_view, token_id, dim, output, error, error_capacity);
    }

    GgufF32TensorView view;
    int fn_ret = gguf_get_f32_tensor_view(file, "token_embd.weight", &view);
    if (!fn_ret) {
        set_error(error, error_capacity, "Qwen2 token_embd.weight F32 data is invalid.");
        return 0;
    }
    const float *row_start = view.data + size_t(token_id) * dim;
    memcpy(output, row_start, (size_t)dim * sizeof(float));
    if (error != nullptr && error_capacity > 0) error[0] = '\0';
    return 1;
}

/* Load one layer's F32 normalization or bias vector. The view borrows GGUF data. */
static int qwen2_get_layer_vector(
    const GgufFile *file,
    int layer,
    const char *suffix,
    int length,
    GgufF32TensorView *view,
    char *error,
    int error_capacity
)
{
    char name[96];

    if (view != nullptr) *view = {};
    if (file == nullptr || layer < 0 || suffix == nullptr || length <= 0 ||
        view == nullptr) {
        set_error(error, error_capacity, "Qwen2 layer-vector arguments are invalid.");
        return 0;
    }
    int name_length = snprintf(name, sizeof(name), "blk.%d.%s", layer, suffix);
    if (name_length < 0 || (size_t)name_length >= sizeof(name)) {
        set_error(error, error_capacity, "Qwen2 layer tensor name is too long.");
        return 0;
    }
    if (!gguf_get_f32_tensor_view(file, name, view) ||
        view->dimension_count != 1 || view->dimensions[0] != (uint64_t)length) {
        if (error != nullptr && error_capacity > 0) {
            snprintf(error, (size_t)error_capacity,
                     "Qwen2 tensor %s must be a valid F32 vector of length %d.",
                     name, length);
        }
        *view = {};
        return 0;
    }
    return 1;
}

int qwen2_weight_matvec(
    const GgufFile *file,
    const char *name,
    const float *input,
    int input_count,
    float *output,
    int output_count,
    char *error,
    int error_capacity
)
{
    GgufF32TensorView view;

    if (file == nullptr || name == nullptr || input == nullptr ||
        output == nullptr || input_count <= 0 || output_count <= 0 || input == output) {
        set_error(error, error_capacity, "Qwen2 weight-matvec arguments are invalid.");
        return 0;
    }
    const GgufTensorInfo *tensor = gguf_find_tensor(file, name);
    if (!tensor_has_shape(tensor, (uint64_t)input_count, (uint64_t)output_count)) {
        if (error != nullptr && error_capacity > 0) {
            snprintf(error, (size_t)error_capacity,
                     "Qwen2 tensor %s is missing or has an invalid matrix shape.", name);
        }
        return 0;
    }
    if (tensor->type != ENGINE_GGML_TYPE_F32) {
        GgufQuantizedTensorView packed;
        if (!gguf_get_quantized_tensor_view(file, name, &packed)) {
            if (error != nullptr && error_capacity > 0) {
                snprintf(error, (size_t)error_capacity,
                         "Qwen2 packed matrix %s has invalid or unsupported data.", name);
            }
            return 0;
        }
        if (!quantized_tensor_matvec_f32(&packed, input, input_count, output, output_count)) {
            if (error != nullptr && error_capacity > 0) {
                snprintf(error, (size_t)error_capacity,
                         "Qwen2 packed matrix-vector multiplication failed for %s.", name);
            }
            return 0;
        }
        if (error != nullptr && error_capacity > 0) error[0] = '\0';
        return 1;
    }
    if (!gguf_get_f32_tensor_view(file, name, &view)) {
        if (error != nullptr && error_capacity > 0) {
            snprintf(error, (size_t)error_capacity,
                     "Qwen2 tensor %s must be a valid F32 matrix with %d columns and %d rows.",
                     name, input_count, output_count);
        }
        return 0;
    }
    if (!tensor_matvec_f32(view.data, output_count, input_count, input, output)) {
        if (error != nullptr && error_capacity > 0) {
            snprintf(error, (size_t)error_capacity,
                     "Qwen2 matrix-vector multiplication failed for %s.", name);
        }
        return 0;
    }
    return 1;
}

/* Add the layer prefix, then use the shared model-weight dispatcher. */
static int qwen2_layer_matvec(
    const GgufFile *file,
    int layer,
    const char *suffix,
    const float *input,
    int input_count,
    float *output,
    int output_count,
    char *error,
    int error_capacity
)
{
    char name[96];
    if (layer < 0 || suffix == nullptr) {
        set_error(error, error_capacity, "Qwen2 layer-matvec arguments are invalid.");
        return 0;
    }
    int name_length = snprintf(name, sizeof(name), "blk.%d.%s", layer, suffix);
    if (name_length < 0 || (size_t)name_length >= sizeof(name)) {
        set_error(error, error_capacity, "Qwen2 layer tensor name is too long.");
        return 0;
    }
    return qwen2_weight_matvec(file, name, input, input_count, output, output_count,
                               error, error_capacity);
}

/* Apply one named projection and bias, using caller-owned scratch for the sum. */
static int qwen2_layer_project(
    const GgufFile *file,
    int layer,
    const char *projection,
    const float *input,
    int input_count,
    float *output,
    int output_count,
    float *scratch,
    char *error,
    int error_capacity
)
{
    char weight_suffix[80];
    char bias_suffix[80];
    GgufF32TensorView bias;

    if (projection == nullptr || scratch == nullptr ||
        scratch == input || scratch == output) {
        set_error(error, error_capacity, "Qwen2 projection arguments are invalid.");
        return 0;
    }
    int weight_length = snprintf(weight_suffix, sizeof(weight_suffix), "%s.weight", projection);
    int bias_length = snprintf(bias_suffix, sizeof(bias_suffix), "%s.bias", projection);
    if (weight_length < 0 || (size_t)weight_length >= sizeof(weight_suffix) ||
        bias_length < 0 || (size_t)bias_length >= sizeof(bias_suffix)) {
        set_error(error, error_capacity, "Qwen2 projection name is too long.");
        return 0;
    }
    if (!qwen2_layer_matvec(file, layer, weight_suffix, input, input_count,
                            output, output_count, error, error_capacity) ||
        !qwen2_get_layer_vector(file, layer, bias_suffix, output_count,
                                &bias, error, error_capacity)) {
        return 0;
    }
    if (!tensor_add_f32(output, bias.data, output_count, scratch)) {
        if (error != nullptr && error_capacity > 0) {
            snprintf(error, (size_t)error_capacity,
                     "Qwen2 bias addition failed for blk.%d.%s.", layer, projection);
        }
        return 0;
    }
    memcpy(output, scratch, (size_t)output_count * sizeof(float));
    return 1;
}

static int qwen2_forward_token(
    const Model *model,
    InferenceState *state,
    int token_id,
    char *error,
    int error_capacity
)
{
    if (model == nullptr || state == nullptr || model->implementation == nullptr ||
        state->implementation == nullptr || state->logits == nullptr) {
        set_error(error, error_capacity, "Qwen2 model or inference state is not initialized.");
        return 0;
    }
    if (token_id < 0 || token_id >= model->config.vocab_size) {
        set_error(error, error_capacity, "Qwen2 input token ID is out of range.");
        return 0;
    }

    Qwen2InferenceState *private_state = (Qwen2InferenceState *)state->implementation;
    const EngineModel *implementation = (const EngineModel *)model->implementation;
    const GgufFile *file = &implementation->gguf;
    GgufF32TensorView view;
    int fn_ret;

    int position = private_state->token_count;
    if (position >= private_state->max_context_tokens) {
        set_error(error, error_capacity, "Qwen2 context capacity is exhausted.");
        return 0;
    }
    OperationProfile *profile = &state->profile;
    OperationTimer forward_timer(profile->enabled, &profile->forward_seconds);
    private_state->token_ids[position] = token_id;
    int prefix_length = position + 1;

    for (int p = 0; p < prefix_length; p++) {
        fn_ret = qwen2_lookup_embedding(
            file, private_state->token_ids[p], model->config.dim, 
            private_state->activations + (size_t)p * model->config.dim,
            error, error_capacity
        );
        if (!fn_ret) {
            return 0;
        }
    }

    // Calculate causal attention
    for (int layer = 0; layer < model->config.n_layers; layer++) {
        for (int i = 0; i < prefix_length; i++) {
            float *act_vector_start = private_state->activations + (size_t)i * model->config.dim;
            fn_ret = qwen2_get_layer_vector(
                file, layer, "attn_norm.weight", model->config.dim,
                &view, error, error_capacity
            );
            if (!fn_ret) {
                return 0;
            }
            {
                OperationTimer timer(profile->enabled, &profile->rmsnorm_seconds);
                fn_ret = tensor_rms_norm_f32(
                    act_vector_start, view.data,
                    model->config.dim, model->config.rms_norm_epsilon,
                    private_state->normalized
                );
            }
            if (!fn_ret) {
                set_error(error, error_capacity, "Qwen2 attention RMSNorm failed.");
                return 0;
            }

            // Project the query vector, then add its bias.
            {
                OperationTimer timer(profile->enabled, &profile->matvec_seconds);
                fn_ret = qwen2_layer_project(
                    file, layer, "attn_q",
                    private_state->normalized, model->config.dim,
                    private_state->query, model->config.dim,
                    private_state->matvec_output, error, error_capacity
                );
            }
            if (!fn_ret) {
                return 0;
            }
            // Project the key vector, then add its bias.
            {
                OperationTimer timer(profile->enabled, &profile->matvec_seconds);
                fn_ret = qwen2_layer_project(
                    file, layer, "attn_k",
                    private_state->normalized, model->config.dim,
                    private_state->key, private_state->key_value_dimension,
                    private_state->matvec_output, error, error_capacity
                );
            }
            if (!fn_ret) {
                return 0;
            }
            // Project the value vector, then add its bias.
            {
                OperationTimer timer(profile->enabled, &profile->matvec_seconds);
                fn_ret = qwen2_layer_project(
                    file, layer, "attn_v",
                    private_state->normalized, model->config.dim,
                    private_state->value, private_state->key_value_dimension,
                    private_state->matvec_output, error, error_capacity
                );
            }
            if (!fn_ret) {
                return 0;
            }
            // Applying RoPE
            {
                OperationTimer timer(profile->enabled, &profile->rope_seconds);
                fn_ret = transformer_apply_rope_f32(
                    private_state->query, model->config.n_heads,
                    private_state->key,
                    model->config.n_kv_heads,
                    private_state->head_dimension,
                    private_state->head_dimension,
                    i, model->config.rope_frequency_base
                );
            }
            if (!fn_ret) {
                set_error(error, error_capacity, "Qwen2 RoPE rotation failed.");
                return 0;
            }
            
            // Store back to our state
            size_t size = (size_t)private_state->key_value_dimension * sizeof(float);
            size_t offset = (size_t)i * private_state->key_value_dimension;
            float *dest = private_state->keys + offset;
            memcpy(dest, private_state->key, size);

            dest = private_state->values + offset;
            memcpy(dest, private_state->value, size);

            // apply attention
            {
                OperationTimer timer(profile->enabled, &profile->attention_seconds);
                fn_ret = transformer_attention_f32(
                    private_state->query,
                    model->config.n_heads,
                    model->config.n_kv_heads,
                    private_state->head_dimension,
                    private_state->keys,
                    private_state->values,
                    i + 1,
                    private_state->attention_probabilities,
                    private_state->attention_output
                );
            }
            if (!fn_ret) {
                set_error(error, error_capacity, "Qwen2 grouped-query attention failed.");
                return 0;
            }

            // Updating token activation
            {
                OperationTimer timer(profile->enabled, &profile->matvec_seconds);
                fn_ret = qwen2_layer_matvec(
                    file, layer, "attn_output.weight",
                    private_state->attention_output, model->config.dim,
                    private_state->matvec_output, model->config.dim,
                    error, error_capacity
                );
            }
            if (!fn_ret) {
                return 0;
            }

            // add to private_state->normalized (temporarily)
            fn_ret = tensor_add_f32(
                private_state->matvec_output, act_vector_start, 
                model->config.dim, private_state->normalized);
            if (!fn_ret) {
                set_error(error, error_capacity, "Qwen2 attention residual addition failed.");
                return 0;
            }
            memcpy(
                act_vector_start, 
                private_state->normalized,
                (size_t)model->config.dim * sizeof(float)
            );

            // Feed-forward network
            fn_ret = qwen2_get_layer_vector(
                file, layer, "ffn_norm.weight", model->config.dim,
                &view, error, error_capacity
            );
            if (!fn_ret) {
                return 0;
            }
            {
                OperationTimer timer(profile->enabled, &profile->rmsnorm_seconds);
                fn_ret = tensor_rms_norm_f32(
                    act_vector_start,
                    view.data,
                    model->config.dim,
                    model->config.rms_norm_epsilon,
                    private_state->normalized
                );
            }
            if (!fn_ret) {
                set_error(error, error_capacity, "Qwen2 FFN RMSNorm failed.");
                return 0;
            }

            {
                OperationTimer timer(profile->enabled, &profile->matvec_seconds);
                fn_ret = qwen2_layer_matvec(
                    file, layer, "ffn_gate.weight",
                    private_state->normalized, model->config.dim,
                    private_state->ffn_gate, model->config.hidden_dim,
                    error, error_capacity
                );
            }
            if (!fn_ret) {
                return 0;
            }

            {
                OperationTimer timer(profile->enabled, &profile->matvec_seconds);
                fn_ret = qwen2_layer_matvec(
                    file, layer, "ffn_up.weight",
                    private_state->normalized, model->config.dim,
                    private_state->ffn_up, model->config.hidden_dim,
                    error, error_capacity
                );
            }
            if (!fn_ret) {
                return 0;
            }

            fn_ret = transformer_swiglu_f32(
                private_state->ffn_gate, 
                private_state->ffn_up, 
                model->config.hidden_dim
            );
            if (!fn_ret) {
                set_error(error, error_capacity, "Qwen2 SwiGLU failed.");
                return 0;
            }
            
            // matvec_output = Wdown * ffn_gate
            {
                OperationTimer timer(profile->enabled, &profile->matvec_seconds);
                fn_ret = qwen2_layer_matvec(
                    file, layer, "ffn_down.weight",
                    private_state->ffn_gate, model->config.hidden_dim,
                    private_state->matvec_output, model->config.dim,
                    error, error_capacity
                );
            }
            if (!fn_ret) {
                return 0;
            }

            // x + matvec_output → temporary buffer
            fn_ret = tensor_add_f32(
                act_vector_start,
                private_state->matvec_output,
                model->config.dim,
                private_state->normalized
            );
            if (!fn_ret) {
                set_error(error, error_capacity, "Qwen2 FFN residual addition failed.");
                return 0;
            }

            // Store the completed sum back into x.
            memcpy(
                act_vector_start,
                private_state->normalized,
                (size_t)model->config.dim * sizeof(float)
            );

        }
    }

    // Only the LAST position supplies next-token logits.
    //normalized = rms_norm(activations[last], output_norm_weight);
    const float *last_activation =
        private_state->activations +
        (size_t)(prefix_length - 1) * model->config.dim;

    if (!gguf_get_f32_tensor_view(file, "output_norm.weight", &view) ||
        view.dimension_count != 1 ||
        view.dimensions[0] != (uint64_t)model->config.dim) {
        set_error(error, error_capacity, "Invalid output_norm.weight.");
        return 0;
    }

    {
        OperationTimer timer(profile->enabled, &profile->rmsnorm_seconds);
        fn_ret = tensor_rms_norm_f32(
            last_activation, view.data,
            model->config.dim, model->config.rms_norm_epsilon,
            private_state->normalized
        );
    }
    if (!fn_ret) {
        set_error(error, error_capacity, "Final RMSNorm failed.");
        return 0;
    }


    {
        OperationTimer timer(profile->enabled, &profile->output_seconds);
        fn_ret = qwen2_weight_matvec(
            file, "output.weight",
            private_state->normalized, model->config.dim,
            state->logits, model->config.vocab_size,
            error, error_capacity
        );
    }
    if (!fn_ret) {
        return 0;
    }

    // Check every logit is finite before committing success.
    for (int j = 0; j < model->config.vocab_size; ++j) {
        if (!isfinite(state->logits[j])) {
            set_error(error, error_capacity, "Output contains a non-finite logit.");
            return 0;
        }
    }

    if (profile->enabled) ++profile->forward_calls;
    state->logits_count = model->config.vocab_size;
    private_state->token_count = prefix_length;
    if (error != nullptr && error_capacity > 0) error[0] = '\0';
    return 1;
}

static void qwen2_state_destroy(InferenceState *state)
{
    Qwen2InferenceState *private_state;
    if (state == nullptr) return;
    private_state = (Qwen2InferenceState *)state->implementation;
    if (private_state != nullptr) {
        free(private_state->token_ids);
        free(private_state->activations);
        free(private_state->keys);
        free(private_state->values);
        free(private_state->normalized);
        free(private_state->query);
        free(private_state->key);
        free(private_state->value);
        free(private_state->attention_probabilities);
        free(private_state->attention_output);
        free(private_state->ffn_gate);
        free(private_state->ffn_up);
        free(private_state->matvec_output);
        free(private_state);
    }
    free(state->logits);
    state->implementation = nullptr;
    state->logits = nullptr;
    state->logits_count = 0;
    memset(&state->profile, 0, sizeof(state->profile));
}

static int qwen2_format_prompt(
    const char *user_prompt, char *output, int output_capacity,
    char *error, int error_capacity)
{
    if (output != nullptr && output_capacity > 0) output[0] = '\0';
    if (user_prompt == nullptr || output == nullptr || output_capacity <= 0) {
        set_error(error, error_capacity, "Qwen2 prompt formatting arguments are invalid.");
        return 0;
    }
    /* Single-user rendering of the local Qwen2.5-Instruct chat template. */
    static const char prefix[] =
        "<|im_start|>system\n"
        "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n";
    static const char suffix[] = "<|im_end|>\n<|im_start|>assistant\n";
    const size_t prefix_length = sizeof(prefix) - 1;
    const size_t suffix_length = sizeof(suffix) - 1;
    const size_t user_length = strlen(user_prompt);
    const size_t overhead = prefix_length + suffix_length + 1;
    /* Subtract from capacity before adding the arbitrary user string length. */
    if ((size_t)output_capacity < overhead ||
        user_length > (size_t)output_capacity - overhead) {
        set_error(error, error_capacity, "Qwen2 formatted prompt exceeds output capacity.");
        return 0;
    }
    memcpy(output, prefix, prefix_length);
    memcpy(output + prefix_length, user_prompt, user_length);
    memcpy(output + prefix_length + user_length, suffix, suffix_length + 1);
    if (error != nullptr && error_capacity > 0) error[0] = '\0';
    return 1;
}

const ArchitectureOps *qwen2_architecture_ops(void)
{
    static const ArchitectureOps ops = {
        "qwen2",
        qwen2_configure_model,
        qwen2_destroy_model_data,
        qwen2_state_init,
        qwen2_forward_token,
        qwen2_state_destroy,
        qwen2_format_prompt
    };

    return &ops;
}
