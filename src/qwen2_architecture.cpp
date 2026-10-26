#include "qwen2_architecture.h"

#include "qwen2_internal.h"
#include "quantization.h"
#include "matrix_backend.h"
#include "operation_timer.h"
#include "tensor.h"
#include "transformer_ops.h"

#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* The public weight helper has no state parameter. Scope its optional profiling
 * context to one forward call; thread_local keeps concurrent requests separate. */
static thread_local OperationProfile *active_weight_profile = nullptr;

class WeightProfileScope {
public:
    explicit WeightProfileScope(OperationProfile *profile)
        : previous_(active_weight_profile)
    {
        active_weight_profile = profile->enabled ? profile : nullptr;
    }
    ~WeightProfileScope() { active_weight_profile = previous_; }
    WeightProfileScope(const WeightProfileScope &) = delete;
    WeightProfileScope &operator=(const WeightProfileScope &) = delete;
private:
    OperationProfile *previous_;
};

static int weight_format_index(uint32_t type)
{
    switch (type) {
        case ENGINE_GGML_TYPE_F32: return PROFILE_F32;
        case ENGINE_GGML_TYPE_Q5_0: return PROFILE_Q5_0;
        case ENGINE_GGML_TYPE_Q8_0: return PROFILE_Q8_0;
        case ENGINE_GGML_TYPE_Q4_K: return PROFILE_Q4_K;
        case ENGINE_GGML_TYPE_Q6_K: return PROFILE_Q6_K;
        default: return -1;
    }
}

static void weight_profile_targets(uint32_t type, const char *name,
                                   double **seconds, unsigned long long **calls)
{
    *seconds = nullptr;
    *calls = nullptr;
    if (active_weight_profile == nullptr) return;
    int index = weight_format_index(type);
    if (index < 0) return;
    bool output = strcmp(name, "output.weight") == 0;
    *seconds = output ? &active_weight_profile->weight_output_seconds[index]
                      : &active_weight_profile->weight_layer_seconds[index];
    *calls = output ? &active_weight_profile->weight_output_calls[index]
                    : &active_weight_profile->weight_layer_calls[index];
}

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
    if (context > SIZE_MAX / dim || context * dim > SIZE_MAX / sizeof(float) ||
        context > SIZE_MAX / kv_dim || context * kv_dim > SIZE_MAX / sizeof(float) ||
        heads > SIZE_MAX / context || heads * context > SIZE_MAX / sizeof(float) ||
        max_projection > SIZE_MAX / QWEN2_PREFILL_CHUNK_SIZE ||
        max_projection * QWEN2_PREFILL_CHUNK_SIZE > SIZE_MAX / sizeof(float) ||
        dim > SIZE_MAX / QWEN2_PREFILL_CHUNK_SIZE ||
        hidden > SIZE_MAX / QWEN2_PREFILL_CHUNK_SIZE ||
        kv_dim > SIZE_MAX / QWEN2_PREFILL_CHUNK_SIZE) {
        set_error(error, error_capacity, "Qwen2 inference-state allocation size overflows.");
        return 0;
    }

    // Number of floats needed for one layer.
    size_t layer_stride = context * kv_dim;
    size_t layers = (size_t)config.n_layers;

    // Check the multiplication before performing it.
    if (layers > SIZE_MAX / layer_stride) {
        set_error(error, error_capacity,
                "Qwen2 KV-cache element count overflows.");
        return 0;
    }

    // Number of floats needed for all layers, in EACH cache.
    size_t cache_count = layers * layer_stride;

    // calloc also needs the total byte count to fit.
    if (cache_count > SIZE_MAX / sizeof(float)) {
        set_error(error, error_capacity,
                "Qwen2 KV-cache byte size overflows.");
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
    ALLOC_FIELD(keys, cache_count, float);
    ALLOC_FIELD(values, cache_count, float);
    ALLOC_FIELD(normalized, QWEN2_PREFILL_CHUNK_SIZE * dim, float);
    ALLOC_FIELD(query, QWEN2_PREFILL_CHUNK_SIZE * dim, float);
    ALLOC_FIELD(key, QWEN2_PREFILL_CHUNK_SIZE * kv_dim, float);
    ALLOC_FIELD(value, QWEN2_PREFILL_CHUNK_SIZE * kv_dim, float);
    ALLOC_FIELD(attention_probabilities, heads * context, float);
    ALLOC_FIELD(attention_output, QWEN2_PREFILL_CHUNK_SIZE * dim, float);
    ALLOC_FIELD(ffn_gate, QWEN2_PREFILL_CHUNK_SIZE * hidden, float);
    ALLOC_FIELD(ffn_up, QWEN2_PREFILL_CHUNK_SIZE * hidden, float);
    ALLOC_FIELD(matvec_output, QWEN2_PREFILL_CHUNK_SIZE * max_projection, float);
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

/* A named GGUF tensor described for the matrix backend. `packed` is the storage
 * that `weight.packed` points to, so keep this struct in place (do not copy it). */
struct Qwen2ResolvedWeight {
    MatrixWeight weight;
    GgufQuantizedTensorView packed;
    uint32_t type;
};

/* Validate the tensor's shape/storage and describe it as a MatrixWeight. This
 * is the only place Qwen2 distinguishes dense F32 from packed weights. */
static int qwen2_resolve_weight(
    const GgufFile *file, const char *name, int input_count, int output_count,
    Qwen2ResolvedWeight *resolved, char *error, int error_capacity)
{
    const GgufTensorInfo *tensor = gguf_find_tensor(file, name);
    if (!tensor_has_shape(tensor, (uint64_t)input_count, (uint64_t)output_count)) {
        if (error != nullptr && error_capacity > 0) {
            snprintf(error, (size_t)error_capacity,
                     "Qwen2 tensor %s is missing or has an invalid matrix shape.", name);
        }
        return 0;
    }
    resolved->type = tensor->type;
    if (tensor->type == ENGINE_GGML_TYPE_F32) {
        GgufF32TensorView view;
        if (!gguf_get_f32_tensor_view(file, name, &view) ||
            !matrix_weight_dense(&resolved->weight, view.data, output_count, input_count)) {
            if (error != nullptr && error_capacity > 0) {
                snprintf(error, (size_t)error_capacity,
                         "Qwen2 tensor %s must be a valid F32 matrix with %d columns and %d rows.",
                         name, input_count, output_count);
            }
            return 0;
        }
        return 1;
    }
    if (!gguf_get_quantized_tensor_view(file, name, &resolved->packed) ||
        !matrix_weight_packed(&resolved->weight, &resolved->packed) ||
        resolved->weight.columns != input_count || resolved->weight.rows != output_count) {
        if (error != nullptr && error_capacity > 0) {
            snprintf(error, (size_t)error_capacity,
                     "Qwen2 packed matrix %s has invalid or unsupported data.", name);
        }
        return 0;
    }
    return 1;
}

static int qwen2_run_backend(
    const MatrixBackend *backend, const MatrixWeight *weight, const float *inputs,
    int batch_count, bool single_vector, float *outputs)
{
    return single_vector
        ? matrix_backend_matvec(backend, weight, inputs, outputs)
        : matrix_backend_matmul(backend, weight, inputs, batch_count, outputs);
}

/* Shared path for one vector (matvec) and a token-major batch (matmul): the same
 * argument checks, weight resolution, profiling, and error handling for both. */
static int qwen2_apply_weight(
    const GgufFile *file, const char *name, const float *inputs, int input_count,
    int batch_count, bool single_vector, float *outputs, int output_count,
    char *error, int error_capacity)
{
    if (file == nullptr || name == nullptr || inputs == nullptr || outputs == nullptr ||
        input_count <= 0 || output_count <= 0 || batch_count <= 0) {
        set_error(error, error_capacity,
                  single_vector ? "Qwen2 weight-matvec arguments are invalid."
                                : "Qwen2 weight-matmul arguments are invalid.");
        return 0;
    }
    Qwen2ResolvedWeight resolved;
    if (!qwen2_resolve_weight(file, name, input_count, output_count,
                              &resolved, error, error_capacity)) {
        return 0;
    }

    const MatrixBackend *backend = matrix_backend_default();
    int succeeded;
    if (active_weight_profile == nullptr) {
        // No clock reads or per-format bookkeeping on the unprofiled path.
        succeeded = qwen2_run_backend(backend, &resolved.weight, inputs,
                                      batch_count, single_vector, outputs);
    } else {
        double *seconds;
        unsigned long long *calls;
        weight_profile_targets(resolved.type, name, &seconds, &calls);
        {
            OperationTimer timer(seconds != nullptr, seconds);
            succeeded = qwen2_run_backend(backend, &resolved.weight, inputs,
                                          batch_count, single_vector, outputs);
        }
        if (succeeded && calls != nullptr) ++*calls;
    }
    if (!succeeded) {
        if (error != nullptr && error_capacity > 0) {
            snprintf(error, (size_t)error_capacity,
                     "Qwen2 %s%s multiplication failed for %s.",
                     resolved.weight.kind == MATRIX_WEIGHT_PACKED ? "packed " : "",
                     single_vector ? "matrix-vector" : "batched matrix", name);
        }
        return 0;
    }
    if (error != nullptr && error_capacity > 0) error[0] = '\0';
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
    return qwen2_apply_weight(file, name, input, input_count, 1, true,
                              output, output_count, error, error_capacity);
}

static int qwen2_weight_matmul(
    const GgufFile *file, const char *name, const float *inputs,
    int input_count, int batch_count, float *outputs, int output_count,
    char *error, int error_capacity)
{
    return qwen2_apply_weight(file, name, inputs, input_count, batch_count, false,
                              outputs, output_count, error, error_capacity);
}

static int qwen2_layer_matmul(
    const GgufFile *file, int layer, const char *suffix,
    const float *inputs, int input_count, int batch_count,
    float *outputs, int output_count, char *error, int error_capacity)
{
    char name[96];
    if (layer < 0 || suffix == nullptr) return 0;
    int length = snprintf(name, sizeof(name), "blk.%d.%s", layer, suffix);
    if (length < 0 || (size_t)length >= sizeof(name)) {
        set_error(error, error_capacity, "Qwen2 layer tensor name is too long.");
        return 0;
    }
    return qwen2_weight_matmul(file, name, inputs, input_count, batch_count,
                              outputs, output_count, error, error_capacity);
}

static int qwen2_layer_project_batch(
    const GgufFile *file, int layer, const char *projection,
    const float *inputs, int input_count, int batch_count,
    float *outputs, int output_count, char *error, int error_capacity)
{
    char weight[80], bias_name[80];
    GgufF32TensorView bias;
    if (projection == nullptr) return 0;
    int wl = snprintf(weight, sizeof(weight), "%s.weight", projection);
    int bl = snprintf(bias_name, sizeof(bias_name), "%s.bias", projection);
    if (wl < 0 || bl < 0 || (size_t)wl >= sizeof(weight) || (size_t)bl >= sizeof(bias_name) ||
        !qwen2_layer_matmul(file, layer, weight, inputs, input_count, batch_count,
                           outputs, output_count, error, error_capacity) ||
        !qwen2_get_layer_vector(file, layer, bias_name, output_count,
                                &bias, error, error_capacity)) return 0;
    for (int batch = 0; batch < batch_count; ++batch)
        for (int i = 0; i < output_count; ++i)
            outputs[(size_t)batch * output_count + i] += bias.data[i];
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

int qwen2_kv_cache_layer_view(
    const Model *model,
    Qwen2InferenceState *private_state,
    int layer,
    int position,
    Qwen2KvCacheLayerView *view,
    char *error,
    int error_capacity
)
{
    if (view != nullptr) memset(view, 0, sizeof(*view));

    if (model == nullptr || private_state == nullptr || view == nullptr ||
        private_state->keys == nullptr || private_state->values == nullptr) {
        set_error(error, error_capacity, "Qwen2 KV-cache storage or view is not initialized.");
        return 0;
    }
    const ModelConfig &config = model->config;
    if (config.dim <= 0 || config.n_layers <= 0 || config.n_heads <= 0 ||
        config.n_kv_heads <= 0 || config.n_kv_heads > config.n_heads ||
        config.dim % config.n_heads != 0 || config.n_heads % config.n_kv_heads != 0 ||
        private_state->max_context_tokens <= 0 ||
        private_state->max_context_tokens > config.seq_len ||
        private_state->head_dimension != config.dim / config.n_heads ||
        private_state->key_value_dimension !=
            (config.dim / config.n_heads) * config.n_kv_heads) {
        set_error(error, error_capacity, "Qwen2 KV-cache model and state dimensions are inconsistent.");
        return 0;
    }
    if (layer < 0 || layer >= config.n_layers ||
        position < 0 || position >= private_state->max_context_tokens) {
        set_error(error, error_capacity, "Qwen2 KV-cache layer or position is out of range.");
        return 0;
    }
    if ((size_t)private_state->max_context_tokens >
        SIZE_MAX / sizeof(float) / (size_t)private_state->key_value_dimension /
            (size_t)config.n_layers) {
        set_error(error, error_capacity, "Qwen2 KV-cache byte size overflows.");
        return 0;
    }
    size_t layer_stride = (size_t)private_state->max_context_tokens * (size_t)private_state->key_value_dimension;
    size_t layer_offset = (size_t)layer * layer_stride;
    size_t token_offset = (size_t)private_state->key_value_dimension * (size_t)position;
    view->keys = private_state->keys + layer_offset;
    view->values = private_state->values + layer_offset;
    view->key_slot = view->keys + token_offset;
    view->value_slot = view->values + token_offset;
    view->token_count = position + 1;
    if (error != nullptr && error_capacity > 0) error[0] = '\0';
    return 1;
}


int qwen2_forward_token_reference(
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
    WeightProfileScope weight_profile_scope(profile);
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

static int qwen2_forward_token_cached_impl(
    const Model *model,
    InferenceState *state,
    int token_id,
    bool produce_logits,
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
    WeightProfileScope weight_profile_scope(profile);
    int prefix_length = position + 1;

    fn_ret = qwen2_lookup_embedding(
        file, token_id, model->config.dim,
        private_state->activations,
        error, error_capacity
    );
    if (!fn_ret) {
        return 0;
    }

    // Calculate multi-head attention with Grouped-Query Attention (GQA)
    for (int layer = 0; layer < model->config.n_layers; layer++) {
        float *act_vector_start = private_state->activations;
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
                position, model->config.rope_frequency_base
            );
        }
        if (!fn_ret) {
            set_error(error, error_capacity, "Qwen2 RoPE rotation failed.");
            return 0;
        }

        // Store back to our state
        Qwen2KvCacheLayerView cache_view;
        if (!qwen2_kv_cache_layer_view(
                model, private_state, layer, position, &cache_view,
                error, error_capacity)) {
            return 0;
        }
        size_t size = (size_t)private_state->key_value_dimension * sizeof(float);
        memcpy(cache_view.key_slot, private_state->key, size);
        memcpy(cache_view.value_slot, private_state->value, size);

        // apply attention
        {
            OperationTimer timer(profile->enabled, &profile->attention_seconds);
            fn_ret = transformer_attention_f32(
                private_state->query,
                model->config.n_heads,
                model->config.n_kv_heads,
                private_state->head_dimension,
                cache_view.keys,
                cache_view.values,
                cache_view.token_count,
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

    // Intermediate prompt tokens still populate their layer KV slots, but
    // their final norm and vocabulary projection would never be used.
    if (!produce_logits) {
        if (profile->enabled) ++profile->forward_calls;
        state->logits_count = 0;
        private_state->token_ids[position] = token_id;
        private_state->token_count = prefix_length;
        if (error != nullptr && error_capacity > 0) error[0] = '\0';
        return 1;
    }

    // Only the last prompt token and later generated tokens need next-token logits.
    if (!gguf_get_f32_tensor_view(file, "output_norm.weight", &view) ||
        view.dimension_count != 1 ||
        view.dimensions[0] != (uint64_t)model->config.dim) {
        set_error(error, error_capacity, "Invalid output_norm.weight.");
        return 0;
    }

    {
        OperationTimer timer(profile->enabled, &profile->rmsnorm_seconds);
        fn_ret = tensor_rms_norm_f32(
            private_state->activations, view.data,
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
    private_state->token_ids[position] = token_id;
    private_state->token_count = prefix_length;
    if (error != nullptr && error_capacity > 0) error[0] = '\0';
    return 1;
}

int qwen2_forward_token_cached(const Model *model, InferenceState *state,
                               int token_id, char *error, int error_capacity)
{
    return qwen2_forward_token_cached_impl(model, state, token_id, true,
                                          error, error_capacity);
}

static int qwen2_forward_token_no_logits(const Model *model, InferenceState *state,
                                         int token_id, char *error, int error_capacity)
{
    return qwen2_forward_token_cached_impl(model, state, token_id, false,
                                          error, error_capacity);
}

/* Layer-first chunked prefill. Packed blocks are decoded once per output row
 * and reused across all activation vectors in the chunk. K/V retains the same
 * [layer][absolute position][KV head][head dimension] layout as decode. */
static int qwen2_prefill_tokens(const Model *model, InferenceState *state,
                               const int *ids, int count, char *error, int error_capacity)
{
    if (model == nullptr || model->implementation == nullptr || state == nullptr ||
        state->implementation == nullptr || state->logits == nullptr ||
        ids == nullptr || count <= 0 || model->config.vocab_size <= 0) {
        set_error(error, error_capacity, "Qwen2 prefill arguments are invalid.");
        return 0;
    }
    Qwen2InferenceState *s = (Qwen2InferenceState *)state->implementation;
    const EngineModel *implementation = (const EngineModel *)model->implementation;
    const GgufFile *file = &implementation->gguf;
    const int old_count = s->token_count;
    const int dim = model->config.dim, hidden = model->config.hidden_dim;
    const int kv_dim = s->key_value_dimension;
    if (old_count < 0 || old_count > s->max_context_tokens ||
        count > s->max_context_tokens - old_count) {
        set_error(error, error_capacity, "Qwen2 prefill context capacity is exhausted.");
        return 0;
    }
    for (int i = 0; i < count; ++i) {
        if (ids[i] < 0 || ids[i] >= model->config.vocab_size) {
            set_error(error, error_capacity, "Qwen2 prefill token ID is out of range.");
            return 0;
        }
    }
    /* The single-token NEON path is already the best decode-shaped kernel. */
    if (count == 1)
        return qwen2_forward_token_cached_impl(model, state, ids[0], true,
                                               error, error_capacity);

    OperationProfile *profile = &state->profile;
    OperationTimer forward_timer(profile->enabled, &profile->forward_seconds);
    WeightProfileScope weight_scope(profile);
    const float *last_activation = nullptr;

    for (int chunk_start = 0; chunk_start < count; chunk_start += QWEN2_PREFILL_CHUNK_SIZE) {
        int batch = count - chunk_start;
        if (batch > QWEN2_PREFILL_CHUNK_SIZE) batch = QWEN2_PREFILL_CHUNK_SIZE;
        for (int b = 0; b < batch; ++b) {
            if (!qwen2_lookup_embedding(file, ids[chunk_start + b], dim,
                                        s->activations + (size_t)b * dim,
                                        error, error_capacity)) goto failure;
        }

        for (int layer = 0; layer < model->config.n_layers; ++layer) {
            GgufF32TensorView norm;
            if (!qwen2_get_layer_vector(file, layer, "attn_norm.weight", dim,
                                        &norm, error, error_capacity)) goto failure;
            {
                OperationTimer timer(profile->enabled, &profile->rmsnorm_seconds);
                for (int b = 0; b < batch; ++b)
                    if (!tensor_rms_norm_f32(s->activations + (size_t)b * dim, norm.data,
                                             dim, model->config.rms_norm_epsilon,
                                             s->normalized + (size_t)b * dim)) goto failure;
            }
            {
                OperationTimer timer(profile->enabled, &profile->matvec_seconds);
                if (!qwen2_layer_project_batch(file, layer, "attn_q", s->normalized, dim,
                                               batch, s->query, dim, error, error_capacity) ||
                    !qwen2_layer_project_batch(file, layer, "attn_k", s->normalized, dim,
                                               batch, s->key, kv_dim, error, error_capacity) ||
                    !qwen2_layer_project_batch(file, layer, "attn_v", s->normalized, dim,
                                               batch, s->value, kv_dim, error, error_capacity))
                    goto failure;
            }
            for (int b = 0; b < batch; ++b) {
                int position = old_count + chunk_start + b;
                {
                    OperationTimer timer(profile->enabled, &profile->rope_seconds);
                    if (!transformer_apply_rope_f32(
                            s->query + (size_t)b * dim, model->config.n_heads,
                            s->key + (size_t)b * kv_dim, model->config.n_kv_heads,
                            s->head_dimension, s->head_dimension, position,
                            model->config.rope_frequency_base)) goto failure;
                }
                Qwen2KvCacheLayerView cache;
                if (!qwen2_kv_cache_layer_view(model, s, layer, position, &cache,
                                               error, error_capacity)) goto failure;
                memcpy(cache.key_slot, s->key + (size_t)b * kv_dim,
                       (size_t)kv_dim * sizeof(float));
                memcpy(cache.value_slot, s->value + (size_t)b * kv_dim,
                       (size_t)kv_dim * sizeof(float));
            }
            /* Causality is expressed by position+1: later K/V slots have been
             * written but are deliberately outside each query's visible prefix. */
            for (int b = 0; b < batch; ++b) {
                int position = old_count + chunk_start + b;
                Qwen2KvCacheLayerView cache;
                if (!qwen2_kv_cache_layer_view(model, s, layer, position, &cache,
                                               error, error_capacity)) goto failure;
                OperationTimer timer(profile->enabled, &profile->attention_seconds);
                if (!transformer_attention_f32(
                        s->query + (size_t)b * dim, model->config.n_heads,
                        model->config.n_kv_heads, s->head_dimension,
                        cache.keys, cache.values, position + 1,
                        s->attention_probabilities,
                        s->attention_output + (size_t)b * dim)) goto failure;
            }
            {
                OperationTimer timer(profile->enabled, &profile->matvec_seconds);
                if (!qwen2_layer_matmul(file, layer, "attn_output.weight",
                                        s->attention_output, dim, batch,
                                        s->matvec_output, dim, error, error_capacity)) goto failure;
            }
            for (int b = 0; b < batch; ++b)
                for (int i = 0; i < dim; ++i)
                    s->activations[(size_t)b * dim + i] += s->matvec_output[(size_t)b * dim + i];

            if (!qwen2_get_layer_vector(file, layer, "ffn_norm.weight", dim,
                                        &norm, error, error_capacity)) goto failure;
            {
                OperationTimer timer(profile->enabled, &profile->rmsnorm_seconds);
                for (int b = 0; b < batch; ++b)
                    if (!tensor_rms_norm_f32(s->activations + (size_t)b * dim, norm.data,
                                             dim, model->config.rms_norm_epsilon,
                                             s->normalized + (size_t)b * dim)) goto failure;
            }
            {
                OperationTimer timer(profile->enabled, &profile->matvec_seconds);
                if (!qwen2_layer_matmul(file, layer, "ffn_gate.weight", s->normalized,
                                        dim, batch, s->ffn_gate, hidden, error, error_capacity) ||
                    !qwen2_layer_matmul(file, layer, "ffn_up.weight", s->normalized,
                                        dim, batch, s->ffn_up, hidden, error, error_capacity))
                    goto failure;
            }
            for (int b = 0; b < batch; ++b)
                if (!transformer_swiglu_f32(s->ffn_gate + (size_t)b * hidden,
                                            s->ffn_up + (size_t)b * hidden, hidden)) goto failure;
            {
                OperationTimer timer(profile->enabled, &profile->matvec_seconds);
                if (!qwen2_layer_matmul(file, layer, "ffn_down.weight", s->ffn_gate,
                                        hidden, batch, s->matvec_output, dim,
                                        error, error_capacity)) goto failure;
            }
            for (int b = 0; b < batch; ++b)
                for (int i = 0; i < dim; ++i)
                    s->activations[(size_t)b * dim + i] += s->matvec_output[(size_t)b * dim + i];
        }
        last_activation = s->activations + (size_t)(batch - 1) * dim;
    }

    {
        GgufF32TensorView norm;
        if (!gguf_get_f32_tensor_view(file, "output_norm.weight", &norm) ||
            norm.dimension_count != 1 || norm.dimensions[0] != (uint64_t)dim) {
            set_error(error, error_capacity, "Invalid output_norm.weight.");
            goto failure;
        }
        {
            OperationTimer timer(profile->enabled, &profile->rmsnorm_seconds);
            if (!tensor_rms_norm_f32(last_activation, norm.data, dim,
                                     model->config.rms_norm_epsilon, s->normalized)) goto failure;
        }
        {
            OperationTimer timer(profile->enabled, &profile->output_seconds);
            if (!qwen2_weight_matvec(file, "output.weight", s->normalized, dim,
                                     state->logits, model->config.vocab_size,
                                     error, error_capacity)) goto failure;
        }
    }
    for (int i = 0; i < model->config.vocab_size; ++i)
        if (!isfinite(state->logits[i])) {
            set_error(error, error_capacity, "Output contains a non-finite logit.");
            goto failure;
        }
    memcpy(s->token_ids + old_count, ids, (size_t)count * sizeof(int));
    s->token_count = old_count + count;
    state->logits_count = model->config.vocab_size;
    if (profile->enabled) profile->forward_calls += (unsigned long long)count;
    if (error != nullptr && error_capacity > 0) error[0] = '\0';
    return 1;

failure:
    s->token_count = old_count;
    state->logits_count = 0;
    if (error != nullptr && error_capacity > 0 && error[0] == '\0')
        set_error(error, error_capacity, "Qwen2 batched prefill failed.");
    return 0;
}

static void qwen2_state_destroy(InferenceState *state)
{
    Qwen2InferenceState *private_state;
    if (state == nullptr) return;
    private_state = (Qwen2InferenceState *)state->implementation;
    if (private_state != nullptr) {
        /* TODO(kv-cache/02): Keep cleanup safe after partial initialization.
         * Expanded contiguous keys/values still need one free each. Borrowed
         * layer-view pointers must never be freed. Clean up any separate
         * reference workspace if introduced.
         */
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
        qwen2_forward_token_cached,
        qwen2_state_destroy,
        qwen2_format_prompt,
        qwen2_forward_token_no_logits,
        qwen2_prefill_tokens
    };

    return &ops;
}
