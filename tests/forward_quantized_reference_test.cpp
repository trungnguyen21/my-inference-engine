#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "qwen2_architecture.h"
#include "qwen2_quantized_test_utils.h"

static const int DIM = 256;
static const int HIDDEN = 512;
static const int KV_DIM = 128;
static const int VOCAB = 3;
static const size_t DATA_CAPACITY = 4 * 1024 * 1024;
enum { PACK_EMBEDDING = 1, PACK_LAYERS = 2, PACK_OUTPUT = 4 };

typedef struct MixedForwardFixture {
    Model model;
    EngineModel implementation;
    GgufTensorInfo tensors[15];
    unsigned char *data;
    size_t used;
    int count;
} MixedForwardFixture;

static GgufTensorInfo *append_tensor(
    MixedForwardFixture *fixture, const char *name, uint32_t type,
    int columns, int rows, size_t bytes
)
{
    fixture->used = (fixture->used + 31) & ~(size_t)31;
    assert(fixture->count < 15 && fixture->used + bytes <= DATA_CAPACITY);
    GgufTensorInfo *tensor = &fixture->tensors[fixture->count++];
    tensor->name = (char *)name;
    tensor->type = type;
    tensor->dimension_count = rows == 0 ? 1 : 2;
    tensor->dimensions[0] = (uint64_t)columns;
    tensor->dimensions[1] = (uint64_t)rows;
    tensor->offset = fixture->used;
    tensor->byte_size = bytes;
    fixture->used += bytes;
    return tensor;
}

static void append_vector(MixedForwardFixture *fixture, const char *name, int length, int bias)
{
    GgufTensorInfo *tensor = append_tensor(
        fixture, name, ENGINE_GGML_TYPE_F32, length, 0, (size_t)length * sizeof(float));
    float *output = (float *)(fixture->data + tensor->offset);
    for (int i = 0; i < length; ++i) output[i] = bias ? (float)(i % 7 - 3) * 0.01f : 1.0f;
}

static void append_matrix(
    MixedForwardFixture *fixture, const char *name,
    uint32_t source_type, int columns, int rows, int packed
)
{
    int block_elements = fixture_block_elements(source_type);
    int block_bytes = fixture_block_bytes(source_type);
    assert(columns % block_elements == 0);
    size_t block_count = (size_t)columns * rows / block_elements;
    size_t bytes = packed ? block_count * block_bytes : (size_t)columns * rows * sizeof(float);
    GgufTensorInfo *tensor = append_tensor(
        fixture, name, packed ? source_type : ENGINE_GGML_TYPE_F32, columns, rows, bytes);
    unsigned char *storage = fixture->data + tensor->offset;
    for (size_t b = 0; b < block_count; ++b) {
        unsigned char encoded[Q6_K_BLOCK_BYTES];
        float expected = fixture_write_block(source_type, (int)b, 1, encoded);
        if (packed) {
            memcpy(storage + b * block_bytes, encoded, (size_t)block_bytes);
        } else {
            /* Independent dense oracle: the block's known constant value. */
            float *dense = (float *)storage;
            for (int j = 0; j < block_elements; ++j) dense[b * block_elements + j] = expected;
        }
    }
}

static void initialize_fixture(MixedForwardFixture *fixture, int packed_parts)
{
    memset(fixture, 0, sizeof(*fixture));
    fixture->data = (unsigned char *)calloc(DATA_CAPACITY, 1);
    assert(fixture->data != nullptr);
    int layers = packed_parts & PACK_LAYERS;
    append_matrix(fixture, "token_embd.weight", ENGINE_GGML_TYPE_Q5_0,
                  DIM, VOCAB, packed_parts & PACK_EMBEDDING);
    append_vector(fixture, "blk.0.attn_norm.weight", DIM, 0);
    append_matrix(fixture, "blk.0.attn_q.weight", ENGINE_GGML_TYPE_Q4_K, DIM, DIM, layers);
    append_matrix(fixture, "blk.0.attn_k.weight", ENGINE_GGML_TYPE_Q8_0, DIM, KV_DIM, layers);
    append_matrix(fixture, "blk.0.attn_v.weight", ENGINE_GGML_TYPE_Q6_K, DIM, KV_DIM, layers);
    append_vector(fixture, "blk.0.attn_q.bias", DIM, 1);
    append_vector(fixture, "blk.0.attn_k.bias", KV_DIM, 1);
    append_vector(fixture, "blk.0.attn_v.bias", KV_DIM, 1);
    append_matrix(fixture, "blk.0.attn_output.weight", ENGINE_GGML_TYPE_Q5_0, DIM, DIM, layers);
    append_vector(fixture, "blk.0.ffn_norm.weight", DIM, 0);
    append_matrix(fixture, "blk.0.ffn_gate.weight", ENGINE_GGML_TYPE_Q5_0, DIM, HIDDEN, layers);
    append_matrix(fixture, "blk.0.ffn_up.weight", ENGINE_GGML_TYPE_Q4_K, DIM, HIDDEN, layers);
    append_matrix(fixture, "blk.0.ffn_down.weight", ENGINE_GGML_TYPE_Q6_K, HIDDEN, DIM, layers);
    append_vector(fixture, "output_norm.weight", DIM, 0);
    append_matrix(fixture, "output.weight", ENGINE_GGML_TYPE_Q8_0,
                  DIM, VOCAB, packed_parts & PACK_OUTPUT);

    fixture->implementation.gguf.data = fixture->data;
    fixture->implementation.gguf.file_size = fixture->used;
    fixture->implementation.gguf.tensor_count = (uint64_t)fixture->count;
    fixture->implementation.gguf.tensors = fixture->tensors;
    fixture->implementation.architecture = qwen2_architecture_ops();
    fixture->model.implementation = &fixture->implementation;
    fixture->model.config.dim = DIM;
    fixture->model.config.hidden_dim = HIDDEN;
    fixture->model.config.n_layers = 1;
    fixture->model.config.n_heads = 4;
    fixture->model.config.n_kv_heads = 2;
    fixture->model.config.vocab_size = VOCAB;
    fixture->model.config.seq_len = 3;
    fixture->model.config.rms_norm_epsilon = 1.0e-6f;
    fixture->model.config.rope_frequency_base = 10000.0f;
}

static void forward_success(MixedForwardFixture *fixture, InferenceState *state, int token)
{
    char error[512] = {};
    int result = qwen2_architecture_ops()->forward_token(&fixture->model, state, token, error, sizeof(error));
    if (!result) fprintf(stderr, "Forward token %d: %s\n", token, error);
    assert(result == 1);
}

static void compare_logits(const InferenceState *actual, const InferenceState *expected)
{
    assert(actual->logits_count == VOCAB && expected->logits_count == VOCAB);
    for (int i = 0; i < VOCAB; ++i) {
        assert(isfinite(actual->logits[i]) && isfinite(expected->logits[i]));
        float tolerance = 0.0001f + 0.001f * fabsf(expected->logits[i]);
        if (fabsf(actual->logits[i] - expected->logits[i]) > tolerance) {
            fprintf(stderr, "Logit %d: actual=%g expected=%g\n", i, actual->logits[i], expected->logits[i]);
            assert(false);
        }
    }
}

static void assert_disabled_profile(const OperationProfile *profile)
{
    assert(profile->enabled == 0);
    assert(profile->matvec_seconds == 0.0 && profile->attention_seconds == 0.0);
    assert(profile->rmsnorm_seconds == 0.0 && profile->rope_seconds == 0.0);
    assert(profile->output_seconds == 0.0 && profile->sampling_seconds == 0.0);
    assert(profile->forward_seconds == 0.0);
    assert(profile->forward_calls == 0 && profile->sampling_calls == 0);
}

int main(int argc, char **argv)
{
    int packed_parts = PACK_EMBEDDING | PACK_LAYERS | PACK_OUTPUT;
    if (argc > 2) return 2;
    if (argc == 2) {
        if (strcmp(argv[1], "embedding") == 0) packed_parts = PACK_EMBEDDING;
        else if (strcmp(argv[1], "layers") == 0) packed_parts = PACK_LAYERS;
        else if (strcmp(argv[1], "output") == 0) packed_parts = PACK_OUTPUT;
        else if (strcmp(argv[1], "f32") == 0) packed_parts = 0;
        else if (strcmp(argv[1], "mixed") != 0) {
            fprintf(stderr, "Usage: %s [embedding|layers|output|mixed|f32]\n", argv[0]);
            return 2;
        }
    }
    MixedForwardFixture actual;
    MixedForwardFixture reference;
    initialize_fixture(&actual, packed_parts);
    initialize_fixture(&reference, 0);
    InferenceState state = {};
    InferenceState expected = {};
    InferenceState profiled = {};
    char error[512] = {};
    const ArchitectureOps *ops = qwen2_architecture_ops();
    assert(ops->state_init(&actual.model, &state, 3, error, sizeof(error)) == 1);
    assert(ops->state_init(&reference.model, &expected, 3, error, sizeof(error)) == 1);
    assert(ops->state_init(&actual.model, &profiled, 3, error, sizeof(error)) == 1);
    profiled.profile.enabled = 1;
    double previous_forward_seconds = 0.0;
    for (int token = 0; token < 2; ++token) {
        forward_success(&reference, &expected, token);
        forward_success(&actual, &state, token);
        compare_logits(&state, &expected);
        forward_success(&actual, &profiled, token);
        assert(memcmp(state.logits, profiled.logits, VOCAB * sizeof(float)) == 0);
        assert_disabled_profile(&state.profile);
        assert(profiled.profile.forward_calls == (unsigned long long)token + 1);
        assert(profiled.profile.forward_seconds > previous_forward_seconds);
        assert(profiled.profile.matvec_seconds > 0.0);
        assert(profiled.profile.attention_seconds > 0.0);
        assert(profiled.profile.rmsnorm_seconds > 0.0);
        assert(profiled.profile.rope_seconds > 0.0);
        assert(profiled.profile.output_seconds > 0.0);
        assert(profiled.profile.sampling_seconds == 0.0);
        unsigned long long calls = (unsigned long long)token + 1;
        if (packed_parts & PACK_LAYERS) {
            assert(profiled.profile.weight_layer_calls[PROFILE_Q5_0] == 2 * calls);
            assert(profiled.profile.weight_layer_calls[PROFILE_Q4_K] == 2 * calls);
            assert(profiled.profile.weight_layer_calls[PROFILE_Q6_K] == 2 * calls);
            assert(profiled.profile.weight_layer_calls[PROFILE_Q8_0] == calls);
            assert(profiled.profile.weight_layer_seconds[PROFILE_Q5_0] > 0.0);
        } else {
            assert(profiled.profile.weight_layer_calls[PROFILE_F32] == 7 * calls);
        }
        int output_format = (packed_parts & PACK_OUTPUT) ? PROFILE_Q8_0 : PROFILE_F32;
        assert(profiled.profile.weight_output_calls[output_format] == calls);
        assert(profiled.profile.weight_output_seconds[output_format] > 0.0);
        double stages = profiled.profile.matvec_seconds + profiled.profile.attention_seconds +
                        profiled.profile.rmsnorm_seconds + profiled.profile.rope_seconds +
                        profiled.profile.output_seconds;
        assert(stages <= profiled.profile.forward_seconds);
        previous_forward_seconds = profiled.profile.forward_seconds;
        assert(((Qwen2InferenceState *)state.implementation)->token_count == token + 1);
    }

    /* Force a late failure, repair the tensor, then retry the same position. */
    GgufTensorInfo *output = &actual.tensors[14];
    assert(strcmp(output->name, "output.weight") == 0);
    output->dimensions[0] -= 1;
    assert(ops->forward_token(&actual.model, &state, 2, error, sizeof(error)) == 0);
    assert(ops->forward_token(&actual.model, &profiled, 2, error, sizeof(error)) == 0);
    assert(profiled.profile.forward_calls == 2);
    assert(error[0] != '\0');
    assert(((Qwen2InferenceState *)state.implementation)->token_count == 2);
    output->dimensions[0] += 1;
    forward_success(&reference, &expected, 2);
    forward_success(&actual, &state, 2);
    forward_success(&actual, &profiled, 2);
    assert(profiled.profile.forward_calls == 3);
    assert(memcmp(state.logits, profiled.logits, VOCAB * sizeof(float)) == 0);
    compare_logits(&state, &expected);
    error[0] = '\0';
    assert(ops->forward_token(&actual.model, &state, 0, error, sizeof(error)) == 0);
    assert(error[0] != '\0');
    assert(((Qwen2InferenceState *)state.implementation)->token_count == 3);

    ops->state_destroy(&state);
    ops->state_destroy(&expected);
    ops->state_destroy(&profiled);
    assert_disabled_profile(&profiled.profile);
    assert(state.implementation == nullptr && state.logits == nullptr);
    assert(expected.implementation == nullptr && expected.logits == nullptr);
    free(actual.data);
    free(reference.data);
    puts("Mixed-weight forward logits, failure/retry, context, and cleanup checks passed.");
    return 0;
}
