#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "engine.h"
#include "architecture.h"
#include "engine_internal.h"
#include "gguf_reader.h"

static const float EPSILON = 0.0001f;
static const int FIXTURE_DIM = 4;
static const int FIXTURE_HIDDEN_DIM = 4;
static const int FIXTURE_VOCAB_SIZE = 3;
static const int FIXTURE_TENSOR_COUNT = 15;

typedef struct ForwardFixture {
    Model model;
    EngineModel implementation;
    GgufTensorInfo tensors[FIXTURE_TENSOR_COUNT];
    unsigned char data[1024];
    uint64_t data_size;
    int tensor_count;
} ForwardFixture;

static int floats_nearly_equal(float actual, float expected)
{
    return fabsf(actual - expected) <= EPSILON;
}

static void add_f32_tensor(
    ForwardFixture *fixture,
    const char *name,
    const float *values,
    uint32_t dimension_count,
    uint64_t dimension_0,
    uint64_t dimension_1
)
{
    GgufTensorInfo *tensor;
    uint64_t element_count = dimension_0;
    uint64_t byte_size;

    if (dimension_count == 2) {
        element_count *= dimension_1;
    }
    byte_size = element_count * sizeof(float);
    assert(fixture->tensor_count < FIXTURE_TENSOR_COUNT);
    assert(fixture->data_size + byte_size <= sizeof(fixture->data));

    tensor = &fixture->tensors[fixture->tensor_count++];
    memset(tensor, 0, sizeof(*tensor));
    tensor->name = (char *)name;
    tensor->dimension_count = dimension_count;
    tensor->dimensions[0] = dimension_0;
    tensor->dimensions[1] = dimension_1;
    tensor->type = ENGINE_GGML_TYPE_F32;
    tensor->offset = fixture->data_size;
    tensor->byte_size = byte_size;
    memcpy(fixture->data + fixture->data_size, values, (size_t)byte_size);
    fixture->data_size += byte_size;
}

static void initialize_fixture(ForwardFixture *fixture)
{
    static const float token_embedding[] = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f
    };
    static const float ones[] = {1.0f, 1.0f, 1.0f, 1.0f};
    static const float zeros_2[] = {0.0f, 0.0f};
    static const float zeros_4x4[] = {
        0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f
    };
    static const float q_bias[] = {1.0f, 0.0f, 0.0f, 1.0f};
    static const float k_bias[] = {1.0f, 0.0f};
    static const float v_weight[] = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f
    };
    static const float identity_4x4[] = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f
    };
    static const float ffn_gate_and_up[] = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f
    };
    static const float ffn_down[] = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f
    };
    static const float output_weight[] = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, -1.0f, 0.0f
    };

    memset(fixture, 0, sizeof(*fixture));
    add_f32_tensor(fixture, "token_embd.weight", token_embedding, 2, 4u, 3u);
    add_f32_tensor(fixture, "blk.0.attn_norm.weight", ones, 1, 4u, 0u);
    add_f32_tensor(fixture, "blk.0.attn_q.weight", zeros_4x4, 2, 4u, 4u);
    add_f32_tensor(fixture, "blk.0.attn_k.weight", zeros_4x4, 2, 4u, 2u);
    add_f32_tensor(fixture, "blk.0.attn_v.weight", v_weight, 2, 4u, 2u);
    add_f32_tensor(fixture, "blk.0.attn_q.bias", q_bias, 1, 4u, 0u);
    add_f32_tensor(fixture, "blk.0.attn_k.bias", k_bias, 1, 2u, 0u);
    add_f32_tensor(fixture, "blk.0.attn_v.bias", zeros_2, 1, 2u, 0u);
    add_f32_tensor(fixture, "blk.0.attn_output.weight", identity_4x4, 2, 4u, 4u);
    add_f32_tensor(fixture, "blk.0.ffn_norm.weight", ones, 1, 4u, 0u);
    add_f32_tensor(fixture, "blk.0.ffn_gate.weight", ffn_gate_and_up, 2, 4u, 4u);
    add_f32_tensor(fixture, "blk.0.ffn_up.weight", ffn_gate_and_up, 2, 4u, 4u);
    add_f32_tensor(fixture, "blk.0.ffn_down.weight", ffn_down, 2, 4u, 4u);
    add_f32_tensor(fixture, "output_norm.weight", ones, 1, 4u, 0u);
    add_f32_tensor(fixture, "output.weight", output_weight, 2, 4u, 3u);
    assert(fixture->tensor_count == FIXTURE_TENSOR_COUNT);

    fixture->implementation.gguf.data = fixture->data;
    fixture->implementation.gguf.file_size = fixture->data_size;
    fixture->implementation.gguf.tensor_data_offset = 0;
    fixture->implementation.gguf.tensor_count = (uint64_t)fixture->tensor_count;
    fixture->implementation.gguf.tensors = fixture->tensors;
    fixture->implementation.architecture = architecture_find("qwen2", 5u);
    assert(fixture->implementation.architecture != nullptr);

    fixture->model.implementation = &fixture->implementation;
    fixture->model.config.dim = FIXTURE_DIM;
    fixture->model.config.hidden_dim = FIXTURE_HIDDEN_DIM;
    fixture->model.config.n_layers = 1;
    fixture->model.config.n_heads = 2;
    fixture->model.config.n_kv_heads = 1;
    fixture->model.config.vocab_size = FIXTURE_VOCAB_SIZE;
    fixture->model.config.seq_len = 3;
    fixture->model.config.rms_norm_epsilon = 1.0f;
    fixture->model.config.rope_frequency_base = 10000.0f;
}

static void assert_logits(const InferenceState *state, float a, float b, float c)
{
    assert(state->logits != nullptr);
    assert(state->logits_count == FIXTURE_VOCAB_SIZE);
    assert(floats_nearly_equal(state->logits[0], a));
    assert(floats_nearly_equal(state->logits[1], b));
    assert(floats_nearly_equal(state->logits[2], c));
}

static void test_model_configuration_validates_output_norm(void)
{
    static const char *keys[] = {
        "qwen2.embedding_length", "qwen2.feed_forward_length", "qwen2.block_count",
        "qwen2.attention.head_count", "qwen2.attention.head_count_kv", "qwen2.context_length",
        "qwen2.attention.layer_norm_rms_epsilon", "qwen2.rope.freq_base"
    };
    ForwardFixture fixture;
    GgufMetadataEntry metadata[8] = {};
    ModelConfig config;
    void *architecture_data = nullptr;
    char error[ENGINE_ERROR_CAPACITY] = {};

    initialize_fixture(&fixture);
    uint32_t values[] = {
        (uint32_t)fixture.model.config.dim, (uint32_t)fixture.model.config.hidden_dim,
        (uint32_t)fixture.model.config.n_layers, (uint32_t)fixture.model.config.n_heads,
        (uint32_t)fixture.model.config.n_kv_heads, (uint32_t)fixture.model.config.seq_len,
        0, 0
    };
    memcpy(&values[6], &fixture.model.config.rms_norm_epsilon, sizeof(float));
    memcpy(&values[7], &fixture.model.config.rope_frequency_base, sizeof(float));
    for (int i = 0; i < 8; ++i) {
        assert(fixture.data_size + sizeof(uint32_t) <= sizeof(fixture.data));
        metadata[i].key = (char *)keys[i];
        metadata[i].type = i < 6 ? 4u : 6u; /* GGUF UINT32 or FLOAT32 */
        metadata[i].value_offset = fixture.data_size;
        for (int byte = 0; byte < 4; ++byte) {
            fixture.data[fixture.data_size++] = (unsigned char)(values[i] >> (byte * 8));
        }
    }
    fixture.implementation.gguf.metadata = metadata;
    fixture.implementation.gguf.metadata_count = 8;
    fixture.implementation.gguf.file_size = fixture.data_size;

    const ArchitectureOps *architecture = fixture.implementation.architecture;
    assert(architecture->configure_model(&fixture.implementation.gguf, FIXTURE_VOCAB_SIZE,
                                         &config, &architecture_data, error, sizeof(error)) == 1);

    GgufTensorInfo *norm = (GgufTensorInfo *)gguf_find_tensor(
        &fixture.implementation.gguf, "output_norm.weight");
    assert(norm != nullptr);
    GgufTensorInfo valid_norm = *norm;
    for (int invalid_case = 0; invalid_case < 3; ++invalid_case) {
        *norm = valid_norm;
        if (invalid_case == 0) norm->name = (char *)"missing_output_norm.weight";
        if (invalid_case == 1) norm->dimension_count = 2;
        if (invalid_case == 2) norm->dimensions[0] += 1;
        error[0] = '\0';
        assert(architecture->configure_model(&fixture.implementation.gguf, FIXTURE_VOCAB_SIZE,
                                             &config, &architecture_data, error, sizeof(error)) == 0);
        assert(strstr(error, "output_norm.weight") != nullptr);
    }
    *norm = valid_norm;
}

static void test_two_token_full_prefix_forward(void)
{
    ForwardFixture fixture;
    InferenceState state;
    char error[ENGINE_ERROR_CAPACITY];

    initialize_fixture(&fixture);
    memset(&state, 0, sizeof(state));
    memset(error, 0, sizeof(error));

    assert(inference_state_init(&fixture.model, &state, 3,
                                error, (int)sizeof(error)) == 1);
    assert(engine_forward_token(&fixture.model, &state, 0,
                                error, (int)sizeof(error)) == 1);
    assert_logits(&state, 1.657019f, 0.0f, -0.457218f);

    assert(engine_forward_token(&fixture.model, &state, 1,
                                error, (int)sizeof(error)) == 1);
    assert_logits(&state, 0.320754f, 1.155113f, -0.241746f);

    inference_state_destroy(&state);
}

static void test_forward_rejects_invalid_token_and_exhausted_context(void)
{
    ForwardFixture fixture;
    InferenceState state;
    char error[ENGINE_ERROR_CAPACITY];

    initialize_fixture(&fixture);
    memset(&state, 0, sizeof(state));
    memset(error, 0, sizeof(error));
    assert(inference_state_init(&fixture.model, &state, 1,
                                error, (int)sizeof(error)) == 1);
    assert(engine_forward_token(&fixture.model, &state, -1,
                                error, (int)sizeof(error)) == 0);
    assert(strstr(error, "token ID") != nullptr);
    error[0] = '\0';
    assert(engine_forward_token(&fixture.model, &state, FIXTURE_VOCAB_SIZE,
                                error, (int)sizeof(error)) == 0);
    assert(strstr(error, "token ID") != nullptr);
    assert(engine_forward_token(&fixture.model, &state, 0,
                                error, (int)sizeof(error)) == 1);
    assert(engine_forward_token(&fixture.model, &state, 1,
                                error, (int)sizeof(error)) == 0);
    assert(strstr(error, "context") != nullptr);

    inference_state_destroy(&state);
}

static void test_attention_failure_reports_error_and_allows_retry(void)
{
    ForwardFixture fixture;
    InferenceState state = {};
    char error[ENGINE_ERROR_CAPACITY] = {};

    initialize_fixture(&fixture);
    assert(inference_state_init(&fixture.model, &state, 1, error, sizeof(error)) == 1);
    const GgufTensorInfo *bias = gguf_find_tensor(
        &fixture.implementation.gguf, "blk.0.attn_q.bias");
    assert(bias != nullptr);
    float saved;
    float invalid = NAN;
    memcpy(&saved, fixture.data + bias->offset, sizeof(float));
    memcpy(fixture.data + bias->offset, &invalid, sizeof(float));
    assert(engine_forward_token(&fixture.model, &state, 0, error, sizeof(error)) == 0);
    assert(strstr(error, "attention failed") != nullptr);

    memcpy(fixture.data + bias->offset, &saved, sizeof(float));
    assert(engine_forward_token(&fixture.model, &state, 0, error, sizeof(error)) == 1);
    assert_logits(&state, 1.657019f, 0.0f, -0.457218f);
    assert(error[0] == '\0');
    inference_state_destroy(&state);
}

int main(void)
{
    printf("Running Milestone 06 F32 forward-reference tests...\n");

    test_model_configuration_validates_output_norm();
    test_two_token_full_prefix_forward();
    test_forward_rejects_invalid_token_and_exhausted_context();
    test_attention_failure_reports_error_and_allows_retry();

    printf("All Milestone 06 F32 forward-reference tests passed.\n");
    return 0;
}
