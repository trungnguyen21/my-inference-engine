#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "engine.h"
#include "architecture.h"
#include "engine_internal.h"
#include "qwen2_internal.h"

static Model make_valid_model(EngineModel *implementation)
{
    Model model;

    memset(implementation, 0, sizeof(*implementation));
    implementation->architecture = architecture_find("qwen2", 5u);
    assert(implementation->architecture != nullptr);
    memset(&model, 0, sizeof(model));
    model.implementation = implementation;
    model.config.dim = 4;
    model.config.hidden_dim = 8;
    model.config.n_layers = 1;
    model.config.n_heads = 2;
    model.config.n_kv_heads = 1;
    model.config.vocab_size = 5;
    model.config.seq_len = 6;
    model.config.rms_norm_epsilon = 1.0e-6f;
    model.config.rope_frequency_base = 10000.0f;
    return model;
}

static void test_state_initializes_every_public_and_private_result(void)
{
    EngineModel model_implementation;
    Model model = make_valid_model(&model_implementation);
    InferenceState state;
    Qwen2InferenceState *private_state;
    char error[ENGINE_ERROR_CAPACITY];

    memset(&state, 0, sizeof(state));
    memset(error, 0, sizeof(error));

    assert(inference_state_init(&model, &state, 3, error, (int)sizeof(error)) == 1);
    assert(state.implementation != nullptr);
    assert(state.logits != nullptr);
    assert(state.logits_count == model.config.vocab_size);

    private_state = (Qwen2InferenceState *)state.implementation;
    assert(private_state->max_context_tokens == 3);
    assert(private_state->token_count == 0);
    assert(private_state->head_dimension == 2);
    assert(private_state->key_value_dimension == 2);
    assert(private_state->token_ids != nullptr);
    assert(private_state->activations != nullptr);
    assert(private_state->keys != nullptr);
    assert(private_state->values != nullptr);
    assert(private_state->normalized != nullptr);
    assert(private_state->query != nullptr);
    assert(private_state->key != nullptr);
    assert(private_state->value != nullptr);
    assert(private_state->attention_probabilities != nullptr);
    assert(private_state->attention_output != nullptr);
    assert(private_state->ffn_gate != nullptr);
    assert(private_state->ffn_up != nullptr);
    assert(private_state->matvec_output != nullptr);

    inference_state_destroy(&state);
    assert(state.implementation == nullptr);
    assert(state.logits == nullptr);
    assert(state.logits_count == 0);
}

static void test_state_rejects_invalid_requests_and_remains_destroyable(void)
{
    EngineModel model_implementation;
    Model model = make_valid_model(&model_implementation);
    InferenceState state;
    char error[ENGINE_ERROR_CAPACITY];

    memset(&state, 0, sizeof(state));
    memset(error, 0, sizeof(error));
    assert(inference_state_init(nullptr, &state, 1, error, (int)sizeof(error)) == 0);
    assert(state.implementation == nullptr);
    assert(state.logits == nullptr);

    assert(inference_state_init(&model, &state, 0, error, (int)sizeof(error)) == 0);
    assert(inference_state_init(&model, &state, model.config.seq_len + 1,
                                error, (int)sizeof(error)) == 0);

    model.config.n_heads = 0;
    assert(inference_state_init(&model, &state, 1, error, (int)sizeof(error)) == 0);

    inference_state_destroy(&state);
    inference_state_destroy(nullptr);
}

int main(void)
{
    printf("Running Milestone 06 inference-state contract tests...\n");

    test_state_initializes_every_public_and_private_result();
    test_state_rejects_invalid_requests_and_remains_destroyable();

    printf("All Milestone 06 inference-state contract tests passed.\n");
    return 0;
}
