#include "engine.h"

#include "architecture.h"
#include "engine_internal.h"
#include "tokenizer.h"

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void set_error(char *error, int capacity, const char *message)
{
    if (error != nullptr && capacity > 0) {
        snprintf(error, (size_t)capacity, "%s", message);
    }
}

int engine_load_model(const AppConfig *config, Model *model,
                      char *error, int error_capacity)
{
    EngineModel *implementation;
    const char *architecture_identifier;
    uint64_t architecture_identifier_length;
    uint32_t token_element_type;
    uint64_t token_count;

    if (model == nullptr) {
        set_error(error, error_capacity, "Model output pointer is null.");
        return 0;
    }
    memset(model, 0, sizeof(*model));
    if (error != nullptr && error_capacity > 0) {
        error[0] = '\0';
    }
    if (config == nullptr) {
        set_error(error, error_capacity, "App configuration is null.");
        return 0;
    }

    implementation = (EngineModel *)calloc(1, sizeof(*implementation));
    if (implementation == nullptr) {
        set_error(error, error_capacity, "Could not allocate model loader state.");
        return 0;
    }
    implementation->gguf.file_descriptor = -1;
    if (!gguf_open(config->model_path, &implementation->gguf,
                   error, error_capacity)) {
        gguf_close(&implementation->gguf);
        free(implementation);
        return 0;
    }

    if (!gguf_get_string(&implementation->gguf, "general.architecture",
                         &architecture_identifier,
                         &architecture_identifier_length)) {
        set_error(error, error_capacity,
                  "GGUF file is missing general.architecture metadata.");
        goto failure;
    }
    implementation->architecture = architecture_find(architecture_identifier,
                                                      architecture_identifier_length);
    if (implementation->architecture == nullptr) {
        set_error(error, error_capacity,
                  "This engine does not have a backend for the GGUF architecture.");
        goto failure;
    }

    if (!gguf_get_array_info(&implementation->gguf, "tokenizer.ggml.tokens",
                             &token_element_type, &token_count) ||
        token_element_type != 8 || token_count == 0 || token_count > INT_MAX) {
        set_error(error, error_capacity,
                  "GGUF tokenizer.ggml.tokens metadata is missing or invalid.");
        goto failure;
    }

    if (implementation->architecture->configure_model == nullptr ||
        !implementation->architecture->configure_model(
            &implementation->gguf, (int)token_count, &model->config,
            &implementation->architecture_data, error, error_capacity)) {
        if (error != nullptr && error_capacity > 0 && error[0] == '\0') {
            set_error(error, error_capacity,
                      "The model-family backend could not configure this GGUF model.");
        }
        goto failure;
    }
    if (!tokenizer_load_from_gguf(config->model_path, &implementation->tokenizer,
                                  model->config.vocab_size,
                                  error, error_capacity)) {
        goto failure;
    }

    model->implementation = implementation;
    if (error != nullptr && error_capacity > 0) {
        error[0] = '\0';
    }
    return 1;

failure:
    tokenizer_destroy(&implementation->tokenizer);
    if (implementation->architecture != nullptr &&
        implementation->architecture->destroy_model_data != nullptr) {
        implementation->architecture->destroy_model_data(
            implementation->architecture_data);
    }
    gguf_close(&implementation->gguf);
    free(implementation);
    memset(model, 0, sizeof(*model));
    return 0;
}

void engine_unload_model(Model *model)
{
    EngineModel *implementation;

    if (model == nullptr) {
        return;
    }
    implementation = (EngineModel *)model->implementation;
    if (implementation != nullptr) {
        tokenizer_destroy(&implementation->tokenizer);
        if (implementation->architecture != nullptr &&
            implementation->architecture->destroy_model_data != nullptr) {
            implementation->architecture->destroy_model_data(
                implementation->architecture_data);
        }
        gguf_close(&implementation->gguf);
        free(implementation);
    }
    memset(model, 0, sizeof(*model));
}

int engine_format_prompt(const Model *model, const char *user_prompt,
                         char *output, int output_capacity,
                         char *error, int error_capacity)
{
    if (output != nullptr && output_capacity > 0) output[0] = '\0';
    if (model == nullptr || model->implementation == nullptr) {
        set_error(error, error_capacity, "Model is not loaded.");
        return 0;
    }
    if (user_prompt == nullptr || output == nullptr || output_capacity <= 0) {
        set_error(error, error_capacity, "Prompt formatting arguments are invalid.");
        return 0;
    }
    const EngineModel *implementation = (const EngineModel *)model->implementation;
    if (implementation->architecture == nullptr ||
        implementation->architecture->format_prompt == nullptr) {
        set_error(error, error_capacity, "The loaded model has no prompt formatter.");
        return 0;
    }
    return implementation->architecture->format_prompt(
        user_prompt, output, output_capacity, error, error_capacity);
}

int inference_state_init(const Model *model, InferenceState *state,
                         int max_context_tokens,
                         char *error, int error_capacity)
{
    const EngineModel *implementation;

    if (state == nullptr) {
        set_error(error, error_capacity, "Inference-state output pointer is null.");
        return 0;
    }
    memset(state, 0, sizeof(*state));
    if (model == nullptr || model->implementation == nullptr) {
        set_error(error, error_capacity, "Model is not loaded.");
        return 0;
    }
    implementation = (const EngineModel *)model->implementation;
    if (implementation->architecture == nullptr ||
        implementation->architecture->state_init == nullptr) {
        set_error(error, error_capacity,
                  "The loaded model has no inference-state backend.");
        return 0;
    }
    return implementation->architecture->state_init(model, state,
                                                    max_context_tokens,
                                                    error, error_capacity);
}

int engine_forward_token(const Model *model, InferenceState *state,
                         int token_id, char *error, int error_capacity)
{
    const EngineModel *implementation;
    const EngineStateHeader *state_header;

    if (model == nullptr || model->implementation == nullptr ||
        state == nullptr || state->implementation == nullptr) {
        set_error(error, error_capacity,
                  "Model or inference state is not initialized.");
        return 0;
    }
    implementation = (const EngineModel *)model->implementation;
    state_header = (const EngineStateHeader *)state->implementation;
    if (implementation->architecture == nullptr ||
        implementation->architecture->forward_token == nullptr ||
        state_header->architecture != implementation->architecture) {
        set_error(error, error_capacity,
                  "Model and inference state do not use the same backend.");
        return 0;
    }
    return implementation->architecture->forward_token(model, state, token_id,
                                                        error, error_capacity);
}

int engine_forward_token_no_logits(const Model *model, InferenceState *state,
                                   int token_id, char *error, int error_capacity)
{
    if (model == nullptr || model->implementation == nullptr ||
        state == nullptr || state->implementation == nullptr) {
        set_error(error, error_capacity, "Model or inference state is not initialized.");
        return 0;
    }
    const EngineModel *implementation = (const EngineModel *)model->implementation;
    const EngineStateHeader *state_header = (const EngineStateHeader *)state->implementation;
    if (implementation->architecture == nullptr ||
        implementation->architecture->forward_token_no_logits == nullptr ||
        state_header->architecture != implementation->architecture) {
        set_error(error, error_capacity,
                  "Model and inference state do not support no-logits forwarding.");
        return 0;
    }
    return implementation->architecture->forward_token_no_logits(
        model, state, token_id, error, error_capacity);
}

void inference_state_destroy(InferenceState *state)
{
    EngineStateHeader *state_header;

    if (state == nullptr) {
        return;
    }
    state_header = (EngineStateHeader *)state->implementation;
    if (state_header != nullptr && state_header->architecture != nullptr &&
        state_header->architecture->state_destroy != nullptr) {
        state_header->architecture->state_destroy(state);
    }
    memset(state, 0, sizeof(*state));
}
