#include <math.h>
#include <stdio.h>
#include <string.h>

#include "app.h"
#include "engine.h"

static int report_error(const char *stage, const char *error)
{
    fprintf(stderr, "%s: %s\n", stage, error[0] != '\0' ? error : "no error message");
    return 1;
}

static int check_logits(const Model *model, const InferenceState *state, char *error, int capacity)
{
    if (state->logits == nullptr || state->logits_count != model->config.vocab_size) {
        snprintf(error, (size_t)capacity, "Forward pass returned an invalid logits buffer.");
        return 0;
    }
    for (int i = 0; i < state->logits_count; ++i) {
        if (!isfinite(state->logits[i])) {
            snprintf(error, (size_t)capacity, "Forward pass produced a non-finite logit at %d.", i);
            return 0;
        }
    }
    return 1;
}

int main(int argc, char **argv)
{
    AppConfig config;
    Model model;
    InferenceState state;
    int token_ids[64];
    int token_count = 0;
    int requested_context;
    int exit_code = 1;
    char error[ENGINE_ERROR_CAPACITY];

    if (argc != 2) {
        fprintf(stderr, "Usage: %s path/to/model.gguf\n", argv[0]);
        return 2;
    }

    memset(&config, 0, sizeof(config));
    memset(&model, 0, sizeof(model));
    memset(&state, 0, sizeof(state));
    snprintf(config.model_path, sizeof(config.model_path), "%s", argv[1]);
    error[0] = '\0';

    if (!engine_load_model(&config, &model, error, (int)sizeof(error))) {
        return report_error("Could not load model", error);
    }
    if (!tokenizer_encode(&model, "Hello", token_ids,
                          (int)(sizeof(token_ids) / sizeof(token_ids[0])),
                          &token_count, error, (int)sizeof(error))) {
        report_error("Could not tokenize fixture", error);
        goto cleanup_model;
    }
    if (token_count <= 0 || token_count >= model.config.seq_len) {
        snprintf(error, sizeof(error), "Fixture token count is outside the model context.");
        report_error("Invalid fixture", error);
        goto cleanup_model;
    }
    requested_context = token_count + 1;
    if (!inference_state_init(&model, &state, requested_context,
                              error, (int)sizeof(error))) {
        report_error("Could not initialize inference state", error);
        goto cleanup_state;
    }

    for (int i = 0; i < token_count; ++i) {
        if (!engine_forward_token(&model, &state, token_ids[i],
                                  error, (int)sizeof(error))) {
            report_error("Forward pass failed", error);
            goto cleanup_state;
        }
    }
    if (!check_logits(&model, &state, error, sizeof(error))) {
        report_error("Forward contract failed", error);
        goto cleanup_state;
    }
    /* Invalid inputs must not consume the one remaining context position. */
    for (int i = 0; i < 2; ++i) {
        error[0] = '\0';
        int invalid_id = i == 0 ? -1 : model.config.vocab_size;
        if (engine_forward_token(&model, &state, invalid_id, error, sizeof(error)) != 0 ||
            error[0] == '\0') {
            report_error("Invalid-token contract failed", "Expected rejection with an error.");
            goto cleanup_state;
        }
    }
    if (!engine_forward_token(&model, &state, token_ids[0], error, sizeof(error)) ||
        !check_logits(&model, &state, error, sizeof(error))) {
        report_error("Forward after rejected tokens failed", error);
        goto cleanup_state;
    }
    error[0] = '\0';
    if (engine_forward_token(&model, &state, token_ids[0], error, sizeof(error)) != 0 ||
        error[0] == '\0') {
        report_error("Context contract failed", "Expected context exhaustion with an error.");
        goto cleanup_state;
    }

    printf("Forwarded %d fixture tokens; produced %d finite logits.\n",
           token_count + 1, state.logits_count);
    exit_code = 0;

cleanup_state:
    inference_state_destroy(&state);
    if (state.implementation != nullptr || state.logits != nullptr || state.logits_count != 0) {
        report_error("State cleanup failed", "Destroyed state still owns storage.");
        exit_code = 1;
    }
cleanup_model:
    engine_unload_model(&model);
    if (model.implementation != nullptr) {
        report_error("Model cleanup failed", "Destroyed model still owns storage.");
        exit_code = 1;
    }
    return exit_code;
}
