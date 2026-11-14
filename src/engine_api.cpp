#include "engine_api.h"

#include "engine.h"
#include "generate.h"
#include "metrics.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct EngineHandle {
    Model model;
};

struct EngineSession {
    EngineHandle *handle;
    InferenceState state;
    int max_context_tokens;
    int *prompt_token_ids; /* max_context_tokens entries */
};

/* Greedy decoding ignores nucleus sampling; pass the no-op value through. */
#define GREEDY_TOP_P 1.0f

static void set_error(char *error, int capacity, const char *message)
{
    if (error != nullptr && capacity > 0) {
        snprintf(error, (size_t)capacity, "%s", message);
    }
}

int engine_error_capacity(void)
{
    return ENGINE_ERROR_CAPACITY;
}

int engine_prompt_capacity(void)
{
    return ENGINE_PROMPT_CAPACITY;
}

EngineHandle *engine_open(const char *gguf_path, char *error, int error_capacity)
{
    EngineHandle *handle;

    if (error != nullptr && error_capacity > 0) {
        error[0] = '\0';
    }
    if (gguf_path == nullptr) {
        set_error(error, error_capacity, "Model path is null.");
        return nullptr;
    }
    if (strlen(gguf_path) >= ENGINE_MODEL_PATH_CAPACITY) {
        set_error(error, error_capacity, "Model path is too long.");
        return nullptr;
    }

    handle = (EngineHandle *)calloc(1, sizeof(*handle));
    if (handle == nullptr) {
        set_error(error, error_capacity, "Could not allocate engine handle.");
        return nullptr;
    }

    if (!engine_load_model(gguf_path, &handle->model, error, error_capacity)) {
        free(handle);
        return nullptr;
    }
    return handle;
}

void engine_close(EngineHandle *handle)
{
    if (handle == nullptr) {
        return;
    }
    engine_unload_model(&handle->model);
    free(handle);
}

int engine_context_limit(const EngineHandle *handle)
{
    return handle == nullptr ? 0 : handle->model.config.seq_len;
}

EngineSession *engine_session_create(EngineHandle *handle, int max_context_tokens,
                                     char *error, int error_capacity)
{
    EngineSession *session;

    if (error != nullptr && error_capacity > 0) {
        error[0] = '\0';
    }
    if (handle == nullptr) {
        set_error(error, error_capacity, "Engine handle is null.");
        return nullptr;
    }
    if (max_context_tokens <= 0 ||
        max_context_tokens > handle->model.config.seq_len) {
        set_error(error, error_capacity,
                  "Requested session context is outside the model's limit.");
        return nullptr;
    }

    session = (EngineSession *)calloc(1, sizeof(*session));
    if (session == nullptr) {
        set_error(error, error_capacity, "Could not allocate engine session.");
        return nullptr;
    }
    session->handle = handle;
    session->max_context_tokens = max_context_tokens;
    session->prompt_token_ids =
        (int *)calloc((size_t)max_context_tokens, sizeof(*session->prompt_token_ids));
    if (session->prompt_token_ids == nullptr) {
        set_error(error, error_capacity, "Could not allocate prompt token buffer.");
        free(session);
        return nullptr;
    }
    if (!inference_state_init(&handle->model, &session->state, max_context_tokens,
                              error, error_capacity)) {
        inference_state_destroy(&session->state);
        free(session->prompt_token_ids);
        free(session);
        return nullptr;
    }
    return session;
}

void engine_session_destroy(EngineSession *session)
{
    if (session == nullptr) {
        return;
    }
    inference_state_destroy(&session->state);
    free(session->prompt_token_ids);
    free(session);
}

int engine_run(EngineSession *session, const char *prompt, int max_tokens,
               float temperature,
               EngineTokenFn on_token, void *user,
               EngineStats *stats, char *error, int error_capacity)
{
    GenerateResult result;
    UsageSnapshot prompt_start_usage;
    int prompt_token_count = 0;

    if (error != nullptr && error_capacity > 0) {
        error[0] = '\0';
    }
    if (stats != nullptr) {
        memset(stats, 0, sizeof(*stats));
        stats->first_token_seconds = -1.0;
    }
    if (session == nullptr || prompt == nullptr || on_token == nullptr) {
        set_error(error, error_capacity, "Generation arguments are null.");
        return 0;
    }
    if (max_tokens <= 0) {
        set_error(error, error_capacity, "max_tokens must be positive.");
        return 0;
    }
    if (temperature != 0.0f) {
        set_error(error, error_capacity,
                  "Only greedy decoding is implemented; temperature must be 0.");
        return 0;
    }
    if (strlen(prompt) >= ENGINE_PROMPT_CAPACITY) {
        set_error(error, error_capacity, "Prompt is too long.");
        return 0;
    }

    /* Reuse this session's buffers; nothing below allocates. */
    inference_state_reset(&session->state);

    metrics_capture(&prompt_start_usage);
    if (!generate_prepare_prompt(&session->handle->model, prompt,
                                 session->prompt_token_ids,
                                 session->max_context_tokens, &prompt_token_count,
                                 error, error_capacity)) {
        return 0;
    }
    if (prompt_token_count + max_tokens > session->max_context_tokens) {
        snprintf(error, (size_t)error_capacity,
                 "Prompt plus requested output (%d tokens) exceeds this session's "
                 "context (%d tokens).",
                 prompt_token_count + max_tokens, session->max_context_tokens);
        return 0;
    }
    if (stats != nullptr) {
        stats->prompt_tokens = prompt_token_count;
    }

    if (!generate_run(&session->handle->model, &session->state,
                      session->prompt_token_ids, prompt_token_count,
                      max_tokens, temperature, GREEDY_TOP_P, &prompt_start_usage,
                      on_token, user, &result, error, error_capacity)) {
        return 0;
    }

    if (stats != nullptr) {
        stats->generated_tokens = result.generated_tokens;
        stats->stopped_early = result.stopped_early;
        stats->first_token_seconds = result.time_to_first_token_seconds;
        stats->generate_seconds =
            metrics_elapsed_seconds(&result.start_usage, &result.end_usage);
        if (prompt_start_usage.available && result.start_usage.available) {
            stats->prefill_seconds =
                metrics_elapsed_seconds(&prompt_start_usage, &result.start_usage);
        }
    }
    return 1;
}
