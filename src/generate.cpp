#include "generate.h"

#include "operation_timer.h"

#include <stdio.h>
#include <string.h>

static void set_error(char *error, int capacity, const char *message)
{
    if (error != nullptr && capacity > 0) {
        snprintf(error, (size_t)capacity, "%s", message);
    }
}

int generate_prepare_prompt(const Model *model, const char *user_prompt,
                            int *token_ids, int token_capacity, int *token_count,
                            char *error, int error_capacity)
{
    char formatted_prompt[GENERATE_FORMATTED_PROMPT_CAPACITY];

    if (model == nullptr || user_prompt == nullptr || token_ids == nullptr ||
        token_count == nullptr || token_capacity <= 0) {
        set_error(error, error_capacity, "Prompt preparation arguments are invalid.");
        return 0;
    }
    *token_count = 0;
    if (!engine_format_prompt(model, user_prompt, formatted_prompt,
                              sizeof(formatted_prompt), error, error_capacity)) {
        return 0;
    }
    if (!tokenizer_encode(model, formatted_prompt, token_ids, token_capacity,
                          token_count, error, error_capacity)) {
        return 0;
    }
    if (*token_count < 0 || *token_count > token_capacity) {
        set_error(error, error_capacity,
                  "Tokenizer returned an invalid prompt token count.");
        return 0;
    }
    if (*token_count == 0) {
        set_error(error, error_capacity,
                  "The prompt must produce at least one token.");
        return 0;
    }
    return 1;
}

int generate_run(const Model *model, InferenceState *state,
                 const int *token_ids, int token_count,
                 int max_tokens, float temperature, float top_p,
                 const UsageSnapshot *prompt_start_usage,
                 GenerateTokenFn on_token, void *user,
                 GenerateResult *result, char *error, int error_capacity)
{
    int generated_tokens = 0;
    int first_token_recorded = 0;

    if (result == nullptr) {
        set_error(error, error_capacity, "Generation result pointer is null.");
        return 0;
    }
    memset(result, 0, sizeof(*result));
    result->time_to_first_token_seconds = -1.0;
    if (model == nullptr || state == nullptr || token_ids == nullptr ||
        token_count <= 0 || max_tokens < 0 || on_token == nullptr) {
        set_error(error, error_capacity, "Generation arguments are invalid.");
        return 0;
    }

    /* Only the final prompt token produces vocabulary logits. */
    if (!engine_prefill_tokens(model, state, token_ids, token_count,
                               error, error_capacity)) {
        return 0;
    }

    metrics_capture(&result->start_usage);
    while (generated_tokens < max_tokens) {
        int token_id = -1;
        char token_text[TOKEN_TEXT_CAPACITY];
        int sample_result;

        if (error != nullptr && error_capacity > 0) {
            error[0] = '\0';
        }
        if (state->logits == nullptr || state->logits_count != model->config.vocab_size) {
            set_error(error, error_capacity,
                      "Model forward pass returned invalid logits.");
            return 0;
        }
        {
            OperationTimer timer(state->profile.enabled, &state->profile.sampling_seconds);
            sample_result = sampler_sample(state->logits, state->logits_count,
                                           temperature, top_p, &token_id,
                                           error, error_capacity);
        }
        if (!sample_result) {
            return 0;
        }
        if (token_id < 0 || token_id >= model->config.vocab_size) {
            set_error(error, error_capacity,
                      "Sampler returned a token ID outside the vocabulary.");
            return 0;
        }
        if (state->profile.enabled) ++state->profile.sampling_calls;
        ++generated_tokens;
        result->generated_tokens = generated_tokens;
        if (tokenizer_is_eos(model, token_id)) {
            break;
        }
        memset(token_text, 0, sizeof(token_text));
        if (!tokenizer_decode_one(model, token_id, token_text,
                                  TOKEN_TEXT_CAPACITY, error, error_capacity)) {
            return 0;
        }
        if (memchr(token_text, '\0', sizeof(token_text)) == nullptr) {
            set_error(error, error_capacity,
                      "Tokenizer returned an unterminated token string.");
            return 0;
        }
        int keep_going = on_token(token_text, user);
        if (!first_token_recorded && token_text[0] != '\0') {
            UsageSnapshot first_token_usage;
            metrics_capture(&first_token_usage);
            first_token_recorded = 1;
            if (prompt_start_usage != nullptr && prompt_start_usage->available &&
                first_token_usage.available &&
                first_token_usage.wall_seconds >= prompt_start_usage->wall_seconds) {
                result->time_to_first_token_seconds =
                    metrics_elapsed_seconds(prompt_start_usage, &first_token_usage);
            }
        }
        if (!keep_going) {
            result->stopped_early = 1;
            break;
        }

        /* The last step samples from logits already in hand; no further decode. */
        if (generated_tokens < max_tokens &&
            !engine_forward_token(model, state, token_id, error, error_capacity)) {
            return 0;
        }
    }
    metrics_capture(&result->end_usage);
    return 1;
}
