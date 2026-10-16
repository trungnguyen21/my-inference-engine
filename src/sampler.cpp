#include "engine.h"
#include "sampler.h"

#include <math.h>
#include <stdio.h>

static void set_error(char *error, int capacity, const char *message)
{
    if (error != NULL && capacity > 0) {
        snprintf(error, (size_t)capacity, "%s", message);
    }
}

int sampler_greedy_sample(const float *logits, int vocab_size, int *token_id,
                          char *error, int error_capacity)
{
    if (token_id != NULL) {
        *token_id = -1;
    }
    if (logits == NULL || vocab_size <= 0 || token_id == NULL) {
        set_error(error, error_capacity, "Greedy sampler arguments are invalid.");
        return 0;
    }
    int best_id = 0;
    for (int i = 0; i < vocab_size; ++i) {
        if (!isfinite(logits[i])) {
            set_error(error, error_capacity, "Greedy sampler received a non-finite logit.");
            return 0;
        }
        /* Strict comparison preserves the first (lowest) ID on exact ties. */
        if (logits[i] > logits[best_id]) best_id = i;
    }
    *token_id = best_id;
    if (error != NULL && error_capacity > 0) error[0] = '\0';
    return 1;
}

int sampler_sample(const float *logits, int vocab_size,
                   float temperature, float top_p, int *token_id,
                   char *error, int error_capacity)
{
    if (token_id != NULL) *token_id = -1;
    if (logits == NULL || vocab_size <= 0 || token_id == NULL ||
        !isfinite(temperature) || temperature < 0.0f || temperature > 1.0f ||
        !isfinite(top_p) || top_p < 0.0f || top_p > 1.0f) {
        set_error(error, error_capacity, "Sampler arguments are invalid.");
        return 0;
    }
    if (temperature == 0.0f) {
        return sampler_greedy_sample(logits, vocab_size, token_id, error, error_capacity);
    }
    /* Optional after first text: temperature scaling, top-p, and RNG policy. */
    set_error(error, error_capacity,
              "Nonzero-temperature sampling is not implemented; use --temperature 0.");
    return 0;
}
