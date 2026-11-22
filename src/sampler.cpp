#include "engine.h"
#include "sampler.h"

#include <math.h>
#include <stdio.h>
#include <algorithm>
#include <random>
#include <vector>

/* Not thread-safe; callers doing nonzero-temperature sampling from multiple
 * threads concurrently would need one engine per thread. */
static std::mt19937 &rng_engine()
{
    static std::mt19937 engine(std::random_device{}());
    return engine;
}

void sampler_seed_rng(unsigned long seed)
{
    rng_engine().seed((std::mt19937::result_type)seed);
}

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

int sampler_apply_repetition_penalty(float *logits, int vocab_size,
                                     const unsigned char *seen_mask,
                                     float penalty)
{
    if (logits == NULL || vocab_size <= 0 || seen_mask == NULL ||
        !isfinite(penalty) || penalty <= 0.0f) {
        return 0;
    }
    if (penalty == 1.0f) {
        return 1;
    }
    for (int i = 0; i < vocab_size; ++i) {
        if (!seen_mask[i]) {
            continue;
        }
        if (logits[i] > 0.0f) {
            logits[i] /= penalty;
        } else {
            logits[i] *= penalty;
        }
    }
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

    /* Softmax with temperature scaling, shifted by the max logit for stability. */
    float max_logit = logits[0];
    for (int i = 0; i < vocab_size; ++i) {
        if (!isfinite(logits[i])) {
            set_error(error, error_capacity, "Sampler received a non-finite logit.");
            return 0;
        }
        if (logits[i] > max_logit) max_logit = logits[i];
    }
    std::vector<double> probabilities((size_t)vocab_size);
    double sum = 0.0;
    for (int i = 0; i < vocab_size; ++i) {
        double p = exp((double)(logits[i] - max_logit) / (double)temperature);
        probabilities[(size_t)i] = p;
        sum += p;
    }
    if (!(sum > 0.0) || !isfinite(sum)) {
        set_error(error, error_capacity, "Sampler probability normalization failed.");
        return 0;
    }
    for (int i = 0; i < vocab_size; ++i) {
        probabilities[(size_t)i] /= sum;
    }

    /* Rank tokens by probability, then keep the smallest highest-probability
     * prefix whose cumulative mass reaches top_p (nucleus sampling). */
    std::vector<int> order((size_t)vocab_size);
    for (int i = 0; i < vocab_size; ++i) order[(size_t)i] = i;
    std::sort(order.begin(), order.end(), [&probabilities](int a, int b) {
        return probabilities[(size_t)a] > probabilities[(size_t)b];
    });

    double cumulative = 0.0;
    size_t nucleus_size = 0;
    for (size_t i = 0; i < order.size(); ++i) {
        cumulative += probabilities[(size_t)order[i]];
        nucleus_size = i + 1;
        if (cumulative >= (double)top_p) break;
    }

    double nucleus_sum = 0.0;
    for (size_t i = 0; i < nucleus_size; ++i) {
        nucleus_sum += probabilities[(size_t)order[i]];
    }

    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    double draw = uniform(rng_engine()) * nucleus_sum;
    double running = 0.0;
    int chosen = order[nucleus_size - 1];
    for (size_t i = 0; i < nucleus_size; ++i) {
        running += probabilities[(size_t)order[i]];
        if (draw <= running) {
            chosen = order[i];
            break;
        }
    }

    *token_id = chosen;
    if (error != NULL && error_capacity > 0) error[0] = '\0';
    return 1;
}
