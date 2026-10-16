#ifndef INFERENCE_SAMPLER_H
#define INFERENCE_SAMPLER_H

/*
 * Choose the largest raw logit, breaking ties by lowest token ID.
 * No softmax, randomness, allocation, or modification of logits is needed.
 * Reject null pointers, nonpositive vocabulary sizes, and every NaN/Inf.
 * Return 1 and write a vocabulary ID only after validating the entire vector.
 * On failure return 0, reset *token_id to -1 when writable, and report an error.
 * A null/zero-capacity error buffer is allowed.
 */
int sampler_greedy_sample(const float *logits, int vocab_size, int *token_id,
                          char *error, int error_capacity);

#endif
