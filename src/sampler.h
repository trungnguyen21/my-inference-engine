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

/*
 * Discourage tokens that already appeared earlier in the sequence, breaking
 * the loops plain greedy decoding falls into. seen_mask has vocab_size
 * entries, one nonzero byte per token ID that has already been emitted
 * (prompt or generated). penalty <= 1.0 is a no-op; otherwise each seen
 * token's logit is divided by penalty when positive, or multiplied by
 * penalty when negative, matching the standard CTRL-style repetition
 * penalty. Modifies logits in place. Returns 0 on invalid arguments.
 */
int sampler_apply_repetition_penalty(float *logits, int vocab_size,
                                     const unsigned char *seen_mask,
                                     float penalty);

#endif
