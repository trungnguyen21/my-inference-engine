#ifndef ENGINE_GENERATE_H
#define ENGINE_GENERATE_H

#include "engine.h"
#include "metrics.h"

/*
 * The single generation loop shared by the command-line front end and the
 * C ABI used by out-of-process servers. It depends only on the public engine
 * primitives declared in engine.h, so a caller may substitute its own.
 *
 * Prompt preparation is separated from generation because the two front ends
 * size their inference state differently: the CLI allocates exactly the
 * context one run needs, while a server allocates a fixed ceiling once and
 * reuses it across requests.
 */

/* Room for a bounded prompt plus the single-user instruction wrapper. */
#define GENERATE_FORMATTED_PROMPT_CAPACITY (ENGINE_PROMPT_CAPACITY + 512)

/*
 * Receive one decoded token piece. text is NUL-terminated and valid only for
 * the duration of the call. Return 1 to continue generating, or 0 to stop
 * early; stopping is reported as success, not as an error.
 */
typedef int (*GenerateTokenFn)(const char *text, void *user);

typedef struct GenerateResult {
    int generated_tokens;            /* includes a terminating EOS, which is never emitted */
    int stopped_early;               /* the callback asked to stop */
    double time_to_first_token_seconds; /* -1.0 when no text was produced */
    UsageSnapshot start_usage;       /* captured after prefill, before decoding */
    UsageSnapshot end_usage;         /* captured after the final token */
} GenerateResult;

/*
 * Apply the model's instruction template to user_prompt and encode the result.
 * token_capacity bounds the output. On success write the count and return 1.
 */
int generate_prepare_prompt(const Model *model, const char *user_prompt,
                            int *token_ids, int token_capacity, int *token_count,
                            char *error, int error_capacity);

/*
 * Prefill token_ids, then sample up to max_tokens further tokens, handing each
 * decoded piece to on_token. state must be initialized with room for
 * token_count + max_tokens positions and positioned at zero; callers reusing a
 * state must call inference_state_reset first.
 *
 * repeat_penalty <= 1.0 disables the repetition penalty; above 1.0, every
 * token already seen in the prompt or generated so far has its logit pulled
 * toward zero before sampling, which keeps greedy decoding from looping.
 *
 * prompt_start_usage is the clock reading taken before prompt preparation; it
 * anchors time-to-first-token so that measurement spans prompt processing.
 * Pass an unavailable snapshot to skip that measurement.
 *
 * Return 1 with result filled in, or 0 with an error. A failed call leaves
 * result's counters valid for whatever was emitted before the failure.
 */
int generate_run(const Model *model, InferenceState *state,
                 const int *token_ids, int token_count,
                 int max_tokens, float temperature, float top_p,
                 float repeat_penalty,
                 const UsageSnapshot *prompt_start_usage,
                 GenerateTokenFn on_token, void *user,
                 GenerateResult *result, char *error, int error_capacity);

#endif
