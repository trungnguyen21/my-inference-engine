#ifndef ARCHITECTURE_H
#define ARCHITECTURE_H

#include <stdint.h>

#include "engine.h"
#include "gguf_reader.h"

/*
 * ArchitectureOps is the boundary between shared engine infrastructure and a
 * model-family implementation. The generic engine selects one set of callbacks
 * from GGUF general.architecture, then never needs to know that family's
 * metadata keys, tensor names, or transformer schedule.
 */
typedef struct ArchitectureOps ArchitectureOps;

typedef int (*ArchitectureConfigureModelFn)(
    const GgufFile *file,
    int vocab_size,
    ModelConfig *config,
    void **architecture_data,
    char *error,
    int error_capacity
);

typedef void (*ArchitectureDestroyModelDataFn)(void *architecture_data);

typedef int (*ArchitectureStateInitFn)(
    const Model *model,
    InferenceState *state,
    int max_context_tokens,
    char *error,
    int error_capacity
);

typedef int (*ArchitectureForwardTokenFn)(
    const Model *model,
    InferenceState *state,
    int token_id,
    char *error,
    int error_capacity
);

typedef void (*ArchitectureStateDestroyFn)(InferenceState *state);

/* Format one user message plus the assistant-generation prefix. */
typedef int (*ArchitectureFormatPromptFn)(
    const char *user_prompt,
    char *output,
    int output_capacity,
    char *error,
    int error_capacity
);

struct ArchitectureOps {
    const char *identifier;
    ArchitectureConfigureModelFn configure_model;
    ArchitectureDestroyModelDataFn destroy_model_data;
    ArchitectureStateInitFn state_init;
    ArchitectureForwardTokenFn forward_token;
    ArchitectureStateDestroyFn state_destroy;
    ArchitectureFormatPromptFn format_prompt;
    /* Advance one cached token without final normalization or vocabulary logits.
     * Only used for intermediate prompt tokens; the next call must produce logits. */
    ArchitectureForwardTokenFn forward_token_no_logits;
};

/*
 * Return callbacks for a known, exact GGUF general.architecture value.
 * identifier is a byte span from the mapped GGUF metadata and need not be
 * NUL-terminated. Return nullptr for an unsupported architecture.
 */
const ArchitectureOps *architecture_find(
    const char *identifier,
    uint64_t identifier_length
);

#endif
