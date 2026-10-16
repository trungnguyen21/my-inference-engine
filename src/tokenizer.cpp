#include "engine.h"

#include "engine_internal.h"
#include "tokenizer.h"

#include <llama.h>

#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* The CLI owns one loaded model today; this count keeps the library lifecycle
   correct if that changes later. */
static int llama_backend_users = 0;

static void llama_log_callback(enum ggml_log_level level, const char *text,
                               void *user_data)
{
    (void)level;
    (void)text;
    (void)user_data;
}

static void set_error(char *error, int capacity, const char *message)
{
    if (error != NULL && capacity > 0) {
        snprintf(error, (size_t)capacity, "%s", message);
    }
}

static const struct llama_vocab *tokenizer_vocab(const TokenizerData *tokenizer)
{
    return (const struct llama_vocab *)tokenizer->library_vocab;
}

int tokenizer_load_from_gguf(const char *model_path, TokenizerData *tokenizer,
                             int expected_vocab_size,
                             char *error, int error_capacity)
{
    struct llama_model_params parameters;
    struct llama_model *library_model;
    const struct llama_vocab *vocab;
    int vocabulary_size;

    if (tokenizer == NULL || model_path == NULL || model_path[0] == '\0' ||
        expected_vocab_size <= 0) {
        set_error(error, error_capacity, "Tokenizer received invalid model metadata.");
        return 0;
    }
    memset(tokenizer, 0, sizeof(*tokenizer));
    // Initialize llama.cpp backend if this is the first tokenizer
    if (llama_backend_users == 0) {
        llama_log_set(llama_log_callback, NULL);
        llama_backend_init();
    }

    // Load the GGUF model
    parameters = llama_model_default_params();
    parameters.vocab_only = true;
    library_model = llama_model_load_from_file(model_path, parameters);
    if (library_model == NULL) {
        if (llama_backend_users == 0) {
            llama_backend_free();
            llama_log_set(NULL, NULL);
        }
        set_error(error, error_capacity,
                  "llama.cpp could not load tokenizer metadata from the GGUF file.");
        return 0;
    }

    // Obtain the vocabulary from the model
    vocab = llama_model_get_vocab(library_model);
    vocabulary_size = vocab == NULL ? 0 : llama_vocab_n_tokens(vocab);
    if (vocabulary_size != expected_vocab_size) {
        llama_model_free(library_model);
        if (llama_backend_users == 0) {
            llama_backend_free();
            llama_log_set(NULL, NULL);
        }
        set_error(error, error_capacity,
                  "Tokenizer vocabulary size does not match this engine's GGUF metadata.");
        return 0;
    }

    tokenizer->library_model = library_model;
    tokenizer->library_vocab = vocab;
    tokenizer->piece_count = vocabulary_size;
    tokenizer->bos_token_id = (int)llama_vocab_bos(vocab);
    tokenizer->eos_token_id = (int)llama_vocab_eos(vocab);
    tokenizer->add_bos_token = llama_vocab_get_add_bos(vocab) ? 1 : 0;
    ++llama_backend_users;
    return 1;
}

void tokenizer_destroy(TokenizerData *tokenizer)
{
    if (tokenizer == NULL) {
        return;
    }
    if (tokenizer->library_model != NULL) {
        llama_model_free((struct llama_model *)tokenizer->library_model);
        if (llama_backend_users > 0) {
            --llama_backend_users;
        }
        if (llama_backend_users == 0) {
            llama_backend_free();
            llama_log_set(NULL, NULL);
        }
    }
    memset(tokenizer, 0, sizeof(*tokenizer));
}

int tokenizer_encode(const Model *model, const char *text, int *token_ids,
                     int token_capacity, int *token_count,
                     char *error, int error_capacity)
{
    const EngineModel *implementation;
    const struct llama_vocab *vocab;
    llama_token *library_tokens;
    size_t text_length;
    int32_t encoded_count;
    int i;

    if (token_count != NULL) {
        *token_count = 0;
    }
    if (model == NULL || model->implementation == NULL || text == NULL ||
        token_ids == NULL || token_count == NULL || token_capacity <= 0) {
        set_error(error, error_capacity, "Tokenizer encoding received invalid arguments.");
        return 0;
    }
    implementation = (const EngineModel *)model->implementation;
    vocab = tokenizer_vocab(&implementation->tokenizer);
    if (vocab == NULL || implementation->tokenizer.piece_count != model->config.vocab_size) {
        set_error(error, error_capacity, "Tokenizer vocabulary is not loaded for this model.");
        return 0;
    }
    text_length = strlen(text);
    if (text_length > INT32_MAX || token_capacity > INT32_MAX ||
        (size_t)token_capacity > SIZE_MAX / sizeof(*library_tokens)) {
        set_error(error, error_capacity, "Tokenizer input or output capacity is too large.");
        return 0;
    }
    library_tokens = (llama_token *)malloc((size_t)token_capacity * sizeof(*library_tokens));
    if (library_tokens == NULL) {
        set_error(error, error_capacity, "Could not allocate temporary tokenizer output.");
        return 0;
    }

    /* Parse the instruction wrapper's role/end markers as special token IDs. */
    encoded_count = llama_tokenize(vocab, text, (int32_t)text_length,
                                   library_tokens, (int32_t)token_capacity,
                                   true, true);
    if (encoded_count < 0) {
        free(library_tokens);
        if (encoded_count == INT32_MIN) {
            set_error(error, error_capacity, "Tokenizer output size overflowed.");
        } else if (error != NULL && error_capacity > 0) {
            snprintf(error, (size_t)error_capacity,
                     "Prompt needs %d token slots, but capacity is %d.",
                     -encoded_count, token_capacity);
        }
        return 0;
    }
    for (i = 0; i < encoded_count; ++i) {
        if (library_tokens[i] < 0 || library_tokens[i] >= model->config.vocab_size) {
            free(library_tokens);
            set_error(error, error_capacity, "Tokenizer returned an ID outside the model vocabulary.");
            return 0;
        }
        token_ids[i] = (int)library_tokens[i];
    }
    free(library_tokens);
    *token_count = (int)encoded_count;
    return 1;
}

int tokenizer_decode_one(const Model *model, int token_id, char *text,
                         int text_capacity, char *error, int error_capacity)
{
    const EngineModel *implementation;
    const struct llama_vocab *vocab;
    int32_t piece_length;

    if (text != NULL && text_capacity > 0) {
        text[0] = '\0';
    }
    if (model == NULL || model->implementation == NULL || text == NULL ||
        text_capacity <= 1 || token_id < 0 || token_id >= model->config.vocab_size) {
        set_error(error, error_capacity, "Tokenizer decoding received invalid arguments.");
        return 0;
    }
    implementation = (const EngineModel *)model->implementation;
    vocab = tokenizer_vocab(&implementation->tokenizer);
    if (vocab == NULL) {
        set_error(error, error_capacity, "Tokenizer vocabulary is not loaded for this model.");
        return 0;
    }

    // This converts the token ID back to the correct output bytes
    piece_length = llama_token_to_piece(vocab, (llama_token)token_id, text,
                                        text_capacity - 1, 0, false);

    // llama.cpp does not append \0 so we did it ourselves
    if (piece_length < 0) {
        if (error != NULL && error_capacity > 0) {
            snprintf(error, (size_t)error_capacity,
                     "Decoded token needs %d bytes, but text capacity is %d.",
                     -piece_length, text_capacity - 1);
        }
        return 0;
    }
    text[piece_length] = '\0';
    return 1;
}

int tokenizer_is_eos(const Model *model, int token_id)
{
    const EngineModel *implementation;
    const struct llama_vocab *vocab;

    if (model == NULL || model->implementation == NULL || token_id < 0 ||
        token_id >= model->config.vocab_size) {
        return 0;
    }
    implementation = (const EngineModel *)model->implementation;
    vocab = tokenizer_vocab(&implementation->tokenizer);
    return vocab != NULL && llama_vocab_is_eog(vocab, (llama_token)token_id);
}
