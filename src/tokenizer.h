#ifndef INFERENCE_TOKENIZER_H
#define INFERENCE_TOKENIZER_H

#include "engine_internal.h"

/* Load only the GGUF vocabulary through the external tokenizer adapter. */
int tokenizer_load_from_gguf(const char *model_path, TokenizerData *tokenizer,
                             int expected_vocab_size,
                             char *error, int error_capacity);
void tokenizer_destroy(TokenizerData *tokenizer);

#endif
