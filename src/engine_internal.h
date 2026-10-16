#ifndef ENGINE_INTERNAL_H
#define ENGINE_INTERNAL_H

#include "gguf_reader.h"

typedef struct ArchitectureOps ArchitectureOps;

/* Every architecture-private inference state begins with this dispatch header. */
typedef struct EngineStateHeader {
    const ArchitectureOps *architecture;
} EngineStateHeader;

typedef struct TokenizerData {
    /* Opaque handles owned by tokenizer.cpp's llama.cpp adapter. */
    void *library_model;
    const void *library_vocab;
    int piece_count;
    int bos_token_id;
    int eos_token_id;
    int add_bos_token;
} TokenizerData;

typedef struct EngineModel {
    GgufFile gguf;
    TokenizerData tokenizer;
    const ArchitectureOps *architecture;
    void *architecture_data;
} EngineModel;

#endif
