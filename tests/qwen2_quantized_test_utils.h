#ifndef QWEN2_QUANTIZED_TEST_UTILS_H
#define QWEN2_QUANTIZED_TEST_UTILS_H

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "quantization.h"
#include "qwen2_internal.h"

/* Test fixtures only: each packed block represents one known constant. */
static inline int fixture_block_elements(uint32_t type)
{
    return type == ENGINE_GGML_TYPE_Q5_0 || type == ENGINE_GGML_TYPE_Q8_0 ? 32 : 256;
}

static inline int fixture_block_bytes(uint32_t type)
{
    switch (type) {
    case ENGINE_GGML_TYPE_Q5_0: return Q5_0_BLOCK_BYTES;
    case ENGINE_GGML_TYPE_Q8_0: return Q8_0_BLOCK_BYTES;
    case ENGINE_GGML_TYPE_Q4_K: return Q4_K_BLOCK_BYTES;
    case ENGINE_GGML_TYPE_Q6_K: return Q6_K_BLOCK_BYTES;
    default: assert(false); return 0;
    }
}

static inline float fixture_write_block(
    uint32_t type, int index, int small_scale, unsigned char *destination
)
{
    static const uint16_t scales[] = {0x3c00, 0x4000, 0x4200, 0x4400, 0x4500, 0x4600};
    uint16_t scale = scales[index % 6];
    float factor = (float)(index % 6 + 1);
    if (small_scale) {
        scale = (uint16_t)(scale - (10 << 10)); /* divide the F16 scale by 1024 */
        factor /= 1024.0f;
    }
    switch (type) {
    case ENGINE_GGML_TYPE_Q5_0: {
        Q5_0Block block = {};
        block.scale_f16_bits = scale; /* packed zero represents signed -16 */
        memcpy(destination, &block, sizeof(block));
        return -16.0f * factor;
    }
    case ENGINE_GGML_TYPE_Q8_0: {
        Q8_0Block block = {};
        block.scale_f16_bits = scale;
        for (int i = 0; i < Q8_0_BLOCK_ELEMENTS; ++i) block.quantized_values[i] = 2;
        memcpy(destination, &block, sizeof(block));
        return 2.0f * factor;
    }
    case ENGINE_GGML_TYPE_Q4_K: {
        Q4_KBlock block = {};
        block.scale_f16_bits = scale;
        memset(block.scale_and_min, 1, sizeof(block.scale_and_min));
        memset(block.packed_nibbles, 0x33, sizeof(block.packed_nibbles));
        memcpy(destination, &block, sizeof(block));
        return 3.0f * factor;
    }
    case ENGINE_GGML_TYPE_Q6_K: {
        Q6_KBlock block = {};
        block.scale_f16_bits = scale;
        memset(block.scales, 1, sizeof(block.scales)); /* packed zero means -32 */
        memcpy(destination, &block, sizeof(block));
        return -32.0f * factor;
    }
    default: assert(false); return 0.0f;
    }
}

static inline uint32_t fixture_parse_type(const char *name)
{
    if (strcmp(name, "q5_0") == 0) return ENGINE_GGML_TYPE_Q5_0;
    if (strcmp(name, "q8_0") == 0) return ENGINE_GGML_TYPE_Q8_0;
    if (strcmp(name, "q4_k") == 0) return ENGINE_GGML_TYPE_Q4_K;
    if (strcmp(name, "q6_k") == 0) return ENGINE_GGML_TYPE_Q6_K;
    fprintf(stderr, "Unknown type %s; use q5_0, q8_0, q4_k, or q6_k.\n", name);
    return 0;
}

typedef struct PackedWeightFixture {
    GgufFile file;
    GgufTensorInfo tensor;
    unsigned char storage[1 + 6 * Q6_K_BLOCK_BYTES];
    float block_values[6];
    int width;
} PackedWeightFixture;

static inline void fixture_init_packed(PackedWeightFixture *fixture, uint32_t type)
{
    memset(fixture, 0, sizeof(*fixture));
    int bytes = fixture_block_bytes(type);
    fixture->width = 2 * fixture_block_elements(type);
    for (int b = 0; b < 6; ++b) {
        fixture->block_values[b] = fixture_write_block(
            type, b, 0, fixture->storage + 1 + b * bytes);
    }
    fixture->tensor.name = (char *)"token_embd.weight";
    fixture->tensor.dimension_count = 2;
    fixture->tensor.dimensions[0] = (uint64_t)fixture->width;
    fixture->tensor.dimensions[1] = 3;
    fixture->tensor.type = type;
    fixture->tensor.offset = 1; /* deliberately unaligned packed storage */
    fixture->tensor.byte_size = (uint64_t)(6 * bytes);
    fixture->file.data = fixture->storage;
    fixture->file.file_size = 1 + fixture->tensor.byte_size;
    fixture->file.tensor_count = 1;
    fixture->file.tensors = &fixture->tensor;
}

#endif
