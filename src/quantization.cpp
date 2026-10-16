#include <math.h>
#include "quantization.h"

int f16_bits_to_f32(uint16_t bits, float *result)
{
    if (result == nullptr) {
        return 0;
    }

    uint16_t sign = (bits >> 15) & 1;
    uint16_t exponent = (bits >> 10) & 0x1f;
    uint16_t fraction = bits & 0x03ff;

    if (exponent == 31) {
        if (fraction == 0) {
            if (!sign) {
                *result = INFINITY;
            } else {
                *result = -INFINITY;
            }
        } else {
            *result = NAN;
        }
    } else if (exponent == 0) {
        // formula for subnormal values (eg: 0.000025)
        // (-1)^sign * (0 + fraction) * 2^-14
        *result = (powf(-1, sign)) * ldexpf((0 + fraction / 1024.0), -14);
    } else {
        // formula: sign * 2^(exponent - 15) * (1 + fraction / 1024)
        *result = (ldexpf(powf(-1, sign), (int64_t) exponent - 15)) * (1 + fraction / 1024.0);
    }

    return 1;
}

int q8_0_dequantize_block(const Q8_0Block *block, float *output)
{
    if ((block == nullptr) || (output == nullptr)) {
        return 0;
    }
    float decoded_scale = 0.0f;
    f16_bits_to_f32(block->scale_f16_bits, &decoded_scale);

    for (int i = 0; i < Q8_0_BLOCK_ELEMENTS; i++) {
        output[i] = decoded_scale * block->quantized_values[i];
    }

    return 1;
}

int q8_0_dot_f32(const Q8_0Block *block, const float *input, float *result)
{
    if ((block == nullptr) || (input == nullptr) || (result == nullptr)) {
        return 0;
    }
    float decoded_scale = 0.0f;
    f16_bits_to_f32(block->scale_f16_bits, &decoded_scale);
    float res = 0.0f;
    for (int i = 0; i < Q8_0_BLOCK_ELEMENTS; i++) {
        res += decoded_scale * block->quantized_values[i] * input[i];
    }
    *result = res;
    return 1;
}

int q5_0_dequantize_block(const Q5_0Block *block, float *output)
{
    if ((block == nullptr) || (output == nullptr)) {
        return 0;
    }
    float decoded_scale = 0.0f;
    f16_bits_to_f32(block->scale_f16_bits, &decoded_scale);

    for (int i = 0; i < Q5_0_BLOCK_ELEMENTS; i++) {
        uint8_t packed = block->packed_nibbles[i % 16];
        uint8_t low_four_bits;
        if (i <= 15) {
            low_four_bits = packed & 0x0F;
        } else {
            low_four_bits = (packed >> 4) & 0x0F;
        }
        bool fifth_bit = (block->high_bits[i / 8] >> (i % 8)) & 1;
        uint8_t encoded = low_four_bits | (fifth_bit << 4);
        int8_t signed_value = encoded - 16;
        output[i] = decoded_scale * signed_value;
    }

    return 1;
}

int q5_0_dot_f32(const Q5_0Block *block, const float *input, float *result)
{
    if ((block == nullptr) || (input == nullptr) || (result == nullptr)) {
        return 0;
    }

    float res = 0.0f;
    float decoded_scale = 0.0f;
    f16_bits_to_f32(block->scale_f16_bits, &decoded_scale);

    for (int i = 0; i < Q5_0_BLOCK_ELEMENTS; i++) {
        uint8_t packed = block->packed_nibbles[i % 16];
        uint8_t low_four_bits;
        if (i <= 15) {
            low_four_bits = packed & 0x0F;
        } else {
            low_four_bits = (packed >> 4) & 0x0F;
        }
        bool fifth_bit = (block->high_bits[i / 8] >> (i % 8)) & 1;
        uint8_t encoded = low_four_bits | (fifth_bit << 4);
        int8_t signed_value = encoded - 16;
        res += decoded_scale * signed_value * input[i];
    }
    *result = res;
    return 1;
}

int q4_k_unpack_scale_min(
    const uint8_t *scale_and_min,
    int subblock_index,
    uint8_t *scale_multiplier,
    uint8_t *min_multiplier
)
{
    if (scale_and_min == nullptr || scale_multiplier == nullptr ||
        min_multiplier == nullptr || subblock_index < 0 ||
        subblock_index >= Q4_K_SUBBLOCK_COUNT) {
        return 0;
    }

    uint8_t scale;
    uint8_t minimum;
    if (subblock_index < 4) {
        scale = scale_and_min[subblock_index] & 0x3f;
        minimum = scale_and_min[subblock_index + 4] & 0x3f;
    } else {
        scale = (scale_and_min[subblock_index + 4] & 0x0f) |
                ((scale_and_min[subblock_index - 4] >> 6) << 4);
        minimum = (scale_and_min[subblock_index + 4] >> 4) |
                  ((scale_and_min[subblock_index] >> 6) << 4);
    }
    *scale_multiplier = scale;
    *min_multiplier = minimum;
    return 1;
}

int q4_k_dequantize_block(const Q4_KBlock *block, float *output)
{
    if (block == nullptr || output == nullptr) {
        return 0;
    }

    float scale_factor = 0.0f;
    float min_factor = 0.0f;
    f16_bits_to_f32(block->scale_f16_bits, &scale_factor);
    f16_bits_to_f32(block->min_f16_bits, &min_factor);

    for (int group = 0; group < 4; ++group) {
        const uint8_t *packed = block->packed_nibbles + group * 32;
        for (int half = 0; half < 2; ++half) {
            const int subblock = group * 2 + half;
            uint8_t scale_multiplier, min_multiplier;
            q4_k_unpack_scale_min(block->scale_and_min, subblock,
                                  &scale_multiplier, &min_multiplier);
            const float scale = scale_factor * scale_multiplier;
            const float minimum = min_factor * min_multiplier;
            float *out = output + group * 64 + half * 32;
            for (int i = 0; i < 32; ++i) {
                const uint8_t byte = packed[i];
                const uint8_t value = half == 0 ? byte & 0x0f : byte >> 4;
                out[i] = scale * value - minimum;
            }
        }
    }
    return 1;
}

int q4_k_dot_f32(const Q4_KBlock *block, const float *input, float *result)
{
    if (block == nullptr || input == nullptr || result == nullptr) {
        return 0;
    }

    float scale_factor = 0.0f;
    float min_factor = 0.0f;
    f16_bits_to_f32(block->scale_f16_bits, &scale_factor);
    f16_bits_to_f32(block->min_f16_bits, &min_factor);

    float sum = 0.0f;
    for (int subblock = 0; subblock < Q4_K_SUBBLOCK_COUNT; ++subblock) {
        uint8_t scale_multiplier, min_multiplier;
        q4_k_unpack_scale_min(block->scale_and_min, subblock,
                              &scale_multiplier, &min_multiplier);
        const float scale = scale_factor * scale_multiplier;
        const float minimum = min_factor * min_multiplier;
        const int group = subblock / 2;
        const int half = subblock % 2;
        const uint8_t *packed = block->packed_nibbles + group * 32;
        const float *values = input + group * 64 + half * 32;
        const int shift = half * 4;
        for (int i = 0; i < 32; ++i) {
            const float decoded = scale * ((packed[i] >> shift) & 0x0f) - minimum;
            sum += decoded * values[i];
        }
    }
    *result = sum;
    return 1;
}

int q6_k_unpack_value(
    const Q6_KBlock *block,
    int index,
    int8_t *quantized_value,
    int8_t *scale_multiplier
)
{
    if (block == nullptr || quantized_value == nullptr ||
        scale_multiplier == nullptr || index < 0 ||
        index >= Q6_K_BLOCK_ELEMENTS) {
        return 0;
    }

    const int group = index / Q6_K_GROUP_ELEMENTS;
    const int local = index % Q6_K_GROUP_ELEMENTS;
    const int region = local / 32;
    const int offset = local % 32;
    const int lower_index = group * 64 + (region % 2) * 32 + offset;
    const int upper_index = group * 32 + offset;
    const int low_shift = (region / 2) * 4;
    const int upper_shift = region * 2;
    const int low = (block->lower_bits[lower_index] >> low_shift) & 0x0f;
    const int upper = (block->upper_bits[upper_index] >> upper_shift) & 0x03;

    *quantized_value = (int8_t)((low | (upper << 4)) - 32);
    *scale_multiplier = block->scales[group * 8 + region * 2 + offset / 16];
    return 1;
}

int q6_k_dequantize_block(const Q6_KBlock *block, float *output)
{
    if (block == nullptr || output == nullptr) {
        return 0;
    }

    float scale_factor = 0.0f;
    f16_bits_to_f32(block->scale_f16_bits, &scale_factor);

    for (int group = 0; group < 2; ++group) {
        for (int region = 0; region < 4; ++region) {
            const int lower_base = group * 64 + (region & 1) * 32;
            const int upper_shift = region * 2;
            const int output_base = group * 128 + region * 32;
            const int scale_base = group * 8 + region * 2;
            for (int offset = 0; offset < 32; ++offset) {
                const int low_shift = (region >> 1) * 4;
                const int low = (block->lower_bits[lower_base + offset] >> low_shift) & 0x0f;
                const int high = (block->upper_bits[group * 32 + offset] >> upper_shift) & 0x03;
                const int quantized = (low | (high << 4)) - 32;
                const int8_t multiplier = block->scales[scale_base + offset / 16];
                output[output_base + offset] =
                    scale_factor * multiplier * quantized;
            }
        }
    }
    return 1;
}

int q6_k_dot_f32(const Q6_KBlock *block, const float *input, float *result)
{
    if (block == nullptr || input == nullptr || result == nullptr) {
        return 0;
    }

    float scale_factor = 0.0f;
    f16_bits_to_f32(block->scale_f16_bits, &scale_factor);

    float sum = 0.0f;
    for (int group = 0; group < 2; ++group) {
        for (int region = 0; region < 4; ++region) {
            const int lower_base = group * 64 + (region & 1) * 32;
            const int upper_shift = region * 2;
            const int input_base = group * 128 + region * 32;
            const int scale_base = group * 8 + region * 2;
            const int low_shift = (region >> 1) * 4;
            for (int offset = 0; offset < 32; ++offset) {
                const int low = (block->lower_bits[lower_base + offset] >> low_shift) & 0x0f;
                const int high = (block->upper_bits[group * 32 + offset] >> upper_shift) & 0x03;
                const int quantized = (low | (high << 4)) - 32;
                const int8_t multiplier = block->scales[scale_base + offset / 16];
                sum += (scale_factor * multiplier * quantized) * input[input_base + offset];
            }
        }
    }

    *result = sum;
    return 1;
}
