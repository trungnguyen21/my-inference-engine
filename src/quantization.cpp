#include <math.h>
#include <string.h>
#include "quantization.h"

#if defined(__aarch64__) && defined(__ARM_NEON)
#include <arm_neon.h>
#elif defined(__x86_64__) || defined(__amd64__)
#include <immintrin.h>
#endif

int f16_bits_to_f32(uint16_t bits, float *result)
{
    if (result == nullptr) return 0;

#if defined(__aarch64__) && defined(__ARM_FEATURE_FP16_SCALAR_ARITHMETIC)
    // memcpy moves the raw encoding without violating aliasing rules. The
    // conversion to float can use ARM's native FP16 -> F32 instruction.
    static_assert(sizeof(__fp16) == sizeof(bits), "Expected 16-bit ARM FP16");
    __fp16 half;
    memcpy(&half, &bits, sizeof(half));
    *result = (float)half;
#else
    // Portable IEEE-754 bit expansion, including signed zero and subnormals.
    // memcpy avoids type-punning through an incompatible pointer type.
    static_assert(sizeof(float) == sizeof(uint32_t), "Expected binary32 float");
    uint32_t sign = ((uint32_t)bits & 0x8000u) << 16;
    uint32_t exponent = (bits >> 10) & 0x1fu;
    uint32_t fraction = bits & 0x03ffu;
    uint32_t expanded;
    if (exponent == 0) {
        if (fraction == 0) {
            expanded = sign;
        } else {
            int power = -14;
            while ((fraction & 0x400u) == 0) {
                fraction <<= 1;
                --power;
            }
            expanded = sign | ((uint32_t)(power + 127) << 23) |
                       ((fraction & 0x3ffu) << 13);
        }
    } else if (exponent == 31) {
        expanded = sign | 0x7f800000u | (fraction << 13);
        // Quiet signaling NaNs before they enter F32 arithmetic.
        if (fraction != 0) expanded |= 0x00400000u;
    } else {
        expanded = sign | ((exponent + 112u) << 23) | (fraction << 13);
    }
    memcpy(result, &expanded, sizeof(expanded));
#endif
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

#if defined(__aarch64__) && defined(__ARM_NEON)
/* Shared by Q8_0 and Q5_0: widen eight signed weights to F32, then
 * accumulate scale * weight * input without materializing a decoded block. */
static inline void quantized_accumulate_8(
    int8x8_t weights, const float *input, float scale,
    float32x4_t *sum_lo, float32x4_t *sum_hi
)
{
    const int16x8_t w16 = vmovl_s8(weights);
    const float32x4_t lo = vcvtq_f32_s32(vmovl_s16(vget_low_s16(w16)));
    const float32x4_t hi = vcvtq_f32_s32(vmovl_s16(vget_high_s16(w16)));
    *sum_lo = vaddq_f32(*sum_lo,
                         vmulq_f32(vmulq_n_f32(lo, scale), vld1q_f32(input)));
    *sum_hi = vaddq_f32(*sum_hi,
                         vmulq_f32(vmulq_n_f32(hi, scale), vld1q_f32(input + 4)));
}

static float q8_0_dot_neon(const Q8_0Block *block, const float *input, float scale)
{
    float32x4_t sum0 = vdupq_n_f32(0.0f);
    float32x4_t sum1 = vdupq_n_f32(0.0f);
    float32x4_t sum2 = vdupq_n_f32(0.0f);
    float32x4_t sum3 = vdupq_n_f32(0.0f);
    for (int i = 0; i < Q8_0_BLOCK_ELEMENTS; i += 16) {
        const int8x16_t weights = vld1q_s8(block->quantized_values + i);
        quantized_accumulate_8(vget_low_s8(weights), input + i, scale, &sum0, &sum1);
        quantized_accumulate_8(vget_high_s8(weights), input + i + 8, scale, &sum2, &sum3);
    }
    /* As with Q5_0, reducing vector lanes changes the summation order. */
    return vaddvq_f32(vaddq_f32(vaddq_f32(sum0, sum1),
                                vaddq_f32(sum2, sum3)));
}
#endif

#if defined(__AVX2__)
static float q8_0_dot_avx2(const Q8_0Block *block, const float *input, float scale)
{
    // Create an accumulator filled with 8 zeros
    __m256 sum = _mm256_setzero_ps();

    // Broadcast the scale to all 8 lanes
    __m256 v_scale = _mm256_set1_ps(scale);

    // Q8_0 has 32 elements. AVX2 processes 8 at a time.
    for (int i = 0; i < Q8_0_BLOCK_ELEMENTS; i += 8) {
        // 1. Load 8 integers (AVX2 has a specific instruction to load and widen 8-bit ints to 32-bit ints)
        __m128i int8_vals = _mm_loadu_si128((const __m128i*)(block->quantized_values + i));
        __m256i int32_vals = _mm256_cvtepi8_epi32(int8_vals);

        // 2. Convert 32-bit integers to 32-bit floats
        __m256 float_weights = _mm256_cvtepi32_ps(int32_vals);

        // 3. Load 8 input activations
        __m256 input_vals = _mm256_loadu_ps(input + i);

        // 4. Math: weights * scale * inputs.
        // We use FMA (Fused Multiply-Add): sum = (weights * scale) * inputs + sum
        __m256 scaled_weights = _mm256_mul_ps(float_weights, v_scale);
        sum = _mm256_fmadd_ps(scaled_weights, input_vals, sum);
    }

    // Finally, add the 8 lanes together into a single float
    // (AVX2 makes this slightly annoying, but here is the standard idiom)
    __m128 sum_half = _mm_add_ps(_mm256_castps256_ps128(sum), _mm256_extractf128_ps(sum, 1));
    sum_half = _mm_hadd_ps(sum_half, sum_half);
    sum_half = _mm_hadd_ps(sum_half, sum_half);

    return _mm_cvtss_f32(sum_half);
}
#endif

int q8_0_dot_f32(const Q8_0Block *block, const float *input, float *result)
{
    if ((block == nullptr) || (input == nullptr) || (result == nullptr)) {
        return 0;
    }
    float decoded_scale = 0.0f;
    f16_bits_to_f32(block->scale_f16_bits, &decoded_scale);
#if defined(__aarch64__) && defined(__ARM_NEON)
    *result = q8_0_dot_neon(block, input, decoded_scale);
#elif defined(__AVX2__)
    *result = q8_0_dot_avx2(block, input, decoded_scale);
#else
    float res = 0.0f;
    for (int i = 0; i < Q8_0_BLOCK_ELEMENTS; i++) {
        res += decoded_scale * block->quantized_values[i] * input[i];
    }
    *result = res;
#endif
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

#if defined(__aarch64__) && defined(__ARM_NEON)
static float q5_0_dot_neon(const Q5_0Block *block, const float *input, float scale)
{
    /* One packed byte holds weight i in its low nibble and weight i+16 in
     * its high nibble. Each half's fifth bits occupy two high_bits bytes. */
    static const uint8_t bit_masks[8] = {1, 2, 4, 8, 16, 32, 64, 128};
    const uint8x8_t masks = vld1_u8(bit_masks);
    const uint8x8_t high_bit = vdup_n_u8(16);
    const int8x8_t center = vdup_n_s8(16);
    float32x4_t sum0 = vdupq_n_f32(0.0f);
    float32x4_t sum1 = vdupq_n_f32(0.0f);
    float32x4_t sum2 = vdupq_n_f32(0.0f);
    float32x4_t sum3 = vdupq_n_f32(0.0f);

    for (int i = 0; i < 16; i += 8) {
        const uint8x8_t packed = vld1_u8(block->packed_nibbles + i);
        const uint8x8_t low_bits = vand_u8(packed, vdup_n_u8(15));
        const uint8x8_t upper_bits = vshr_n_u8(packed, 4);
        const uint8x8_t low_fifth = vand_u8(
            vtst_u8(vdup_n_u8(block->high_bits[i / 8]), masks), high_bit);
        const uint8x8_t upper_fifth = vand_u8(
            vtst_u8(vdup_n_u8(block->high_bits[2 + i / 8]), masks), high_bit);
        const int8x8_t low = vsub_s8(
            vreinterpret_s8_u8(vorr_u8(low_bits, low_fifth)), center);
        const int8x8_t upper = vsub_s8(
            vreinterpret_s8_u8(vorr_u8(upper_bits, upper_fifth)), center);
        quantized_accumulate_8(low, input + i, scale, &sum0, &sum1);
        quantized_accumulate_8(upper, input + 16 + i, scale, &sum2, &sum3);
    }
    /* Vector lanes accumulate in a different order from the scalar loop. */
    return vaddvq_f32(vaddq_f32(vaddq_f32(sum0, sum1),
                                vaddq_f32(sum2, sum3)));
}
#endif

#if defined(__AVX2__)
static inline void q5_0_accumulate_8_avx2(
    __m128i weights, const float *input, __m256 v_scale,
    __m256 *sum
)
{
    // Widen 8 signed 8-bit integers to 32-bit integers, then convert to floats
    __m256i int32_vals = _mm256_cvtepi8_epi32(weights);
    __m256 float_weights = _mm256_cvtepi32_ps(int32_vals);

    // Load 8 input activations
    __m256 input_vals = _mm256_loadu_ps(input);

    // Math: weights * scale * inputs + sum
    __m256 scaled_weights = _mm256_mul_ps(float_weights, v_scale);
    *sum = _mm256_fmadd_ps(scaled_weights, input_vals, *sum);
}

static float q5_0_dot_avx2(const Q5_0Block *block, const float *input, float scale)
{
    __m256 sum0 = _mm256_setzero_ps();
    __m256 sum1 = _mm256_setzero_ps();
    __m256 sum2 = _mm256_setzero_ps();
    __m256 sum3 = _mm256_setzero_ps();

    __m256 v_scale = _mm256_set1_ps(scale);
    __m128i center = _mm_set1_epi8(16);
    __m128i mask_low = _mm_set1_epi8(0x0F);

    // Q5_0 block has 32 elements. The 16 packed bytes hold all 32 values' low 4 bits.
    // i=0 corresponds to the first 8 packed bytes.
    for (int i = 0; i < 16; i += 8) {
        // Load 8 packed bytes (which contain 16 weights: 8 low, 8 high)
        // Since we only want to load 64 bits (8 bytes), we use _mm_loadl_epi64.
        __m128i packed = _mm_loadl_epi64((const __m128i*)(block->packed_nibbles + i));

        // Extract low and high nibbles
        __m128i low_bits = _mm_and_si128(packed, mask_low);
        __m128i upper_bits = _mm_and_si128(_mm_srli_epi16(packed, 4), mask_low);

        // The fifth bits for the entire block are in block->high_bits (4 bytes).
        // Each bit in high_bits corresponds to a weight.
        // For i=0, we need bits 0-7 for low, and bits 16-23 for upper.
        // For i=8, we need bits 8-15 for low, and bits 24-31 for upper.

        // Load the single byte containing the 8 fifth bits we need.
        uint8_t low_fifth_byte = block->high_bits[i / 8];
        uint8_t upper_fifth_byte = block->high_bits[2 + i / 8];

        // To combine them efficiently in AVX2, we broadcast the single byte to a 64-bit vector,
        // then use a shuffle to isolate each bit, but a simpler portable AVX2 way for 8 bits
        // is to unpack them manually into an __m128i register.

        // We can create a vector where each byte has the corresponding bit isolated.
        __m128i bit_masks = _mm_setr_epi8(1, 2, 4, 8, 16, 32, 64, 128, 0, 0, 0, 0, 0, 0, 0, 0);
        __m128i v_low_fifth_byte = _mm_set1_epi8(low_fifth_byte);
        __m128i v_upper_fifth_byte = _mm_set1_epi8(upper_fifth_byte);

        // Test bits: if the bit is set, the result is the bit mask itself. We want 0 or 16.
        // Compare with mask to get 0xFF if set, 0x00 if not.
        __m128i low_fifth = _mm_and_si128(v_low_fifth_byte, bit_masks);
        __m128i upper_fifth = _mm_and_si128(v_upper_fifth_byte, bit_masks);

        // _mm_cmpeq_epi8 gives 0xFF if equal, 0x00 if not. We compare against bit_masks.
        low_fifth = _mm_cmpeq_epi8(low_fifth, bit_masks);
        upper_fifth = _mm_cmpeq_epi8(upper_fifth, bit_masks);

        // Now mask with 16 to set the fifth bit value (0 or 16)
        __m128i bit_16 = _mm_set1_epi8(16);
        low_fifth = _mm_and_si128(low_fifth, bit_16);
        upper_fifth = _mm_and_si128(upper_fifth, bit_16);

        // Combine nibbles and fifth bit, subtract 16
        __m128i low = _mm_sub_epi8(_mm_or_si128(low_bits, low_fifth), center);
        __m128i upper = _mm_sub_epi8(_mm_or_si128(upper_bits, upper_fifth), center);

        // Accumulate into our running sums
        if (i == 0) {
            q5_0_accumulate_8_avx2(low, input, v_scale, &sum0);
            q5_0_accumulate_8_avx2(upper, input + 16, v_scale, &sum2);
        } else {
            q5_0_accumulate_8_avx2(low, input + 8, v_scale, &sum1);
            q5_0_accumulate_8_avx2(upper, input + 24, v_scale, &sum3);
        }
    }

    // Combine the 4 sums into one
    __m256 sum = _mm256_add_ps(_mm256_add_ps(sum0, sum1), _mm256_add_ps(sum2, sum3));

    // Horizontal addition to reduce the 8 lanes to a single float
    __m128 sum_half = _mm_add_ps(_mm256_castps256_ps128(sum), _mm256_extractf128_ps(sum, 1));
    sum_half = _mm_hadd_ps(sum_half, sum_half);
    sum_half = _mm_hadd_ps(sum_half, sum_half);

    return _mm_cvtss_f32(sum_half);
}
#endif

int q5_0_dot_f32(const Q5_0Block *block, const float *input, float *result)
{
    if ((block == nullptr) || (input == nullptr) || (result == nullptr)) {
        return 0;
    }

    float decoded_scale = 0.0f;
    f16_bits_to_f32(block->scale_f16_bits, &decoded_scale);
#if defined(__aarch64__) && defined(__ARM_NEON)
    *result = q5_0_dot_neon(block, input, decoded_scale);
#elif defined(__AVX2__)
    *result = q5_0_dot_avx2(block, input, decoded_scale);
#else
    float res = 0.0f;
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
#endif
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

#if defined(__aarch64__) && defined(__ARM_NEON)
/* Q4_K has an affine (scale * nibble - minimum) transform per 32 values. */
static inline void q4_k_accumulate_8(
    uint8x8_t nibbles, const float *input, float scale, float minimum,
    float32x4_t *sum_lo, float32x4_t *sum_hi
)
{
    const uint16x8_t w16 = vmovl_u8(nibbles);
    const float32x4_t lo = vcvtq_f32_u32(vmovl_u16(vget_low_u16(w16)));
    const float32x4_t hi = vcvtq_f32_u32(vmovl_u16(vget_high_u16(w16)));
    const float32x4_t bias = vdupq_n_f32(minimum);
    *sum_lo = vaddq_f32(*sum_lo, vmulq_f32(
        vsubq_f32(vmulq_n_f32(lo, scale), bias), vld1q_f32(input)));
    *sum_hi = vaddq_f32(*sum_hi, vmulq_f32(
        vsubq_f32(vmulq_n_f32(hi, scale), bias), vld1q_f32(input + 4)));
}

static float q4_k_dot_neon(
    const Q4_KBlock *block, const float *input,
    float scale_factor, float min_factor
)
{
    float32x4_t sums[4] = {
        vdupq_n_f32(0.0f), vdupq_n_f32(0.0f),
        vdupq_n_f32(0.0f), vdupq_n_f32(0.0f)
    };
    for (int subblock = 0; subblock < Q4_K_SUBBLOCK_COUNT; ++subblock) {
        uint8_t scale_multiplier, min_multiplier;
        q4_k_unpack_scale_min(block->scale_and_min, subblock,
                              &scale_multiplier, &min_multiplier);
        const float scale = scale_factor * scale_multiplier;
        const float minimum = min_factor * min_multiplier;
        const uint8_t *packed = block->packed_nibbles + (subblock / 2) * 32;
        const float *values = input + subblock * 32;
        for (int i = 0; i < 32; i += 16) {
            const uint8x16_t bytes = vld1q_u8(packed + i);
            const uint8x16_t nibbles = (subblock & 1)
                ? vshrq_n_u8(bytes, 4)
                : vandq_u8(bytes, vdupq_n_u8(15));
            q4_k_accumulate_8(vget_low_u8(nibbles), values + i,
                              scale, minimum, &sums[0], &sums[1]);
            q4_k_accumulate_8(vget_high_u8(nibbles), values + i + 8,
                              scale, minimum, &sums[2], &sums[3]);
        }
    }
    return vaddvq_f32(vaddq_f32(vaddq_f32(sums[0], sums[1]),
                                vaddq_f32(sums[2], sums[3])));
}
#endif

int q4_k_dot_f32(const Q4_KBlock *block, const float *input, float *result)
{
    if (block == nullptr || input == nullptr || result == nullptr) {
        return 0;
    }

    float scale_factor = 0.0f;
    float min_factor = 0.0f;
    f16_bits_to_f32(block->scale_f16_bits, &scale_factor);
    f16_bits_to_f32(block->min_f16_bits, &min_factor);

#if defined(__aarch64__) && defined(__ARM_NEON)
    *result = q4_k_dot_neon(block, input, scale_factor, min_factor);
#else
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
#endif
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

#if defined(__aarch64__) && defined(__ARM_NEON)
static float q6_k_dot_neon(const Q6_KBlock *block, const float *input, float scale_factor)
{
    float32x4_t sums[4] = {
        vdupq_n_f32(0.0f), vdupq_n_f32(0.0f),
        vdupq_n_f32(0.0f), vdupq_n_f32(0.0f)
    };
    for (int group = 0; group < 2; ++group) {
        for (int region = 0; region < 4; ++region) {
            const int lower_base = group * 64 + (region & 1) * 32;
            const int upper_base = group * 32;
            const int input_base = group * 128 + region * 32;
            const int scale_base = group * 8 + region * 2;
            /* Negative vshl shift counts select a right shift. */
            const int8x16_t low_shift = vdupq_n_s8(-((region >> 1) * 4));
            const int8x16_t high_shift = vdupq_n_s8(-(region * 2));
            for (int half = 0; half < 2; ++half) {
                const int offset = half * 16;
                const uint8x16_t lower = vld1q_u8(block->lower_bits + lower_base + offset);
                const uint8x16_t upper = vld1q_u8(block->upper_bits + upper_base + offset);
                const uint8x16_t lo = vandq_u8(vshlq_u8(lower, low_shift), vdupq_n_u8(15));
                const uint8x16_t hi = vshlq_n_u8(
                    vandq_u8(vshlq_u8(upper, high_shift), vdupq_n_u8(3)), 4);
                const int8x16_t weights = vsubq_s8(
                    vreinterpretq_s8_u8(vorrq_u8(lo, hi)), vdupq_n_s8(32));
                const float scale = scale_factor * block->scales[scale_base + half];
                const float *values = input + input_base + offset;
                quantized_accumulate_8(vget_low_s8(weights), values,
                                       scale, &sums[0], &sums[1]);
                quantized_accumulate_8(vget_high_s8(weights), values + 8,
                                       scale, &sums[2], &sums[3]);
            }
        }
    }
    return vaddvq_f32(vaddq_f32(vaddq_f32(sums[0], sums[1]),
                                vaddq_f32(sums[2], sums[3])));
}
#endif

int q6_k_dot_f32(const Q6_KBlock *block, const float *input, float *result)
{
    if (block == nullptr || input == nullptr || result == nullptr) {
        return 0;
    }

    float scale_factor = 0.0f;
    f16_bits_to_f32(block->scale_f16_bits, &scale_factor);

#if defined(__aarch64__) && defined(__ARM_NEON)
    *result = q6_k_dot_neon(block, input, scale_factor);
#else
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
#endif
    return 1;
}
