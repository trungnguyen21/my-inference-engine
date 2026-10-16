#ifndef QUANTIZATION_H
#define QUANTIZATION_H

#include <stdint.h>

#define Q8_0_BLOCK_ELEMENTS 32
#define Q8_0_BLOCK_BYTES 34

#define Q5_0_BLOCK_ELEMENTS 32
#define Q5_0_BLOCK_BYTES 22

#define Q4_K_BLOCK_ELEMENTS 256
#define Q4_K_SUBBLOCK_ELEMENTS 32
#define Q4_K_SUBBLOCK_COUNT 8
#define Q4_K_SCALE_AND_MIN_BYTES 12
#define Q4_K_BLOCK_BYTES 144

#define Q6_K_BLOCK_ELEMENTS 256
#define Q6_K_GROUP_ELEMENTS 128
#define Q6_K_SCALE_COUNT 16
#define Q6_K_LOWER_BITS_BYTES 128
#define Q6_K_UPPER_BITS_BYTES 64
#define Q6_K_BLOCK_BYTES 210

/*
 * This matches one GGML Q8_0 block in a GGUF tensor.
 *
 * scale_f16_bits holds an IEEE 754 binary16 scale. quantized_values contains
 * 32 signed integer values. The original float value at index i is:
 *
 *     f16_to_f32(scale_f16_bits) * quantized_values[i]
 */
typedef struct Q8_0Block {
    uint16_t scale_f16_bits;
    int8_t quantized_values[Q8_0_BLOCK_ELEMENTS];
} Q8_0Block;

static_assert(sizeof(Q8_0Block) == Q8_0_BLOCK_BYTES,
              "Q8_0Block must match GGUF's 34-byte layout");

/*
 * Decode IEEE 754 binary16 bits into one F32 value.
 *
 * bits is the raw 16-bit pattern stored in a GGUF scale field. result points
 * to caller-owned storage for the decoded float. Handle zero, subnormal,
 * normal, infinity, and NaN bit patterns. Return 0 only when result is null;
 * otherwise write the decoded value and return 1.
 */
int f16_bits_to_f32(uint16_t bits, float *result);

/*
 * Decode one Q8_0 block into 32 ordinary F32 values.
 *
 * block points to the 34 bytes read from a Q8_0 tensor. output points to a
 * caller-provided array with room for exactly Q8_0_BLOCK_ELEMENTS floats.
 * Decode block->scale_f16_bits, then write:
 *
 *     output[i] = decoded_scale * block->quantized_values[i]
 *
 * for every index from 0 through 31. Do not allocate memory. Return 0 when
 * block or output is null; otherwise return 1.
 */
int q8_0_dequantize_block(const Q8_0Block *block, float *output);

/*
 * Compute the dot product between one Q8_0 block and a 32-element F32 vector.
 *
 * block contains the quantized model weights. input points to exactly 32 F32
 * activation values. result points to caller-owned storage for one float.
 * Calculate the Q8_0 scale once, then accumulate:
 *
 *     decoded_scale * block->quantized_values[i] * input[i]
 *
 * for all 32 elements. This avoids allocating a temporary decoded block.
 * Return 0 when block, input, or result is null; otherwise return 1.
 */
int q8_0_dot_f32(const Q8_0Block *block, const float *input, float *result);

/*
 * This matches one GGML Q5_0 block in a GGUF tensor.
 *
 * A Q5_0 block represents 32 signed, five-bit integer values. One byte in
 * packed_nibbles describes two values: its low nibble is value i and its high
 * nibble is value i + 16. high_bits supplies the fifth bit for each value.
 * Bit i of high_bits[i / 8] belongs to value i.
 *
 * Rebuild an integer value as follows:
 *
 *     encoded = four_low_bits | (fifth_bit << 4)
 *     signed_value = encoded - 16
 *     decoded_value = f16_to_f32(scale_f16_bits) * signed_value
 */
typedef struct Q5_0Block {
    uint16_t scale_f16_bits;
    uint8_t high_bits[4];
    uint8_t packed_nibbles[Q5_0_BLOCK_ELEMENTS / 2];
} Q5_0Block;

static_assert(sizeof(Q5_0Block) == Q5_0_BLOCK_BYTES,
              "Q5_0Block must match GGUF's 22-byte layout");

/*
 * Decode one Q5_0 block into 32 ordinary F32 values.
 *
 * block points to the 22 bytes read from a Q5_0 tensor. output points to a
 * caller-provided array with Q5_0_BLOCK_ELEMENTS floats. Decode the F16 scale
 * once. For each packed_nibbles byte at index i, extract the low nibble for
 * output[i] and the high nibble for output[i + 16]. For each, read its fifth
 * bit from high_bits, combine it with the nibble, subtract 16 to center the
 * five-bit unsigned value into the range -16 through 15, then multiply by the
 * decoded scale. Do not allocate memory. Return 0 when block or output is
 * null; otherwise return 1.
 */
int q5_0_dequantize_block(const Q5_0Block *block, float *output);

/*
 * Compute the dot product between one Q5_0 block and a 32-element F32 vector.
 *
 * block contains packed Q5_0 model weights. input points to exactly
 * Q5_0_BLOCK_ELEMENTS F32 activation values. result points to caller-owned
 * storage for one float. 
 * Reconstruct each signed five-bit value exactly as in
 * q5_0_dequantize_block, multiply it by the decoded scale and input element,
 * and add it to a local accumulator initialized to 0.0f. Write the final sum
 * to *result. Do not allocate or call q5_0_dequantize_block, since this
 * function should calculate directly from the packed block. Return 0 when
 * block, input, or result is null; otherwise return 1.
 */
int q5_0_dot_f32(const Q5_0Block *block, const float *input, float *result);

/*
 * This matches one GGML Q4_K block in a GGUF tensor.
 *
 * Q4_K is a 256-weight super-block split into eight 32-weight sub-blocks.
 * Each weight is an unsigned 4-bit value. Each sub-block has its own scale
 * multiplier and minimum multiplier, each stored as a packed unsigned 6-bit
 * value. scale_f16_bits and min_f16_bits are shared F16 factors.
 *
 * The decoded value for a weight in sub-block s is:
 *
 *     scale = f16_to_f32(scale_f16_bits) * scale_multiplier[s]
 *     minimum = f16_to_f32(min_f16_bits) * min_multiplier[s]
 *     decoded_value = scale * four_bit_value - minimum
 *
 * packed_nibbles is arranged as four groups of 64 weights. Each group has 32
 * bytes: low nibbles encode the first 32 weights, and high nibbles encode the
 * next 32 weights. Therefore packed_nibbles[32 * group + offset] supplies:
 *
 *     low nibble  -> weight 64 * group + offset
 *     high nibble -> weight 64 * group + 32 + offset
 */
typedef struct Q4_KBlock {
    uint16_t scale_f16_bits;
    uint16_t min_f16_bits;
    uint8_t scale_and_min[Q4_K_SCALE_AND_MIN_BYTES];
    uint8_t packed_nibbles[Q4_K_BLOCK_ELEMENTS / 2];
} Q4_KBlock;

static_assert(sizeof(Q4_KBlock) == Q4_K_BLOCK_BYTES,
              "Q4_KBlock must match GGUF's 144-byte layout");

/*
 * Unpack the 6-bit scale and minimum multipliers for one Q4_K sub-block.
 *
 * scale_and_min points to the 12-byte field in a Q4_KBlock. subblock_index
 * selects a sub-block from 0 through Q4_K_SUBBLOCK_COUNT - 1. The two output
 * pointers receive the corresponding values from 0 through 63.
 *
 * For sub-blocks 0 through 3:
 *
 *     scale = scale_and_min[index]     & 0x3f
 *     min   = scale_and_min[index + 4] & 0x3f
 *
 * For sub-blocks 4 through 7, let index be subblock_index:
 *
 *     scale = (scale_and_min[index + 4] & 0x0f)
 *           | ((scale_and_min[index - 4] >> 6) << 4)
 *     min   = (scale_and_min[index + 4] >> 4)
 *           | ((scale_and_min[index] >> 6) << 4)
 *
 * Return 0 when any pointer is null or subblock_index is outside 0 through 7.
 * Otherwise write both values and return 1. Do not allocate memory.
 */
int q4_k_unpack_scale_min(
    const uint8_t *scale_and_min,
    int subblock_index,
    uint8_t *scale_multiplier,
    uint8_t *min_multiplier
);

/*
 * Decode one Q4_K block into Q4_K_BLOCK_ELEMENTS F32 values.
 *
 * block points to one 144-byte Q4_K block. output points to caller-provided
 * storage for 256 floats. Decode the two F16 shared factors once. Then, for
 * each of the eight sub-blocks, use q4_k_unpack_scale_min to obtain its two
 * 6-bit multipliers and reconstruct its 32 four-bit values from the correct
 * half of the correct 32-byte packed group. Write values in their natural
 * output order, indices 0 through 255. Do not allocate memory. Return 0 when
 * block or output is null; otherwise return 1.
 */
int q4_k_dequantize_block(const Q4_KBlock *block, float *output);

/*
 * Compute the dot product between one Q4_K block and a 256-element F32 vector.
 *
 * block contains packed Q4_K weights. input points to exactly 256 F32
 * activation values. result points to caller-owned storage for one float.
 * Reconstruct each scale, minimum, and four-bit value as in
 * q4_k_dequantize_block. Multiply the decoded weight by input[index] and add
 * it to a local accumulator initialized to 0.0f. Write the final accumulator
 * to *result. Do not allocate or call q4_k_dequantize_block. Return 0 when
 * block, input, or result is null; otherwise return 1.
 */
int q4_k_dot_f32(const Q4_KBlock *block, const float *input, float *result);

/*
 * This matches one GGML Q6_K block in a GGUF tensor.
 *
 * Q6_K stores 256 signed six-bit values. lower_bits holds the lower four bits
 * for two values per byte. upper_bits holds four two-bit fields per byte.
 * scales contains one signed 8-bit multiplier for every 16 decoded values.
 * scale_f16_bits is the shared F16 factor for all 16 scale multipliers.
 *
 * The block has two 128-value groups. Within each group, lower_bits and
 * upper_bits use the same 32-element offset l for four output regions:
 *
 *     region 0: output l       -> lower_bits[l] low nibble,  upper bits 0..1
 *     region 1: output l + 32  -> lower_bits[l+32] low nibble, upper bits 2..3
 *     region 2: output l + 64  -> lower_bits[l] high nibble, upper bits 4..5
 *     region 3: output l + 96  -> lower_bits[l+32] high nibble,upper bits 6..7
 *
 * Combine the four low bits with the two upper bits to produce 0 through 63,
 * then subtract 32 to obtain the signed integer range -32 through 31.
 */
typedef struct Q6_KBlock {
    uint8_t lower_bits[Q6_K_LOWER_BITS_BYTES];
    uint8_t upper_bits[Q6_K_UPPER_BITS_BYTES];
    int8_t scales[Q6_K_SCALE_COUNT];
    uint16_t scale_f16_bits;
} Q6_KBlock;

static_assert(sizeof(Q6_KBlock) == Q6_K_BLOCK_BYTES,
              "Q6_KBlock must match GGUF's 210-byte layout");

/*
 * Reconstruct the integer and scale multiplier for one Q6_K weight.
 *
 * block points to one Q6_K block. index selects one value from 0 through 255.
 * quantized_value receives the reconstructed signed six-bit integer from -32
 * through 31. scale_multiplier receives the signed 8-bit value from the
 * matching 16-element scale region.
 *
 * First calculate group = index / 128, local = index % 128, region = local /
 * 32, and offset = local % 32. Use the layout table above to extract the four
 * low bits and the matching two upper bits. For the scale index use:
 *
 *     group * 8 + region * 2 + offset / 16
 *
 * Return 0 when block or either output pointer is null, or when index is
 * outside 0 through 255. Otherwise write both results and return 1. Do not
 * allocate memory.
 */
int q6_k_unpack_value(
    const Q6_KBlock *block,
    int index,
    int8_t *quantized_value,
    int8_t *scale_multiplier
);

/*
 * Decode one Q6_K block into Q6_K_BLOCK_ELEMENTS F32 values.
 *
 * block points to one 210-byte Q6_K block. output points to caller-provided
 * storage for 256 floats. Decode scale_f16_bits once. For every output index,
 * use q6_k_unpack_value, then write:
 *
 *     output[index] = decoded_f16_scale * scale_multiplier * quantized_value
 *
 * Return 0 when block or output is null; otherwise return 1. Do not allocate
 * memory.
 */
int q6_k_dequantize_block(const Q6_KBlock *block, float *output);

/*
 * Compute the dot product between one Q6_K block and a 256-element F32 vector.
 *
 * block contains packed Q6_K weights. input points to exactly 256 F32
 * activation values. result points to caller-owned storage for one float.
 * Decode the shared F16 scale once. Reconstruct every quantized integer and
 * its scale multiplier, multiply them with input[index], and add to a local
 * accumulator initialized to 0.0f. Write the final accumulator to *result.
 * Do not allocate or call q6_k_dequantize_block. Return 0 when block, input,
 * or result is null; otherwise return 1.
 */
int q6_k_dot_f32(const Q6_KBlock *block, const float *input, float *result);

#endif
