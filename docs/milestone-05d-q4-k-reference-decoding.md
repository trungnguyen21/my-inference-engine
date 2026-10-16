# Milestone 05D: Q4_K Reference Decoding

**Status: Complete.** The focused `q4_k_reference_test` passed.

## Goal

Decode one Q4_K GGUF block into F32 weights and calculate its dot product with
a 256-element activation vector.

Q4_K is the main weight format behind the `Q4_K_M` name of your local Qwen
model. Completing this milestone does not run the model yet, but it teaches
the packed format needed for many of its largest tensors.

## High-level model

Q8_0 gave 32 weights one shared scale. Q5_0 added a fifth packed bit. Q4_K
groups **256 weights** into eight 32-weight sub-blocks. Every sub-block has:

- a six-bit scale multiplier;
- a six-bit minimum multiplier; and
- 32 four-bit quantized values.

The entire 256-weight super-block shares two F16 factors: one scales the
sub-block scale multipliers, and one scales the sub-block minimum multipliers.

```text
F16 scale + F16 minimum + 12 packed metadata bytes + 128 packed weight bytes
      |            |                  |                       |
      |            |                  |                       +-- 256 nibbles
      |            |                  +-- 8 scales + 8 minimums, 6 bits each
      |            +-- base factor for each sub-block minimum
      +-- base factor for each sub-block scale
```

For a weight in sub-block `s`:

```text
scale   = F16_scale * scale_multiplier[s]
minimum = F16_minimum * minimum_multiplier[s]
weight  = scale * four_bit_value - minimum
```

The offset is subtracted because the stored minimum represents the magnitude
of a negative offset. Unlike Q5_0, the four-bit value itself is unsigned:
from 0 through 15.

## Why this matters

The model stores Q4_K blocks in 144 bytes. The same 256 F32 weights would
require 1,024 bytes. It is the first format in this project with separate
parameters for different 32-weight regions, which is why it is useful before
building a quantized matrix-vector operation.

## Files

```text
include/quantization.h       Q4_K layout and API contracts
src/quantization.cpp         Q4_K stubs for you to implement
tests/q4_k_reference_test.cpp
```

## Implementation order

1. Implement `q4_k_unpack_scale_min` first. It reads one of eight scale/min
   pairs from the 12 metadata bytes. The function contract gives the exact
   two cases for sub-block indices 0–3 and 4–7.
2. Run the test. The first group checks all eight unpacked pairs before any
   256-weight decoding is attempted.
3. Implement `q4_k_dequantize_block`. Work through one 32-weight sub-block at
   a time. Find the packed group, extract the required nibble, then apply that
   sub-block’s scale and minimum.
4. Implement `q4_k_dot_f32` from the packed data. Use a local `0.0f` sum;
   do not allocate a decoded array or call the decoder.
5. Run the focused test after each step. Its `assert` runner stops on the
   first failure.

## Definition of done

- [x] `sizeof(Q4_KBlock)` is 144 bytes.
- [x] All eight packed scale/minimum pairs decode into the expected 6-bit
      values.
- [x] The low and high nibbles map to the correct halves of every 64-weight
      group.
- [x] All eight 32-weight sub-blocks use their own scale and minimum.
- [x] The decoder writes 256 expected F32 values.
- [x] The direct dot product matches the known F32 result and overwrites a
      nonzero initial result.
- [x] Invalid pointers and invalid sub-block indices return `0`.
- [x] `q4_k_reference_test` passes.

## Out of scope

- Mapping a full Q4_K tensor to a matrix shape;
- Q6_K decoding;
- optimized SIMD or multithreaded quantized math;
- calling quantized kernels from `engine_forward_token`.

## Next milestone

Decode Q6_K blocks, the remaining packed format in the local model before
connecting quantized tensor views to matrix-vector operations.
