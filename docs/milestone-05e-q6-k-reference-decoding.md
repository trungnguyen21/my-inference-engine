# Milestone 05E: Q6_K Reference Decoding

**Status: Complete.** The focused `q6_k_reference_test` passed.

## Goal

Decode one Q6_K GGUF block into F32 weights and calculate its dot product with
a 256-element activation vector.

Q6_K is the last packed storage format present in the local Qwen GGUF file.
After Q4_K and Q6_K are complete, every packed block type in that file has a
small reference decoder and direct dot-product function.

## High-level model

Q6_K stores 256 signed six-bit integers. It splits every integer into four
low bits and two high bits. It also gives every 16 weights a signed scale.

```text
128 lower-bit bytes + 64 upper-bit bytes + 16 signed scales + F16 factor
         |                    |                  |                |
         +--------------------+------------------+----------------+
                                      256 weights
```

Each 128-weight half has four 32-weight regions. For one offset `l` from 0 to
31, a single `upper_bits` byte holds the two high bits for four positions:

```text
bits 0–1: position l       bits 2–3: position l + 32
bits 4–5: position l + 64  bits 6–7: position l + 96
```

Each reconstructed six-bit value is centered by subtracting 32:

```text
encoded_value = low_four_bits | (upper_two_bits << 4)
signed_value  = encoded_value - 32
weight         = F16_factor * signed_scale * signed_value
```

## Why this matters

Q6_K stores 256 weights in 210 bytes rather than the 1,024 bytes required by
F32. It is the final storage format your model needs and combines every packing
idea from the previous decoder exercises: packed nibbles, packed bit fields,
and per-region scales.

## Files

```text
include/quantization.h       Q6_K layout and implementation contracts
src/quantization.cpp         Q6_K stubs for you to implement
tests/q6_k_reference_test.cpp
```

## Implementation order

1. Implement `q6_k_unpack_value`. The test checks a value from every scale
   region and both 128-weight halves before decoding starts.
2. Implement `q6_k_dequantize_block` by applying the helper to all 256 output
   indices and multiplying by the shared decoded F16 factor.
3. Implement `q6_k_dot_f32` directly from packed values using a local sum.
   Do not allocate a decoded array or call the decoder.
4. Run the focused test after each step. Its `assert` runner stops at the
   first failed assertion.

## Definition of done

- [x] `sizeof(Q6_KBlock)` is 210 bytes.
- [x] The helper reconstructs all four two-bit fields and all 16 scale regions.
- [x] The decoder writes 256 expected F32 values.
- [x] The direct dot product matches the known F32 result and overwrites a
      nonzero initial result.
- [x] Invalid pointers and invalid indices return `0`.
- [x] `q6_k_reference_test` passes.

## Out of scope

- Mapping a full Q6_K tensor to matrix rows;
- SIMD, multithreading, BLAS, or GPU code;
- calling Q6_K kernels from `engine_forward_token`.

## Next milestone

Expose any supported packed GGUF tensor as a safe, read-only mapped view.
