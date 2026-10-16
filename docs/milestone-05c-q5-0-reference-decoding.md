# Milestone 05C: Q5_0 Reference Decoding

**Status: Complete.** The focused `q5_0_reference_test` passed. The fixture
can be strengthened by setting a fifth bit in `high_bits[2]` as well.

## Goal

Decode one Q5_0 GGUF block into floats and calculate its dot product with a
32-element activation vector.

Your local Qwen GGUF contains Q5_0 tensors. This milestone builds on Q8_0 by
introducing bit packing: each stored value has five bits, but those bits are
split between two arrays in the block.

## High-level model

Q8_0 stored one signed byte for every weight. Q5_0 stores a signed five-bit
integer for every weight. Five bits cannot fit neatly into bytes, so the block
stores the low four bits and fifth bits separately:

```text
F16 scale + 4 bytes fifth bits + 16 bytes paired low four-bit values
               |                        |
               |                        +-- byte i: values i and i + 16
               +-- bit i: fifth bit for value i
```

For a value at index `i`:

```text
low_four_bits = packed_nibbles[i % 16]
fifth_bit     = bit (i % 8) of high_bits[i / 8]
encoded        = low_four_bits | (fifth_bit << 4)    // 0 through 31
signed_value   = encoded - 16                         // -16 through 15
weight         = F16_scale * signed_value
```

When `i` is from 16 through 31, use the **high nibble** of
`packed_nibbles[i - 16]`; for index 0 through 15, use the low nibble.

The block layout is exactly 22 bytes:

```text
2 bytes scale + 4 bytes fifth bits + 16 bytes packed nibbles = 22 bytes
```

## Why this matters

Real model weights trade some numerical precision for a much smaller file and
lower memory use. Q5_0 stores 32 weights in 22 bytes. The same F32 weights
would take 128 bytes. Decoding this format teaches the two operations every
quantized kernel needs: unpack a stored integer and apply a scale.

## Files

```text
include/quantization.h       API contracts and implementation instructions
src/quantization.cpp         Q5_0 function stubs for you to implement
tests/q5_0_reference_test.cpp
```

The exact input, output, return-value, and no-allocation requirements are in
`quantization.h`, next to each function declaration.

## Implementation order

1. Read `Q5_0Block` and confirm its size is 22 bytes.
2. Implement `q5_0_dequantize_block`. Start with the low nibble, then add the
   fifth bit, center the integer by subtracting 16, and finally multiply by
   the existing F16 scale converter.
3. Implement `q5_0_dot_f32` directly from the packed values. Use a local sum
   set to `0.0f`; do not call the decoder or allocate a temporary array.
4. Run `q5_0_reference_test`. The `assert` runner stops at the first failure,
   so address one assertion at a time.

## Definition of done

- [x] `sizeof(Q5_0Block)` is 22 bytes.
- [x] Both nibbles in each packed byte map to the correct output positions.
- [x] Fifth bits from all four `high_bits` bytes map to the correct values.
- [x] Q5_0 blocks decode to the expected 32 F32 values.
- [x] The direct Q5_0 dot product matches the known F32 result and overwrites
      a nonzero initial result.
- [x] Invalid pointers return `0`.
- [x] `q5_0_reference_test` passes.

## Out of scope

- Reading a full Q5_0 tensor directly from the model;
- Q4_K, Q6_K, or Q4_K_M decoding;
- optimized SIMD or multithreaded quantized math;
- calling quantized kernels from `engine_forward_token`.

## Next milestone

Inspect and decode a Q4_K block, the quantization family used by the model's
main Q4_K_M weight tensors.
