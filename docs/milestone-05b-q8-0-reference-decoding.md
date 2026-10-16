# Milestone 05B: Q8_0 Reference Decoding

**Status: Complete.** The focused `q8_0_reference_test` passed.

## Goal

Decode one Q8_0 GGUF block into floats and calculate its dot product with a
32-element activation vector.

Q8_0 is the simplest packed weight format used by the local Qwen GGUF file.
It provides the first bridge from a model's compressed bytes to the F32 tensor
kernels you already wrote.

## High-level model

A model stores many weights. F32 stores every weight directly as a 4-byte
float. Q8_0 stores 32 nearby weights as small signed integers plus one shared
scale:

```text
scale (F16) + 32 signed int8 values
        |
        v
decoded_weight[i] = scale * integer[i]
```

For example, scale `0.5` and integer `-4` represent float weight `-2.0`.
The full block is 34 bytes, while 32 F32 weights would require 128 bytes.

The Q8_0 decoder sits between the GGUF file and a future matrix-vector kernel:

```text
GGUF Q8_0 bytes --> Q8_0Block --> decoded values or direct dot product
                                              |
                                              v
                                      activation vector
```

## Files

```text
include/quantization.h     API contracts and implementation instructions
src/quantization.cpp       your function bodies
tests/q8_0_reference_test.cpp
```

The detailed input, output, and calculation requirements are deliberately in
`quantization.h`, beside the function declarations.

## Implementation order

1. Implement `f16_bits_to_f32`. Start with normal values such as `1.0`, `0.5`,
   and `-2.0`, then support zero and subnormal values.
2. Implement `q8_0_dequantize_block` by decoding the scale once and applying
   it to all 32 signed integers.
3. Implement `q8_0_dot_f32` directly. Decode the scale once and accumulate
   all 32 scaled integer/input products. Do not create a temporary float array.
4. Run the Q8_0 test target and fix the first failing assertion each time.

## Definition of done

- [x] `sizeof(Q8_0Block)` is 34 bytes.
- [x] Binary16 values decode correctly for normal values, zero, and a
      subnormal value.
- [x] Q8_0 blocks decode to the expected 32 F32 values.
- [x] The direct Q8_0 dot product matches the known F32 result.
- [x] Invalid pointers return `0`.
- [x] `q8_0_reference_test` passes.

## Out of scope

- Reading a full Q8_0 tensor directly from the model;
- Q5_0, Q4_K, and Q6_K decoding;
- optimized SIMD or multithreaded quantized math;
- calling quantized kernels from `engine_forward_token`.

## Next milestone

Decode Q5_0 blocks, which add packed high bits to the per-value low-bit data.
