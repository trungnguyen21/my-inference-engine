# Milestone 04A: Reference F32 CPU Kernels

## Goal

Write small, correct CPU implementations of three operations used throughout a
transformer:

1. vector addition;
2. dot product;
3. matrix-vector multiplication.

This is a correctness milestone. Do not optimize, use SIMD, read GGUF tensor
data, handle quantization, or modify `engine_forward_token` yet.

## Why this comes next

After tokenization, the engine has prompt token IDs but cannot transform them
into activations. Every Qwen layer eventually reduces to combinations of these
operations. A tiny float reference version gives you something trustworthy to
compare against when you later read Q4_K, Q5_0, Q6_K, and Q8_0 weights.

```text
weights + activation vector
          |
          v
matrix-vector multiplication
          |
          v
new activation vector
```

## Scope

Create these files:

```text
include/tensor.h
src/tensor.cpp
tests/tensor_reference_test.cpp
```

Add a separate `tensor_reference_test` executable to `CMakeLists.txt`. Keep it
independent of model loading, the tokenizer adapter, and terminal UI code.

## Required API

Use procedural, C-style functions. The exact names may vary, but the contracts
should remain equivalent.

| Operation | Inputs | Output | Mathematical rule |
| --- | --- | --- | --- |
| Vector add | `left`, `right`, `length` | `output` | `output[i] = left[i] + right[i]` |
| Dot product | `left`, `right`, `length` | one `float` | `sum(left[i] * right[i])` |
| Matrix-vector product | matrix, `rows`, `columns`, input vector | output vector | `output[row] = sum(matrix[row, column] * input[column])` |

All three functions should:

- accept `const` pointers for inputs they do not modify;
- receive explicit sizes rather than rely on hidden global state;
- allocate no memory internally;
- return failure for null pointers or non-positive sizes;
- never write beyond the caller-provided output range.

For this milestone, require `output` to be separate from input buffers. You can
decide whether to support aliasing later, after the basic behavior is correct.

## Matrix layout

Use ordinary row-major storage for this reference kernel:

```text
matrix = [
  row 0, column 0; row 0, column 1; ...
  row 1, column 0; row 1, column 1; ...
]
```

The element at `row`, `column` is located at:

```text
row * columns + column
```

For this milestone, treat this as a local reference-kernel convention. GGUF
tensor dimensions and quantized storage will need an explicit translation in a
later milestone; do not mix that problem into this one.

## Required checks

Write a tiny executable test program with fixed arrays and an approximate float
comparison. Use an epsilon such as `0.00001f`.

| Check | Input | Expected output |
| --- | --- | --- |
| Vector add | `[1, -2, 3] + [4, 5, -6]` | `[5, 3, -3]` |
| Dot product | `[1, 2, 3] dot [4, 5, 6]` | `32` |
| Matrix-vector | `[[1, 2, 3], [4, 5, 6]] × [1, 0, -1]` | `[-2, -2]` |
| Invalid input | null pointer or zero length | function reports failure and does not crash |

Do not print a full framework report. A simple pass/fail message and a nonzero
exit code on failure are enough.

## Build and run

After adding the CMake target:

```sh
./build.sh
cmake --build build --target tensor_reference_test
./build/tensor_reference_test
```

The regular inference executable must continue to build:

```sh
./build/inference --model models/qwen-2.5.gguf \
  --prompt "What is the capital of France?" --max-tokens 8
```

It should still stop at the inference-state stub. This milestone does not add
model inference yet.

## Definition of done

- [ ] `tensor.h` documents the input, output, and storage contracts.
- [ ] `tensor.cpp` implements vector addition, dot product, and matrix-vector
      multiplication without heap allocation.
- [ ] The reference test covers every operation and invalid inputs.
- [ ] The reference test passes.
- [ ] The regular project build still succeeds.
- [ ] You can explain why matrix-vector output has one value per matrix row.

## Out of scope today

- RMSNorm, SiLU, softmax, and RoPE;
- Q4_K, Q5_0, Q6_K, or Q8_0 dequantization;
- multithreading, BLAS, SIMD, or GPU work;
- KV-cache allocation;
- changes to `engine_forward_token`.

## Next milestone

Add RMSNorm, SiLU, elementwise multiplication, and stable softmax to the same
reference-kernel module. Only after those float operations are verified should
you start reading quantized GGUF tensor blocks.
