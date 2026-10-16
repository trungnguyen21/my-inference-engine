# Milestone 04B: Transformer F32 Operations

## Goal

Implement and test four float operations used inside a Qwen transformer layer:

1. RMS normalization;
2. SiLU activation;
3. elementwise vector multiplication;
4. numerically stable softmax.

Keep these as small CPU reference kernels. Do not modify model loading,
quantized weights, the KV cache, or `engine_forward_token` in this milestone.

## Where this fits

The kernels you have now provide basic vector and matrix work. These operations
fill in the non-matrix parts of a transformer layer:

```text
activations
    |
    +--> RMSNorm --> Q, K, V matrix-vector products --> attention scores
                                                        |
                                                     softmax
                                                        |
                                                   attention output

residual activations
    |
    +--> RMSNorm --> gate projection --> SiLU --+
                    up projection --------------*--> down projection
```

`*` is elementwise multiplication. Qwen's feed-forward network uses this
gated form: `SiLU(gate) * up`.

## API contracts

All functions belong in `include/tensor.h` and `src/tensor.cpp`.

Every function must:

- return `1` on success and `0` for invalid arguments;
- reject null pointers and non-positive lengths;
- allocate no memory;
- require the caller to give an output buffer distinct from the input buffers.

### RMSNorm

```text
mean_square = sum(input[i] * input[i]) / length
scale       = 1 / sqrt(mean_square + epsilon)
output[i]   = input[i] * scale * weight[i]
```

`input` is the activation vector. `weight` contains one learned RMSNorm weight
per element. Require `epsilon > 0` and return failure otherwise.

Suggested starter check:

```text
input    = [1, 1]
weight   = [2, 3]
epsilon  = 3
expected = [1, 1.5]
```

### SiLU

```text
output[i] = input[i] / (1 + exp(-input[i]))
```

Include `<math.h>` in `tensor.cpp` when you implement this. Test `0`, a
positive value, and a negative value. Approximate comparisons are required.

### Elementwise multiplication

```text
output[i] = left[i] * right[i]
```

This is the gate operation in the feed-forward network. A useful check is:

```text
[2, -3, 0.5] * [4, 2, -6] = [8, -6, -3]
```

### Stable softmax

Softmax turns attention scores into probabilities:

```text
maximum   = max(input)
output[i] = exp(input[i] - maximum)
sum       = sum(output)
output[i] = output[i] / sum
```

Subtracting `maximum` prevents overflow. For example, direct exponentiation
of `1000` overflows a `float`, while softmax of `[1000, 1001]` remains valid.

Suggested checks:

```text
softmax([0, 0])       = [0.5, 0.5]
softmax([1000, 1001]) = [0.2689414, 0.7310586]
```

## Implementation order

1. Add test cases for each API before replacing its TODO body.
2. Implement `tensor_multiply_f32`; it has the same validation pattern as
   `tensor_add_f32`.
3. Implement `tensor_silu_f32` and verify approximate values.
4. Implement `tensor_softmax_f32` in three passes: find maximum, write
   exponentials and accumulate their sum, then divide each output element.
5. Implement `tensor_rms_norm_f32`: accumulate squared values, calculate one
   normalization scale, then write the weighted outputs.
6. Add invalid-input assertions for every new function.

## Definition of done

- [x] Each declared tensor function has a correct reference implementation.
- [x] Tests cover ordinary inputs and invalid arguments for every function.
- [x] Softmax handles `[1000, 1001]` without overflow and sums to about `1`.
- [x] The `tensor_reference_test` target passes.
- [ ] You can explain why RMSNorm uses one scale for the whole vector and why
      softmax subtracts its maximum.

## Out of scope

- RoPE and attention masking;
- Q4_K, Q5_0, Q6_K, or Q8_0 weights;
- KV-cache allocation;
- SIMD, BLAS, GPU code, or multithreading;
- wiring these operations into `engine_forward_token`.

## Next milestone

Read F32 GGUF tensor data into a safe view and compare it against these
reference kernels. Quantized tensor blocks follow after the F32 path is
understood.
