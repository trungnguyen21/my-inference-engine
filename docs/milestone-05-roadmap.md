# Milestone 05: Quantized GGUF Weights

## Purpose

This milestone takes model bytes from the mapped GGUF file through to a
reference matrix-vector result. It deliberately keeps the work outside
`engine_forward_token`, so each storage and math step can be checked in a
small test before transformer code uses it.

## Storage types in the local Qwen GGUF

| GGML type | Meaning | Block values | Block bytes | Milestone |
| --- | --- | ---: | ---: | --- |
| F32 (`0`) | ordinary floats | 1 | 4 | 05A |
| Q5_0 (`6`) | 5-bit values plus F16 scale | 32 | 22 | 05C |
| Q8_0 (`8`) | signed 8-bit values plus F16 scale | 32 | 34 | 05B |
| Q4_K (`12`) | 4-bit K-quant super-block | 256 | 144 | 05D |
| Q6_K (`14`) | 6-bit K-quant super-block | 256 | 210 | 05E |

`Q4_K_M` describes the model’s overall quantization recipe. It does not mean
every tensor has the Q4_K type, which is why all five rows matter.

## Sequence

| Stage | Scope | Status |
| --- | --- | --- |
| 05A | Safe mapped F32 views | Complete |
| 05B | Q8_0 block decode and direct dot product | Complete |
| 05C | Q5_0 block decode and direct dot product | Complete |
| 05D | Q4_K block decode and direct dot product | Complete |
| 05E | Q6_K block decode and direct dot product | Complete |
| 05F | Safe mapped views for supported packed tensors | Complete |
| 05G | Dispatch a packed matrix row to its direct dot kernel | Complete |
| 05H | Check each storage path against a real local GGUF tensor | Complete |

## Completion boundary

Milestone 05 is complete when a two-dimensional tensor from the mapped model
can produce one F32 output per matrix row through the appropriate storage-type
kernel, and every input type found in the local file is supported.

The next large milestone can then build transformer operations by selecting
named tensors and calling this matrix-vector path. The first version will
recompute token history and use temporary K/V arrays; RoPE, attention,
residuals, and sampling still need implementation.
