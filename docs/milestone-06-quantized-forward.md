# Milestone 06: Quantized Forward Integration

## Scope and current behavior

The complete F32 forward pass is the reference implementation. This work
connects the existing Q5_0, Q8_0, Q4_K, and Q6_K kernels to that same schedule.
Normalization weights and Q/K/V biases remain F32. Sampling is Milestone 07.

Two internal entry points are declared in
[`qwen2_internal.h`](../src/qwen2_internal.h) and defined in
[`qwen2_architecture.cpp`](../src/qwen2_architecture.cpp):

- `qwen2_decode_quantized_embedding_row`: validates shape, layout, counts,
  and byte sizes before decoding only the selected row with aligned block copies.
- `qwen2_weight_matvec`: validates the tensor, keeps the F32 path, and dispatches
  packed weights to `quantized_tensor_matvec_f32` with errors naming the tensor.

Both are already connected to the forward pass. The final `output.weight`
projection and all layer matrices use the same weight dispatcher. The build
targets include the quantization and packed-matvec sources.

Three CTest checkpoints are enabled and labelled `milestone06-quantized`.
All three pass, along with the original 12 regression tests. These are ordinary
regression tests, not `WILL_FAIL` tests.

## Work item 1: Decode one quantized embedding row

Entry point:

```cpp
int qwen2_decode_quantized_embedding_row(
    const GgufQuantizedTensorView *matrix,
    int token_id,
    int dim,
    float *output,
    char *error,
    int error_capacity
);
```

The caller has obtained a packed view of `token_embd.weight`. Its dimensions
are `[dim, vocab_size]` in GGUF order. The function must write only the selected
token's `dim` embedding values into the existing activation buffer.

Requirements:

- Reject null input/data/output, nonpositive dimensions, and an ID outside the
  row range. An ID equal to the row count is invalid.
- Require a two-dimensional matrix whose column count equals `dim`.
- Support Q5_0, Q8_0, Q4_K, and Q6_K. Validate that the view's block element
  count, byte count, element count, and block count match its type and shape.
- Require each row to contain whole blocks. Total tensor divisibility alone
  is insufficient: a 48-column Q8_0 matrix cannot be addressed as whole rows.
- Check products and byte offsets for overflow before pointer arithmetic.
  Read within the storage described by the validated view.
- Packed bytes may be unaligned. Copy each block to the matching local block
  struct before calling an existing decoder from `quantization.h`.
- Decode only the selected row. Allocate no memory and leave the source
  unchanged. The caller provides room for exactly `dim` floats.
- Validate arguments/layout before writing. Return `1` only after completing
  the row; on failure return `0` with a useful error. A null error buffer or
  zero error capacity must remain safe.

Numerical and boundary checks are in
[`qwen2_weight_reference_test.cpp`](../tests/qwen2_weight_reference_test.cpp).
For every type, the fixture contains three rows with two different blocks per
row. First, middle, and last row checks catch incorrect row offsets, block
strides, and incomplete copies. Output sentinels catch extra writes; a byte
snapshot catches source mutation. Invalid views, IDs, widths, layouts, and
unaligned storage are included.

```sh
# All embedding cases.
./build/qwen2_weight_reference_test embedding

# Focus on one format while implementing it.
./build/qwen2_weight_reference_test embedding q5_0
./build/qwen2_weight_reference_test embedding q8_0
./build/qwen2_weight_reference_test embedding q4_k
./build/qwen2_weight_reference_test embedding q6_k

# Negative cases verify rejection before any output is written.
./build/qwen2_weight_reference_test embedding invalid
```

## Work item 2: Dispatch packed matrix-vector multiplication

Entry point:

```cpp
int qwen2_weight_matvec(
    const GgufFile *file,
    const char *name,
    const float *input,
    int input_count,
    float *output,
    int output_count,
    char *error,
    int error_capacity
);
```

`name` is a complete GGUF tensor name. It may be a layer weight such as
`blk.0.attn_k.weight` or the global `output.weight`. The operation is:

```text
matrix [output_count rows × input_count columns]
    × input [input_count]
    → output [output_count]
```

The existing code validates arguments and matrix shape, loads an F32 or packed
view, and preserves the F32 path. Packed execution uses the existing
`quantized_tensor_matvec_f32` function. Propagate its
failure with an error naming the tensor. Return success only after all output
values have been written. Keep the input/output buffers separate and allocate
no memory. Do not expand complete matrices into temporary float arrays.

The matrix tests cover F32, all four packed types, rectangular matrices,
multiple blocks per row, layer and global names, unsupported types, invalid
shapes/counts, truncated storage, null arguments, and output sentinels. The
expected results come from known fixture values rather than the dispatcher
under test.

```sh
./build/qwen2_weight_reference_test matvec
./build/qwen2_weight_reference_test matvec q4_k
./build/qwen2_weight_reference_test matvec f32
./build/qwen2_weight_reference_test matvec invalid
```

## Work item 3: Verify integration and lifecycle

[`forward_quantized_reference_test.cpp`](../tests/forward_quantized_reference_test.cpp)
builds a one-layer model with `dim=256`, `hidden_dim=512`, four query heads,
two KV heads, and a three-token vocabulary. The fixture includes every packed
type, F32 biases/norms, a rectangular FFN, and a global output projection.

The same known weights are also stored as an F32 model. Compare every logit
across three successive forwards, including full-prefix recomputation. The
tolerance is `0.0001 + 0.001 * abs(reference_logit)` to allow accumulation-order
differences between direct packed and dense products.

The test also corrupts the output tensor shape after two successful calls,
requires a failure without advancing position, repairs it, and retries. It
then checks context exhaustion and destroys both inference states.

```sh
# Fixture sanity check: the F32-only mode remains available.
./build/forward_quantized_reference_test f32

# Isolate each connection while implementing.
./build/forward_quantized_reference_test embedding
./build/forward_quantized_reference_test layers
./build/forward_quantized_reference_test output

# Final synthetic checkpoint: all packed paths together.
./build/forward_quantized_reference_test mixed
```

The existing [`real_model_forward_check.cpp`](../tests/real_model_forward_check.cpp)
is the acceptance check against `models/qwen-2.5.gguf`. It checks tokenization,
prefill, finite vocabulary-sized logits, rejection of invalid IDs without
consuming context, one further valid forward, context exhaustion, and cleared
state/model ownership after cleanup. It passes with the local model and
reports two forwarded tokens and 151,936 finite logits.

## Build and acceptance commands

```sh
cmake --build build --parallel

# Completed-code checks: 12 earlier regressions plus 3 app/dispatch checks.
ctest --test-dir build \
  -LE 'milestone06-quantized|milestone07-exercises|requires-local-model' \
  --output-on-failure

# Packed embedding, dispatcher, and mixed-weight forward regressions.
ctest --test-dir build -L milestone06-quantized --output-on-failure

# Run the actual model check directly once the synthetic checkpoints pass.
./build/real_model_forward_check models/qwen-2.5.gguf

# Enable local-model checks, then run through Milestone 06 only.
cmake -S . -B build -DENGINE_TEST_MODEL_PATH=models/qwen-2.5.gguf
cmake --build build --parallel
ctest --test-dir build -LE milestone07 --output-on-failure
```

The [next milestone](milestone-07-first-generated-text.md) has separate greedy
selection and prompt-formatting exercises. Its real CLI smoke test is also
enabled by `ENGINE_TEST_MODEL_PATH`, but is excluded by the Milestone 06 command.

## Completion checklist

- [x] Declare and connect the embedding-row decoder.
- [x] Share one matrix dispatcher between layer and output projections.
- [x] Preserve the F32 implementation and connect packed-view lookup.
- [x] Register numerical helper and mixed-weight forward checkpoints.
- [x] Prepare the real-model error/lifecycle acceptance checks.
- [x] Implement packed embedding-row decoding for all four types.
- [x] Implement packed matrix-vector dispatch.
- [x] Pass all three synthetic checkpoints and existing regressions.
- [x] Pass the real-model schema and forward checks against the local model.

Milestone 06 is complete: the synthetic and local-model acceptance checks pass.
The focused format and integration modes above remain useful when changing
these implementations.
