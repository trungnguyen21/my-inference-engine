# Milestone 05G: Quantized Matrix-Vector Dispatch

## Goal

Multiply a two-dimensional packed GGUF tensor by an F32 vector without first
decoding the whole matrix into F32 memory.

This is the final reference-math step before a transformer layer can use model
weight tensors. It combines a 05F mapped tensor view with the direct block dot
products from 05B–05E.

## Data layout

GGML stores a two-dimensional matrix shape as:

```text
dimensions[0] = columns in each row
dimensions[1] = rows / output values
```

The packed bytes for one row are contiguous. If a row has 896 columns and uses
Q8_0, it contains `896 / 32 = 28` Q8_0 blocks. Sum the dot product for each
block to produce one output value for that row.

```text
packed row blocks + F32 input slices --> block dot products --> one row sum
```

## API

`quantized_tensor_matvec_f32` is in `src/quantized_matvec.h`. Its declaration
contains the complete validation, layout, alignment, and no-allocation rules.

The raw view pointer is only byte-aligned by contract. Copy each packed block
into a local correctly aligned block structure using `memcpy` before passing it
to the matching block-dot function.

## Implementation order

1. Validate pointers, two-dimensional shape, integer-sized dimensions, counts,
   supported type, and block metadata.
2. Calculate `blocks_per_row` and validate the total block count.
3. Implement Q8_0 dispatch first. The two-row test confirms that each row uses
   a distinct segment of packed model bytes.
4. Add Q5_0, Q4_K, and Q6_K dispatch with the same row/block structure.
5. Run the focused test after each addition. It stops at the first `assert`.

## Definition of done

- [x] Q8_0, Q5_0, Q4_K, and Q6_K each dispatch to their direct block-dot path.
- [x] A two-row Q8_0 fixture produces two distinct expected outputs.
- [x] Each remaining format produces its expected one-row output.
- [x] Invalid pointers, type, dimensions, counts, and block metadata return
      `0`.
- [x] `quantized_matvec_reference_test` passes.

## Out of scope

- F32 matrices; use the existing `tensor_matvec_f32` path.
- Any SIMD, BLAS, multithreading, GPU, or heap-allocated decoded matrices.
- Transformer layer scheduling and `engine_forward_token`.

## Next milestone

Perform an opt-in check against real tensors in the local Qwen GGUF file, then
begin the transformer-forward milestone.
