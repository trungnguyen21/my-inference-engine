# Milestone 05F: Quantized GGUF Tensor Views

## Goal

Safely expose one Q5_0, Q8_0, Q4_K, or Q6_K tensor from the mapped GGUF file
as a read-only view of packed blocks.

This mirrors 05A’s F32 view, but the `data` pointer remains raw bytes because
the correct C structure depends on the tensor’s GGML type.

## Why this comes next

The block decoders operate on one block. A model weight matrix contains many
blocks stored back-to-back in the mapped GGUF data section. This view carries
the pointer, shape, type, and block layout needed to locate those blocks
without copying the model into RAM.

```text
GGUF tensor directory --> validated packed tensor view --> block decoder
       name/type/shape             data + layout             Q4_K/Q6_K/etc.
```

## API

`GgufQuantizedTensorView` and `gguf_get_quantized_tensor_view` are declared
in `src/gguf_reader.h`. Their full contract is beside the declaration.

The view records:

- the read-only mapped byte pointer;
- the GGML type;
- the original GGUF dimensions and total element count;
- values per packed block, bytes per block, and total block count.

The view is valid only until `gguf_close` is called on its `GgufFile`.

## Implementation order

1. Clear the output view before checking other arguments.
2. Find the tensor and accept only the four packed types present in this
   model: Q5_0, Q8_0, Q4_K, and Q6_K.
3. Validate dimensions and calculate the element count without overflow.
4. Use the existing internal type-layout helper to obtain block dimensions.
5. Check exact block divisibility and exact byte size, then store the mapped
   address and metadata in the view.
6. Run the fixture test. It has no dependency on the downloaded model.

## Definition of done

- [x] A valid fixture works for all four supported packed types.
- [x] The returned view has the correct pointer, type, shape, element count,
      block size, and block count.
- [x] Missing names, F32 requests, non-divisible shapes, and bad byte sizes
      fail and clear the output view.
- [x] `gguf_quantized_view_test` passes.

## Out of scope

- Decoding a block; that is covered by 05B–05E.
- Computing a matrix-vector result.
- Modifying model loading or `engine_forward_token`.

## Next milestone

Use the type and block layout in this view to dispatch each matrix row to its
matching direct packed dot-product function.
