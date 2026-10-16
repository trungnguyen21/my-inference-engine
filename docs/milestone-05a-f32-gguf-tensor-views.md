# Milestone 05A: F32 GGUF Tensor Views

**Status: Complete.** The fixture tests and manual Qwen normalization-vector
check are recorded below.

## Goal

Safely expose an F32 tensor already stored in the memory-mapped GGUF file as a
read-only view. Start with Qwen's F32 RMSNorm weights.

The view does not copy any model bytes. It points into the mapped GGUF file,
so it remains valid only while `GgufFile` stays open.

## Why this comes next

Your reference kernels operate on `const float *` inputs. The model file is
the source of those floats. Before handling packed Q4_K, Q5_0, Q6_K, and Q8_0
weights, prove that the engine can locate and safely read a simple F32 tensor.

```text
GGUF tensor directory
        |
        v
name, type, dimensions, byte offset
        |
        v
validated pointer into mapped tensor-data section
        |
        v
const float * used by RMSNorm
```

## What already exists

`GgufFile` already owns the mapped file. `GgufTensorInfo` already records a
tensor's name, type, dimensions, byte offset, and computed byte size.

These helpers already work:

```cpp
const GgufTensorInfo *gguf_find_tensor(const GgufFile *file, const char *name);
const void *gguf_tensor_data(const GgufFile *file, const GgufTensorInfo *tensor);
```

This milestone builds a narrow, typed wrapper around them. Do not change model
loading or call the tensor math kernels from `engine_forward_token` yet.

## New internal API

Add this private view type to `src/gguf_reader.h`:

```cpp
typedef struct GgufF32TensorView {
    const float *data;
    uint32_t dimension_count;
    uint64_t dimensions[4];
    uint64_t element_count;
} GgufF32TensorView;
```

Then add a function with this contract:

```cpp
int gguf_get_f32_tensor_view(
    const GgufFile *file,
    const char *name,
    GgufF32TensorView *view
);
```

It returns `1` on success and `0` when the file, name, or output view is null;
the tensor is absent; the tensor is not GGML type `0` (`F32`); its dimensions
are invalid; the element count overflows; or the mapped data pointer is not
available.

On success:

- `view->data` points to the first F32 element in GGUF storage;
- `view->dimension_count` and `view->dimensions` match the directory entry;
- `view->element_count` equals the product of the dimensions;
- the caller must not write through `view->data` or keep the view after
  `gguf_close`.

Clear the output view before validation so callers never receive stale values
after a failure.

## Implementation steps

1. Add the type and function declaration to `src/gguf_reader.h`.
2. In `gguf_reader.cpp`, zero the output view and validate all three function
   pointers.
3. Find the tensor by name. Require its type to be F32.
4. Multiply dimensions carefully into `element_count`; reject zero dimensions
   and integer overflow.
5. Verify `tensor->byte_size == element_count * sizeof(float)` without an
   overflowing multiplication.
6. Obtain the data address with `gguf_tensor_data`, store it as a read-only
   F32 pointer, and copy shape information into the view.
7. Keep the function free of allocations and model-specific names.

## First real-model target

Use an F32 normalization tensor from the local Qwen model, such as a layer's
attention-normalization or feed-forward-normalization weight. Its one
dimension should equal `ModelConfig.dim` (`896` for this model).

Read a few elements only to confirm they are finite float values. You are
checking pointer safety and tensor metadata here, not validating Qwen's model
quality.

## Tests

Create a small `gguf_f32_view_test` executable separate from
`tensor_reference_test`.

It should cover a small in-memory `GgufFile` fixture first, so the test does
not depend on a downloaded model:

- a valid F32 tensor returns its data pointer, dimensions, and element count;
- a missing name returns failure and clears the output view;
- a non-F32 tensor returns failure;
- a zero dimension or overflowing dimension product returns failure;
- a mismatched byte size returns failure.

Then add a manual check using `models/qwen-2.5.gguf`:

- open the file with `gguf_open`;
- obtain an F32 normalization-weight view;
- assert a one-dimensional shape of `896` and finite sample values;
- close the file only after the view is no longer used.

Keep the real-model path out of normal CTest registration until the test suite
has a portable model fixture.

## Definition of done

- [x] `GgufF32TensorView` has a clear lifetime contract.
- [x] `gguf_get_f32_tensor_view` rejects invalid requests without stale output.
- [x] The in-memory fixture covers valid F32, missing, wrong-type, invalid
      shape, and bad byte-size cases.
- [x] A manual Qwen check reads one F32 normalization vector safely.
- [x] No forward-pass or quantized-weight code has been added.

## Out of scope

- copying the full model into RAM;
- F16 conversion and all quantized block decoding;
- matrix-vector products over model weights;
- Qwen block tensor lookup or `engine_forward_token` changes.

## Next milestone

Define and test one quantized block layout, beginning with Q8_0 because it is
the simplest packed format in this GGUF file. Compare its decoded values and
dot product against your F32 reference kernels.
