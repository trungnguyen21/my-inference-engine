# Milestone 05H: Real-Model Quantized Integration Check

This document records a completed historical acceptance check. The real-model
executable has been removed; current automated tests use small synthetic
fixtures. See [current test commands](../README.md#tests).

## Goal

Validate the completed packed-tensor path against the downloaded Qwen GGUF:

```text
mapped GGUF tensor --> packed tensor view --> direct block dots --> row output
```

This is an opt-in executable, not a CTest target. A normal test suite should
not require a multi-gigabyte model file at a fixed local path.

## What the executable checks

`real_model_quantized_check` opens the path supplied on its command line. For
each packed GGML type present in this model—Q5_0, Q8_0, Q4_K, and Q6_K—it:

1. finds a two-dimensional tensor of that type;
2. obtains a `GgufQuantizedTensorView` by name;
3. creates a small deterministic F32 activation vector;
4. forms a coherent view of the tensor’s first row; and
5. calculates one finite output through `quantized_tensor_matvec_f32`.

Checking one row keeps the command quick. The fixture test in 05G checks that
multiple rows start at distinct packed-byte offsets.

## Run

For current regression coverage of the packed tensor and matvec paths, run:

```sh
ctest --test-dir build \
  -R 'gguf_quantized_view_test|quantized_matvec_reference_test' --output-on-failure
```

The historical executable printed the selected tensor name, type, row width, and calculated value for
each format. The values are a structural check, so they do not have a fixed
expected number across different model files.

## Definition of done

- [x] The executable opens the local Qwen GGUF successfully.
- [x] It locates and views a matrix for Q5_0, Q8_0, Q4_K, and Q6_K.
- [x] Each selected first row produces a finite F32 result.
- [x] The command exits with code 0.

## Completion of Milestone 05

After this check passes, the project has a verified read path for every storage
type found in the downloaded model, from mapped bytes to reference matrix-vector
results. The next milestone can select Qwen’s named weights and schedule the
operations of one transformer layer.
