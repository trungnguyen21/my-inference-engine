# Milestone 06 Roadmap: Full-Prefix Transformer Forward Pass

## Status

The full F32 and packed forward paths are implemented. State allocation,
full-prefix recomputation, attention, FFN, final normalization, and logits pass
the numerical fixtures. The embedding decoder supports Q5_0, Q8_0, Q4_K, and
Q6_K; the shared weight dispatcher supports packed layer and output matrices.
The local schema and real-model forward checks pass; Milestone 06 is complete.

Each call retains token IDs and rebuilds activations and temporary K/V arrays.
No K/V results are reused between calls, and the forward path allocates no
memory after state initialization.

Latest verification:

- The full CMake build succeeds, including the state and schema test targets.
- All 20 model-free regression tests pass, including packed embeddings,
  packed projection dispatch, mixed-weight logits, greedy selection, prompt
  formatting, and app sequencing.
- Packed embedding and projection checks also pass AddressSanitizer and
  UndefinedBehaviorSanitizer.
- The local schema check passes. The real forward check processes two tokens,
  produces 151,936 finite logits, rejects invalid IDs without consuming context,
  checks context exhaustion, and clears owned state/model storage.
- The CLI produces `Hello` for `Say hello.` with a one-token limit, and
  `The capital of France is Paris.` in the full smoke check. All 23 checks pass
  with the local model enabled (2026-10-16).

Function contracts, fixtures, focused commands, and acceptance criteria are in
[Quantized Forward Integration](milestone-06-quantized-forward.md).

## Stage 1: Complete the Qwen model schema

The generic loader selects the Qwen2 backend. The items below are implemented
in `src/qwen2_architecture.cpp`; generic `src/engine.cpp` stays free
of Qwen metadata and tensor names.

- [x] Add checked GGUF floating-point metadata access for RMS epsilon and RoPE
      frequency base.
- [x] Retain those values with the loaded model configuration.
- [x] Add F32 layer-vector, matrix-vector, and projection-with-bias helpers
      with name/shape checks and useful errors.
- [x] Extend matrix-vector dispatch to packed weights for both layer and final
      output projections.
- [x] Validate every required Qwen tensor shape at model-load time, including
      Q/K/V biases and the global `output_norm.weight` vector.

Evidence: the direct real-model schema check passes. The F32 fixture also tests
that loading rejects a missing final-normalization tensor, a matrix in its
place, or a vector of the wrong length.

**Checkpoint:** an incomplete or wrongly shaped model fails during loading,
before any state allocation or forward call.

## Stage 2: Allocate and own inference state

- [x] Define the private state structure behind `InferenceState.implementation`.
- [x] Calculate checked sizes for token history, full-sequence activations,
      temporary K/V arrays, logits, and scratch buffers.
- [x] Implement `inference_state_init` and `inference_state_destroy`.
- [x] Track token-history length and reject tokens after `max_context_tokens`.
- [x] Ensure failed initialization leaves a destroyable zero or partial state.

Passing test: `inference_state_contract_test` checks the public logits buffer
and every required `Qwen2InferenceState` workspace pointer after successful
initialization, plus invalid context and destruction behavior.

**Checkpoint:** state initialization succeeds for a valid context, fails for
invalid lengths, and does not allocate during a later forward call.

## Stage 3: Implement and prove local transformer operations

- [x] Implement `transformer_apply_rope_f32` using split-half Qwen RoPE.
- [x] Implement `transformer_attention_f32` with scale, stable softmax, and
      grouped-query head mapping.
- [x] Implement `transformer_swiglu_f32`.
- [x] Register and pass `transformer_ops_reference_test` with CTest.

**Checkpoint:** the helper test passes its fixed numerical examples and invalid
argument checks.

## Stage 4: Schedule one complete token forward pass

The scheduling items below are verified on F32 and mixed packed-weight
fixtures. Stage 5 adds verification against the local GGUF.

- [x] Implement F32 embedding lookup.
- [x] Schedule attention norm, Q/K/V projections and biases, RoPE, temporary K/V writes,
      grouped-query attention, output projection, and attention residual.
- [x] Schedule FFN norm, gate/up projections, SwiGLU, down projection, and FFN
      residual.
- [x] Apply final norm and F32 output projection to fill the vocabulary logits.
- [x] Recompute all stored tokens and advance history length only after the
      entire call succeeds.
- [x] Report invalid token IDs, exhausted context, and failed numerical
      operations without advancing the state position.

Passing test: `forward_reference_test` builds a one-layer F32 Qwen-shaped
fixture in memory. It checks fixed logits after token `0`, then token `1`,
where the second result requires recomputing both prefix positions. It also
checks invalid token IDs, context exhaustion, final-norm schema validation, and
an attention failure followed by a successful retry.

**Checkpoint:** a one-layer in-memory F32 fixture produces expected logits for
two successive tokens and proves each call recomputes the complete prefix.

## Stage 5: Prove the actual model path

- [x] Add the tiny F32 forward fixture as `forward_reference_test` and register
      it with CTest.
- [x] Pass the full build and the 12 existing regression checks.
- [x] Pass the schema check against `models/qwen-2.5.gguf` when run directly.
- [x] Obtain and validate a packed embedding tensor view; report malformed or
      unsupported packed data.
- [x] Connect the row decoder and shared weight dispatcher, register three
      quantized checkpoints, and document their contracts and acceptance checks.
- [x] Decode the selected quantized embedding row into the activation buffer.
- [x] Connect `quantized_tensor_matvec_f32` to layer and final output
      projections, retaining the F32 path for reference fixtures.
- [x] Pass the embedding, matrix-dispatch, and mixed-weight forward checkpoints.
- [x] Run the regressions and every Milestone 06 test with the local model enabled.
- [x] Run `real_model_forward_check` on `models/qwen-2.5.gguf` and confirm a
      short prompt produces a finite logits vector with the correct vocabulary
      length.
- [x] Check state destruction after success and rejected initialization, and
      successful retry after a failed forward call.
- [x] Verify the complete real-model lifecycle through successful forward and
      forced failure.

**Checkpoint:** packed embedding rows and matrix-vector projections now pass
both focused tests and the mixed-weight forward comparison. The commands below
run synthetic unit and integration checks. Historical real-model acceptance
checks have been removed from the build and test suite.

## Commands

```sh
./build.sh

# All automated checks use synthetic fixtures or scripted backends.
ctest --test-dir build --output-on-failure

# Run the three packed-forward regressions independently.
ctest --test-dir build -L milestone06-quantized --output-on-failure

# Run the small cross-component checks separately.
ctest --test-dir build -L '^integration$' --output-on-failure
```

The [Milestone 07 guide](milestone-07-first-generated-text.md) describes
the implemented greedy sampler, prompt formatting, and generation loop.
