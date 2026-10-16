# Milestone 06: Full-Prefix Qwen Forward Pass

## Goal

Implement the engine path that accepts one token ID and produces the logits
that predict its successor. The state retains token IDs. Each call reruns the
model over the complete token history, including tokens processed by earlier
calls. It does not reuse saved K/V results between calls.

```text
token history --> embeddings --> 24 Qwen transformer layers --> final norm --> logits
                                      |                         |
                                      +-- temporary K/V arrays -+
```

This is one milestone. The work stages below are an implementation order and
test plan, rather than separate milestones to be signed off independently.

## Current progress

The F32 reference path is implemented: checked model schema, state lifecycle,
embedding lookup, attention with Q/K/V biases and RoPE, SwiGLU FFN, both
residuals, final normalization, and vocabulary logits. The two-token fixture
passes its expected numerical results and tests errors and retry behavior.

The packed embedding-row decoder and shared F32/packed weight dispatcher are
implemented for Q5_0, Q8_0, Q4_K, and Q6_K. All 20 model-free checks pass,
including the embedding, matrix dispatch, and mixed-weight logits comparisons.
The packed helper tests also pass AddressSanitizer and UndefinedBehaviorSanitizer.
The local schema and real forward checks pass, including 151,936 finite logits
and context/lifecycle error cases. The CLI produces `Hello` for `Say hello.`
with a one-token limit. Real-model CTest checks are opt-in through
`ENGINE_TEST_MODEL_PATH`. See the [roadmap](milestone-06-roadmap.md) and
[quantized integration contracts](milestone-06-quantized-forward.md).

## Architecture boundary

`engine.cpp` is shared infrastructure: it opens GGUF, reads
`general.architecture`, selects a callback table, loads the tokenizer, and
dispatches state/forward lifecycle calls. It must not contain `qwen2.*`
metadata keys or `blk.*` tensor names.

`qwen2_architecture.cpp` and `qwen2_internal.h` own the Qwen2 implementation:
its scalar metadata, tensor schema, state allocation, and forward schedule.
The numerical functions in `transformer_ops.cpp` and the tensor/quantization
packages remain shared helpers that a backend may call. Adding another model
family means adding another architecture module and registering it in
`architecture.cpp`.

## Completion contract

After the milestone:

```text
engine_forward_token(model, state, token_id, error, error_capacity)
```

must do all of the following on success:

1. consume exactly one valid token ID at the current state position;
2. append the token ID to the state history and recompute every position in it;
3. write one finite F32 logit for every vocabulary entry to `state.logits`;
4. set `state.logits_count` to `model.config.vocab_size`;
5. advance the state position by one; and
6. allocate no memory in the per-token path.

`inference_state_init` allocates token-history storage, all activation buffers,
the logits buffer, and temporary K/V workspace up front. The temporary K/V
arrays are rebuilt during each full-prefix pass and reused for each layer;
they are not retained as results for the next call. `inference_state_destroy`
frees those owned allocations and remains safe after failed partial setup.

Sampling and text output stay outside this milestone. The existing app already
has the correct prefill/decode sequencing and will use the produced logits once
the forward path works.

## Qwen2 tensor schema

The current Qwen2 backend supports Qwen2 GGUF models. Extend its model loading
to retain the following scalar metadata in the internal model configuration:

| Metadata key | Used for |
| --- | --- |
| `qwen2.embedding_length` | activation width `dim` |
| `qwen2.feed_forward_length` | FFN width `hidden_dim` |
| `qwen2.block_count` | number of layers |
| `qwen2.attention.head_count` | query-head count |
| `qwen2.attention.head_count_kv` | key/value-head count |
| `qwen2.context_length` | maximum sequence positions |
| `qwen2.attention.layer_norm_rms_epsilon` | RMSNorm epsilon |
| `qwen2.rope.freq_base` | RoPE frequency base |

The last two keys are floating-point GGUF metadata. Add a checked floating-point
metadata reader before model loading uses them; do not replace them with guessed
constants.

For each layer `L`, validate these tensor names and shapes while loading. GGML
stores a matrix as `[input columns, output rows]`.

| Tensor | Logical operation | Required shape |
| --- | --- | --- |
| `token_embd.weight` | token ID to initial activation | `[dim, vocab_size]` |
| `blk.L.attn_norm.weight` | attention RMSNorm | `[dim]` |
| `blk.L.attn_q.weight` | normalized activation to Q | `[dim, dim]` |
| `blk.L.attn_k.weight` | normalized activation to K | `[dim, n_kv_heads * head_dim]` |
| `blk.L.attn_v.weight` | normalized activation to V | `[dim, n_kv_heads * head_dim]` |
| `blk.L.attn_q.bias` | Q bias | `[dim]` |
| `blk.L.attn_k.bias` | K bias | `[n_kv_heads * head_dim]` |
| `blk.L.attn_v.bias` | V bias | `[n_kv_heads * head_dim]` |
| `blk.L.attn_output.weight` | attended values to activation | `[dim, dim]` |
| `blk.L.ffn_norm.weight` | FFN RMSNorm | `[dim]` |
| `blk.L.ffn_gate.weight` | FFN gate projection | `[dim, hidden_dim]` |
| `blk.L.ffn_up.weight` | FFN up projection | `[dim, hidden_dim]` |
| `blk.L.ffn_down.weight` | FFN down projection | `[hidden_dim, dim]` |
| `output_norm.weight` | final RMSNorm | `[dim]` |
| `output.weight` | final activation to vocabulary logits | `[dim, vocab_size]` |

The local Qwen2.5 0.5B file includes the Q, K, and V bias tensors. Add each
bias after its matching matrix-vector result and before RoPE or attention.

Use `gguf_get_f32_tensor_view` plus `tensor_matvec_f32` for F32 tensors and
`gguf_get_quantized_tensor_view` plus `quantized_tensor_matvec_f32` for
Q5_0, Q8_0, Q4_K, and Q6_K tensors. Keep this selection in one internal helper
so every layer uses identical shape checks and dispatch behavior.

## State layout

Let `head_dim = dim / n_heads`. The state stores token IDs received so far.
During a forward call, it uses temporary arrays for the current layer's K/V
vectors, laid out as:

```text
keys[position][kv_head][head_dim]
values[position][kv_head][head_dim]
```

Each temporary array contains:

```text
max_context_tokens * n_kv_heads * head_dim
```

F32 values per temporary array. Calculate every multiplication with overflow
checks before allocating. Keep activation data for all positions so the full
sequence can pass through each layer. The state also needs reusable F32 buffers
for normalized activations, Q/K/V, attention output/probabilities, FFN gate/up,
and matrix-vector output. Store token count and `max_context_tokens`.

For each layer, process positions from first to last. Write each position's K/V
into the temporary arrays, then let attention read positions `0` through the
current position. Later positions are not read, so attention remains causal.
The arrays are overwritten for the next layer and rebuilt from scratch on the
next call to `engine_forward_token`.

## Per-token algorithm

When a new token arrives, first check that it fits, then append it to state
history and reload embeddings for the complete history. For each layer, process
each position `p` from first to last:

1. RMS-normalize that position's activation with `attn_norm.weight`.
2. Project it into Q, K, and V; then add the Q/K/V biases.
3. Reshape Q as `n_heads * head_dim` and K/V as
   `n_kv_heads * head_dim`.
4. Apply RoPE to Q and K at position `p`; write K/V into the temporary arrays
   at position `p`.
5. Run grouped-query attention over temporary-array positions `0..p`.
6. Project the attended heads with `attn_output.weight` and add the result to
   this position's residual activation.
7. RMS-normalize with `ffn_norm.weight`.
8. Project gate and up, apply SwiGLU, project with `ffn_down.weight`, and add
   the result as a residual.

After all layers finish, RMS-normalize the final position with
`output_norm.weight`, project it through `output.weight` into `state.logits`,
and advance the state token count. On failure, leave the token count unchanged.

Use separate scratch buffers whenever a tensor helper requires non-aliasing
input and output. On a failure, return `0`, write a specific error where the
public API provides one, and leave the state position unchanged.

## Transformer operation helpers

[`transformer_ops.h`](../src/transformer_ops.h) contains these functions and
their detailed contracts. `transformer_attention_f32` now reads temporary
full-prefix key/value arrays and calculates each query head's scores,
probabilities, and weighted output. RoPE and SwiGLU are separate helpers.

| Function | Role in forward pass |
| --- | --- |
| `transformer_apply_rope_f32` | rotates Q and K by token position |
| `transformer_attention_f32` | computes scaled grouped-query attention from temporary key/value arrays |
| `transformer_swiglu_f32` | turns FFN gate/up projections into the FFN input |

`inference_state_init`, `engine_forward_token`, and `inference_state_destroy`
remain the public orchestration boundary in `include/engine.h`.

## Tests and acceptance evidence

Every Milestone 06 test is registered with CTest as soon as its source exists.
Enabled checks fail normally for incomplete or incorrect implementation; none
are marked as expected failures. The two Milestone 06 real-model checks are registered as
disabled unless CMake receives `ENGINE_TEST_MODEL_PATH`. Their disabled status
does not establish that real-model inference works.

| Check | Evidence it provides | Command when its implementation is ready |
| --- | --- | --- |
| `transformer_ops_reference_test` | numerical RoPE, grouped-query attention, SwiGLU, and invalid-argument behavior | `ctest --test-dir build -R transformer_ops_reference_test --output-on-failure` |
| `real_model_schema_check` | model loader retains RMSNorm/RoPE metadata from the local Qwen GGUF | `./build/real_model_schema_check models/qwen-2.5.gguf` |
| `inference_state_contract_test` | state allocation, public logits, private workspace layout, invalid requests, and destruction | `ctest --test-dir build -R inference_state_contract_test --output-on-failure` |
| `forward_reference_test` | exact known logits across successive full-prefix recomputations for a one-layer Qwen-shaped fixture | `ctest --test-dir build -R forward_reference_test --output-on-failure` |
| `qwen2_embedding_reference_test` | selected packed rows for all four formats, block strides, unaligned bytes, boundaries, and invalid views | `ctest --test-dir build -R '^qwen2_embedding_reference_test$' --output-on-failure` |
| `qwen2_matvec_reference_test` | F32/packed dispatch, rectangular weights, layer/global names, malformed storage, and invalid arguments | `ctest --test-dir build -R '^qwen2_matvec_reference_test$' --output-on-failure` |
| `forward_quantized_reference_test` | mixed packed/F32 forward versus a dense oracle, failed-forward retry, context, and cleanup | `ctest --test-dir build -R '^forward_quantized_reference_test$' --output-on-failure` |
| `real_model_forward_check` | actual Qwen tokenization, prefill, F32 logits buffer shape, and finite output | configure `ENGINE_TEST_MODEL_PATH`, then run `ctest --test-dir build -R real_model_forward_check --output-on-failure` |
| Full suite | regression protection for milestones 04 and 05 | `ctest --test-dir build --output-on-failure` |

The F32 fixture must include a nonzero Q/K/V path, two positions, grouped query
heads, and a nonzero FFN gate/up path. It should assert exact expected logits
for two successive calls, so a merely finite real-model result cannot hide a
wrong RoPE, attention, residual, or history-recompute result.

## Definition of done

- [x] Model loading retains checked RMS epsilon and RoPE frequency metadata.
- [x] Model loading validates every required tensor shape, including Q/K/V
      biases and the global final-normalization vector, before reporting success.
- [x] `inference_state_init` allocates checked token history, workspace, and
      logits storage; destruction safely releases partial state.
- [x] RoPE, grouped-query attention, and SwiGLU pass their reference test.
- [x] The F32 forward path recomputes full token history without per-token
      allocation and advances the token count only after success.
- [x] `forward_reference_test` checks exact logits from successive full-prefix
      calls, and is registered as a CTest target.
- [x] Forward integrates quantized embedding-row decoding and packed
      matrix-vector dispatch for layer and output projections.
- [x] The quantized embedding, weight-dispatch, and mixed-weight forward
      reference checkpoints pass.
- [x] The real local Qwen GGUF check produces one finite vocabulary-logit vector
      after a short prefill.
- [x] The completed CTest suite passes with all Milestone 06 tests enabled.

## Out of scope

- Greedy, temperature, and top-p sampling.
- The Qwen instruction chat template.
- SIMD, multithreading, GPU kernels, or performance optimization.
- Multi-sequence batching and speculative decoding.

## Next milestone

The [Milestone 07 guide](milestone-07-first-generated-text.md) now provides the
implemented greedy selection and prompt formatting, tests, and contracts. The app
prefill/decode loop is already wired and tested with a scripted backend. Follow
the [complete text roadmap](roadmap-to-generated-text.md) through real output;
performance work can follow afterward.
