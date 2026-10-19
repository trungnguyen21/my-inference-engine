# KV-cache implementation exercise

The cached forward path is implemented and selected for generation. The original
full-prefix path is exposed internally as `qwen2_forward_token_reference` for
independent comparisons. `forward_reference_test` compares both paths on three
tokens using a tiny in-memory fixture. Multilayer and failure/retry coverage
remain follow-up work for the permanent test suite.

Find every change point with:

```sh
rg -n 'TODO\(kv-cache/' src include tests
```

Work through these items in order. Each number matches comments beside the code.

| TODO | Start here | Work to complete |
| --- | --- | --- |
| 01 | `src/qwen2_internal.h`: `Qwen2InferenceState` | Plan persistent `[layer][position][KV head][head dimension]` K/V. Keep singular `key`/`value` as scratch. |
| 02 | `src/qwen2_architecture.cpp`: `qwen2_state_init`, `qwen2_state_destroy` | Check every allocation-size multiplication; allocate both caches for all layers and requested capacity; retain safe partial cleanup. |
| 03 | `qwen2_kv_cache_layer_view` stub | Validate bounds and expose borrowed layer-base/current-slot pointers. Use capacity for layer stride and position + 1 for active length. |
| 04 | `qwen2_forward_token_cached` stub and existing forward loops | Embed only the appended token. Run its activation through each layer once, reusing the existing math and profiling scopes. |
| 05 | Existing RoPE call | Rotate only the new Q and K at absolute position `token_count`; do not rotate V or old cached keys. |
| 06 | Existing K/V copies and attention call | Store new K/V in this layer's current slot; attend over this layer's cached positions 0 through the current position. |
| 07 | Final normalization and token-count commit | Use the current activation for logits. Commit history/token count only after finite logits; preserve old slots on failure. |
| 08 | Architecture callback, `include/engine.h`, `src/transformer_ops.h` | Switch to the cached callback after verification. Update contracts and project documentation to describe the actual active behavior. |
| 09 | Existing inference-state and forward tests | Add multilayer reference parity, cache immutability, failure/retry, lifecycle, context, quantized-weight, and profiling checks. |

## The two stubs

`qwen2_kv_cache_layer_view` returns a `Qwen2KvCacheLayerView` with:

- `keys` / `values`: the start of one layer's cache at position zero.
- `key_slot` / `value_slot`: where this token's new K/V must be written.
- `token_count`: active attention length, including this token.

The view owns nothing. Its pointers borrow memory from `Qwen2InferenceState`.
The helper validates the model/state, computes offsets into the all-layer
allocation, and clears the output view when returning an error.

`qwen2_forward_token_cached` performs incremental forward execution. Use
`qwen2_forward_token_reference` immediately above it as the full-prefix oracle,
with separate states for the two schedules.

## Workspace and reference order

Expanding K/V allocations first is compatible with the old forward function,
which will still use its temporary first-layer-sized workspace. Its output is
not a persistent cache; do not mix full-prefix and cached execution in one state.
Create separate states for reference and cached comparisons, and let cached
execution populate its own state from the first token.

Keep the current `context * dim` activation allocation while the reference
depends on it. After parity checks, give the test reference separate full-prefix
workspace and compact production activations to `dim`. Shrinking that allocation
while the old prefix loop still writes into it would cause an out-of-bounds write.

Before the callback switch, make the full-prefix reference callable from tests.
Compare every logit after each appended token using the same weights and IDs.
Packed-vs-F32 execution alone is not an independent scheduling reference.

## Suggested checkpoints

1. Implement checked allocation and layer views. Check at least two layers,
   multiple positions, invalid bounds, and cleanup.
2. Implement cached execution and compare with the existing F32 expected logits.
3. Compare against full-prefix execution using a multilayer fixture with grouped
   query attention and identical F32 or packed weights on both sides.
4. Fail after cache writes, repair the fixture, and retry with a different token.
   Confirm old entries remain unchanged and the retry matches a clean reference.
5. Verify context rejection, independent states, and enabled/disabled profiling.
6. Verify the callback, update contracts/README/project guide, and run the
   automated fixture suite before measuring prefill/decode performance manually.

The existing app prompt loop already forwards tokens in order and can fill the
cache without a new prefill API. Batched prefill is a separate follow-up.

Build and run the focused existing checks with:

```sh
./build.sh
ctest --test-dir build --output-on-failure \
  -R 'inference_state_contract|transformer_ops_reference|forward_reference|forward_quantized_reference|app_generation_contract'
```

The TODOs in tests are prompts for future cache tests, not passing placeholders
claiming that caching has been verified.
