# Batched prefill implementation plan

**Status:** The checked API, transactional Qwen2 prefill, bounded layer-first
chunks, and F32/Q5_0/Q8_0/Q4_K/Q6_K batched projection dispatch are implemented.
Chunks currently contain up to eight positions. Packed blocks are copied once
per output row/block and applied to every activation in the chunk through the
existing format-specific NEON dot kernels. Attention remains causal and runs
one query position at a time after the chunk's K/V slots are populated. The
single-token cached path remains the decode path and correctness oracle.

## Goal and non-goals

**Goal:** reduce time to first generated token by processing bounded chunks of prompt positions through each Qwen2 layer and reusing a weight block across several activation vectors. Preserve the existing single-token cached forward path for generation and as a correctness baseline. Compute final vocabulary logits only once, for the last prompt token.

**Current baseline:** `src/app.cpp` submits the prompt through `engine_prefill_tokens`. Qwen2 processes bounded chunks layer-first, batches the weight projections, applies RoPE at absolute positions, writes each token's K/V into the persistent cache, and computes vocabulary logits only for the final prompt position. There is **no full-prefix layer recomputation to eliminate** in the active path. A new API or a layer-first schedule by itself will not improve speed if it still calls the existing matvec for every token: batched weight reuse is the performance milestone.

**Out of scope:** changing decode kernels/NEON, changing GGUF weights, GPU offload, multi-request batching, paged attention, and redesigning causal attention. Keep F32 activations and the existing scalar attention implementation initially.

## Blast radius / areas of damage

| Surface | Files | Change and principal risk |
| --- | --- | --- |
| App prefill | `src/app.cpp`, `tests/app_generation_contract_test.cpp` | Replace only the prompt loop with a checked `engine_prefill_tokens` call. Sampling, EOS, generated-token forwarding, and output remain unchanged. Risk: wrong last-token logits or changed failure cleanup. |
| Generic engine boundary | `include/engine.h`, `src/engine.cpp`, `src/architecture.h` | Add a batch-prefill callback and checked dispatch (including model/state/backend identity). Retain existing `engine_forward_token` and no-logits paths for decode/tests. Risk: invalid counts, mismatched architecture state, or unclear failure semantics. |
| Qwen2 scheduling | `src/qwen2_architecture.cpp`, optionally `src/qwen2_internal.h` | Add a layer-first, chunked prefill path alongside `qwen2_forward_token_cached_impl`. Risk: RoPE positions, causal visibility inside chunks, GQA cache offsets, residual ordering, and retries after failure. This is the **highest correctness risk**. |
| Workspace/ownership | `src/qwen2_internal.h`, state init/destroy in `src/qwen2_architecture.cpp`, `tests/inference_state_contract_test.cpp` | Allocate bounded `[chunk, ...]` scratch once at state init; check all element and byte multiplications for overflow; free on normal and partial-init paths. Risk: excess RAM, aliasing, leaks, or per-token allocation. |
| Batched math | New `src/` packed-batched kernel and header; possibly `include/tensor.h`, `src/tensor.cpp`; `CMakeLists.txt` and kernel tests | F32 baseline and Q5_0/Q8_0/Q4_K/Q6_K block-reuse kernels. Risk: GGUF row/block layout, numeric drift versus NEON matvec, unsupported types, or worse speed for small chunks. Do not change current decode kernels. |
| Timing and docs | `include/metrics.h`, `src/terminal.cpp`, `README.md`, profiling tests | Distinguish prefill from decode time; existing per-format call counts will not be directly comparable after batching. Risk: reporting a throughput gain when only prefill changed. |

No changes should be needed in tokenizer encoding, sampling, model-file parsing, or the existing decode KV layout. Add specific files above only when the corresponding milestone begins.

## Fixed contracts before coding

- Proposed API: `engine_prefill_tokens(model, state, ids, count, error, capacity)`. Require non-null pointers, `count > 0`, valid token IDs, matching model/state backends, and enough context for **all** IDs before writing. Add an architecture callback rather than hard-code Qwen2 in `src/engine.cpp`. If an architecture has no prefill callback, either explicitly reject it or deliberately implement the existing sequential fallback; choose and test one behavior. Initially Qwen2 can implement the callback.
- Success: append all IDs exactly once, populate K/V for every new position in every layer, leave finite logits for the **last** ID, set `logits_count = vocab_size`, and advance `token_count` by `count`. A subsequent `engine_forward_token` must work without rebuilding those K/V entries.
- Failure: do not publish a partially advanced `token_count` or valid logits. Reject invalid IDs/capacity up front; if processing fails later, slots beyond the old count may contain scratch values but must remain invisible. Retrying the same batch must overwrite them. Stage new `token_ids` or write them only on commit. Define how existing valid logits are invalidated on failure, and test that contract.
- Never allocate per token/chunk inside a successful forward; all scratch must be reserved at state initialization. Cap the scratch chunk size independently of prompt length. Keep the current cached forward entry points and full-prefix reference tests intact.
- An initial chunk size can be a small compile-time constant (e.g. 8); include `B=1` and tail chunks of length 1 in tests. Only make it configurable after finding a useful range on the M2.

## Scheduling and shapes

Let `T` be prompt positions, `B` the current chunk length, `D=dim`, `H=hidden_dim`, and `K=n_kv_heads * head_dimension`. Store chunk activations in token-major rows, `X[B,D]`. A GGUF weight has dimensions `[in,out]`, with each contiguous packed row corresponding to one output feature. A batched projection computes `Y[b,o] = sum_j W[o,j] * X[b,j]` and writes `Y[B,out]`. This convention avoids quietly transposing GGUF weights.

For each chunk `[start, start+B)`:

1. Look up its `B` embeddings into the chunk activation buffer.
2. **For each layer:** normalize all `B` activations; project Q `[B,D]`, K/V `[B,K]`, and add their biases. Apply RoPE at each token's **absolute** position. Store that layer's K/V for every position in the chunk using the existing `[layer][position][KV head][head dimension]` layout.
3. For each position `p` in the chunk, compute attention with exactly `p+1` cached positions. Keys/values for later positions may already be written but must **not** be visible to token `p`. Preserve the existing query-head to KV-head mapping in `transformer_attention_f32`.
4. Batch the attention-output projection, add the per-token residual, then batch FFN norm, gate/up projections, SwiGLU, down projection, and final residual. Feed the resulting `[B,D]` activations to the next layer. Do not apply one token's layer `L+1` before the other tokens finish layer `L` if the batched buffers rely on their layer-`L` results.
5. Process the next chunk, reusing the allocated buffers. After the final chunk and final layer, normalize only its last activation and run `output.weight` **once**. Commit the batch token count and logits on success.

Existing single-vector scratch fields cannot hold all `B` queries, outputs, or FFN intermediates. Define explicit sizes for each new buffer (including `[B,H]` gate/up); keep context-length-sized attention score scratch from growing into a full `[T,T]` allocation. Check aliasing across projections and residuals before reusing storage.

## Step-by-step milestones and gates

### 0. Record baseline and oracle (no new fast kernel)

- Fix model/prompt IDs, Release build, `OMP_NUM_THREADS=4`, generation limit, and temperature. Record repeated **unprofiled** time to first token, generation tok/s, and peak RAM for short and long prompts. Record prefill and decode separately; current `--profile` combines both.
- Use the existing cached token-at-a-time path (no logits except last) as the primary parity oracle; retain `qwen2_forward_token_reference` as an independent tiny-fixture check.
- **Gate:** repeatable baseline and tests for one token, two tokens, and a prompt crossing intended chunk boundaries.

### 1. Add the prefill API with a sequential Qwen2 implementation

- Add generic dispatch, callback, app call, and contract tests. Initially call the existing no-logits cached path for `count-1` IDs, and the logits-producing path for the last. This changes control flow without claiming a performance improvement.
- Prevalidate the entire batch, save the original token count, and on any sequential-forward failure restore that count and invalidate `logits_count`. The attempted calls only write slots beyond the original count; a retry overwrites them. This gives the externally visible all-or-nothing state contract even while the fallback advances its internal count between calls. Keep attempted-work profiling separate from committed-token counts.
- **Gate:** identical generated text and final logits; backend mismatch, null/zero count, invalid IDs, insufficient capacity, and app cleanup tests pass. Existing decode path unchanged.

### 2. Introduce bounded layer-first chunk scheduling, still using matvec

- Allocate scratch once, move the Qwen2 layer schedule to `[chunk][layer][token]`, and use existing per-token projection helpers until the schedule is correct. Keep an opt-in or test-only dispatch path, leaving the sequential fallback available.
- Copy or factor existing Qwen2 layer operations carefully; do not delete the reference or cached single-token code during the first parity pass. Compare cache entries per layer/position as well as final logits. When introducing the new schedule, implement the final all-or-nothing commit contract and test late failures/retry.
- **Gate:** parity for `B=1`, exact multiples and non-multiple tails, existing prefix + appended batch, absolute RoPE positions, GQA (`n_heads > n_kv_heads`), causal within-chunk masking, and next-token decode. Timing may be unchanged or worse; correctness comes first.

### 3. Add and validate batched projection kernels

- First define a small F32 batched projection contract; compare it with `B` existing F32 matvec calls for rectangular weights, `B=1`, and tail chunks.
- Add packed block-reuse kernels for the actual GGUF formats **Q5_0, Q8_0, Q4_K, Q6_K**. For an output row and packed weight block, decode/load once and accumulate into `B` independent output rows. Keep per-token accumulation order as close as possible to the scalar path. A single-type whole-matrix dispatch should select the appropriate kernel; preserve unsupported/invalid-view error handling.
- Compare packed batched output with repeated current packed matvecs, using documented tolerances for possible NEON reassociation. Test buffer boundaries, input/output non-aliasing, quantized block/row alignment, and odd tails. No per-call heap allocation.
- **Gate:** batched kernel tests plus cached-vs-batched end-to-end logits and a subsequent decode match within tolerance for every supported format.

### 4. Integrate and measure before making batched prefill default

- Add a size-based fallback: `B=1` and very short prompts may be faster on the existing optimized NEON matvecs. Compare e.g. chunk sizes 1/4/8/16 under fixed build/thread settings; avoid assuming the largest is best on a CPU.
- Report prefill and decode separately, and optionally separate prefill packed-kernel timings. The existing `weight_*_calls` counts kernel invocations, which change meaning under batching; update report labels before using them to compare old/new versions.
- Benchmark repeated unprofiled TTFT for short/medium/long prompts, generated text parity, generation tok/s, and RAM. Turn on the new path by default only where measurements show a repeatable benefit without decode regression; otherwise retain the proven sequential path.
- **Gate:** all automated tests pass, malformed weights and allocation failures do not corrupt committed state, real GGUF produces matching outputs, and performance improvement exceeds run-to-run noise.

## Explicit rollback boundaries

- Milestones 1 and 2 should land independently of new numeric kernels; the existing `engine_forward_token` path remains selectable for correctness and fallback.
- Each quantization format in milestone 3 should be independently dispatched/fallback-capable. A slow or failing Q4_K kernel must not prevent using a faster Q5_0 kernel.
- If batched kernels do not beat NEON matvec at realistic prompt lengths, keep the correctness-tested prefill API/schedule only if it simplifies later work; do not force the slower path into the CLI.
