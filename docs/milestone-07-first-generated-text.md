# Milestone 07: First Generated Text

## Goal and prerequisite

Generate a short, readable answer from the local Qwen2.5-Instruct GGUF using
your forward pass, deterministic greedy selection, the existing tokenizer,
and the existing terminal app. Finish the
[quantized forward checkpoints](milestone-06-quantized-forward.md) first.

The greedy sampler and single-user prompt formatter are implemented. The
generation loop, token decoding, EOS detection, cleanup, and metrics are wired
and tested. The sections below describe their contracts and verification,
without adding performance work. All 23 checks pass with the local model
enabled. Actual outputs include `Hello` for `Say hello.` and
`The capital of France is Paris.` for the full CLI smoke check (2026-10-16).

## 07A: Greedy next-token selection

The implementation is in `src/sampler.cpp`; its contract is also in
`src/sampler.h`:

```cpp
int sampler_greedy_sample(
    const float *logits, int vocab_size, int *token_id,
    char *error, int error_capacity);
```

The input is the latest forward call's vocabulary-sized array of raw scores.
The output is the ID of the largest score. A larger score means a preferred
next token. Break exact ties by choosing the lowest ID. Negative scores and
very large finite scores are valid; probabilities and softmax are unnecessary
for this selection.

- Return `1` only after inspecting a valid, entirely finite vector.
- Reject null logits, null output ID, nonpositive vocabulary size, and any
  NaN or positive/negative infinity, including entries after the apparent winner.
- Return `0` with a useful error and set `*token_id = -1` when writable.
- Leave the logits unchanged. Use no allocation and no random state.
- Support a null error buffer or a nonpositive error capacity. Respect small
  writable error capacities without overwriting neighboring bytes.

The public `sampler_sample` already validates its arguments and dispatches
`temperature == 0` to this helper. `top_p` may be any finite value in `[0, 1]`
on this path; it has no effect on greedy selection. Positive-temperature
sampling deliberately returns a message directing you to `--temperature 0`.

```sh
cmake --build build --target sampler_greedy_test
ctest --test-dir build -R '^sampler_greedy_test$' --output-on-failure

# Already implemented: public argument validation and deferred-mode error.
ctest --test-dir build -R '^sampler_validation_test$' --output-on-failure
```

`tests/sampler_greedy_test.cpp` covers an ordinary winner, all-negative scores,
ties at zero and later IDs, signed zero, a singleton, finite float extremes,
invalid entries at every position, input immutability, and public dispatch.

**Checkpoint:** the logits now produce one valid token ID deterministically.

## 07B: Format a single instruction prompt

`qwen2_format_prompt` is implemented near the end of
`src/qwen2_architecture.cpp`. Its signature is:

```cpp
static int qwen2_format_prompt(
    const char *user_prompt, char *output, int output_capacity,
    char *error, int error_capacity);
```

The callback is registered in `ArchitectureOps`. The public
`engine_format_prompt` checks the loaded model and dispatches to it.
`app_run` calls that public function before `tokenizer_encode`, while displaying
the original user prompt. Keep Qwen-specific strings inside the Qwen backend.

The local GGUF's `tokenizer.chat_template` was inspected. For one user message,
no tools, the default system message, and an assistant-generation prefix, the
exact result is:

```text
<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
<|im_start|>user
{user_prompt}<|im_end|>
<|im_start|>assistant
```

`{user_prompt}` means the user's original bytes, with no added whitespace
before the following `<|im_end|>`. There is a newline after the final
`assistant` header, followed immediately by the terminating NUL. Do not
append an assistant answer or an end token. Do not tokenize here: the existing
tokenizer handles special-token strings when encoding the completed prompt.

The adapter now enables special-token parsing. The real-model schema check
also verifies that each ChatML boundary marker encodes as one token and that
the end-of-message token is recognized as an end-of-generation ID.

- Copy the entire result to a separate output buffer; input/output overlap is
  outside this contract.
- Preserve UTF-8 bytes, whitespace, and embedded newlines. Empty user content
  is allowed; a null user pointer is invalid.
- Include the NUL when checking capacity. Exact fit succeeds; one byte too
  short fails. Do not return a truncated prompt as success.
- Reject a null output or nonpositive output capacity. Clear `output[0]` on
  failure if the output is writable, and provide an error where possible.
- Never write outside either supplied capacity; no allocation is required.

The app reserves `PROMPT_CAPACITY + 512` bytes for the result, enough for its
bounded user prompt and this wrapper. Count the formatted prompt's tokens
when checking context capacity: system text and role markers also take tokens.

The formatter intentionally covers this local Qwen2.5-Instruct template. It is
not an arbitrary Jinja renderer. Supporting another model's system defaults,
custom templates, tools, or multi-turn history is later work.

```sh
cmake --build build --target prompt_format_reference_test
ctest --test-dir build -R '^prompt_format_reference_test$' --output-on-failure

# Already implemented: generic dispatch and validation, with a fake callback.
ctest --test-dir build -R '^prompt_dispatch_test$' --output-on-failure
```

`tests/prompt_format_reference_test.cpp` checks exact output bytes, empty and
multilingual content, a maximum CLI-length message, exact-fit and insufficient
capacities, guards around the output, invalid pointers, and architecture dispatch.

**Checkpoint:** a user message encodes as an instruction followed by the start
of the assistant response.

## 07C: Verify the existing generation loop

The implemented `app_run` in `src/app.cpp` already performs this sequence:

```text
load -> format -> tokenize -> check total context -> initialize state
forward every prompt ID once -> keep final prefill logits
sample -> check ID -> stop if EOS -> decode and print
forward that generated ID only if another sample is needed -> repeat
report metrics -> destroy state -> unload model
```

The last prompt token's logits predict the first output token. The final
sampled token does not need another forward call when the token limit is
reached. EOS is never decoded, printed, or forwarded. The metrics convention
counts every sampled token, including EOS; the number of printed pieces may
therefore be smaller than the reported token count.

```sh
cmake --build build --target app_generation_contract_test
ctest --test-dir build -R '^app_generation_contract_test$' --output-on-failure
```

This passing test compiles the real app with a scripted backend, tokenizer,
terminal, and clock. It proves formatted text reaches the tokenizer, prompt
IDs are not repeated, later samples use fresh logits, one-token and multi-token
limits work, and EOS stops immediately or after a piece. It also checks invalid
logits/IDs, decode failure and missing NUL, short context, bad prompt counts,
failures at load/format/tokenize/init/forward/sample, cleanup after partial
initialization, CLI help, and invalid arguments. Scripted text is a control-flow
fixture, not actual model generation.

**Checkpoint:** no new loop implementation is needed; keep this regression green.

## 07D: Run the real model

Follow the final commands in the
[roadmap to generated text](roadmap-to-generated-text.md#final-acceptance).
The opt-in `real_text_generation_check` invokes the built CLI with the local
GGUF, a short question, a 16-token limit, and temperature zero. It rejects a
failed exit, missing/whitespace-only output, leaked ChatML/end markers, missing
completion/metrics, or a token count outside the limit. It has no speed gate
and does not require an exact answer string.

```sh
cmake -S . -B build -DENGINE_TEST_MODEL_PATH=models/qwen-2.5.gguf
cmake --build build --parallel
ctest --test-dir build -R '^real_text_generation_check$' --output-on-failure
```

Read the response as well as running the smoke test. Finite logits and visible
characters alone cannot establish model correctness. A coherent answer to
the question plus the numerical forward fixtures provides stronger evidence.

To return to the model-free test configuration, run:

```sh
cmake -S . -B build -DENGINE_TEST_MODEL_PATH=
```

## Definition of done

- [x] Finish Milestone 06, including packed fixtures and real-model logits.
- [x] Pass greedy selection and the public sampler routing tests.
- [x] Pass exact single-user prompt formatting and architecture dispatch tests.
- [x] Prepare and pass deterministic app sequencing, EOS, limit, and error tests.
- [x] Pass all 23 registered checks with `ENGINE_TEST_MODEL_PATH` configured.
- [x] See readable, relevant output from the real CLI with temperature zero.
- [x] Check a one-token limit, a second prompt, and an invalid path with the CLI.

Optimization and stochastic sampling are outside this first-text milestone.
