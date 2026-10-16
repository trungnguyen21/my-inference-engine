# Build a CPU Qwen Inference Engine: a hands-on course

## Destination

Build a procedural C-style C++ program that loads your local
`models/qwen-2.5.gguf`, answers **"What is the capital of France?"**, and
reports token/s, CPU use, and RAM use. You will implement the inference path;
llama.cpp supplies tokenizer behavior only and never executes model layers.

The model is already trained. We are not training GPT or changing its weights.
Inference means applying the stored weights to a prompt to predict one next
token, then repeating that process.

```text
[ Your Prompt Text ]
       │
       ▼
 1. TOKENIZER          --> Converts text to Token IDs (e.g., "The" -> 42, "cat" -> 109)
       │
       ▼
 2. EMBEDDING LAYER    --> Looks up weights to turn Token IDs into Vectors (Shape: [Seq_Len, Hidden_Dim])
       │
       ▼
 3. TRANSFORMER BLOCKS (Repeated N times)
    │
    ├──► A. Linear Projections --> Multiplies word vectors by weight matrices to get Q, K, and V
    │                              (Query, Key, Value vectors for attention)
    │
    ├──► B. THE RoPE STEP     --> You apply the rotation matrices to Q and K right here!
    │                              (Before computing attention scores)
    │
    ├──► C. Self-Attention    --> Q changes K to find relationships; applies weights to V
    │                              (Each pass builds temporary K and V arrays)
    │
    └──► D. Feed-Forward      --> Final linear layers and activations pass data to the next block
       │
       ▼
 4. LM_HEAD (Output)   --> Turns final vectors back into probabilities over the vocabulary
       │
       ▼
 5. SAMPLING & DECODE  --> Picks the next Token ID -> Feeds it back to Step 1 (Looping)

```

## How to use this course

Each lesson has one idea, one build task, and an observable check. Do them in
order. Keep the implementation deliberately slow and readable at first;
measure and optimize only after the result is correct.

The project checkpoints are kept in `PROJECT_GUIDE.md`. A checked item means
code exists and has been exercised with the real Qwen file.

## Before lesson 1: the minimum vocabulary

| Term | Meaning in this project |
| --- | --- |
| Token | An integer representing a text piece, such as a word fragment or punctuation. |
| Vocabulary | The table mapping every supported token ID to a text piece. Qwen has 151,936 entries. |
| Embedding | A row of numbers representing one token at the start of the model. |
| Activation | A temporary vector of numbers while the model processes one token. |
| Weight | A learned number stored in the model file. It does not change during inference. |
| Logit | One unnormalized score per possible next token. The largest logit is the greedy choice. |
| Layer | One repeated transformer block. This Qwen model has 24. |
| Key/value workspace | Temporary vectors used to calculate attention in the current full-prefix pass. This first version rebuilds them on every forward call. |
| Quantization | A compact representation of weights. This Q4_K_M file uses several packed formats, not only 32-bit floats. |

## Lesson 1: understand the generation loop

### Concept

A causal language model predicts the next token from the tokens before it.
For `[A, B, C]`, a forward pass over `C` produces scores for the token after
`C`. Sampling chooses that next token `D`. Forwarding `D` produces scores for
the token after `D`.

There are two phases:

```text
prefill: prompt token 1 -> prompt token 2 -> ... -> prompt token N -> logits
decode:  sample logits -> generated token 1 -> forward -> next logits -> ...
```

The final prompt token is forwarded once during prefill. Its logits are used to
pick the first generated token; do not forward it again.

### Build task

Read `app_run` in `src/app.cpp`. It already has the correct prefill/decode
control flow. Trace the lifetime of `Model`, `InferenceState`, and
`prompt_token_ids` through the `cleanup:` label.

### Done when

You can explain why `engine_forward_token` is called once per input token and
why its output is logits for the *following* token.

## Lesson 2: load and inspect the model file

### Concept

GGUF is a container format. It contains a header, typed metadata, a tensor
directory, then the raw tensor bytes. It is not the inference engine itself.

```text
GGUF header | metadata | tensor directory | aligned tensor data
```

The loader memory-maps the model instead of copying its 469 MB contents. The
operating system then loads file pages when your matrix code actually reads
them.

### Build task

Read, then run, the existing loader:

```sh
./build.sh
./build/inference --model models/qwen-2.5.gguf \
  --prompt "What is the capital of France?" --max-tokens 8
```

Study `engine_load_model` in `src/engine.cpp` and `gguf_open` in
`src/gguf_reader.cpp`. Locate the metadata that becomes `Model.config`:

```text
dim=896, hidden_dim=4864, layers=24
query heads=14, KV heads=2, vocabulary=151936, context=32768
```

### Done when

The command reaches `Full-prefix inference state initialization is not implemented.` This
proves the file was opened, structurally checked, recognized as Qwen2, and its
prompt was encoded before it reached the next missing layer.

## Lesson 3: use a general GGUF tokenizer adapter

### Concept

The model receives token IDs, never C strings. Qwen uses a BPE-style tokenizer:
start with byte-level pieces, repeatedly merge adjacent pieces that form a
known vocabulary token, then return their IDs. The project delegates that
model-specific behavior to llama.cpp so you can focus on the inference engine.

### Build task

Read `src/tokenizer.cpp`. It loads the GGUF with `vocab_only = true`, then uses
llama.cpp to encode prompts, decode a token piece, and identify end-of-
generation tokens. It does not create a llama.cpp context or call its forward
pass.

Trace the adapter boundary:

```text
app.cpp -> tokenizer_encode -> llama_tokenize
app.cpp -> tokenizer_decode_one -> llama_token_to_piece
app.cpp -> tokenizer_is_eos -> llama_vocab_is_eog
```

### Done when

The real Qwen command reaches the inference-state stub. This proves prompt encoding
completed; do not expect a response until the forward pass exists.

## Lesson 4: write small float math first

### Concept

The transformer is mostly vector operations and matrix-vector products. A
matrix-vector product produces one output number per matrix row:

```text
y[row] = sum over column of matrix[row, column] * x[column]
```

Write a plain float reference version before handling quantized Qwen weights.
Correctness is much easier to establish with a tiny 2-by-3 matrix than a
151,936-by-896 output matrix.

### Build task

Create `src/tensor.cpp` and unit-sized hand checks for:

1. vector addition and scalar multiplication;
2. dot product and matrix-vector multiplication;
3. RMS normalization;
4. SiLU activation and elementwise multiplication;
5. stable softmax.

Use fixed input arrays and expected values written in the test/check program.

Complete [Milestone 04A reference-kernel spec](docs/milestone-04a-reference-f32-kernels.md), then continue with [Milestone 04B transformer operations](docs/milestone-04b-transformer-operations.md).

### Done when

Every small result agrees with a hand calculation. Keep these reference kernels
even after introducing optimized kernels.

## Lesson 5: read quantized Qwen tensors

### Concept

Q4_K_M is a model-file recipe, not one universal data type. Your file has F32,
Q5_0, Q8_0, Q4_K, and Q6_K tensors. Each quantized block stores packed values
plus scale information. You must either dequantize blocks to floats or compute
dot products directly from their packed representation.

### Build task

For each format, define a C-style block struct whose byte size exactly matches
the GGUF reader’s storage-size table. Start with F32. Then implement one
format at a time and compare a small dot product against a trusted float
reference. Use direct quantized dot products for the eventual CPU path.

Complete [Milestone 05A: F32 GGUF tensor views](docs/milestone-05a-f32-gguf-tensor-views.md), then continue with [Milestone 05B: Q8_0 reference decoding](docs/milestone-05b-q8-0-reference-decoding.md).

### Done when

`token_embd.weight`, normalization weights, and output weights can be located
and read without out-of-bounds access. A reference test proves each block
decoder returns sensible values.

## Lesson 6: build one Qwen transformer block

### Concept

For a single token position, a Qwen2 block approximately does:

```text
x = x + attention(RMSNorm(x))
x = x + MLP(RMSNorm(x))
```

Attention creates query (Q), key (K), and value (V) vectors. Q compares with
past K vectors to produce attention weights; those weights combine past V
vectors. Qwen uses rotary positional embeddings (RoPE) to encode positions and
grouped-query attention because it has 14 query heads but only 2 KV heads.

The MLP projects from `dim` to `hidden_dim`, applies SiLU gating, then projects
back to `dim`.

### Build task

Add every Qwen-specific field required by the forward pass to `ModelConfig`
(for example RMS epsilon and RoPE frequency base). Locate the 24 copies of
each layer tensor by their GGUF names. Implement, in this order:

1. token embedding lookup;
2. RMSNorm and residual addition;
3. Q/K/V projections and RoPE;
4. temporary key/value arrays and causal attention;
5. output projection;
6. gated MLP;
7. final RMSNorm and vocabulary projection.

### Done when

`engine_forward_token` accepts one token ID, increments the current position,
and writes exactly `vocab_size` logits.

## Lesson 7: recompute the full prefix

### Concept

At position `t`, attention needs keys and values from positions `0` through
`t`. For this first implementation, each forward call recomputes those vectors
from the saved token IDs. The current layer stores them in temporary arrays
while it processes the prefix. Their logical shape during a call is:

```text
keys[position][n_kv_heads * head_size]
values[position][n_kv_heads * head_size]
```

For this model, `n_kv_heads * head_size` is `2 * 64 = 128` floats per layer
and position per cache.

### Build task

Define an internal `EngineState` with token-history storage, full-sequence
activation buffers, and temporary contiguous K/V arrays. Calculate allocation
sizes using checked multiplication before calling `calloc`. Track the number
of stored tokens and reject attempts to exceed the requested context.

### Done when

Two calls to `engine_forward_token` leave two token IDs in history. The second
call recomputes both positions and returns logits for the second one. A later
KV cache can avoid recomputing the first position.

## Lesson 8: sample and print generated tokens

### Concept

Logits are scores. Greedy sampling chooses the largest score. Temperature and
top-p turn scores into a controlled random choice.

Start with greedy decoding because it is deterministic and makes debugging
possible. Add temperature and top-p only after greedy output is correct.

### Build task

Implement `sampler_sample` as greedy argmax when `temperature == 0`. Then add
temperature scaling, stable softmax, sorting/filtering for top-p, and a seeded
random-number generator. Print decoded pieces as soon as each token is chosen.

### Done when

With greedy sampling, the same prompt produces the same token IDs on repeated
runs. Stop at EOS, the requested token limit, or context exhaustion.

## Lesson 9: prove the end-to-end path and optimize CPU use

### Build task

Use this deterministic prompt first:

```sh
./build/inference --model models/qwen-2.5.gguf \
  --prompt "What is the capital of France?" \
  --max-tokens 16 --temperature 0 --top-p 1
```

Confirm the generated text identifies Paris. Then profile the scalar code,
optimize the quantized matrix-vector kernels, and only then consider threading
the largest independent output-row computations.

### Done when

The local executable answers the prompt, reports generated token count and
token/s, and runs without an external inference library.

