# Common terminologies

### Text and tokens
1. **Token**:
- An integer representing a piece of text. A token can be a word, part of a word, punctuation, whitespace, or a special control marker.
2. **Token ID**
- The integer used internally to represent one token.
3. **Vocabulary**
- The complete token table. Your Qwen model has 151,936 possible token IDs.
4. **Tokenizer**
- The component that converts text into token IDs and IDs back into text. It must match the tokenizer used during model training.
5. **BPE, or Byte Pair Encoding**
- The tokenizer algorithm used by Qwen’s GPT-2-style tokenizer. It starts with small text pieces and repeatedly joins frequent adjacent pairs into larger pieces.
6. **Special token**
- A token with a control meaning instead of ordinary text. Examples include beginning-of-sequence, end-of-sequence, or Qwen chat markers such as <|im_start|>.
7. **EOS, or end-of-sequence token**
- A token telling generation to stop.

### Numbers inside a model
1. Tensor
- A block of numbers with dimensions. A vector is a one-dimensional tensor; a matrix is two-dimensional.
2. Shape
- The size of each tensor dimension. 
3. Dimension (dim)
- The length of the main activation vector flowing through the model.
4. Activation
- Temporary numbers calculated while processing a token. Unlike weights, activations change for every prompt and generated token.
5. Embedding
- The first activation for a token. The engine looks up the token’s row in token_embd.weight.
6. Logits
- One raw score per vocabulary token. After the forward pass, Qwen produces 151,936 logits. The largest logit is its strongest next-token prediction.

### Transformer layer
Generally, the process looks something like this:
```
input
  -> normalize
  -> attention
  -> add residual connection
  -> normalize
  -> MLP
  -> add residual connection
```

1. Attention:
- The mechanism that lets the current token decides which earlier token is the most relevant
2. Q, K , V (Query, Key, Value):
- 3 vectors created from an activation: 
    - Query: what the current token is looking for
    - Key: what each earlier token offers
    - Value: information carried by each earlier token
- Attention compares the current query against earlier keys, then combines their values
3. Attention head: 
- A processing unit in a transformer attention layer - its job is to calculate the attention score from Q, K, V vectors to determine the correlation between tokens in a sequence
- Can be multiple + done in parallel
4. KV cache:
- A saved copy of key and value vectors from earlier tokens. Reusing it avoids recalculating those vectors.
- This project’s first forward pass does not use a KV cache: it saves token IDs and recalculates the complete prefix each time. A cache can be added later as an optimization.


### Generation:
1. Forward pass: Processing the known token history through the model to produce logits for the next token. This first version recalculates the whole history each time.
2. Prefill: forwarding the prompt tokens in order. Each call recalculates the prefix so far; the final prompt token produces logits for the first generated token.
3. Decode: generating one token at a time after prefill
4. Sampling: choose one token from logits
5. Greedy sampling: always choose the largest logit
