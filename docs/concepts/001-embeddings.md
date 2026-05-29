# Concepts

---

## Embeddings

### The problem

After tokenization, each character is represented as an integer — `'A'=0`, `'B'=1`, `' '=32`, and so on. But integers carry false meaning. The number 32 is not "closer" to 31 than to 0 in any linguistically meaningful way. You can't do useful math on them.

The model needs a representation where similar characters end up geometrically close to each other, and where arithmetic on the representation reflects something real about language.

---

### The intuition

Imagine placing every character somewhere in a large open space — 128-dimensional, not 3-dimensional, but the idea is the same. Characters that behave similarly in language end up near each other. Uppercase letters cluster together. Vowels cluster together. Punctuation that ends sentences clusters together.

This space is not designed by hand. The model starts with characters placed randomly, and training gradually pushes them into positions where the arrangement helps predict the next character. The positions — 128 numbers per character — are the embeddings.

---

### How it works, step by step

**Step 1 — Token embeddings**

The tokenizer produced a sequence of integers, for example `[58, 46, 43, 1, 41, 39]` for `"the ca"` — where `'t'=58`, `'h'=46`, `'e'=43`, `' '=1`, `'c'=41`, `'a'=39`. The model's first job is to convert each integer into a vector of 128 numbers:

```python
# transformer.py line 56
self.token_emb = nn.Embedding(config.vocab_size, config.n_embd)
```

`nn.Embedding(65, 128)` generates a lookup table with random values — a matrix of shape `(65, 128)`. Each row is one character's embedding. Passing the integer `46` returns row 46: a vector of 128 numbers.

```
nn.Embedding(65, 128)  — shape: (65 chars × 128 dims), initialised with small random values

          dim_0   dim_1   dim_2   dim_3   dim_4   dim_5  ...  dim_127
          ──────  ──────  ──────  ──────  ──────  ──────       ───────
'\n' [ 0]  0.012  -0.019   0.003   0.021  -0.008   0.014  ...   0.017
' '  [ 1] -0.005   0.014  -0.022   0.007   0.018  -0.011  ...  -0.003
'!'  [ 2]  0.024  -0.003   0.011  -0.017   0.002   0.019  ...   0.009
'$'  [ 3] -0.018   0.021   0.009   0.004  -0.014   0.006  ...   0.011
 ...
'A'  [13]  0.007  -0.011   0.018  -0.006   0.023  -0.015  ...  -0.016
'B'  [14] -0.013   0.008  -0.004   0.019  -0.007   0.022  ...   0.014
 ...
'a'  [39]  0.019  -0.007   0.013  -0.021   0.005   0.017  ...  -0.008
'b'  [40] -0.002   0.016  -0.018   0.003   0.012  -0.009  ...   0.021
 ...
'z'  [64]  0.011  -0.023   0.006   0.018  -0.013   0.003  ...  -0.019

          ↑ all 128 values per row are random at init — meaningless until training adjusts them
```

When you pass a single integer — say `46` (`'h'`) — the embedding returns row 46: one vector of 128 numbers.

When you pass a full sequence — `[58, 46, 43, 1, 41, 39]` — the embedding performs that lookup for every integer simultaneously, returning one row per integer and stacking them:

```
input:  [58,  46,  43,   1,  41,  39]
         't'  'h'  'e'  ' '  'c'  'a'

each integer indexes one row from the (65, 128) table:

output: [ row_58 ]   ← 't' → 128 numbers
        [ row_46 ]   ← 'h' → 128 numbers
        [ row_43 ]   ← 'e' → 128 numbers
        [ row_1  ]   ← ' ' → 128 numbers
        [ row_41 ]   ← 'c' → 128 numbers
        [ row_39 ]   ← 'a' → 128 numbers

shape: (6, 128)  — six vectors of 128 numbers each
```

The (65, 128) table is never consumed or reduced — it stays intact. You are just reading rows out of it, one per token in the sequence.

**Step 2 — Position embeddings**

There is a problem: attention treats the sequence as a set, not a sequence. If you scramble the order of the tokens before passing them in, the attention computation produces the exact same result. The model has no sense of order.

To fix this, we give every position its own learned vector and add it to the token embedding:

```python
# transformer.py line 57
self.pos_emb = nn.Embedding(config.block_size, config.n_embd)

# transformer.py line 81-82
positions = torch.arange(T, device=idx.device)
x = self.drop(self.token_emb(idx) + self.pos_emb(positions))
```

`nn.Embedding(256, 128)` is another lookup table — this time 256 rows, one per position. Position 0 gets a vector, position 1 gets a different vector, and so on.

These are added to the token embeddings element-wise. The result is a single vector per token that encodes both *what the character is* and *where it sits in the sequence*.

**Step 3 — The combined vector enters the model**

After adding the token embedding and the position embedding together, `x` has shape `(B, T, 128)` — batch size × sequence length × embedding dimension. This is what flows into the transformer blocks. Everything from here operates on these 128-number vectors, never on raw integers again.

---

### Why addition and not concatenation?

You could concatenate token and position embeddings into a 256-dim vector instead of adding them into a 128-dim one. Addition is preferred because it keeps the dimension constant — every layer downstream expects 128 numbers — and in practice it works just as well. The model learns to encode both types of information within the same 128 dimensions.

There is some interference, and this is one reason transformers aren't perfect. Concatenation would lose zero information in theory, but it doubles the dimension, which doubles the cost of every subsequent operation. Addition is a practical tradeoff — good enough in practice, and the training process compensates for the imperfection.

---

### PyTorch's role here

| What we need | How PyTorch provides it |
|---|---|
| The token lookup table | `nn.Embedding(vocab_size, n_embd)` — allocates a `(65, 128)` matrix as a learnable parameter |
| The position lookup table | `nn.Embedding(block_size, n_embd)` — same structure, one row per position |
| Looking up a row | Passing an integer tensor to the embedding returns the corresponding rows |
| Learning the embeddings | Autograd — the embedding vectors are parameters, so `loss.backward()` + `optimizer.step()` adjusts them like any other weight |

The embedding tables start as small random numbers and are updated at every training step alongside all other parameters. By step 5000, the 128-number vector for `'e'` has been nudged thousands of times into a position that helps the model predict what comes after `'e'` in Shakespeare.
