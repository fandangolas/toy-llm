# The Toy Model

The hand-built language model at the centre of this project: a character-level, GPT-style decoder-only transformer, deliberately tiny so every part stays understandable. This document is its spec sheet and results.

For *how* the pieces work, see [architecture.md](architecture.md) (system design + forward pass) and [concepts/](concepts/) (embeddings, attention). This document covers *what it is* and *what it produces*.

---

## Architecture

A character-level, GPT-style decoder-only transformer.

| Hyperparameter | Value |
|---|---|
| Embedding dimension (`n_embd`) | 128 |
| Context length (`block_size`) | 256 characters |
| Transformer layers (`n_layer`) | 4 |
| Attention heads (`n_head`) | 4 |
| Dropout | 0.1 |
| Vocabulary | 65 characters |
| Total parameters | ~832K |

Each block is `LayerNorm → causal self-attention → residual → LayerNorm → MLP → residual`. The token embedding and output projection share one weight matrix (weight tying). These values live in `GPTConfig` in `model/transformer.py`.

---

## Training

| Setting | Value |
|---|---|
| Dataset | Tiny Shakespeare (~1 MB) |
| Steps | 5000 |
| Batch size | 32 |
| Optimizer | AdamW, lr `1e-3`, gradient clipping at 1.0 |
| Eval + sample | every 500 steps |

Run it with:

```bash
python -m model.train
```

The script downloads the dataset on the first run, builds the tokenizer, trains, prints loss and a text sample every 500 steps, and saves `data/checkpoint.pt` at the end. On a modern CPU this takes 10–15 minutes; on a GPU, under 2. Generation uses temperature scaling and top-k sampling.

---

## Results

Actual output from a run:

```
Vocab size: 65 characters
Model parameters: 832,384  |  Device: cpu

step     0  |  train loss 4.2100  |  val loss 4.2113
--- sample ---
z''tC.f-cMnu?z!JsI&!lRbKI!ISjt nRUWsqLQQ.kKbwb-tLPnBQutZpCNRV&Boob...
--------------

step  5000  |  train loss 1.3272  |  val loss 1.5571
--- sample ---
ANGELO:
I would you all make myself provost,
Fasticion for what him vainity disipers
Tell him athe rivecolate the son ame,
--------------
```

Loss drops from **4.21** (random guessing over 65 characters ≈ `ln(65)`) to **~1.33**. By the end the model produces recognizable Shakespearean structure — character names, dialogue layout, archaic words — having never been told a single grammar rule.

---

## What "good" looks like here

This is a **base completion model**, not a chat model: it continues text in the style it learned, it does not answer questions. Prompt it with the start of something Shakespearean (`ROMEO:`, `To be, or not to`) and it continues in kind.

Its output looks like Shakespeare at a glance but is semantically empty — and that is expected for a model this size (832K parameters) at the character level. It learned spelling, word shapes, and play formatting, but it is far too small to learn meaning. That ceiling is the point: the model is small enough to understand completely.

To actually talk to it (or swap in a real model through the same interface), see the harness usage in the [README](../README.md).
