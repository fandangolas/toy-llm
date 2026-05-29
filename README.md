# llm-toy-project

A GPT-style language model built from scratch in PyTorch — and, eventually, the small system around it: retrieval (RAG) and a harness that connects the pieces.

The goal is not a useful model. The goal is to **understand how an LLM works under the hood** by building one in a handful of readable files, with no frameworks doing invisible work.

> This is a learning project. The model is tiny, the dataset is Shakespeare, and nothing here is production-ready. That is the point.

---

## The bigger picture

The project is three loosely-coupled components, built in order. Only the first exists today.

| Component | What it does | Status |
|---|---|---|
| **LLM** | A decoder-only transformer that generates text one character at a time | ✅ Built (`model/`) |
| **Harness** | Orchestrates the flow and hides the LLM behind a swappable interface | ⬜ Planned |
| **RAG** | Retrieves relevant text and feeds it into the prompt at generation time | ⬜ Planned |

The harness is the only piece that knows about the others, which is what lets each be built and understood in isolation. The full design — with diagrams of the system, the model internals, the provider abstraction, and the RAG flow — lives in **[docs/architecture.md](docs/architecture.md)**.

---

## Project structure

```
llm-toy-project/
├── model/
│   ├── tokenizer.py      # Character-level tokenizer (encode, decode, save, load)
│   ├── attention.py      # Causal multi-head self-attention
│   ├── transformer.py    # GPTConfig, MLP, Block, GPT (embeddings → blocks → head → generate)
│   └── train.py          # Downloads data, trains, samples, saves a checkpoint
├── docs/
│   ├── architecture.md       # System design + mermaid diagrams + decisions
│   └── concepts/
│       ├── 001-embeddings.md # How tokens become vectors
│       └── 002-qkv.md        # How Query / Key / Value attention works
├── tests/
│   └── test_model.py     # 14 tests across all four model files
├── data/                 # Created at runtime — dataset, vocab.json, checkpoint.pt
└── requirements.txt
```

---

## How to run it

Install the single dependency:

```bash
pip install -r requirements.txt
```

Train the model:

```bash
python -m model.train
```

The script downloads the Tiny Shakespeare dataset on the first run (~1 MB), trains for 5000 steps, prints loss and a text sample every 500 steps, and saves `data/checkpoint.pt` at the end.

**Actual output from a run:**

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

Loss drops from **4.21** (random guessing over 65 characters ≈ `ln(65)`) to **~1.33**. By the end the model produces recognizable Shakespearean structure — character names, dialogue layout, archaic words — having never been told a single grammar rule. On a modern CPU this takes 10–15 minutes; on a GPU, under 2.

Run the tests:

```bash
python -m pytest tests/
```

---

## The model in brief

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

Each block is `LayerNorm → causal self-attention → residual → LayerNorm → MLP → residual`. The token embedding and output projection share one weight matrix (weight tying). Training uses AdamW with gradient clipping; generation uses temperature scaling and top-k sampling. See **[docs/architecture.md](docs/architecture.md)** for the annotated forward-pass and block diagrams.

---

## Documentation

Read these in roughly this order:

1. **[docs/architecture.md](docs/architecture.md)** — the system. How the LLM, harness, and RAG fit together; the forward pass and transformer block drawn out; the provider abstraction that makes the model swappable; the end-to-end RAG flow; and the reasoning behind every "keep it simple" decision.

2. **[docs/concepts/001-embeddings.md](docs/concepts/001-embeddings.md)** — why raw token integers aren't enough, how the token and position embedding tables work, and what the combined vector looks like before it enters the transformer.

3. **[docs/concepts/002-qkv.md](docs/concepts/002-qkv.md)** — how each token produces Query, Key, and Value vectors, how dot-product scores become attention weights through softmax, and why the three are kept as separate projections.

Each concept doc follows the same shape: the problem, the intuition, the step-by-step mechanics, and how PyTorch implements it.

---

## Tests

`tests/test_model.py` holds 14 tests, one section per source file. The notable ones:

- **Causal masking** — runs a full sequence and a prefix, then asserts the prefix output matches the first *N* positions of the full run. This is the real proof that future tokens never leak backward.
- **Weight tying** — asserts `lm_head.weight is token_emb.weight` (identity, not equality) so a copy would fail the test.
- **Next-token invariant** — `get_batch` returns `y` as `x` shifted one position right (`y[:, :-1] == x[:, 1:]`), the relationship that makes next-character prediction work.

---

## Status & roadmap

1. ✅ **LLM** — tokenizer, attention, transformer, training loop, tests.
2. ⬜ **Harness skeleton** — define `LLMProvider`, wrap the trained checkpoint in a `ToyLLMProvider`.
3. ⬜ **Real provider** — add an Ollama or API-based provider and compare against the toy model through the same interface.
4. ⬜ **RAG** — chunking, an embedding function, a minimal vector store, and `retrieve()`.
5. ⬜ **Connect everything** — harness calls RAG, builds the prompt, calls the provider, returns the answer.

---

## License

Released under the [MIT License](LICENSE).
