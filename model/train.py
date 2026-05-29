import os
import urllib.request

import torch
from torch import Tensor

from model.tokenizer import CharTokenizer
from model.transformer import GPT, GPTConfig

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
BATCH_SIZE = 32
BLOCK_SIZE = 256
MAX_ITERS = 5000
EVAL_ITERS = 200
EVAL_EVERY = 500
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
DATA_PATH = os.path.join(DATA_DIR, "input.txt")
VOCAB_PATH = os.path.join(DATA_DIR, "vocab.json")
CKPT_PATH = os.path.join(DATA_DIR, "checkpoint.pt")
DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def download_data(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"Downloading Shakespeare dataset → {path}")
    urllib.request.urlretrieve(DATA_URL, path)
    print("Download complete.")


def get_batch(split: str, train_data: Tensor, val_data: Tensor) -> tuple[Tensor, Tensor]:
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([data[i : i + BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i + 1 : i + BLOCK_SIZE + 1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)


@torch.no_grad()
def estimate_loss(model: GPT, train_data: Tensor, val_data: Tensor) -> dict[str, float]:
    model.eval()
    losses: dict[str, float] = {}
    for split in ("train", "val"):
        total = 0.0
        for _ in range(EVAL_ITERS):
            x, y = get_batch(split, train_data, val_data)
            _, loss = model(x, y)
            total += loss.item()
        losses[split] = total / EVAL_ITERS
    model.train()
    return losses


def train() -> None:
    if not os.path.exists(DATA_PATH):
        download_data(DATA_PATH)

    with open(DATA_PATH) as f:
        text = f.read()

    tokenizer = CharTokenizer(text)
    tokenizer.save(VOCAB_PATH)
    print(f"Vocab size: {tokenizer.vocab_size} characters")

    tokens = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    split = int(0.9 * len(tokens))
    train_data, val_data = tokens[:split], tokens[split:]

    config = GPTConfig(vocab_size=tokenizer.vocab_size, block_size=BLOCK_SIZE)
    model = GPT(config).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}  |  Device: {DEVICE}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    for step in range(MAX_ITERS + 1):
        if step % EVAL_EVERY == 0:
            losses = estimate_loss(model, train_data, val_data)
            print(f"step {step:5d}  |  train loss {losses['train']:.4f}  |  val loss {losses['val']:.4f}")

            # Print a short sample to watch the model improve
            seed = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
            sample_tokens = model.generate(seed, max_new_tokens=200, temperature=0.8, top_k=40)
            print("--- sample ---")
            print(tokenizer.decode(sample_tokens[0].tolist()))
            print("--------------")

        if step == MAX_ITERS:
            break

        x, y = get_batch("train", train_data, val_data)
        _, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    torch.save({"model": model.state_dict(), "config": config.__dict__}, CKPT_PATH)
    print(f"\nSaved checkpoint → {CKPT_PATH}")


if __name__ == "__main__":
    train()
