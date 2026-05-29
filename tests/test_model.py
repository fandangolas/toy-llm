"""
Test suite for the GPT-style toy LLM project.

Organised into four sections that mirror the source modules:
  - CharTokenizer        (model/tokenizer.py)
  - CausalSelfAttention  (model/attention.py)
  - GPT / transformer    (model/transformer.py)
  - Train utilities      (model/train.py)

All tests use small configs so the full suite runs in a few seconds on CPU.
"""

import math
import pytest
import torch

from model.tokenizer import CharTokenizer
from model.attention import CausalSelfAttention
from model.transformer import GPT, GPTConfig
from model.train import get_batch, BATCH_SIZE, BLOCK_SIZE


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

TINY_CONFIG = GPTConfig(
    vocab_size=10,
    block_size=32,
    n_layer=2,
    n_head=2,
    n_embd=16,
    dropout=0.0,  # deterministic for shape/causal tests
)


@pytest.fixture()
def tiny_model() -> GPT:
    torch.manual_seed(0)
    model = GPT(TINY_CONFIG)
    model.eval()
    return model


@pytest.fixture()
def tiny_attention() -> CausalSelfAttention:
    torch.manual_seed(0)
    attn = CausalSelfAttention(
        n_embd=16,
        n_head=2,
        block_size=32,
        dropout=0.0,
    )
    attn.eval()
    return attn


# ---------------------------------------------------------------------------
# CharTokenizer
# ---------------------------------------------------------------------------


class TestCharTokenizer:
    def test_build_vocab_size_equals_unique_character_count(self):
        text = "hello world"
        unique_chars = len(set(text))  # ' ', 'd', 'e', 'h', 'l', 'o', 'r', 'w' = 8

        tokenizer = CharTokenizer(text)

        assert tokenizer.vocab_size == unique_chars

    def test_encode_decode_round_trips_original_string(self):
        text = "the quick brown fox"
        tokenizer = CharTokenizer(text)

        recovered = tokenizer.decode(tokenizer.encode(text))

        assert recovered == text

    def test_encode_raises_on_character_absent_from_vocab(self):
        tokenizer = CharTokenizer("abc")

        with pytest.raises(KeyError):
            tokenizer.encode("abcZ")  # 'Z' was never in the training text

    def test_save_and_load_preserves_encode_decode_behaviour(self, tmp_path):
        original_text = "sphinx of black quartz"
        tokenizer = CharTokenizer(original_text)
        vocab_file = tmp_path / "vocab.json"

        tokenizer.save(str(vocab_file))

        reloaded = CharTokenizer()
        reloaded.load(str(vocab_file))

        assert reloaded.vocab_size == tokenizer.vocab_size
        assert reloaded.encode(original_text) == tokenizer.encode(original_text)
        assert reloaded.decode(tokenizer.encode(original_text)) == original_text


# ---------------------------------------------------------------------------
# CausalSelfAttention
# ---------------------------------------------------------------------------


class TestCausalSelfAttention:
    def test_output_shape_matches_input_shape(self, tiny_attention):
        B, T, C = 2, 8, 16
        x = torch.randn(B, T, C)

        with torch.no_grad():
            out = tiny_attention(x)

        assert out.shape == (B, T, C)

    def test_causal_mask_prefix_output_matches_full_sequence(self, tiny_attention):
        """
        If the causal mask is correct, the attention output for a prefix of
        length T_prefix must be bit-for-bit identical to the first T_prefix
        positions of the full-sequence output.  Future tokens must not leak.
        """
        torch.manual_seed(42)
        full_len = 8
        prefix_len = 5
        x = torch.randn(1, full_len, 16)

        with torch.no_grad():
            full_out = tiny_attention(x)
            prefix_out = tiny_attention(x[:, :prefix_len, :])

        assert torch.allclose(prefix_out, full_out[:, :prefix_len, :])

    def test_forward_works_with_batch_size_greater_than_one(self, tiny_attention):
        B = 4
        x = torch.randn(B, 8, 16)

        with torch.no_grad():
            out = tiny_attention(x)

        assert out.shape == (B, 8, 16)


# ---------------------------------------------------------------------------
# GPT / transformer
# ---------------------------------------------------------------------------


class TestGPT:
    def test_forward_without_targets_returns_logits_and_none_loss(self, tiny_model):
        idx = torch.zeros(2, 8, dtype=torch.long)

        logits, loss = tiny_model(idx)

        assert loss is None
        assert logits.shape == (2, 8, TINY_CONFIG.vocab_size)

    def test_forward_with_targets_returns_finite_positive_scalar_loss(self, tiny_model):
        idx = torch.zeros(2, 8, dtype=torch.long)
        targets = torch.zeros(2, 8, dtype=torch.long)

        logits, loss = tiny_model(idx, targets)

        assert loss is not None
        assert loss.shape == ()           # scalar
        assert loss.item() > 0
        assert math.isfinite(loss.item())

    def test_generate_extends_sequence_by_exactly_max_new_tokens(self, tiny_model):
        prompt_len = 5
        max_new_tokens = 10
        idx = torch.zeros(1, prompt_len, dtype=torch.long)

        generated = tiny_model.generate(idx, max_new_tokens=max_new_tokens)

        assert generated.shape == (1, prompt_len + max_new_tokens)

    def test_lm_head_and_token_embedding_share_the_same_weight_tensor(self, tiny_model):
        assert tiny_model.lm_head.weight is tiny_model.token_emb.weight

    def test_parameter_count_is_in_expected_ballpark_for_tiny_config(self):
        """
        For n_layer=2, n_head=2, n_embd=16, block_size=32, vocab_size=10 the
        total is ~7 136 (verified analytically).  We bound it loosely enough to
        survive minor architecture tweaks but tightly enough to catch accidental
        layer duplication or config misuse.
        """
        model = GPT(TINY_CONFIG)
        n_params = sum(p.numel() for p in model.parameters())

        assert 5_000 < n_params < 10_000


# ---------------------------------------------------------------------------
# Train utilities
# ---------------------------------------------------------------------------


class TestGetBatch:
    """
    get_batch uses the module-level BATCH_SIZE and BLOCK_SIZE constants from
    model/train.py.  We build a data tensor that is long enough for the sampler
    (len > BLOCK_SIZE) but we don't need it to be large — a tight lower bound
    keeps the test self-contained.
    """

    @pytest.fixture()
    def train_data(self) -> torch.Tensor:
        # Needs at least BLOCK_SIZE + 1 tokens so randint has at least one valid index.
        n_tokens = BLOCK_SIZE * 2 + 10
        return torch.arange(n_tokens, dtype=torch.long)

    @pytest.fixture()
    def val_data(self, train_data) -> torch.Tensor:
        return train_data  # shape doesn't matter for split routing in these tests

    def test_get_batch_returns_correct_shape(self, train_data, val_data):
        x, y = get_batch("train", train_data, val_data)

        assert x.shape == (BATCH_SIZE, BLOCK_SIZE)
        assert y.shape == (BATCH_SIZE, BLOCK_SIZE)

    def test_get_batch_y_is_x_shifted_right_by_one(self, train_data, val_data):
        """
        For every row b in the batch:
          x[b] = data[i : i + BLOCK_SIZE]
          y[b] = data[i+1 : i + BLOCK_SIZE + 1]

        Therefore y[b, :-1] == x[b, 1:] for every row — the core next-token
        prediction invariant.  We verify this on the CPU-resident tensors so
        the test does not depend on DEVICE.
        """
        x, y = get_batch("train", train_data, val_data)
        x_cpu = x.cpu()
        y_cpu = y.cpu()

        assert torch.equal(y_cpu[:, :-1], x_cpu[:, 1:])
