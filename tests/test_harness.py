"""
Test suite for the harness layer.

Covers:
  - LLMProvider protocol recognition (isinstance via @runtime_checkable)
  - ToyLLMProvider.generate return type and length contract
  - Empty-prompt seeding behaviour
  - Unknown-character validation (ValueError, not KeyError)
  - from_checkpoint round-trip with a tiny in-memory checkpoint

All tests use tiny model configs so the suite runs in a few seconds on CPU.
"""

import pytest
import torch

from model.tokenizer import CharTokenizer
from model.transformer import GPT, GPTConfig

from harness import LLMProvider, ToyLLMProvider


# ---------------------------------------------------------------------------
# Shared tiny model and tokenizer
# ---------------------------------------------------------------------------

CORPUS = "hello world shakespeare"
# Unique sorted chars: ' ', 'a', 'd', 'e', 'h', 'k', 'l', 'o', 'p', 'r',
#                      's', 'w' → 13 distinct characters

TINY_HARNESS_CONFIG = GPTConfig(
    vocab_size=len(set(CORPUS)),   # determined after tokenizer build below
    block_size=32,
    n_layer=2,
    n_head=2,
    n_embd=16,
    dropout=0.0,
)


@pytest.fixture()
def tokenizer() -> CharTokenizer:
    return CharTokenizer(CORPUS)


@pytest.fixture()
def tiny_model(tokenizer: CharTokenizer) -> GPT:
    torch.manual_seed(0)
    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=32,
        n_layer=2,
        n_head=2,
        n_embd=16,
        dropout=0.0,
    )
    model = GPT(config)
    model.eval()
    return model


@pytest.fixture()
def provider(tiny_model: GPT, tokenizer: CharTokenizer) -> ToyLLMProvider:
    return ToyLLMProvider(tiny_model, tokenizer)


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class TestLLMProviderProtocol:
    def test_toy_provider_is_recognized_as_llm_provider_via_isinstance(
        self, provider: ToyLLMProvider
    ):
        """
        LLMProvider is @runtime_checkable so isinstance works structurally —
        any object with a matching generate() signature passes.
        """
        assert isinstance(provider, LLMProvider)

    def test_arbitrary_object_with_correct_signature_is_recognized(self):
        """
        Demonstrates the Protocol is structural, not nominal.  A plain class
        that is never told about LLMProvider satisfies the check if it has
        the right generate() signature.
        """

        class StubProvider:
            def generate(self, prompt: str, max_tokens: int) -> str:
                return "stub"

        assert isinstance(StubProvider(), LLMProvider)

    def test_object_without_generate_is_not_recognized(self):
        class NotAProvider:
            pass

        assert not isinstance(NotAProvider(), LLMProvider)


# ---------------------------------------------------------------------------
# ToyLLMProvider.generate return type and length contract
# ---------------------------------------------------------------------------


class TestToyLLMProviderGenerate:
    def test_generate_returns_a_string(self, provider: ToyLLMProvider):
        result = provider.generate("hello", max_tokens=5)

        assert isinstance(result, str)

    def test_continuation_length_equals_max_tokens(self, provider: ToyLLMProvider):
        """
        Character-level model: 1 token == 1 character, so the continuation
        string must be exactly max_tokens characters long.
        """
        max_tokens = 10

        result = provider.generate("hello", max_tokens=max_tokens)

        assert len(result) == max_tokens

    def test_continuation_length_equals_max_tokens_for_various_lengths(
        self, provider: ToyLLMProvider
    ):
        for max_tokens in (1, 5, 20, 50):
            result = provider.generate("hello", max_tokens=max_tokens)
            assert len(result) == max_tokens, (
                f"Expected {max_tokens} chars, got {len(result)} "
                f"for max_tokens={max_tokens}"
            )

    def test_generate_only_returns_continuation_not_the_prompt(
        self, provider: ToyLLMProvider
    ):
        """
        The provider must strip the prompt from the output.  The caller
        already knows the prompt; returning it again would be redundant.
        """
        prompt = "hel"
        max_tokens = 7

        result = provider.generate(prompt, max_tokens=max_tokens)

        assert len(result) == max_tokens


# ---------------------------------------------------------------------------
# Empty prompt seeding
# ---------------------------------------------------------------------------


class TestEmptyPromptSeeding:
    def test_empty_prompt_returns_string(self, provider: ToyLLMProvider):
        result = provider.generate("", max_tokens=5)

        assert isinstance(result, str)

    def test_empty_prompt_returns_correct_length(self, provider: ToyLLMProvider):
        max_tokens = 15

        result = provider.generate("", max_tokens=max_tokens)

        assert len(result) == max_tokens


# ---------------------------------------------------------------------------
# Unknown-character validation
# ---------------------------------------------------------------------------


class TestUnknownCharacterValidation:
    def test_unknown_character_raises_value_error_not_key_error(
        self, provider: ToyLLMProvider
    ):
        """
        The raw KeyError from CharTokenizer.encode must be caught and converted
        to a ValueError with a user-friendly message.  KeyError must never
        escape the provider boundary.
        """
        with pytest.raises(ValueError):
            provider.generate("hello\x00", max_tokens=5)  # NUL is not in CORPUS

    def test_error_message_names_the_offending_character(
        self, provider: ToyLLMProvider
    ):
        offending_char = "Z"  # definitely not in CORPUS

        with pytest.raises(ValueError, match="Z"):
            provider.generate(f"hello{offending_char}", max_tokens=5)

    def test_error_message_mentions_training_corpus(
        self, provider: ToyLLMProvider
    ):
        with pytest.raises(ValueError, match="(?i)vocab|train|corpus"):
            provider.generate("hello\x00", max_tokens=5)

    def test_key_error_does_not_propagate(self, provider: ToyLLMProvider):
        """
        Explicit guard: the raw KeyError from encode() must not leak.
        """
        with pytest.raises(ValueError):
            try:
                provider.generate("helloZ", max_tokens=5)
            except KeyError as exc:
                raise AssertionError(
                    f"KeyError escaped ToyLLMProvider.generate: {exc!r}"
                ) from exc


# ---------------------------------------------------------------------------
# from_checkpoint round-trip
# ---------------------------------------------------------------------------


class TestFromCheckpointRoundTrip:
    def test_from_checkpoint_loads_and_generates(
        self, tokenizer: CharTokenizer, tiny_model: GPT, tmp_path
    ):
        """
        Save a tiny checkpoint dict and vocab file in the same format used by
        model/train.py, reload via from_checkpoint, and confirm generate works.
        """
        ckpt_path = tmp_path / "checkpoint.pt"
        vocab_path = tmp_path / "vocab.json"

        config = tiny_model.config
        torch.save(
            {"model": tiny_model.state_dict(), "config": config.__dict__},
            str(ckpt_path),
        )
        tokenizer.save(str(vocab_path))

        loaded_provider = ToyLLMProvider.from_checkpoint(
            checkpoint_path=str(ckpt_path),
            vocab_path=str(vocab_path),
            device="cpu",
        )

        result = loaded_provider.generate("hello", max_tokens=10)

        assert isinstance(result, str)
        assert len(result) == 10

    def test_from_checkpoint_produces_deterministic_output_with_seed(
        self, tokenizer: CharTokenizer, tiny_model: GPT, tmp_path
    ):
        """
        With the same random seed and a fixed model, generate must produce
        the same output twice — confirming no non-deterministic state leaks.
        """
        ckpt_path = tmp_path / "checkpoint.pt"
        vocab_path = tmp_path / "vocab.json"

        torch.save(
            {"model": tiny_model.state_dict(), "config": tiny_model.config.__dict__},
            str(ckpt_path),
        )
        tokenizer.save(str(vocab_path))

        provider = ToyLLMProvider.from_checkpoint(
            checkpoint_path=str(ckpt_path),
            vocab_path=str(vocab_path),
            device="cpu",
        )

        torch.manual_seed(42)
        result_a = provider.generate("hello", max_tokens=20)
        torch.manual_seed(42)
        result_b = provider.generate("hello", max_tokens=20)

        assert result_a == result_b

    def test_from_checkpoint_config_matches_saved_config(
        self, tokenizer: CharTokenizer, tiny_model: GPT, tmp_path
    ):
        ckpt_path = tmp_path / "checkpoint.pt"
        vocab_path = tmp_path / "vocab.json"

        config = tiny_model.config
        torch.save(
            {"model": tiny_model.state_dict(), "config": config.__dict__},
            str(ckpt_path),
        )
        tokenizer.save(str(vocab_path))

        loaded_provider = ToyLLMProvider.from_checkpoint(
            checkpoint_path=str(ckpt_path),
            vocab_path=str(vocab_path),
            device="cpu",
        )

        assert loaded_provider.model.config.vocab_size == config.vocab_size
        assert loaded_provider.model.config.block_size == config.block_size
        assert loaded_provider.model.config.n_embd == config.n_embd
