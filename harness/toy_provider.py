"""
ToyLLMProvider — adapter that wraps our hand-built GPT and CharTokenizer.

This is the only place in the codebase that bridges strings <-> token IDs
for the toy model.  The model/ package stays clean: it knows nothing about
prompts, providers, or the harness.
"""

from __future__ import annotations

import torch

from model.tokenizer import CharTokenizer
from model.transformer import GPT, GPTConfig

from harness.provider import LLMProvider


class ToyLLMProvider:
    """Implements :class:`LLMProvider` by wrapping our GPT + CharTokenizer.

    Parameters
    ----------
    model:
        A GPT instance (already constructed; will be moved to ``device`` and
        put into eval mode).
    tokenizer:
        A CharTokenizer whose vocab matches the model's vocab_size.
    device:
        ``"cuda"`` or ``"cpu"``.  ``None`` resolves to CUDA when available,
        otherwise CPU.
    temperature:
        Softmax temperature passed to ``GPT.generate``.  Lower values make
        the output more deterministic; higher values increase randomness.
    top_k:
        Top-k truncation passed to ``GPT.generate``.  ``None`` disables it.
    """

    def __init__(
        self,
        model: GPT,
        tokenizer: CharTokenizer,
        device: str | None = None,
        temperature: float = 0.8,
        top_k: int | None = 40,
    ) -> None:
        self._device = self._resolve_device(device)
        self.model = model.to(self._device)
        self.model.eval()
        self._tokenizer = tokenizer
        self._temperature = temperature
        self._top_k = top_k

    # ------------------------------------------------------------------
    # LLMProvider interface
    # ------------------------------------------------------------------

    def generate(self, prompt: str, max_tokens: int) -> str:
        """Return *only* the generated continuation (not the prompt itself).

        Parameters
        ----------
        prompt:
            Input text.  Every character must be present in the tokenizer's
            vocabulary.  An empty string is valid — generation is seeded from
            token id 0, matching the convention used in ``model/train.py``.
        max_tokens:
            Number of new characters to generate.  Because this is a
            character-level model, 1 token == 1 character, so
            ``len(result) == max_tokens`` is guaranteed.

        Raises
        ------
        ValueError
            If ``prompt`` contains characters absent from the training vocab.
        """
        if prompt == "":
            # Seed with a single token (id 0) as train.py does.
            # The seed occupies one position in the sequence, so the slice
            # starts at index 1 to return exactly max_tokens new characters.
            idx = torch.zeros((1, 1), dtype=torch.long, device=self._device)
            prompt_length = 1
        else:
            self._validate_prompt_characters(prompt)
            token_ids = self._tokenizer.encode(prompt)
            idx = torch.tensor(
                [token_ids], dtype=torch.long, device=self._device
            )
            prompt_length = len(token_ids)

        with torch.no_grad():
            full_sequence = self.model.generate(
                idx,
                max_new_tokens=max_tokens,
                temperature=self._temperature,
                top_k=self._top_k,
            )

        new_token_ids = full_sequence[0, prompt_length:].tolist()
        return self._tokenizer.decode(new_token_ids)

    # ------------------------------------------------------------------
    # Class method factory
    # ------------------------------------------------------------------

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str = "data/checkpoint.pt",
        vocab_path: str = "data/vocab.json",
        device: str | None = None,
        temperature: float = 0.8,
        top_k: int | None = 40,
    ) -> "ToyLLMProvider":
        """Build a :class:`ToyLLMProvider` from a saved checkpoint and vocab.

        The checkpoint must be a dict with keys:
          - ``"model"``: a state_dict
          - ``"config"``: a dict of :class:`~model.transformer.GPTConfig` fields

        The vocab file must be the JSON format produced by
        :meth:`~model.tokenizer.CharTokenizer.save`.
        """
        resolved_device = cls._resolve_device(device)
        map_location = torch.device(resolved_device)

        checkpoint = torch.load(
            checkpoint_path,
            map_location=map_location,
            weights_only=True,
        )

        tokenizer = CharTokenizer()
        tokenizer.load(vocab_path)

        config = GPTConfig(**checkpoint["config"])
        model = GPT(config)
        model.load_state_dict(checkpoint["model"])

        return cls(
            model=model,
            tokenizer=tokenizer,
            device=resolved_device,
            temperature=temperature,
            top_k=top_k,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_device(device: str | None) -> str:
        if device is not None:
            return device
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _validate_prompt_characters(self, prompt: str) -> None:
        """Raise ValueError naming any characters absent from the vocab.

        We inspect ``_stoi`` directly because the public ``encode()`` raises
        on the *first* unknown character only.  We want to report *all* unknown
        characters in one shot so the user can fix them in a single iteration.
        """
        unknown = [ch for ch in prompt if ch not in self._tokenizer._stoi]
        if unknown:
            unique_unknown = sorted(set(unknown))
            char_list = ", ".join(repr(ch) for ch in unique_unknown)
            raise ValueError(
                f"Prompt contains character(s) not in the training vocab: "
                f"{char_list}. "
                f"This toy model only knows the characters from its training "
                f"corpus. Remove or replace the unknown character(s) and try again."
            )

