"""
OllamaProvider — an LLMProvider backed by a locally running Ollama server.

Talks to Ollama's HTTP API (default http://localhost:11434) using only the
Python standard library, so it adds no new dependencies. It implements the same
``generate(prompt, max_tokens) -> str`` interface as every other provider, so it
is a drop-in replacement for ToyLLMProvider — the call site never changes.

Prerequisites (outside Python):
    1. Install Ollama:   https://ollama.com
    2. Pull a model:     ollama pull qwen2.5-coder:14b
    3. Ollama serves automatically after install; or run ``ollama serve``.

Note: unlike the character-level toy model, Ollama models are sub-word
tokenized, so ``max_tokens`` counts model tokens, not characters — the returned
string will NOT have length == max_tokens.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

DEFAULT_HOST = "http://localhost:11434"


class OllamaProvider:
    """Implements the LLMProvider interface against a local Ollama server.

    Parameters
    ----------
    model:
        The Ollama model name, e.g. ``"qwen2.5-coder:14b"``. Must already be
        pulled (``ollama pull <model>``).
    host:
        Base URL of the Ollama server. Defaults to ``http://localhost:11434``.
    temperature:
        Sampling temperature passed through to Ollama.
    top_k:
        Top-k cutoff passed through to Ollama. ``None`` omits it (Ollama default).
    timeout:
        Per-request timeout in seconds. Large models on CPU can be slow, so the
        default is generous.
    """

    def __init__(
        self,
        model: str,
        host: str = DEFAULT_HOST,
        temperature: float = 0.8,
        top_k: int | None = 40,
        timeout: float = 120.0,
    ) -> None:
        self.model = model
        self._host = host.rstrip("/")
        self._temperature = temperature
        self._top_k = top_k
        self._timeout = timeout

    # ------------------------------------------------------------------
    # LLMProvider interface
    # ------------------------------------------------------------------

    def generate(self, prompt: str, max_tokens: int) -> str:
        """Generate a completion for ``prompt`` and return it as a string.

        Raises
        ------
        RuntimeError
            If the Ollama server is unreachable, or returns an HTTP error
            (e.g. the model has not been pulled). The message is actionable.
        """
        options: dict = {
            "num_predict": max_tokens,
            "temperature": self._temperature,
        }
        if self._top_k is not None:
            options["top_k"] = self._top_k

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": options,
        }

        data = self._post("/api/generate", payload)
        return data.get("response", "")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _post(self, path: str, payload: dict) -> dict:
        url = f"{self._host}{path}"
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = self._extract_error(exc)
            raise RuntimeError(
                f"Ollama returned HTTP {exc.code} for {path}: {detail}. "
                f"Is the model '{self.model}' pulled? Try:  ollama pull {self.model}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Could not reach Ollama at {self._host}: {exc.reason}. "
                f"Is Ollama running? Start it with `ollama serve`."
            ) from exc

    @staticmethod
    def _extract_error(exc: urllib.error.HTTPError) -> str:
        """Pull the human-readable error message out of an Ollama HTTP error."""
        try:
            payload = json.loads(exc.read().decode("utf-8"))
            return payload.get("error", exc.reason)
        except Exception:
            return getattr(exc, "reason", str(exc))
