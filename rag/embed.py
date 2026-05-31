"""
embed.py — Embedder protocol and OllamaEmbedder implementation.

The Embedder Protocol is the single contract that the Retriever depends on.
OllamaEmbedder is the only concrete implementation today; a future embedder
(OpenAI, a local SentenceTransformer, etc.) would implement the same one-method
interface without touching the Retriever.

HTTP communication reuses the exact same stdlib pattern as harness/ollama_provider.py:
urllib + json, no third-party HTTP library, clear actionable RuntimeErrors.

Prerequisites:
    1. Install Ollama: https://ollama.com
    2. Pull the embedding model: ollama pull nomic-embed-text
    3. Ollama serves automatically after install; or run `ollama serve`.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Protocol, runtime_checkable

DEFAULT_EMBED_HOST = "http://localhost:11434"
DEFAULT_EMBED_MODEL = "nomic-embed-text"


@runtime_checkable
class Embedder(Protocol):
    """Contract for turning a list of strings into a list of float vectors.

    Any object that exposes this single method satisfies the Protocol.
    The Protocol is @runtime_checkable so ``isinstance(obj, Embedder)`` works
    structurally without inheritance — identical to the LLMProvider approach
    in harness/provider.py.
    """

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text, in the same order."""
        ...


class OllamaEmbedder:
    """Implements :class:`Embedder` against a local Ollama server.

    Sends one POST to ``{host}/api/embeddings`` per text and collects the
    ``"embedding"`` field from each response. This is intentionally simple:
    one request per text makes batching visible and the control flow trivial
    to follow.

    Parameters
    ----------
    model:
        Ollama model name. Must already be pulled (``ollama pull <model>``).
    host:
        Base URL of the Ollama server.
    timeout:
        Per-request timeout in seconds. Embedding is fast but the first
        request may need to load the model into memory.
    """

    def __init__(
        self,
        model: str = DEFAULT_EMBED_MODEL,
        host: str = DEFAULT_EMBED_HOST,
        timeout: float = 120.0,
    ) -> None:
        self.model = model
        self._host = host.rstrip("/")
        self._timeout = timeout

    # ------------------------------------------------------------------
    # Embedder interface
    # ------------------------------------------------------------------

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per text in *texts*.

        Raises
        ------
        RuntimeError
            If the Ollama server is unreachable or the model has not been
            pulled. The message includes actionable hints.
        """
        return [self._embed_one(text) for text in texts]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _embed_one(self, text: str) -> list[float]:
        payload = {"model": self.model, "prompt": text}
        data = self._post("/api/embeddings", payload)
        return data["embedding"]

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
        try:
            payload = json.loads(exc.read().decode("utf-8"))
            return payload.get("error", exc.reason)
        except Exception:
            return getattr(exc, "reason", str(exc))
