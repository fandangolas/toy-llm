"""
Tests for OllamaProvider.

The HTTP layer is mocked with pytest's monkeypatch so these tests never touch a
real Ollama server. They verify the request we send, how we parse the response,
and that failures become clear, actionable RuntimeErrors.
"""

import email.message
import io
import json
import urllib.error
import urllib.request

import pytest

from harness.ollama_provider import OllamaProvider
from harness.provider import LLMProvider


class _FakeResponse:
    """Minimal stand-in for the object returned by urllib.request.urlopen."""

    def __init__(self, payload: dict):
        self._body = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _http_error(code: int, payload: dict) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        url="http://localhost:11434/api/generate",
        code=code,
        msg="error",
        hdrs=email.message.Message(),
        fp=io.BytesIO(json.dumps(payload).encode("utf-8")),
    )


def test_ollama_provider_satisfies_llmprovider_protocol():
    assert isinstance(OllamaProvider(model="qwen2.5-coder:14b"), LLMProvider)


def test_generate_returns_response_field(monkeypatch):
    def fake_urlopen(request, timeout=None):
        return _FakeResponse({"response": "Hello there!", "done": True})

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    provider = OllamaProvider(model="qwen2.5-coder:14b")
    result = provider.generate("hi", max_tokens=50)

    assert result == "Hello there!"


def test_generate_sends_expected_payload(monkeypatch):
    captured: dict = {}

    def fake_urlopen(request, timeout=None):
        captured["url"] = request.full_url
        captured["method"] = request.get_method()
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return _FakeResponse({"response": "ok"})

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    provider = OllamaProvider(model="qwen2.5-coder:14b", temperature=0.5, top_k=20)
    provider.generate("write a function", max_tokens=128)

    assert captured["url"] == "http://localhost:11434/api/generate"
    assert captured["method"] == "POST"
    body = captured["body"]
    assert body["model"] == "qwen2.5-coder:14b"
    assert body["prompt"] == "write a function"
    assert body["stream"] is False
    assert body["options"]["num_predict"] == 128
    assert body["options"]["temperature"] == 0.5
    assert body["options"]["top_k"] == 20


def test_top_k_is_omitted_when_none(monkeypatch):
    captured: dict = {}

    def fake_urlopen(request, timeout=None):
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return _FakeResponse({"response": "ok"})

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    provider = OllamaProvider(model="m", top_k=None)
    provider.generate("hi", max_tokens=10)

    assert "top_k" not in captured["body"]["options"]


def test_generate_raises_clear_error_when_unreachable(monkeypatch):
    def fake_urlopen(request, timeout=None):
        raise urllib.error.URLError("Connection refused")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    provider = OllamaProvider(model="m")
    with pytest.raises(RuntimeError) as exc_info:
        provider.generate("hi", max_tokens=10)

    message = str(exc_info.value)
    assert "Ollama" in message
    assert "serve" in message  # actionable hint to start the server


def test_generate_surfaces_model_not_found(monkeypatch):
    def fake_urlopen(request, timeout=None):
        raise _http_error(404, {"error": "model 'm' not found, try pulling it first"})

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    provider = OllamaProvider(model="m")
    with pytest.raises(RuntimeError) as exc_info:
        provider.generate("hi", max_tokens=10)

    message = str(exc_info.value).lower()
    assert "not found" in message
    assert "pull" in message  # actionable hint to pull the model
