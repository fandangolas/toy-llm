"""
harness — orchestration and LLM provider abstraction layer.

Public surface:
  LLMProvider    — a typing.Protocol; implement generate() to plug in any LLM.
  ToyLLMProvider — the adapter that wraps our hand-built GPT + CharTokenizer.
  OllamaProvider — talks to a local Ollama server over HTTP (stdlib only).
  RAGPipeline    — retrieves context from rag/ and generates via a provider.
"""

from harness.provider import LLMProvider
from harness.toy_provider import ToyLLMProvider
from harness.ollama_provider import OllamaProvider
from harness.pipeline import RAGPipeline

__all__ = ["LLMProvider", "ToyLLMProvider", "OllamaProvider", "RAGPipeline"]
