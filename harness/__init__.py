"""
harness — orchestration and LLM provider abstraction layer.

Public surface:
  LLMProvider   — a typing.Protocol; implement generate() to plug in any LLM.
  ToyLLMProvider — the adapter that wraps our hand-built GPT + CharTokenizer.
"""

from harness.provider import LLMProvider
from harness.toy_provider import ToyLLMProvider

__all__ = ["LLMProvider", "ToyLLMProvider"]
