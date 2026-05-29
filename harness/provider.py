"""
LLMProvider — the swappable generation interface.

Any object that exposes ``generate(prompt: str, max_tokens: int) -> str``
satisfies this Protocol.  Because the Protocol is @runtime_checkable,
``isinstance(obj, LLMProvider)`` works structurally without inheritance.

Adding a new provider
---------------------
Create a class in its own module and implement exactly one method::

    class OllamaProvider:
        def generate(self, prompt: str, max_tokens: int) -> str:
            # call the local Ollama HTTP API and return the generated text
            ...

No base class, no import of this module, and no registration step is needed.
The harness accepts anything that has a matching ``generate`` method.
"""

from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMProvider(Protocol):
    def generate(self, prompt: str, max_tokens: int) -> str:
        ...
