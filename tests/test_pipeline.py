"""
Tests for harness/pipeline.py — the RAGPipeline orchestrator.

Uses a fake retriever (fixed passages) and a fake provider (records the
prompt, returns a canned answer). No network, no model, no database.

The pipeline is the only place that imports both rag and a provider —
these tests confirm the integration contract without testing either side's
implementation details.
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeRetriever:
    """Returns a fixed list of passages for any query."""

    def __init__(self, passages: list[str]):
        self._passages = passages

    def retrieve(self, query: str, k: int = 4) -> list[str]:
        return self._passages[:k]


class _FakeProvider:
    """Records the prompt it receives; returns a canned answer."""

    def __init__(self, answer: str = "The answer is 42."):
        self._answer = answer
        self.last_prompt: str | None = None

    def generate(self, prompt: str, max_tokens: int) -> str:
        self.last_prompt = prompt
        return self._answer


# ---------------------------------------------------------------------------
# RAGPipeline tests
# ---------------------------------------------------------------------------


class TestRAGPipeline:
    def _make_pipeline(self, passages: list[str], answer: str = "canned answer"):
        from harness.pipeline import RAGPipeline

        retriever = _FakeRetriever(passages)
        provider = _FakeProvider(answer)
        return RAGPipeline(retriever=retriever, provider=provider), provider

    def test_ask_returns_provider_output(self):
        pipeline, _ = self._make_pipeline(
            passages=["Some context."],
            answer="The generated answer.",
        )

        result = pipeline.ask("What is the answer?")

        assert result == "The generated answer."

    def test_prompt_sent_to_provider_contains_retrieved_context(self):
        pipeline, provider = self._make_pipeline(
            passages=["First passage.", "Second passage."]
        )

        pipeline.ask("What happened?")

        assert provider.last_prompt is not None
        assert "First passage." in provider.last_prompt
        assert "Second passage." in provider.last_prompt

    def test_prompt_sent_to_provider_contains_the_question(self):
        pipeline, provider = self._make_pipeline(passages=["Some context."])

        question = "What does the document say?"
        pipeline.ask(question)

        assert question in provider.last_prompt

    def test_prompt_contains_both_context_and_question(self):
        """
        The augmented prompt must combine retrieval results with the question
        so the LLM has both pieces of information in a single call.
        """
        passages = ["The sky is blue.", "Water is wet."]
        pipeline, provider = self._make_pipeline(passages=passages)

        question = "Tell me about nature."
        pipeline.ask(question)

        prompt = provider.last_prompt
        assert "The sky is blue." in prompt
        assert "Water is wet." in prompt
        assert question in prompt

    def test_k_controls_how_many_passages_are_retrieved(self):
        """
        ask(k=1) should retrieve only 1 passage even if more are available.
        The provider's prompt should contain exactly the passage(s) for that k.
        """
        passages = ["Passage A.", "Passage B.", "Passage C."]
        pipeline, provider = self._make_pipeline(passages=passages)

        pipeline.ask("Question?", k=1)

        prompt = provider.last_prompt
        assert "Passage A." in prompt
        assert "Passage B." not in prompt
        assert "Passage C." not in prompt

    def test_max_tokens_is_forwarded_to_provider(self):
        """
        ask() should forward max_tokens to provider.generate so callers can
        control the response length.
        """

        class _RecordingProvider:
            def __init__(self):
                self.received_max_tokens: int | None = None

            def generate(self, prompt: str, max_tokens: int) -> str:
                self.received_max_tokens = max_tokens
                return "ok"

        from harness.pipeline import RAGPipeline

        provider = _RecordingProvider()
        pipeline = RAGPipeline(
            retriever=_FakeRetriever(["context"]),
            provider=provider,
        )

        pipeline.ask("question?", max_tokens=42)

        assert provider.received_max_tokens == 42

    def test_empty_retrieval_still_produces_a_prompt(self):
        """
        When the store is empty (retrieve returns []), the pipeline must not
        crash — it should call provider.generate with a prompt that still
        contains the question.
        """
        pipeline, provider = self._make_pipeline(passages=[])

        result = pipeline.ask("Can this work with no context?")

        assert result == "canned answer"
        assert "Can this work with no context?" in provider.last_prompt

    def test_prompt_template_is_readable_and_structured(self):
        """
        The augmented prompt should follow the expected template structure,
        making it easy for humans to read and for LLMs to follow.
        """
        pipeline, provider = self._make_pipeline(
            passages=["The cat sat on the mat."]
        )

        pipeline.ask("Where did the cat sit?")

        prompt = provider.last_prompt
        # The template should mark context and question clearly
        assert "Context" in prompt or "context" in prompt
        assert "Question" in prompt or "question" in prompt or "?" in prompt
