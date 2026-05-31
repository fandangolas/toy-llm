"""
pipeline.py — RAGPipeline: the only place that imports both rag and a provider.

This module is the integration point between retrieval and generation.
Everything else in the codebase keeps those two concerns isolated:
  - rag/ knows nothing about providers or generation.
  - Providers know nothing about retrieval or chunking.
  - RAGPipeline knows about both, and only about their interfaces.

The prompt template lives here, in one obvious place, and nowhere else.
"""

from __future__ import annotations

_PROMPT_TEMPLATE = (
    "Use the following context to answer the question.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n"
    "Answer:"
)


class RAGPipeline:
    """Retrieve relevant context then generate an answer.

    Parameters
    ----------
    retriever:
        Any object with a ``retrieve(query: str, k: int) -> list[str]`` method.
        In production this is a :class:`rag.retriever.Retriever`; in tests it
        is a lightweight fake.
    provider:
        Any object satisfying the :class:`~harness.provider.LLMProvider`
        Protocol — i.e. any object with ``generate(prompt, max_tokens) -> str``.
    """

    def __init__(self, retriever, provider) -> None:
        self._retriever = retriever
        self._provider = provider

    def ask(self, question: str, k: int = 4, max_tokens: int = 300) -> str:
        """Retrieve *k* passages, build an augmented prompt, and generate.

        Parameters
        ----------
        question:
            The user's question.
        k:
            Number of context passages to retrieve.
        max_tokens:
            Maximum tokens for the generated answer; forwarded to the provider.

        Returns
        -------
        str
            The generated answer from the provider.
        """
        passages = self._retriever.retrieve(question, k=k)
        prompt = self._build_prompt(passages, question)
        return self._provider.generate(prompt, max_tokens)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_prompt(passages: list[str], question: str) -> str:
        """Assemble the augmented prompt from retrieved passages and the question.

        The template is defined once at module level so it is easy to find,
        read, and change without hunting through method bodies.
        """
        context = "\n\n".join(passages) if passages else "(no context retrieved)"
        return _PROMPT_TEMPLATE.format(context=context, question=question)
