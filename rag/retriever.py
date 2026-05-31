"""
retriever.py — Retriever: the top-level RAG coordinator inside rag/.

Retriever depends on two abstractions:
  - Embedder (from rag.embed): converts text → vectors.
  - VectorStore (from rag.store): persists and searches vectors.

Neither is a concrete class here. This is dependency injection on both
interfaces, so the storage backend and the embedding model are both swappable
without touching the Retriever at all.

The Retriever has two public methods:
  - index(documents): chunk + embed + persist. Run once (or when docs change).
  - retrieve(query, k): embed the query + search. Run per user question.
"""

from __future__ import annotations

from rag.chunk import chunk_text
from rag.embed import Embedder
from rag.store import VectorStore


class Retriever:
    """Coordinates chunking, embedding, and retrieval against any backend.

    Parameters
    ----------
    embedder:
        Any object satisfying the :class:`~rag.embed.Embedder` Protocol.
        Responsible for turning text into float vectors.
    store:
        Any object satisfying the :class:`~rag.store.VectorStore` Protocol.
        Responsible for persisting and searching vectors.
    """

    def __init__(self, embedder: Embedder, store: VectorStore) -> None:
        self._embedder = embedder
        self._store = store

    def index(self, documents: list[tuple[str, str]]) -> None:
        """Chunk, embed, and store *documents*.

        Parameters
        ----------
        documents:
            A list of (source, text) pairs. *source* is a human-readable
            identifier (e.g. a file path). *text* is the raw document content.

        Each document is split into overlapping chunks first, then all chunks
        are embedded in one :meth:`Embedder.embed` call per document.
        Keeping the embed call per-document (rather than one giant batch)
        makes memory usage predictable and progress visible.
        """
        for source, text in documents:
            chunks = chunk_text(text)
            if not chunks:
                continue
            vectors = self._embedder.embed(chunks)
            self._store.add(
                texts=chunks,
                vectors=vectors,
                sources=[source] * len(chunks),
            )

    def retrieve(self, query: str, k: int = 4) -> list[str]:
        """Return the *k* text chunks most relevant to *query*.

        Parameters
        ----------
        query:
            The user's question or search string.
        k:
            Number of chunks to return.

        Returns
        -------
        list[str]
            The retrieved text chunks, ordered by descending cosine similarity.
            Scores are not exposed; callers receive plain text, which is what
            the prompt template needs.
        """
        query_vecs = self._embedder.embed([query])
        query_vec = query_vecs[0]
        results = self._store.search(query_vec, k)
        return [text for text, _score in results]
