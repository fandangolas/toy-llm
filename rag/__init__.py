"""
rag — Retrieval-Augmented Generation components.

Public surface
--------------
chunk_text      Split a document into overlapping text chunks.
Embedder        Protocol: embed(texts) -> list[list[float]].
OllamaEmbedder  Concrete embedder using a local Ollama server.
VectorStore     Protocol: add / search / count.
SqliteStore     Concrete backend: SQLite persistence + torch cosine search.
Retriever       Coordinates chunking, embedding, and retrieval.

Boundary contract
-----------------
This package uses only the Python standard library and torch.
It has no dependency on model/ or harness/. The harness is the only component
that bridges rag/ and a provider — rag/ is standalone and returns plain text.
"""

from rag.chunk import chunk_text
from rag.embed import Embedder, OllamaEmbedder
from rag.store import VectorStore, SqliteStore
from rag.retriever import Retriever

__all__ = [
    "chunk_text",
    "Embedder",
    "OllamaEmbedder",
    "VectorStore",
    "SqliteStore",
    "Retriever",
]
