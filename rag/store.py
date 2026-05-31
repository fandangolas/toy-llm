"""
store.py — VectorStore protocol and SqliteStore implementation.

DESIGN: TWO-LAYER SEPARATION
=============================
SQLite is the *persistence* layer only. It stores text and embedding BLOBs,
handles durability, and survives process restarts. SQLite does no vector math.

PyTorch (torch) is the *similarity* layer only. At query time, all stored
vectors are loaded into a (N, D) tensor, L2-normalised, and compared to the
query via matrix multiplication (cosine similarity). torch.topk picks the top-k.

This split is the key concept: persistence and similarity computation are
separate concerns handled by separate tools, each chosen for what it is good at.
A future backend — Postgres/pgvector, FAISS, Weaviate — would swap the
persistence layer without touching the similarity math, or swap both together
behind the same VectorStore interface.

SWAPPABILITY
=============
The VectorStore Protocol is the single contract that Retriever depends on.
SqliteStore is the only concrete backend today. To add a new backend, e.g.
Postgres with pgvector:

    class PgvectorStore:
        def add(self, texts, vectors, sources=None): ...
        def search(self, query_vec, k): ...
        def count(self): ...

That class satisfies VectorStore structurally (Protocol, runtime_checkable) —
no inheritance, no registration, no change to Retriever or any caller.

SERIALIZATION
==============
Vectors are stored as BLOBs using Python's stdlib `array` module:
    array('f', vec).tobytes()   → float32 bytes → SQLite BLOB
    array('f').frombytes(blob)  → float32 bytes → list[float]
float32 is sufficient for cosine similarity and matches the precision that most
embedding models advertise.
"""

from __future__ import annotations

import array
import sqlite3
from typing import Protocol, runtime_checkable

import torch

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS chunks (
    id        INTEGER PRIMARY KEY,
    source    TEXT,
    text      TEXT    NOT NULL,
    embedding BLOB    NOT NULL,
    dim       INTEGER NOT NULL
)
"""


@runtime_checkable
class VectorStore(Protocol):
    """Contract for storing and retrieving embedded text chunks.

    Any object that exposes these three methods satisfies the Protocol.
    The Protocol is @runtime_checkable so ``isinstance(obj, VectorStore)``
    works structurally — identical to the LLMProvider / Embedder approach.

    Adding a new backend (Postgres/pgvector, FAISS, etc.) means writing a
    class that implements these three methods. The Retriever never needs to
    change.
    """

    def add(
        self,
        texts: list[str],
        vectors: list[list[float]],
        sources: list[str] | None = None,
    ) -> None:
        """Persist *texts* alongside their embedding *vectors*.

        Parameters
        ----------
        texts:
            The original text chunks, in the same order as *vectors*.
        vectors:
            One embedding vector per text chunk.
        sources:
            Optional document identifier for each chunk (e.g. file path).
            If provided, must have the same length as *texts*.
        """
        ...

    def search(self, query_vec: list[float], k: int) -> list[tuple[str, float]]:
        """Return the *k* chunks most similar to *query_vec*.

        Returns
        -------
        list[tuple[str, float]]
            (text, cosine_similarity) pairs, ordered descending by similarity.
        """
        ...

    def count(self) -> int:
        """Return the total number of stored chunks."""
        ...


class SqliteStore:
    """VectorStore backed by a single SQLite file.

    SQLite stores the text and the embedding BLOB. PyTorch computes cosine
    similarity at query time by loading all stored vectors into a tensor,
    L2-normalising, and doing a matrix multiply. SQLite does no vector math.

    Parameters
    ----------
    path:
        Path to the SQLite database file. The file and its parent directory
        must be writable. Created if it does not exist.
    """

    def __init__(self, path: str = "data/rag.db") -> None:
        self._conn = sqlite3.connect(path)
        self._conn.execute(_CREATE_TABLE)
        self._conn.commit()

    # ------------------------------------------------------------------
    # VectorStore interface
    # ------------------------------------------------------------------

    def add(
        self,
        texts: list[str],
        vectors: list[list[float]],
        sources: list[str] | None = None,
    ) -> None:
        """Persist *texts* and their *vectors* as rows in the chunks table.

        Vectors are serialized to float32 BLOBs via the stdlib ``array`` module.
        """
        if sources is None:
            sources = [None] * len(texts)  # type: ignore[list-item]

        rows = [
            (source, text, _vec_to_blob(vec), len(vec))
            for text, vec, source in zip(texts, vectors, sources)
        ]
        self._conn.executemany(
            "INSERT INTO chunks (source, text, embedding, dim) VALUES (?, ?, ?, ?)",
            rows,
        )
        self._conn.commit()

    def search(self, query_vec: list[float], k: int) -> list[tuple[str, float]]:
        """Return the top-*k* most similar chunks using cosine similarity.

        Algorithm:
          1. Load all stored (text, embedding BLOB) rows from SQLite.
          2. Deserialize each BLOB to a list[float] via array('f').
          3. Stack into a (N, D) torch tensor.
          4. L2-normalise both the matrix rows and the query vector.
          5. Cosine similarities = normalised_matrix @ normalised_query.
          6. torch.topk selects the highest-scoring k indices.

        SQLite performs no vector arithmetic — it is a pure storage retrieval.
        """
        rows = self._conn.execute(
            "SELECT text, embedding FROM chunks"
        ).fetchall()

        if not rows:
            return []

        texts = [row[0] for row in rows]
        raw_vecs = [_blob_to_vec(row[1]) for row in rows]

        matrix = torch.tensor(raw_vecs, dtype=torch.float32)   # (N, D)
        query = torch.tensor(query_vec, dtype=torch.float32)    # (D,)

        matrix_norm = _l2_normalize_rows(matrix)               # (N, D)
        query_norm = _l2_normalize_vec(query)                   # (D,)

        similarities = matrix_norm @ query_norm                 # (N,)

        actual_k = min(k, len(texts))
        top_scores, top_indices = torch.topk(similarities, actual_k)

        return [
            (texts[idx.item()], score.item())
            for idx, score in zip(top_indices, top_scores)
        ]

    def count(self) -> int:
        """Return the number of stored chunks."""
        row = self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
        return row[0]

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Drop and recreate the chunks table, removing all stored data."""
        self._conn.execute("DROP TABLE IF EXISTS chunks")
        self._conn.execute(_CREATE_TABLE)
        self._conn.commit()

    def close(self) -> None:
        """Close the underlying database connection."""
        self._conn.close()


# ---------------------------------------------------------------------------
# Private serialization helpers
# ---------------------------------------------------------------------------


def _vec_to_blob(vec: list[float]) -> bytes:
    """Serialize a float vector to a float32 BLOB via the stdlib array module."""
    return array.array("f", vec).tobytes()


def _blob_to_vec(blob: bytes) -> list[float]:
    """Deserialize a float32 BLOB to a list of floats."""
    result = array.array("f")
    result.frombytes(blob)
    return result.tolist()


def _l2_normalize_rows(matrix: torch.Tensor) -> torch.Tensor:
    """L2-normalise each row of *matrix* to unit length."""
    norms = matrix.norm(dim=1, keepdim=True).clamp(min=1e-12)
    return matrix / norms


def _l2_normalize_vec(vec: torch.Tensor) -> torch.Tensor:
    """L2-normalise a 1-D vector to unit length."""
    norm = vec.norm().clamp(min=1e-12)
    return vec / norm
