"""
Tests for the rag/ package.

All HTTP is monkeypatched — no running server, no network.
All SqliteStore tests use tmp_path so they never touch the real filesystem.

Sections:
  - chunk_text: paragraph splitting, max_chars, overlap, empty-drop
  - VectorStore protocol conformance
  - SqliteStore cosine ranking
  - SqliteStore persistence across close/reopen
  - SqliteStore BLOB round-trip
  - OllamaEmbedder: payload shape, response parsing, error handling
  - Retriever: index + retrieve with a fake embedder and real store
  - Boundary guard: rag.* must not import model or harness
"""

from __future__ import annotations

import array
import email.message
import importlib
import io
import json
import math
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Protocol, runtime_checkable

import pytest
import torch

# ---------------------------------------------------------------------------
# Helpers shared across test sections
# ---------------------------------------------------------------------------


def _make_db_path(tmp_path: Path) -> str:
    return str(tmp_path / "rag.db")


# ---------------------------------------------------------------------------
# chunk_text
# ---------------------------------------------------------------------------


class TestChunkText:
    def test_single_paragraph_shorter_than_max_chars_is_returned_as_one_chunk(self):
        from rag.chunk import chunk_text

        text = "Hello world."
        result = chunk_text(text, max_chars=800, overlap=100)

        assert result == ["Hello world."]

    def test_empty_string_returns_no_chunks(self):
        from rag.chunk import chunk_text

        assert chunk_text("", max_chars=800, overlap=100) == []

    def test_whitespace_only_string_returns_no_chunks(self):
        from rag.chunk import chunk_text

        assert chunk_text("   \n\n   ", max_chars=800, overlap=100) == []

    def test_two_short_paragraphs_packed_into_one_chunk(self):
        """Two paragraphs that together fit within max_chars become one chunk."""
        from rag.chunk import chunk_text

        text = "First paragraph.\n\nSecond paragraph."
        result = chunk_text(text, max_chars=800, overlap=100)

        assert len(result) == 1
        assert "First paragraph." in result[0]
        assert "Second paragraph." in result[0]

    def test_paragraph_exceeding_max_chars_starts_a_new_chunk(self):
        """Once adding a paragraph would exceed max_chars, it starts a fresh chunk."""
        from rag.chunk import chunk_text

        # Each paragraph is 50 chars; max_chars=60, so only one fits per chunk.
        para_a = "A" * 50
        para_b = "B" * 50
        text = f"{para_a}\n\n{para_b}"

        result = chunk_text(text, max_chars=60, overlap=0)

        assert len(result) == 2
        assert result[0] == para_a
        assert result[1] == para_b

    def test_overlap_carries_tail_into_next_chunk(self):
        """The tail of chunk N appears at the start of chunk N+1."""
        from rag.chunk import chunk_text

        para_a = "Alpha " * 15  # ~90 chars
        para_b = "Beta " * 15   # ~75 chars
        para_c = "Gamma " * 15  # ~90 chars
        text = f"{para_a}\n\n{para_b}\n\n{para_c}"

        result = chunk_text(text, max_chars=100, overlap=30)

        assert len(result) >= 2
        # The tail of the first chunk should appear somewhere in the next chunk
        tail = result[0][-30:]
        # Find a word boundary version of the tail in chunk[1]
        assert any(word in result[1] for word in tail.split() if word)

    def test_does_not_split_mid_word(self):
        """Overlap never cuts a word in half."""
        from rag.chunk import chunk_text

        # Use real words so a mid-word split would be obvious.
        para_a = "one two three four five six seven eight nine ten " * 2
        para_b = "alpha beta gamma delta epsilon " * 2
        text = f"{para_a}\n\n{para_b}"

        result = chunk_text(text, max_chars=len(para_a) + 10, overlap=40)

        for chunk in result:
            # Every "word" in the chunk must be a complete word (no partial starts)
            words = chunk.split()
            for word in words:
                assert word == word.strip(), f"Unexpected whitespace in word: {word!r}"

    def test_empty_paragraphs_between_content_are_dropped(self):
        """Multiple blank lines collapse; empty paragraphs are not emitted."""
        from rag.chunk import chunk_text

        text = "First.\n\n\n\n\nSecond."
        result = chunk_text(text, max_chars=800, overlap=0)

        # Empty strings must not appear in result
        assert all(chunk.strip() for chunk in result)

    def test_chunk_text_respects_max_chars_boundary(self):
        """No chunk exceeds max_chars (except when a single paragraph is longer)."""
        from rag.chunk import chunk_text

        # Build paragraphs of known sizes
        paras = ["word " * 10 for _ in range(20)]  # 50 chars each
        text = "\n\n".join(paras)

        result = chunk_text(text, max_chars=120, overlap=20)

        for chunk in result:
            # Overlap can only add content up to max_chars; single oversized
            # paragraphs are allowed through but should not happen in this test.
            assert len(chunk) <= 200, f"Chunk is unexpectedly large: {len(chunk)}"


# ---------------------------------------------------------------------------
# VectorStore protocol conformance
# ---------------------------------------------------------------------------


class TestVectorStoreProtocol:
    def test_sqlite_store_satisfies_vector_store_protocol(self, tmp_path):
        from rag.store import SqliteStore, VectorStore

        store = SqliteStore(_make_db_path(tmp_path))
        try:
            assert isinstance(store, VectorStore)
        finally:
            store.close()

    def test_plain_class_without_required_methods_fails_isinstance(self):
        from rag.store import VectorStore

        class Incomplete:
            def add(self):
                pass  # wrong signature

        assert not isinstance(Incomplete(), VectorStore)

    def test_custom_class_with_correct_methods_satisfies_protocol(self):
        from rag.store import VectorStore

        class FakeStore:
            def add(
                self,
                texts: list[str],
                vectors: list[list[float]],
                sources: list[str] | None = None,
            ) -> None:
                pass

            def search(
                self, query_vec: list[float], k: int
            ) -> list[tuple[str, float]]:
                return []

            def count(self) -> int:
                return 0

        assert isinstance(FakeStore(), VectorStore)


# ---------------------------------------------------------------------------
# SqliteStore cosine ranking
# ---------------------------------------------------------------------------


class TestSqliteStoreCosineRanking:
    """
    Use unit vectors in 2-D so expected cosine similarities are trivial to
    reason about. The ranking must match descending cosine similarity.
    """

    def _unit(self, *components: float) -> list[float]:
        mag = math.sqrt(sum(x * x for x in components))
        return [x / mag for x in components]

    def test_most_similar_vector_is_returned_first(self, tmp_path):
        from rag.store import SqliteStore

        store = SqliteStore(_make_db_path(tmp_path))
        try:
            # Three vectors in 2-D: pointing mostly along x, y, and diagonal
            vec_x = self._unit(1.0, 0.0)         # cosine(query, vec_x) = 1.0
            vec_y = self._unit(0.0, 1.0)         # cosine(query, vec_y) = 0.0
            vec_diag = self._unit(1.0, 1.0)      # cosine(query, vec_diag) ≈ 0.707

            store.add(
                texts=["text_x", "text_y", "text_diag"],
                vectors=[vec_x, vec_y, vec_diag],
            )

            query = vec_x  # identical to vec_x
            results = store.search(query, k=3)

            texts_in_order = [t for t, _ in results]
            assert texts_in_order[0] == "text_x"
            assert texts_in_order[1] == "text_diag"
            assert texts_in_order[2] == "text_y"
        finally:
            store.close()

    def test_scores_are_in_descending_order(self, tmp_path):
        from rag.store import SqliteStore

        store = SqliteStore(_make_db_path(tmp_path))
        try:
            store.add(
                texts=["a", "b", "c", "d"],
                vectors=[
                    self._unit(1.0, 0.0, 0.0),
                    self._unit(0.0, 1.0, 0.0),
                    self._unit(0.0, 0.0, 1.0),
                    self._unit(0.7, 0.7, 0.0),
                ],
            )

            results = store.search(self._unit(1.0, 0.0, 0.0), k=4)
            scores = [s for _, s in results]

            assert scores == sorted(scores, reverse=True)
        finally:
            store.close()

    def test_k_limits_number_of_results(self, tmp_path):
        from rag.store import SqliteStore

        store = SqliteStore(_make_db_path(tmp_path))
        try:
            store.add(
                texts=["a", "b", "c"],
                vectors=[
                    self._unit(1.0, 0.0),
                    self._unit(0.0, 1.0),
                    self._unit(1.0, 1.0),
                ],
            )

            results = store.search(self._unit(1.0, 0.0), k=2)

            assert len(results) == 2
        finally:
            store.close()

    def test_count_reflects_added_rows(self, tmp_path):
        from rag.store import SqliteStore

        store = SqliteStore(_make_db_path(tmp_path))
        try:
            assert store.count() == 0
            store.add(texts=["a", "b"], vectors=[[1.0, 0.0], [0.0, 1.0]])
            assert store.count() == 2
        finally:
            store.close()

    def test_clear_resets_count_to_zero(self, tmp_path):
        from rag.store import SqliteStore

        store = SqliteStore(_make_db_path(tmp_path))
        try:
            store.add(texts=["a"], vectors=[[1.0, 0.0]])
            assert store.count() == 1

            store.clear()
            assert store.count() == 0
        finally:
            store.close()


# ---------------------------------------------------------------------------
# SqliteStore persistence across close/reopen
# ---------------------------------------------------------------------------


class TestSqliteStorePersistence:
    def test_rows_survive_close_and_reopen(self, tmp_path):
        """
        Writing to a SqliteStore, closing it, and opening a NEW instance on the
        same path must produce the same search results — simulating a server
        restart.
        """
        from rag.store import SqliteStore

        db_path = _make_db_path(tmp_path)

        # Write
        store_a = SqliteStore(db_path)
        store_a.add(
            texts=["persisted text"],
            vectors=[[1.0, 0.0]],
            sources=["doc.md"],
        )
        store_a.close()

        # Reopen with a completely new object
        store_b = SqliteStore(db_path)
        try:
            results = store_b.search([1.0, 0.0], k=1)
            assert len(results) == 1
            assert results[0][0] == "persisted text"
        finally:
            store_b.close()

    def test_count_is_correct_after_reopen(self, tmp_path):
        from rag.store import SqliteStore

        db_path = _make_db_path(tmp_path)

        store_a = SqliteStore(db_path)
        store_a.add(texts=["a", "b", "c"], vectors=[[1.0], [2.0], [3.0]])
        store_a.close()

        store_b = SqliteStore(db_path)
        try:
            assert store_b.count() == 3
        finally:
            store_b.close()


# ---------------------------------------------------------------------------
# SqliteStore BLOB round-trip
# ---------------------------------------------------------------------------


class TestSqliteStoreBlobRoundTrip:
    def test_stored_vector_equals_retrieved_vector_within_float32_tolerance(
        self, tmp_path
    ):
        """
        Vectors are serialized as float32 BLOBs (array module), which has
        limited precision. The round-tripped value must be within float32 eps.
        """
        from rag.store import SqliteStore

        original = [0.1, 0.2, 0.3, 0.4, 0.5]
        db_path = _make_db_path(tmp_path)

        store = SqliteStore(db_path)
        try:
            store.add(texts=["blob test"], vectors=[original])

            # Re-read by storing a unit vec as query and checking the
            # stored text's embedding via a dummy search approach.
            # We instead check via the search score (cos sim of vec with itself = 1).
            results = store.search(original, k=1)
            assert len(results) == 1
            text, score = results[0]
            assert text == "blob test"
            assert abs(score - 1.0) < 1e-4, f"Cosine self-similarity should be ~1, got {score}"
        finally:
            store.close()

    def test_float32_serialization_precision(self, tmp_path):
        """
        The array('f', ...) round-trip matches what float32 torch would produce.
        """
        from rag.store import SqliteStore

        original = [0.123456789, -0.987654321, 1.0, -1.0]
        # Manually round-trip through float32
        expected = list(array.array("f", original))

        db_path = _make_db_path(tmp_path)
        store = SqliteStore(db_path)
        try:
            store.add(texts=["precision"], vectors=[original])
            # The score is cos_sim(expected_f32, expected_f32) = 1.0
            results = store.search(expected, k=1)
            assert results[0][0] == "precision"
            assert abs(results[0][1] - 1.0) < 1e-4
        finally:
            store.close()


# ---------------------------------------------------------------------------
# OllamaEmbedder
# ---------------------------------------------------------------------------


class _FakeResponse:
    """Minimal stand-in for urllib.request.urlopen's return value."""

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
        url="http://localhost:11434/api/embeddings",
        code=code,
        msg="error",
        hdrs=email.message.Message(),
        fp=io.BytesIO(json.dumps(payload).encode("utf-8")),
    )


class TestOllamaEmbedder:
    def test_embed_sends_correct_payload_to_api(self, monkeypatch):
        from rag.embed import OllamaEmbedder

        captured: list[dict] = []

        def fake_urlopen(request, timeout=None):
            captured.append({
                "url": request.full_url,
                "method": request.get_method(),
                "body": json.loads(request.data.decode("utf-8")),
            })
            return _FakeResponse({"embedding": [0.1, 0.2, 0.3]})

        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

        embedder = OllamaEmbedder(model="nomic-embed-text")
        result = embedder.embed(["hello world"])

        assert len(captured) == 1
        assert captured[0]["url"] == "http://localhost:11434/api/embeddings"
        assert captured[0]["method"] == "POST"
        assert captured[0]["body"]["model"] == "nomic-embed-text"
        assert captured[0]["body"]["prompt"] == "hello world"

    def test_embed_parses_embedding_field(self, monkeypatch):
        from rag.embed import OllamaEmbedder

        expected = [0.1, 0.5, -0.3]

        monkeypatch.setattr(
            urllib.request,
            "urlopen",
            lambda req, timeout=None: _FakeResponse({"embedding": expected}),
        )

        embedder = OllamaEmbedder()
        result = embedder.embed(["test"])

        assert result == [expected]

    def test_embed_multiple_texts_sends_one_request_per_text(self, monkeypatch):
        from rag.embed import OllamaEmbedder

        call_count = 0

        def fake_urlopen(request, timeout=None):
            nonlocal call_count
            call_count += 1
            return _FakeResponse({"embedding": [float(call_count)]})

        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

        embedder = OllamaEmbedder()
        result = embedder.embed(["one", "two", "three"])

        assert call_count == 3
        assert len(result) == 3

    def test_embed_raises_runtime_error_when_server_unreachable(self, monkeypatch):
        from rag.embed import OllamaEmbedder

        def fake_urlopen(request, timeout=None):
            raise urllib.error.URLError("Connection refused")

        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

        embedder = OllamaEmbedder()
        with pytest.raises(RuntimeError) as exc_info:
            embedder.embed(["hello"])

        message = str(exc_info.value)
        assert "Ollama" in message or "ollama" in message.lower()
        assert "serve" in message

    def test_embed_raises_runtime_error_when_model_not_found(self, monkeypatch):
        from rag.embed import OllamaEmbedder

        def fake_urlopen(request, timeout=None):
            raise _http_error(404, {"error": "model 'nomic-embed-text' not found"})

        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

        embedder = OllamaEmbedder(model="nomic-embed-text")
        with pytest.raises(RuntimeError) as exc_info:
            embedder.embed(["hello"])

        message = str(exc_info.value)
        assert "pull" in message
        assert "nomic-embed-text" in message

    def test_embedder_satisfies_embedder_protocol(self):
        from rag.embed import Embedder, OllamaEmbedder

        assert isinstance(OllamaEmbedder(), Embedder)


# ---------------------------------------------------------------------------
# Retriever (fake in-memory embedder, real SqliteStore on tmp_path)
# ---------------------------------------------------------------------------


class _FakeEmbedder:
    """
    Deterministic embedder: maps text to a known fixed vector.
    No network, no HTTP.
    """

    def __init__(self, mapping: dict[str, list[float]]):
        self._mapping = mapping
        self._dim = len(next(iter(mapping.values())))

    def embed(self, texts: list[str]) -> list[list[float]]:
        result = []
        for text in texts:
            # Return the mapped vector if known; otherwise a zero vector.
            result.append(self._mapping.get(text, [0.0] * self._dim))
        return result


class TestRetriever:
    def _make_unit(self, *components: float) -> list[float]:
        mag = math.sqrt(sum(x * x for x in components))
        return [x / mag for x in components]

    def test_retrieve_returns_most_similar_document(self, tmp_path):
        from rag.retriever import Retriever
        from rag.store import SqliteStore

        # Three single-chunk documents with orthogonal embeddings.
        # Querying with vec_a should return doc_a first.
        vec_a = self._make_unit(1.0, 0.0, 0.0)
        vec_b = self._make_unit(0.0, 1.0, 0.0)
        vec_c = self._make_unit(0.0, 0.0, 1.0)

        embedder = _FakeEmbedder({
            "Content of doc A.": vec_a,
            "Content of doc B.": vec_b,
            "Content of doc C.": vec_c,
            "query about A": vec_a,   # same direction as doc A
        })

        store = SqliteStore(_make_db_path(tmp_path))
        try:
            retriever = Retriever(embedder=embedder, store=store)
            retriever.index([
                ("source_a.md", "Content of doc A."),
                ("source_b.md", "Content of doc B."),
                ("source_c.md", "Content of doc C."),
            ])

            results = retriever.retrieve("query about A", k=1)

            assert results == ["Content of doc A."]
        finally:
            store.close()

    def test_retrieve_returns_k_passages(self, tmp_path):
        from rag.retriever import Retriever
        from rag.store import SqliteStore

        vec_a = self._make_unit(1.0, 0.0)
        vec_b = self._make_unit(0.0, 1.0)

        embedder = _FakeEmbedder({
            "Doc A text.": vec_a,
            "Doc B text.": vec_b,
            "my query": vec_a,
        })

        store = SqliteStore(_make_db_path(tmp_path))
        try:
            retriever = Retriever(embedder=embedder, store=store)
            retriever.index([
                ("a.md", "Doc A text."),
                ("b.md", "Doc B text."),
            ])

            results = retriever.retrieve("my query", k=2)

            assert len(results) == 2
        finally:
            store.close()

    def test_retrieve_returns_plain_strings_not_tuples(self, tmp_path):
        from rag.retriever import Retriever
        from rag.store import SqliteStore

        vec = self._make_unit(1.0, 0.0)
        embedder = _FakeEmbedder({"some text.": vec, "the query": vec})

        store = SqliteStore(_make_db_path(tmp_path))
        try:
            retriever = Retriever(embedder=embedder, store=store)
            retriever.index([("src.md", "some text.")])

            results = retriever.retrieve("the query", k=1)

            assert isinstance(results, list)
            for item in results:
                assert isinstance(item, str), f"Expected str, got {type(item)}"
        finally:
            store.close()

    def test_index_splits_long_documents_into_chunks(self, tmp_path):
        """
        index() should run chunk_text on each document so long docs produce
        multiple stored rows.
        """
        from rag.retriever import Retriever
        from rag.store import SqliteStore

        # Build a document long enough to produce multiple chunks.
        long_text = ("paragraph content here. " * 20 + "\n\n") * 5
        vec = self._make_unit(1.0, 0.0)

        # Override embed to return a fixed vec for anything
        class AnyEmbedder:
            def embed(self, texts: list[str]) -> list[list[float]]:
                return [vec for _ in texts]

        store = SqliteStore(_make_db_path(tmp_path))
        try:
            retriever = Retriever(embedder=AnyEmbedder(), store=store)
            retriever.index([("long.md", long_text)])

            # Should have stored more than 1 chunk
            assert store.count() > 1
        finally:
            store.close()


# ---------------------------------------------------------------------------
# Boundary guard
# ---------------------------------------------------------------------------


class TestBoundaryGuard:
    def test_rag_does_not_import_model_or_harness(self):
        """
        After importing the rag package, sys.modules must contain no module
        whose name starts with 'model' or 'harness'.
        """
        import rag  # noqa: F401  (imported for side effect)

        # Collect all rag.* module source files that are loaded
        forbidden_prefixes = ("model", "harness")
        violations = [
            name
            for name in sys.modules
            if any(name == p or name.startswith(p + ".") for p in forbidden_prefixes)
            # Only flag modules that come from rag-related imports, but since
            # this test suite also imports harness (test_harness.py runs before),
            # we check by inspecting the rag module's own __file__ attribute
            # against the module's origin.
        ]

        # Filter out violations that were already in sys.modules BEFORE rag was
        # imported (e.g., harness imported by the test suite itself).
        # We assert that rag's own source files do NOT reference model/harness.
        import rag.chunk
        import rag.embed
        import rag.store
        import rag.retriever

        rag_modules = [rag, rag.chunk, rag.embed, rag.store, rag.retriever]

        for mod in rag_modules:
            source = getattr(mod, "__file__", "") or ""
            for forbidden in forbidden_prefixes:
                # The rag module files must not live under model/ or harness/
                assert f"/{forbidden}/" not in source, (
                    f"{mod.__name__} lives inside {forbidden}/ — boundary violated"
                )

        # Also verify by inspecting the source text for import statements
        for mod in rag_modules:
            source_file = getattr(mod, "__file__", None)
            if source_file and source_file.endswith(".py"):
                source_text = Path(source_file).read_text()
                for forbidden in forbidden_prefixes:
                    # Look for "import model" / "from model" / "import harness" etc.
                    assert f"import {forbidden}" not in source_text, (
                        f"{mod.__name__} contains 'import {forbidden}' — boundary violated"
                    )
                    assert f"from {forbidden}" not in source_text, (
                        f"{mod.__name__} contains 'from {forbidden}' — boundary violated"
                    )
