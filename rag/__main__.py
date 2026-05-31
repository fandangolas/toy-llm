"""
rag/__main__.py — retrieval-only demo.

    python -m rag "<query>" [--rebuild] [--k N]

Builds a SqliteStore("data/rag.db") + OllamaEmbedder + Retriever.
If the store is empty or --rebuild is passed, indexes all docs/**/*.md files
using their relative path as the source label.
Then prints the top-k retrieved passages with their sources.

This module is intentionally retrieval-only — it does NOT import any provider.
The retrieve → generate path lives in harness/pipeline.py.

Prerequisites:
    ollama pull nomic-embed-text
    ollama serve  (or it starts automatically)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# rag/ is standalone: no model/, no harness/ imported here.
from rag.embed import OllamaEmbedder
from rag.retriever import Retriever
from rag.store import SqliteStore

_DOCS_GLOB = "docs/**/*.md"
_DEFAULT_DB = "data/rag.db"


def _find_doc_files(project_root: Path) -> list[Path]:
    return sorted(project_root.glob(_DOCS_GLOB))


def _index_docs(retriever: Retriever, project_root: Path) -> None:
    doc_files = _find_doc_files(project_root)
    if not doc_files:
        print("No docs/**/*.md files found — nothing to index.", file=sys.stderr)
        return

    print(f"Indexing {len(doc_files)} file(s)...")
    documents = []
    for path in doc_files:
        source = str(path.relative_to(project_root))
        text = path.read_text(encoding="utf-8")
        documents.append((source, text))

    retriever.index(documents)
    print("Indexing complete.")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m rag",
        description="Retrieve passages from indexed docs using semantic search.",
    )
    parser.add_argument("query", help="The query to search for.")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Re-index docs even if the store is non-empty.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=4,
        help="Number of passages to retrieve (default: 4).",
    )
    parser.add_argument(
        "--db",
        default=_DEFAULT_DB,
        help=f"Path to the SQLite database (default: {_DEFAULT_DB}).",
    )
    parser.add_argument(
        "--model",
        default="nomic-embed-text",
        help="Ollama embedding model (default: nomic-embed-text).",
    )
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent

    try:
        embedder = OllamaEmbedder(model=args.model)
        store = SqliteStore(args.db)
        retriever = Retriever(embedder=embedder, store=store)

        should_index = args.rebuild or store.count() == 0
        if should_index:
            if args.rebuild and store.count() > 0:
                print("--rebuild: clearing existing index.")
                store.clear()
            _index_docs(retriever, project_root)

        print(f"\nQuery: {args.query!r}")
        print(f"Top {args.k} passages:\n")

        passages = retriever.retrieve(args.query, k=args.k)
        if not passages:
            print("(no results)")
        else:
            for i, passage in enumerate(passages, start=1):
                print(f"--- Passage {i} ---")
                print(passage)
                print()

    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    finally:
        # store may not be bound if SqliteStore constructor raised
        try:
            store.close()
        except NameError:
            pass


main()
