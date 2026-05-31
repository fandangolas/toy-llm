"""
chunk.py — split a document into overlapping text chunks.

The algorithm is intentionally simple and transparent:
  1. Split the document on blank lines to get paragraphs.
  2. Greedily pack paragraphs into a chunk until the next paragraph would
     exceed max_chars.
  3. When a chunk is full, carry the last ~overlap characters of its tail
     into the next chunk as leading context, cutting only at a word boundary
     so no word is ever split mid-way.

This mirrors how a human would read and excerpt a document: finish a thought,
then repeat a little context before starting the next excerpt.
"""

from __future__ import annotations

import re


def chunk_text(
    text: str,
    max_chars: int = 800,
    overlap: int = 100,
) -> list[str]:
    """Split *text* into overlapping chunks of at most *max_chars* characters.

    Parameters
    ----------
    text:
        Raw document text. Blank lines separate paragraphs.
    max_chars:
        Soft upper bound on chunk length. A single paragraph that is longer
        than max_chars is stored as its own chunk (no mid-paragraph splits).
    overlap:
        Approximate number of tail characters carried into the next chunk for
        context continuity. The actual carry is trimmed to a word boundary, so
        the value is a maximum, not an exact count.

    Returns
    -------
    list[str]
        Non-empty chunks. Empty strings are never returned.
    """
    paragraphs = _split_paragraphs(text)
    if not paragraphs:
        return []

    chunks: list[str] = []
    current_parts: list[str] = []
    current_len: int = 0

    for para in paragraphs:
        # Joining paragraphs with a single newline between them inside a chunk.
        join_cost = 1 if current_parts else 0
        para_cost = join_cost + len(para)

        if current_parts and current_len + para_cost > max_chars:
            # Flush the current chunk and start the next one with overlap tail.
            chunk = "\n".join(current_parts)
            chunks.append(chunk)
            tail = _word_boundary_tail(chunk, overlap)
            current_parts = [tail, para] if tail else [para]
            current_len = len(tail) + (1 if tail else 0) + len(para)
        else:
            current_parts.append(para)
            current_len += para_cost

    if current_parts:
        chunk = "\n".join(current_parts)
        if chunk.strip():
            chunks.append(chunk)

    return [c for c in chunks if c.strip()]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _split_paragraphs(text: str) -> list[str]:
    """Split *text* on one or more blank lines; drop empty results."""
    raw = re.split(r"\n\s*\n", text)
    return [p.strip() for p in raw if p.strip()]


def _word_boundary_tail(text: str, overlap: int) -> str:
    """Return the last ~*overlap* characters of *text*, trimmed to a word start.

    Cutting at a word boundary ensures no word is emitted in two halves across
    consecutive chunks. If *overlap* >= len(text), the full text is returned.
    """
    if overlap <= 0 or not text:
        return ""
    if overlap >= len(text):
        return text

    candidate = text[-overlap:]
    # Walk forward until we land on a word-start (preceded by whitespace or
    # the beginning of the candidate string).
    space_pos = candidate.find(" ")
    if space_pos == -1:
        # No space in the overlap window — return the whole candidate.
        return candidate.strip()
    # Skip any leading partial word.
    return candidate[space_pos:].lstrip()
