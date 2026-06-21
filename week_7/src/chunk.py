"""Text chunking strategies.

Chunk quality is one of the biggest levers on retrieval accuracy, so we
expose several strategies behind a common interface and benchmark them in
the ablation study (see ``notebooks/analysis.ipynb``). Every function takes
raw text and returns a list of chunk strings — no external dependencies.
"""
from __future__ import annotations

import re
from typing import Callable


def _words(text: str) -> list[str]:
    return text.split()


def fixed_size_chunks(text: str, chunk_size: int = 512, overlap: int = 64) -> list[str]:
    """Split text into fixed windows of ``chunk_size`` words with ``overlap``.

    The classic RAG baseline: simple and fast. The overlap reduces the chance
    that an answer spanning a window boundary is split across two chunks and
    lost to retrieval.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    words = _words(text)
    if not words:
        return []
    step = chunk_size - overlap
    chunks: list[str] = []
    for start in range(0, len(words), step):
        window = words[start : start + chunk_size]
        if window:
            chunks.append(" ".join(window))
        if start + chunk_size >= len(words):
            break
    return chunks


def _sentences(text: str) -> list[str]:
    """Naive but dependency-free sentence splitter."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def recursive_chunks(text: str, chunk_size: int = 512, overlap: int = 64) -> list[str]:
    """Pack whole sentences into ~``chunk_size``-word chunks with sentence-level
    overlap.

    Respecting sentence boundaries keeps each chunk semantically coherent
    (unlike blind fixed-size slicing that can cut mid-sentence). This mirrors
    the idea behind LangChain's ``RecursiveCharacterTextSplitter`` but is
    word-based and has no dependencies. Sentences longer than ``chunk_size``
    are hard-split as a fallback.
    """
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for sentence in _sentences(text):
        n = len(sentence.split())

        # A single over-long sentence: flush, then hard-split it.
        if n > chunk_size:
            if current:
                chunks.append(" ".join(current))
                current, current_len = [], 0
            chunks.extend(fixed_size_chunks(sentence, chunk_size, overlap))
            continue

        # Adding this sentence would overflow -> close the current chunk and
        # seed the next one with a trailing-sentence overlap for continuity.
        if current_len + n > chunk_size and current:
            chunks.append(" ".join(current))
            overlap_sents: list[str] = []
            overlap_len = 0
            for prev in reversed(current):
                pn = len(prev.split())
                if overlap_len + pn > overlap:
                    break
                overlap_sents.insert(0, prev)
                overlap_len += pn
            current, current_len = overlap_sents[:], overlap_len

        current.append(sentence)
        current_len += n

    if current:
        chunks.append(" ".join(current))
    return chunks


# Registry so the pipeline / ablations can select a strategy by name.
CHUNKERS: dict[str, Callable[..., list[str]]] = {
    "fixed": fixed_size_chunks,
    "recursive": recursive_chunks,
}


def chunk_text(
    text: str,
    strategy: str = "recursive",
    chunk_size: int = 512,
    overlap: int = 64,
) -> list[str]:
    """Dispatch to the named chunking strategy."""
    if strategy not in CHUNKERS:
        raise ValueError(f"Unknown strategy {strategy!r}; choose from {list(CHUNKERS)}")
    return CHUNKERS[strategy](text, chunk_size=chunk_size, overlap=overlap)
