import re
from typing import List, Optional, Tuple

from storage.db import get_cached_chunks, set_cached_chunks


def chunk_text(
    text: str,
    chunk_size: int = 1200,
    overlap: int = 150,
    file_bytes: Optional[bytes] = None,
) -> List[str]:
    """
    Chunk by approximate word count with overlap.
    If file_bytes is provided, results are cached by file hash so
    repeated uploads of the same document skip re-chunking entirely.
    """
    if file_bytes is not None:
        cached = get_cached_chunks(file_bytes)
        if cached is not None:
            return cached

    text = re.sub(r"\s+", " ", text).strip()
    words = text.split()

    chunks: List[str] = []
    i = 0
    while i < len(words):
        chunk_words = words[i : i + chunk_size]
        chunks.append(" ".join(chunk_words))
        i += max(1, chunk_size - overlap)

    if file_bytes is not None:
        set_cached_chunks(file_bytes, chunks)

    return chunks


def score_chunk(chunk: str, query: str) -> float:
    """
    Simple relevance score: counts query keyword overlaps.
    Pre-computes query word set to avoid re-computing per chunk.
    """
    chunk_l = chunk.lower()
    q_words = set(re.findall(r"[a-zA-ZА-Яа-я0-9\-]{3,}", query.lower()))
    if not q_words:
        return 0.0
    hits = sum(1 for w in q_words if w in chunk_l)
    return hits / len(q_words)


def select_top_chunks(text: str, query: str, top_k: int = 6) -> List[Tuple[int, str]]:
    chunks = chunk_text(text)
    scored = [(i, c, score_chunk(c, query)) for i, c in enumerate(chunks)]
    scored.sort(key=lambda x: x[2], reverse=True)
    top = scored[:top_k]
    return [(i, c) for i, c, _ in top]
