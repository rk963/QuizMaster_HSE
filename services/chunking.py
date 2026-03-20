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
    if not text:
        return []

    words = text.split()

    chunks: List[str] = []
    step = max(1, chunk_size - overlap)
    i = 0

    while i < len(words):
        chunk_words = words[i:i + chunk_size]
        chunk = " ".join(chunk_words).strip()
        if chunk:
            chunks.append(chunk)
        i += step

    if file_bytes is not None:
        set_cached_chunks(file_bytes, chunks)

    return chunks


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-ZА-Яа-яЁё0-9\-]{3,}", text.lower())


def score_chunk(chunk: str, query: str) -> float:
    """
    Better relevance score than simple substring overlap.

    Combines:
    - unique keyword overlap
    - frequency overlap
    - exact phrase bonus for longer query terms
    - mild length normalization
    """
    chunk_l = chunk.lower()
    chunk_tokens = _tokenize(chunk_l)
    query_tokens = _tokenize(query)

    if not chunk_tokens or not query_tokens:
        return 0.0

    chunk_set = set(chunk_tokens)
    query_set = set(query_tokens)

    # 1) Unique overlap ratio
    unique_overlap = len(chunk_set & query_set) / max(1, len(query_set))

    # 2) Frequency overlap ratio
    chunk_freq_hits = sum(chunk_tokens.count(q) for q in query_set)
    freq_overlap = chunk_freq_hits / max(1, len(query_tokens))

    # 3) Bonus for longer exact terms appearing as substrings
    long_terms = [q for q in query_set if len(q) >= 6]
    phrase_hits = sum(1 for term in long_terms if term in chunk_l)
    phrase_bonus = phrase_hits / max(1, len(long_terms))

    # 4) Mild normalization so very tiny chunks do not dominate
    length_factor = min(1.0, len(chunk_tokens) / 120.0)

    score = (
        0.5 * unique_overlap +
        0.3 * freq_overlap +
        0.2 * phrase_bonus
    ) * (0.75 + 0.25 * length_factor)

    return score


def select_top_chunks(
    text: str,
    query: str,
    top_k: int = 6,
    file_bytes: Optional[bytes] = None,
) -> List[Tuple[int, str]]:
    chunks = chunk_text(text, file_bytes=file_bytes)
    if not chunks:
        return []

    scored = [(i, c, score_chunk(c, query)) for i, c in enumerate(chunks)]
    scored.sort(key=lambda x: x[2], reverse=True)

    top = scored[:top_k]
    return [(i, c) for i, c, _ in top]