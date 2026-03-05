import re
from typing import List, Tuple


def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 150) -> List[str]:
    """
    Chunk by approximate word count with overlap.
    """
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split()

    chunks = []
    i = 0
    while i < len(words):
        chunk_words = words[i : i + chunk_size]
        chunks.append(" ".join(chunk_words))
        i += max(1, chunk_size - overlap)

    return chunks


def score_chunk(chunk: str, query: str) -> float:
    """
    Very simple relevance score: counts query keyword overlaps.
    """
    chunk_l = chunk.lower()
    q_words = [w for w in re.findall(r"[a-zA-ZА-Яа-я0-9\-]{3,}", query.lower())]
    if not q_words:
        return 0.0
    hits = sum(1 for w in set(q_words) if w in chunk_l)
    return hits / max(1, len(set(q_words)))


def select_top_chunks(text: str, query: str, top_k: int = 6) -> List[Tuple[int, str]]:
    chunks = chunk_text(text)
    scored = [(i, c, score_chunk(c, query)) for i, c in enumerate(chunks)]
    scored.sort(key=lambda x: x[2], reverse=True)
    top = scored[:top_k]
    return [(i, c) for i, c, _ in top]