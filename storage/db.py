import hashlib
import json
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# In-memory document chunk cache
# Key: sha256 of file bytes  →  Value: list of chunks
# Survives within a single Streamlit session / process lifetime.
# ---------------------------------------------------------------------------
_chunk_cache: Dict[str, List[str]] = {}


def get_cached_chunks(file_bytes: bytes) -> Optional[List[str]]:
    key = hashlib.sha256(file_bytes).hexdigest()
    return _chunk_cache.get(key)


def set_cached_chunks(file_bytes: bytes, chunks: List[str]) -> None:
    key = hashlib.sha256(file_bytes).hexdigest()
    _chunk_cache[key] = chunks


def clear_chunk_cache() -> None:
    _chunk_cache.clear()
