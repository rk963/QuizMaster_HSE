from typing import Any, Dict, List, Tuple
from .base_agent import BaseAgent
from services.chunking import chunk_text, score_chunk


class RetrieverAgent(BaseAgent):
    def __init__(self):
        super().__init__("RetrieverAgent")

    def _build_query(self, state: Dict[str, Any]) -> str:
        plan = state.get("plan", {})
        difficulty = plan.get("difficulty", "medium")
        focus = " ".join(plan.get("focus", []))
        distribution = plan.get("distribution", {})
        return (
            f"{difficulty} "
            f"{focus} "
            f"fact {distribution.get('fact', 0)} "
            f"concept {distribution.get('concept', 0)} "
            f"application {distribution.get('application', 0)}"
        )

    def _select_diverse_chunks(
        self,
        text: str,
        query: str,
        top_k: int = 6,
        file_bytes: bytes = None,
    ) -> List[Tuple[int, str]]:
        # Pass file_bytes so chunk_text can use the cache
        chunks = chunk_text(text, file_bytes=file_bytes)

        scored = [(idx, chunk, score_chunk(chunk, query)) for idx, chunk in enumerate(chunks)]
        scored.sort(key=lambda x: x[2], reverse=True)

        selected: List[Tuple[int, str]] = []
        selected_indices: set = set()

        for idx, chunk, _score in scored:
            if len(selected) >= top_k:
                break
            if any(abs(idx - prev_idx) <= 1 for prev_idx in selected_indices):
                continue
            selected.append((idx, chunk))
            selected_indices.add(idx)

        # Fill remaining slots if diversity filter was too aggressive
        if len(selected) < top_k:
            for idx, chunk, _score in scored:
                if len(selected) >= top_k:
                    break
                if idx not in selected_indices:
                    selected.append((idx, chunk))
                    selected_indices.add(idx)

        return selected

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        text = state["text"]
        n_questions = int(state.get("n_questions", 5))
        file_bytes = state.get("file_bytes")  # may be None for pasted text

        query = self._build_query(state)
        top_k = min(6, max(4, n_questions))
        top_chunks = self._select_diverse_chunks(
            text, query=query, top_k=top_k, file_bytes=file_bytes
        )

        state["retrieved_chunks"] = top_chunks
        state["allowed_chunk_ids"] = {idx for idx, _ in top_chunks}

        max_context_chars = 4500
        parts = []
        current_len = 0
        for idx, chunk in top_chunks:
            piece = f"CHUNK {idx}:\n{chunk}\n\n"
            if current_len + len(piece) > max_context_chars:
                break
            parts.append(piece)
            current_len += len(piece)

        state["retrieved_text"] = "".join(parts)
        return state
