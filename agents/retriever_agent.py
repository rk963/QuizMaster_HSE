import re
from collections import Counter
from typing import Any, Dict, List, Tuple

from .base_agent import BaseAgent
from services.chunking import chunk_text, score_chunk


class RetrieverAgent(BaseAgent):
    def __init__(self):
        super().__init__("RetrieverAgent")

    def _extract_keywords_from_text(self, text: str, top_n: int = 40) -> List[str]:
        """
        Build a retrieval-friendly keyword list from the uploaded text itself.
        This makes retrieval grounded in the actual document instead of generic plan words.
        """
        words = re.findall(r"[A-Za-zА-Яа-яЁё0-9\-]{4,}", text.lower())

        stopwords = {
            # English
            "this", "that", "with", "from", "have", "been", "were", "their", "there",
            "about", "into", "than", "then", "they", "them", "such", "also", "only",
            "some", "more", "most", "many", "much", "very", "used", "using", "use",
            "what", "when", "where", "which", "while", "will", "would", "should",
            "could", "your", "ours", "ourselves", "between", "through", "each",
            "because", "after", "before", "under", "over", "topic", "study", "material",
            "question", "questions", "answer", "answers", "correct", "incorrect",
            "fact", "facts", "concept", "concepts", "application", "applications",
            "important", "definition", "definitions", "medium", "easy", "hard",
            "text", "chapter", "lecture", "article",

            # Russian
            "это", "этот", "эта", "эти", "того", "того", "также", "только", "очень",
            "когда", "тогда", "после", "перед", "между", "через", "который", "которая",
            "которые", "может", "могут", "должен", "должны", "такой", "такая", "такие",
            "есть", "были", "было", "быть", "если", "или", "для", "при", "над", "под",
            "как", "что", "чтобы", "где", "какой", "какая", "какие", "текст", "тема",
            "вопрос", "вопросы", "ответ", "ответы", "правильный", "неверный",
            "важный", "важные", "определение", "определения", "понятие", "понятия",
            "применение", "материал", "лекция", "статья",

            # Common filler
            "http", "https", "www", "com", "org", "pdf", "docx", "txt",
        }

        filtered = [w for w in words if w not in stopwords and not w.isdigit()]
        counts = Counter(filtered)

        return [word for word, _ in counts.most_common(top_n)]

    def _build_query(self, state: Dict[str, Any]) -> str:
        """
        Build the retrieval query primarily from the uploaded text itself.
        We keep a little optional plan info, but the core is document-grounded.
        """
        text = state.get("text", "") or ""
        normalized_text = " ".join(text.split())

        # Main signal: keywords from the actual uploaded text
        keywords = self._extract_keywords_from_text(normalized_text, top_n=40)

        # Secondary signal: a short prefix of the document
        # This helps preserve named entities and exact phrasing from the source.
        prefix = normalized_text[:1200]

        # Optional: lightly include plan focus if present
        plan = state.get("plan", {}) or {}
        focus_terms = " ".join(plan.get("focus", []))

        query_parts = []
        if keywords:
            query_parts.append(" ".join(keywords))
        if focus_terms.strip():
            query_parts.append(focus_terms.strip())
        if prefix.strip():
            query_parts.append(prefix.strip())

        query = " ".join(query_parts).strip()

        # Final fallback for very short or empty text
        if not query:
            difficulty = plan.get("difficulty", "medium")
            distribution = plan.get("distribution", {})
            query = (
                f"{difficulty} "
                f"fact {distribution.get('fact', 0)} "
                f"concept {distribution.get('concept', 0)} "
                f"application {distribution.get('application', 0)}"
            )

        return query

    def _select_diverse_chunks(
        self,
        text: str,
        query: str,
        top_k: int = 6,
        file_bytes: bytes = None,
    ) -> List[Tuple[int, str]]:
        chunks = chunk_text(text, file_bytes=file_bytes)

        if not chunks:
            return []

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

        top_k = min(8, max(4, n_questions))
        top_chunks = self._select_diverse_chunks(
            text,
            query=query,
            top_k=top_k,
            file_bytes=file_bytes,
        )

        state["retrieved_chunks"] = top_chunks
        state["allowed_chunk_ids"] = {idx for idx, _ in top_chunks}

        # Give the generator a bit more grounded context than before
        max_context_chars = 7000
        parts = []
        current_len = 0

        for idx, chunk in top_chunks:
            piece = f"CHUNK {idx}:\n{chunk}\n\n"
            if current_len + len(piece) > max_context_chars:
                break
            parts.append(piece)
            current_len += len(piece)

        # Fallback: if retrieval failed for any reason, still pass source text
        if not parts:
            fallback_text = text[:max_context_chars]
            state["retrieved_text"] = f"CHUNK 0:\n{fallback_text}\n\n"
            state["retrieved_chunks"] = [(0, fallback_text)]
            state["allowed_chunk_ids"] = {0}
        else:
            state["retrieved_text"] = "".join(parts)

        return state