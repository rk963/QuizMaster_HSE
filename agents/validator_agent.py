import re
from typing import Any, Dict, List
from .base_agent import BaseAgent


def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _keyword_set(text: str) -> set:
    stopwords = {
        "a", "an", "the", "is", "are", "was", "were", "what", "which",
        "how", "why", "when", "where", "who", "of", "in", "on", "at",
        "to", "for", "and", "or", "be", "has", "have", "had",
        "с", "в", "на", "из", "и", "или", "что", "как", "где",
        "когда", "какой", "какая", "какие", "это", "не", "по", "за",
    }
    words = _normalize(text).split()
    return {w for w in words if w not in stopwords and len(w) > 2}


def _too_similar(a: str, b: str, threshold: float = 0.6) -> bool:
    a_kw = _keyword_set(a)
    b_kw = _keyword_set(b)

    if not a_kw or not b_kw:
        return False

    inter = a_kw & b_kw
    union = a_kw | b_kw
    if not union:
        return False

    return (len(inter) / len(union)) >= threshold


class ValidatorAgent(BaseAgent):
    def __init__(self):
        super().__init__("ValidatorAgent")

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        quiz = state.get("quiz", [])
        allowed_chunk_ids = state.get("allowed_chunk_ids", set())

        errors: List[str] = []
        valid_questions = []

        seen_questions = []

        if not isinstance(quiz, list):
            state["validation_errors"] = ["Quiz output is not a list."]
            state["is_valid"] = False
            return state

        for i, q in enumerate(quiz, start=1):
            if not isinstance(q, dict):
                errors.append(f"Question {i}: item is not an object.")
                continue

            question = str(q.get("question", "")).strip()
            choices = q.get("choices", [])
            correct = str(q.get("correct_answer", "")).strip()
            explanation = str(q.get("explanation", "")).strip()
            source_chunks = q.get("source_chunks", [])

            if not question:
                errors.append(f"Question {i}: empty question.")
                continue

            if not isinstance(choices, list) or len(choices) != 4:
                errors.append(f"Question {i}: must have exactly 4 choices.")
                continue

            cleaned_choices = [str(c).strip() for c in choices]

            if len(set(c.lower() for c in cleaned_choices)) != 4:
                errors.append(f"Question {i}: choices must be unique.")
                continue

            if correct not in cleaned_choices:
                match = next((c for c in cleaned_choices if c.lower() == correct.lower()), None)
                if match:
                    correct = match
                else:
                    errors.append(f"Question {i}: correct answer not found in choices.")
                    continue

            if any(_too_similar(question, prev) for prev in seen_questions):
                errors.append(f"Question {i}: too similar to another question.")
                continue

            normalized_chunks = []
            if isinstance(source_chunks, list):
                for ch in source_chunks:
                    try:
                        ch_i = int(ch)
                        if ch_i in allowed_chunk_ids and ch_i not in normalized_chunks:
                            normalized_chunks.append(ch_i)
                    except Exception:
                        pass

            valid_questions.append({
                "id": len(valid_questions) + 1,
                "question": question,
                "choices": cleaned_choices,
                "correct_answer": correct,
                "explanation": explanation[:240] if explanation else "",
                "source_chunks": normalized_chunks,
            })

            seen_questions.append(question)

        requested = int(state.get("n_questions", 5))

        state["quiz"] = valid_questions
        state["validation_errors"] = errors
        state["is_valid"] = len(valid_questions) >= requested and len(errors) == 0
        state["valid_count"] = len(valid_questions)

        return state