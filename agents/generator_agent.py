from typing import Any, Dict

from .base_agent import BaseAgent
from services.quiz_generator_ollama import generate_quiz_ollama


class GeneratorAgent(BaseAgent):
    def __init__(self):
        super().__init__("GeneratorAgent")

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        text = state.get("text", "") or ""
        retrieved_text = state.get("retrieved_text", "") or ""
        allowed_chunk_ids = state.get("allowed_chunk_ids", set()) or set()
        n_questions = int(state.get("n_questions", 5))
        language = state.get("language", "English")
        stop_event = state.get("stop_event")

        # Fallback: if retrieval returned nothing useful,
        # use a compact slice of the original text so generation
        # still stays grounded in the uploaded material.
        if not retrieved_text.strip():
            fallback_text = text[:7000].strip()
            if fallback_text:
                retrieved_text = f"CHUNK 0:\n{fallback_text}\n\n"
                allowed_chunk_ids = {0}

        quiz = generate_quiz_ollama(
            text=text,
            source_text=retrieved_text,
            allowed_chunk_ids=allowed_chunk_ids,
            n_questions=n_questions,
            language=language,
            stop_event=stop_event,
            use_internal_retrieval=False,
        )

        state["quiz"] = quiz
        return state