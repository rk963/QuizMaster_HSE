from typing import Any, Dict
from .base_agent import BaseAgent
from services.quiz_generator_ollama import generate_quiz_ollama


class GeneratorAgent(BaseAgent):
    def __init__(self):
        super().__init__("GeneratorAgent")

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        quiz = generate_quiz_ollama(
            text=state["text"],
            source_text=state["retrieved_text"],
            allowed_chunk_ids=state["allowed_chunk_ids"],
            n_questions=state["n_questions"],
            language=state["language"],
            stop_event=state.get("stop_event"),
            use_internal_retrieval=False,
        )
        state["quiz"] = quiz
        return state