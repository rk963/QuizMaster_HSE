from typing import Any, Dict, List, Set
from .base_agent import BaseAgent
from services.quiz_generator_ollama import generate_quiz_ollama


def _simple_normalize(text: str) -> str:
    return " ".join(text.lower().strip().split())


class RepairAgent(BaseAgent):
    def __init__(self):
        super().__init__("RepairAgent")

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        current_quiz = state.get("quiz", [])
        requested = int(state.get("n_questions", 5))
        missing = requested - len(current_quiz)

        if missing <= 0:
            return state

        existing_question_keys: Set[str] = set()
        existing_questions_text: List[str] = []

        for q in current_quiz:
            question_text = str(q.get("question", "")).strip()
            if question_text:
                existing_questions_text.append(question_text)
                existing_question_keys.add(_simple_normalize(question_text))

        repaired_quiz = generate_quiz_ollama(
            text=state["text"],
            source_text=state["retrieved_text"],
            allowed_chunk_ids=state["allowed_chunk_ids"],
            n_questions=missing,
            language=state["language"],
            stop_event=state.get("stop_event"),
            existing_question_keys=existing_question_keys,
            existing_questions_text=existing_questions_text,
            use_internal_retrieval=False,
        )

        if isinstance(repaired_quiz, list):
            state["quiz"] = current_quiz + repaired_quiz

        return state