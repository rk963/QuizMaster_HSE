from typing import Any, Dict
from .base_agent import BaseAgent


class PlannerAgent(BaseAgent):
    def __init__(self):
        super().__init__("PlannerAgent")

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        n_questions = int(state.get("n_questions", 5))
        text = state.get("text", "")

        text_len = len(text)

        if text_len < 2500:
            difficulty = "easy"
        elif text_len < 12000:
            difficulty = "medium"
        else:
            difficulty = "medium-hard"

        fact_count = max(1, round(n_questions * 0.3))
        concept_count = max(1, round(n_questions * 0.4))
        application_count = max(1, n_questions - fact_count - concept_count)

        if fact_count + concept_count + application_count > n_questions:
            application_count = max(0, n_questions - fact_count - concept_count)

        state["plan"] = {
            "difficulty": difficulty,
            "distribution": {
                "fact": fact_count,
                "concept": concept_count,
                "application": application_count,
            },
            "focus": [
                "key definitions",
                "important concepts",
                "practical understanding",
            ],
            "avoid": [
                "duplicate questions",
                "trivial wording",
                "outside knowledge",
                "very similar answer options",
            ],
        }
        return state