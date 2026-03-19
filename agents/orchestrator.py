from typing import Any, Dict, List, Optional

from .planner_agent import PlannerAgent
from .retriever_agent import RetrieverAgent
from .generator_agent import GeneratorAgent
from .validator_agent import ValidatorAgent
from .repair_agent import RepairAgent


def run_quiz_pipeline(
    text: str,
    n_questions: int = 5,
    language: str = "English",
    stop_event=None,
    file_bytes: Optional[bytes] = None,   # NEW: enables chunk cache
) -> List[Dict[str, Any]]:
    state: Dict[str, Any] = {
        "text": text,
        "file_bytes": file_bytes,          # passed to RetrieverAgent for cache
        "n_questions": n_questions,
        "language": language,
        "stop_event": stop_event,
        "plan": None,
        "retrieved_chunks": [],
        "retrieved_text": "",
        "allowed_chunk_ids": set(),
        "quiz": [],
        "validation_errors": [],
        "is_valid": False,
        "valid_count": 0,
    }

    # Stage 1 — PlannerAgent  (pure logic, no I/O, instant)
    state = PlannerAgent().run(state)

    # Stage 2 — RetrieverAgent  (CPU, uses chunk cache if file_bytes set)
    state = RetrieverAgent().run(state)

    # Stage 3 — GeneratorAgent  (LLM call — the only slow step)
    state = GeneratorAgent().run(state)

    # Stage 4 — ValidatorAgent
    state = ValidatorAgent().run(state)

    # Stage 5 — RepairAgent only when we're actually short on questions
    if not state.get("is_valid", False):
        if state.get("valid_count", 0) < n_questions:
            state = RepairAgent().run(state)
            state = ValidatorAgent().run(state)

    return state.get("quiz", [])[:n_questions]
