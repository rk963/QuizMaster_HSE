from typing import Any, Dict, List, Optional

from services.quiz_generator_ollama import GenerationStoppedError
from .planner_agent import PlannerAgent
from .retriever_agent import RetrieverAgent
from .generator_agent import GeneratorAgent
from .validator_agent import ValidatorAgent
from .repair_agent import RepairAgent


def _ensure_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Make sure required keys always exist, so later stages do not fail
    because of missing fields.
    """
    defaults = {
        "plan": None,
        "retrieved_chunks": [],
        "retrieved_text": "",
        "allowed_chunk_ids": set(),
        "quiz": [],
        "validation_errors": [],
        "is_valid": False,
        "valid_count": 0,
    }
    for key, value in defaults.items():
        if key not in state:
            state[key] = value
    return state


def _check_stop(state: Dict[str, Any]) -> None:
    stop_event = state.get("stop_event")
    if stop_event is not None and stop_event.is_set():
        raise GenerationStoppedError("Generation stopped by user.")


def run_quiz_pipeline(
    text: str,
    n_questions: int = 5,
    language: str = "English",
    stop_event=None,
    file_bytes: Optional[bytes] = None,
) -> List[Dict[str, Any]]:
    state: Dict[str, Any] = {
        "text": text,
        "file_bytes": file_bytes,
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

    state = _ensure_state(state)
    _check_stop(state)

    # Stage 1 — PlannerAgent
    state = PlannerAgent().run(state)
    state = _ensure_state(state)
    _check_stop(state)

    # Stage 2 — RetrieverAgent
    state = RetrieverAgent().run(state)
    state = _ensure_state(state)
    _check_stop(state)

    # Stage 3 — GeneratorAgent
    state = GeneratorAgent().run(state)
    state = _ensure_state(state)
    _check_stop(state)

    # Stage 4 — ValidatorAgent
    state = ValidatorAgent().run(state)
    state = _ensure_state(state)
    _check_stop(state)

    # Stage 5 — RepairAgent only if still short on valid questions
    if not state.get("is_valid", False):
        if state.get("valid_count", 0) < n_questions:
            state = RepairAgent().run(state)
            state = _ensure_state(state)
            _check_stop(state)

            state = ValidatorAgent().run(state)
            state = _ensure_state(state)
            _check_stop(state)

    return state.get("quiz", [])[:n_questions]