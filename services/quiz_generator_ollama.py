import json
import os
import re
from threading import Event
from typing import Any, Dict, List, Optional, Set, Tuple

import requests

from services.chunking import select_top_chunks


class GenerationStoppedError(Exception):
    """Raised when quiz generation is manually stopped."""
    pass


OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
MODEL_NAME = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")

MAX_QUESTIONS = 10  # hard cap enforced in generate_quiz_ollama()


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def _compact_text(text: str, max_chars: int = 12000) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars]


def _normalize(text: str) -> str:
    """Lowercase, strip punctuation and extra whitespace — used for dedup."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _keyword_set(text: str) -> Set[str]:
    """Return the significant words of a question (common stop-words removed)."""
    stopwords = {
        # English
        "a", "an", "the", "is", "are", "was", "were", "what", "which",
        "how", "why", "when", "where", "who", "of", "in", "on", "at",
        "to", "for", "and", "or", "be", "has", "have", "had",
        # Russian
        "с", "в", "на", "из", "и", "или", "что", "как", "где",
        "когда", "какой", "какая", "какие", "это", "не", "по", "за",
    }
    words = _normalize(text).split()
    return {w for w in words if w not in stopwords and len(w) > 2}


def _is_too_similar(new_q: str, existing_questions: List[str], threshold: float = 0.55) -> bool:
    """
    Returns True if new_q shares too many keywords with any existing question
    (Jaccard similarity >= threshold).
    """
    new_kw = _keyword_set(new_q)
    if not new_kw:
        return False

    for eq in existing_questions:
        eq_kw = _keyword_set(eq)
        if not eq_kw:
            continue
        intersection = new_kw & eq_kw
        union = new_kw | eq_kw
        if len(union) == 0:
            continue
        if len(intersection) / len(union) >= threshold:
            return True

    return False


# ---------------------------------------------------------------------------
# Ollama API
# ---------------------------------------------------------------------------

def _check_stop(stop_event: Optional[Event]) -> None:
    if stop_event is not None and stop_event.is_set():
        raise GenerationStoppedError("Generation stopped by user.")


def _looks_like_letter_choices(choices: List[str]) -> bool:
    norm = [c.strip().upper() for c in choices]
    return set(norm) <= {"A", "B", "C", "D"} and len(norm) == 4


def _extract_json_array(text: str) -> Any:
    text = text.strip()
    text = re.sub(r"^```json", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"^```", "", text).strip()
    text = re.sub(r"```$", "", text).strip()

    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Model did not return a JSON array.")

    return json.loads(text[start:end + 1])


def _ollama_chat(
    prompt: str,
    temperature: float = 0.2,
    stop_event: Optional[Event] = None,
    model_name: Optional[str] = None,
    num_ctx: int = 8192,
    num_predict: int = 3072,
    timeout: int = 900,
) -> str:
    payload = {
        "model": model_name or MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a precise quiz generation system. "
                    "Output only valid JSON. "
                    "No markdown. No explanations outside JSON."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "stream": True,
        "options": {
            "temperature": temperature,
            "top_p": 0.9,
            "num_ctx": num_ctx,
            "num_predict": num_predict,
        },
    }

    _check_stop(stop_event)

    output_parts: List[str] = []

    with requests.post(OLLAMA_URL, json=payload, stream=True, timeout=timeout) as response:
        response.raise_for_status()

        for line in response.iter_lines(decode_unicode=True):
            _check_stop(stop_event)

            if not line:
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            message = data.get("message", {})
            content = message.get("content", "")

            if content:
                output_parts.append(content)

            if data.get("done", False):
                break

    _check_stop(stop_event)
    return "".join(output_parts).strip()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_quiz(
    items: Any,
    n_questions: int,
    allowed_chunk_ids: Set[int],
    existing_question_keys: Set[str],
    existing_questions_text: List[str],
) -> Tuple[List[Dict[str, Any]], bool]:
    """
    Validate model output and return (new_valid_questions, needs_retry).
    Applies both exact-key dedup and semantic (Jaccard) dedup.
    Also rejects questions that are not grounded in allowed source chunks.
    """
    if not isinstance(items, list):
        raise ValueError("Model output is not a list.")

    out: List[Dict[str, Any]] = []
    needs_retry = False

    for it in items:
        if not isinstance(it, dict):
            needs_retry = True
            continue

        q = str(it.get("question", "")).strip()
        choices = it.get("choices", [])
        correct = str(it.get("correct_answer", "")).strip()
        explanation = str(it.get("explanation", "")).strip()

        # ---- structural checks ----
        if not q or not isinstance(choices, list) or len(choices) != 4:
            needs_retry = True
            continue

        choices = [str(c).strip() for c in choices]

        if len(set(c.lower() for c in choices)) < 4:
            needs_retry = True
            continue

        if _looks_like_letter_choices(choices):
            needs_retry = True
            continue

        if any(len(c) < 3 for c in choices):
            needs_retry = True
            continue

        # ---- correct_answer match with fuzzy heal ----
        if correct not in choices:
            correct_lower = correct.lower()
            matched = next((c for c in choices if c.lower() == correct_lower), None)
            if matched:
                correct = matched
            else:
                needs_retry = True
                continue

        # ---- exact dedup ----
        q_key = _normalize(q)
        if q_key in existing_question_keys:
            continue

        # ---- semantic dedup ----
        all_so_far = existing_questions_text + [r["question"] for r in out]
        if _is_too_similar(q, all_so_far):
            needs_retry = True
            continue

        # ---- source chunks grounding ----
        source_chunks = it.get("source_chunks", [])
        if not isinstance(source_chunks, list):
            source_chunks = []

        seen_ids: Set[int] = set()
        normalized_chunks: List[int] = []
        for x in source_chunks:
            try:
                xi = int(x)
                if xi in allowed_chunk_ids and xi not in seen_ids:
                    normalized_chunks.append(xi)
                    seen_ids.add(xi)
            except Exception:
                continue

        # Reject ungrounded questions
        if not normalized_chunks:
            needs_retry = True
            continue

        # Keep explanation compact
        if len(explanation) > 240:
            explanation = explanation[:240].rsplit(" ", 1)[0] + "..."

        out.append(
            {
                "id": 0,  # renumbered later
                "question": q,
                "choices": choices,
                "correct_answer": correct,
                "explanation": explanation,
                "source_chunks": normalized_chunks,
            }
        )
        existing_question_keys.add(q_key)

        if len(out) >= n_questions:
            break

    return out, needs_retry


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _build_prompt(
    source_text: str,
    language: str,
    n_questions: int,
    existing_questions: Optional[List[str]] = None,
) -> str:
    already_used = ""
    if existing_questions:
        lines = "\n".join(f"  {i + 1}. {q}" for i, q in enumerate(existing_questions))
        already_used = f"""
ALREADY GENERATED QUESTIONS — do NOT repeat, rephrase, or paraphrase ANY of these:
{lines}

Each new question MUST cover a DIFFERENT fact, concept, or detail from the source text.
"""

    grounding_rules = """
10. Use ONLY information explicitly present in the SOURCE TEXT.
11. Do NOT invent facts, names, dates, definitions, or explanations not supported by the SOURCE TEXT.
12. For every question, source_chunks must contain at least one valid chunk ID from the provided source.
13. Do NOT create generic world-knowledge questions. Every question must be grounded in the provided source.
"""

    if language == "Russian":
        return f"""
You are a precise quiz generator that ALWAYS outputs in the requested LANGUAGE.

TARGET LANGUAGE: Russian
Everything below MUST be written EXCLUSIVELY in Russian:
  - Questions, all 4 answer choices, correct answer, explanation.

If the SOURCE TEXT is in a different language, translate facts accurately into Russian.

STRICT RULES:
1. Generate EXACTLY {n_questions} multiple-choice questions from the SOURCE TEXT only.
2. Each question must have exactly 4 full-text answer choices — NEVER single letters like A/B/C/D.
3. "correct_answer" must be the EXACT full text of one of the 4 choices (copy it precisely).
4. Explanation: 1-2 sentences in Russian, grounded in the source text.
5. Every question must test a DIFFERENT fact or concept — absolutely no near-duplicates.
6. Spread questions across as many different topics in the source as possible.
7. Include "source_chunks": [list of integer chunk IDs used] for each question.
8. Output ONLY a valid JSON array — no markdown fences, no extra text before or after.
9. Questions must be clear, factual, and directly answerable from the source.
{grounding_rules}
{already_used}
OUTPUT FORMAT:
[
  {{
    "question": "...",
    "choices": ["choice 1", "choice 2", "choice 3", "choice 4"],
    "correct_answer": "choice 2",
    "explanation": "...",
    "source_chunks": [0]
  }}
]

SOURCE TEXT (chunks with IDs):
{source_text}
""".strip()

    return f"""
You are a precise quiz generator that ALWAYS outputs in the requested LANGUAGE.

TARGET LANGUAGE: English
Everything below MUST be written EXCLUSIVELY in English:
  - Questions, all 4 answer choices, correct answer, explanation.

If the SOURCE TEXT is in a different language, translate facts accurately into English.

STRICT RULES:
1. Generate EXACTLY {n_questions} multiple-choice questions from the SOURCE TEXT only.
2. Each question must have exactly 4 full-text answer choices — NEVER single letters like A/B/C/D.
3. "correct_answer" must be the EXACT full text of one of the 4 choices (copy it precisely).
4. Explanation: 1-2 sentences in English, grounded in the source text.
5. Every question must test a DIFFERENT fact or concept — absolutely no near-duplicates.
6. Spread questions across as many different topics in the source as possible.
7. Include "source_chunks": [list of integer chunk IDs used] for each question.
8. Output ONLY a valid JSON array — no markdown fences, no extra text before or after.
9. Questions must be clear, factual, and directly answerable from the source.
{grounding_rules}
{already_used}
OUTPUT FORMAT:
[
  {{
    "question": "...",
    "choices": ["choice 1", "choice 2", "choice 3", "choice 4"],
    "correct_answer": "choice 2",
    "explanation": "...",
    "source_chunks": [0]
  }}
]

SOURCE TEXT (chunks with IDs):
{source_text}
""".strip()


# ---------------------------------------------------------------------------
# Source preparation helpers
# ---------------------------------------------------------------------------

def prepare_generation_source(
    text: str,
    n_questions: int,
    top_k: Optional[int] = None,
    max_source_chars: Optional[int] = None,
) -> Tuple[str, Set[int], List[Tuple[int, str]]]:
    """
    Prepare compact source text plus allowed chunk IDs.
    Useful for multi-agent pipelines where retrieval happens before generation.
    """
    n_questions = min(max(1, int(n_questions)), MAX_QUESTIONS)
    query_seed = text[:400]

    effective_top_k = top_k if top_k is not None else max(6, n_questions + 2)
    top_chunks = select_top_chunks(text, query_seed, top_k=effective_top_k)

    allowed_chunk_ids = {idx for idx, _ in top_chunks}
    source_text = "\n\n---\n\n".join(f"CHUNK {idx}:\n{ch}" for idx, ch in top_chunks)

    effective_max_chars = (
        max_source_chars
        if max_source_chars is not None
        else min(14400, 1600 * n_questions)
    )
    source_text = _compact_text(source_text, max_chars=effective_max_chars)

    return source_text, allowed_chunk_ids, top_chunks


# ---------------------------------------------------------------------------
# Core generation from prepared source
# ---------------------------------------------------------------------------

def generate_quiz_from_source_text(
    source_text: str,
    n_questions: int = 5,
    language: str = "English",
    stop_event: Optional[Event] = None,
    allowed_chunk_ids: Optional[Set[int]] = None,
    existing_question_keys: Optional[Set[str]] = None,
    existing_questions_text: Optional[List[str]] = None,
    temperature_schedule: Optional[List[float]] = None,
    model_name: Optional[str] = None,
    num_ctx: int = 8192,
    num_predict: int = 3072,
    timeout: int = 900,
) -> List[Dict[str, Any]]:
    """
    Generate quiz from already-prepared source text.
    Best entry point for multi-agent pipelines.
    """
    _check_stop(stop_event)

    n_questions = min(max(1, int(n_questions)), MAX_QUESTIONS)
    allowed_chunk_ids = allowed_chunk_ids or set()
    existing_question_keys = existing_question_keys or set()
    existing_questions_text = list(existing_questions_text or [])

    final_quiz: List[Dict[str, Any]] = []

    temp_schedule = temperature_schedule or [0.2, 0.35, 0.5, 0.65]

    for temperature in temp_schedule:
        _check_stop(stop_event)

        remaining = n_questions - len(final_quiz)
        if remaining <= 0:
            break

        current_existing_questions = existing_questions_text + [q["question"] for q in final_quiz]

        prompt = _build_prompt(
            source_text=source_text,
            language=language,
            n_questions=remaining,
            existing_questions=current_existing_questions,
        )

        try:
            raw = _ollama_chat(
                prompt,
                temperature=temperature,
                stop_event=stop_event,
                model_name=model_name,
                num_ctx=num_ctx,
                num_predict=num_predict,
                timeout=timeout,
            )
            data = _extract_json_array(raw)
        except GenerationStoppedError:
            raise
        except Exception:
            continue

        new_questions, needs_retry = _validate_quiz(
            data,
            n_questions=remaining,
            allowed_chunk_ids=allowed_chunk_ids,
            existing_question_keys=existing_question_keys,
            existing_questions_text=current_existing_questions,
        )

        final_quiz.extend(new_questions)

        if len(final_quiz) >= n_questions:
            break

        if not needs_retry and not new_questions:
            break

    for i, q in enumerate(final_quiz, start=1):
        q["id"] = i

    return final_quiz[:n_questions]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_quiz_ollama(
    text: str,
    n_questions: int = 5,
    language: str = "English",
    stop_event: Optional[Event] = None,
    *,
    source_text: Optional[str] = None,
    allowed_chunk_ids: Optional[Set[int]] = None,
    existing_question_keys: Optional[Set[str]] = None,
    existing_questions_text: Optional[List[str]] = None,
    temperature_schedule: Optional[List[float]] = None,
    use_internal_retrieval: bool = True,
    top_k: Optional[int] = None,
    max_source_chars: Optional[int] = None,
    model_name: Optional[str] = None,
    num_ctx: int = 8192,
    num_predict: int = 3072,
    timeout: int = 900,
) -> List[Dict[str, Any]]:
    """
    Backward-compatible public function.

    Old usage:
        generate_quiz_ollama(text=raw_text, ...)

    Multi-agent usage:
        generate_quiz_ollama(
            text=raw_text,
            source_text=prepared_source,
            allowed_chunk_ids=allowed_ids,
            use_internal_retrieval=False,
            ...
        )
    """
    _check_stop(stop_event)

    n_questions = min(max(1, int(n_questions)), MAX_QUESTIONS)

    prepared_source_text = source_text
    prepared_allowed_chunk_ids = allowed_chunk_ids

    if prepared_source_text is None or (use_internal_retrieval and prepared_allowed_chunk_ids is None):
        prepared_source_text, prepared_allowed_chunk_ids, _ = prepare_generation_source(
            text=text,
            n_questions=n_questions,
            top_k=top_k,
            max_source_chars=max_source_chars,
        )

    return generate_quiz_from_source_text(
        source_text=prepared_source_text,
        n_questions=n_questions,
        language=language,
        stop_event=stop_event,
        allowed_chunk_ids=prepared_allowed_chunk_ids or set(),
        existing_question_keys=existing_question_keys,
        existing_questions_text=existing_questions_text,
        temperature_schedule=temperature_schedule,
        model_name=model_name,
        num_ctx=num_ctx,
        num_predict=num_predict,
        timeout=timeout,
    )