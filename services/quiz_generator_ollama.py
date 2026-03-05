# services/quiz_generator_ollama.py
# Step C implemented: each question includes "source_chunks" (chunk IDs used)

import json
import requests
import re
from typing import Any, Dict, List, Tuple

from services.chunking import select_top_chunks

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "qwen2.5:14b"


def _compact_text(text: str, max_chars: int = 12000) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars]


def _looks_like_letter_choices(choices: List[str]) -> bool:
    norm = [c.strip().upper() for c in choices]
    return set(norm) <= {"A", "B", "C", "D"} and len(norm) == 4


def _extract_json_array(text: str) -> Any:
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Model did not return a JSON array.")
    return json.loads(text[start : end + 1])


def _ollama_chat(prompt: str, temperature: float = 0.2) -> str:
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are a precise quiz generation system. Output only valid JSON."},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {"temperature": temperature},
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=300)
    r.raise_for_status()
    return r.json()["message"]["content"].strip()


def _validate_quiz(items: Any, n_questions: int, allowed_chunk_ids: set[int]) -> Tuple[List[Dict[str, Any]], bool]:
    """
    Returns (quiz, needs_retry). needs_retry=True if we detect low-quality output.
    Step C: validates and keeps `source_chunks`.
    """
    if not isinstance(items, list):
        raise ValueError("Model output is not a list.")

    out: List[Dict[str, Any]] = []
    needs_retry = False

    for it in items:
        if not isinstance(it, dict):
            continue

        q = str(it.get("question", "")).strip()
        choices = it.get("choices", [])
        correct = str(it.get("correct_answer", "")).strip()
        explanation = str(it.get("explanation", "")).strip()

        source_chunks = it.get("source_chunks", [])
        if not isinstance(source_chunks, list):
            source_chunks = []
        # normalize chunk ids to ints and keep only allowed ones
        normalized_chunks: List[int] = []
        for x in source_chunks:
            try:
                xi = int(x)
                if xi in allowed_chunk_ids:
                    normalized_chunks.append(xi)
            except Exception:
                continue
        # de-dup and keep stable order
        seen = set()
        normalized_chunks = [x for x in normalized_chunks if not (x in seen or seen.add(x))]

        if not q or not isinstance(choices, list) or len(choices) != 4:
            continue

        choices = [str(c).strip() for c in choices]

        # Reject letter-only choices
        if _looks_like_letter_choices(choices):
            needs_retry = True
            continue

        # Reject very short options
        if any(len(c) < 3 for c in choices):
            needs_retry = True
            continue

        # Correct answer must be one of the choices (full text)
        if correct not in choices:
            if correct.strip().upper() in {"A", "B", "C", "D"}:
                needs_retry = True
            continue

        out.append(
            {
                "id": len(out) + 1,
                "question": q,
                "choices": choices,
                "correct_answer": correct,
                "explanation": explanation,
                "source_chunks": normalized_chunks,  # Step C
            }
        )

        if len(out) >= n_questions:
            break

    if not out:
        return [], True

    return out, needs_retry


def generate_quiz_ollama(text: str, n_questions: int = 5, language: str = "English") -> List[Dict[str, Any]]:
    # Step B (RAG-lite): select top chunks
    query_seed = text[:400]
    top_chunks = select_top_chunks(text, query_seed, top_k=6)

    allowed_chunk_ids = {idx for idx, _ in top_chunks}

    source_text = "\n\n---\n\n".join([f"CHUNK {idx}:\n{ch}" for idx, ch in top_chunks])
    source_text = _compact_text(source_text, max_chars=12000)

    prompt = f"""
LANGUAGE: {language}

TASK:
Create up to {n_questions} multiple-choice questions (MCQ) based ONLY on the SOURCE TEXT chunks.

STRICT RULES (VERY IMPORTANT):
- Use ONLY information from the SOURCE TEXT.
- Each question must have exactly 4 answer choices.
- Each choice MUST be a full text answer (at least 3 characters), NOT just "A", "B", "C", "D".
- Do NOT output letter labels as choices.
- correct_answer MUST be the full text of the correct option and must match one of the choices exactly.
- Provide a short explanation grounded in the SOURCE TEXT.
- Avoid trivial keyword presence questions; prefer concepts/definitions/understanding.
- IMPORTANT (Step C): Add "source_chunks": an array of CHUNK numbers you used to answer (e.g., [0, 3]).
  Only use CHUNK ids that appear in the SOURCE TEXT below.

OUTPUT FORMAT:
Return ONLY valid JSON (no markdown, no comments), exactly like:

[
  {{
    "question": "Full question text",
    "choices": ["Option 1 text", "Option 2 text", "Option 3 text", "Option 4 text"],
    "correct_answer": "Option 2 text",
    "explanation": "Short explanation based on the SOURCE TEXT",
    "source_chunks": [0, 2]
  }}
]

SOURCE TEXT:
{source_text}
""".strip()

    # First attempt
    raw = _ollama_chat(prompt, temperature=0.2)
    data = _extract_json_array(raw)
    quiz, needs_retry = _validate_quiz(data, n_questions, allowed_chunk_ids)

    # Retry once if needed (force compliance)
    if needs_retry or len(quiz) < max(1, n_questions // 2):
        retry_prompt = prompt + "\n\nFINAL REMINDER: choices must be real text options (not A/B/C/D) and include source_chunks."
        raw2 = _ollama_chat(retry_prompt, temperature=0.0)
        data2 = _extract_json_array(raw2)
        quiz2, _ = _validate_quiz(data2, n_questions, allowed_chunk_ids)
        if quiz2:
            return quiz2

    return quiz