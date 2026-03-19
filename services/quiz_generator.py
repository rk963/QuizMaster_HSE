# services/quiz_generator.py

import os
import re
import time
import random
from typing import Any, Dict, List, Optional

from google import genai
from google.genai import types

MODEL_NAME = "gemini-flash-latest"


# -----------------------------
# Helpers
# -----------------------------
def _compact_text(text: str, max_chars: int = 12000) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars]


def _is_transient_error(e: Exception) -> bool:
    msg = str(e)
    return (
        "503" in msg
        or "UNAVAILABLE" in msg
        or "429" in msg
        or "RESOURCE_EXHAUSTED" in msg
        or "Rate limit" in msg
    )


def _call_gemini_with_retry(
    client: genai.Client,
    *,
    prompt: str,
    config: types.GenerateContentConfig,
    attempts: int = 5,
) -> Any:
    last_err: Optional[Exception] = None
    for i in range(attempts):
        try:
            return client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=config,
            )
        except Exception as e:
            last_err = e
            if not _is_transient_error(e) or i == attempts - 1:
                raise
            sleep_s = min(2 ** i, 16) + random.random()
            time.sleep(sleep_s)
    raise last_err  # should not reach


def _validate_quiz(items: Any, n_questions: int) -> List[Dict[str, Any]]:
    """
    items is expected to be a list of dicts:
    {question:str, choices:[str,str,str,str], correct_answer:str, explanation:str}
    """
    if not isinstance(items, list):
        raise ValueError("Model output is not a list of questions.")

    out: List[Dict[str, Any]] = []
    for it in items:
        if not isinstance(it, dict):
            continue

        q = str(it.get("question", "")).strip()
        choices = it.get("choices", [])
        correct = str(it.get("correct_answer", "")).strip()
        explanation = str(it.get("explanation", "")).strip()

        if not q:
            continue
        if not isinstance(choices, list) or len(choices) != 4:
            continue

        choices = [str(c).strip() for c in choices]
        if correct not in choices:
            continue

        out.append(
            {
                "id": len(out) + 1,
                "question": q,
                "choices": choices,
                "correct_answer": correct,
                "explanation": explanation,
            }
        )

        if len(out) >= n_questions:
            break

    if not out:
        raise ValueError("No valid questions were produced.")
    return out


# -----------------------------
# Function calling schema
# -----------------------------
def _quiz_function_declaration() -> types.FunctionDeclaration:
    return types.FunctionDeclaration(
        name="create_quiz",
        description="Create a multiple-choice quiz strictly based on the provided source text.",
        parameters={
            "type": "OBJECT",
            "properties": {
                "questions": {
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "question": {"type": "STRING"},
                            "choices": {
                                "type": "ARRAY",
                                "items": {"type": "STRING"},
                                "minItems": 4,
                                "maxItems": 4,
                            },
                            "correct_answer": {"type": "STRING"},
                            "explanation": {"type": "STRING"},
                        },
                        "required": ["question", "choices", "correct_answer", "explanation"],
                    },
                }
            },
            "required": ["questions"],
        },
    )


def _extract_function_call_args(resp: Any) -> Optional[Dict[str, Any]]:
    """
    Try to read function call args from the response.
    google-genai responses typically store tool calls in candidates[0].content.parts
    """
    try:
        cand = resp.candidates[0]
        parts = cand.content.parts
        for p in parts:
            # function_call may exist as attribute
            fc = getattr(p, "function_call", None)
            if fc and getattr(fc, "name", "") == "create_quiz":
                return fc.args  # dict
    except Exception:
        return None
    return None


# -----------------------------
# Public API
# -----------------------------
def generate_quiz_gemini(
    text: str,
    n_questions: int = 5,
    language: str = "English",
    api_key: Optional[str] = None,
) -> List[Dict[str, Any]]:
    api_key = api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Missing GEMINI_API_KEY.")

    client = genai.Client(api_key=api_key)
    source_text = _compact_text(text)

    system_instruction = (
        "You are QuizMaster. Create high-quality multiple-choice questions strictly from SOURCE TEXT. "
        "Do not use outside knowledge. If the text is insufficient, return fewer questions."
    )

    prompt = f"""
LANGUAGE: {language}

TASK:
Create up to {n_questions} MCQ questions based ONLY on the SOURCE TEXT.

STRICT RULES:
- Every question must be answerable from the SOURCE TEXT only.
- Exactly 4 options per question.
- correct_answer must match one option exactly.
- explanation must be grounded in the SOURCE TEXT (no outside facts).
- Avoid trivial 'keyword presence' questions. Prefer concepts/definitions/understanding.
- Return the result by calling the function create_quiz(questions=[...]).
- If the SOURCE TEXT is too short, return fewer questions.

SOURCE TEXT:
{source_text}
""".strip()

    tool = types.Tool(function_declarations=[_quiz_function_declaration()])

    cfg = types.GenerateContentConfig(
        system_instruction=system_instruction,
        tools=[tool],
        # Force the model to use function calling
        tool_config=types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(
                mode="ANY",
                allowed_function_names=["create_quiz"],
            )
        ),
        temperature=0.2,
        max_output_tokens=1800,
    )

    resp = _call_gemini_with_retry(client, prompt=prompt, config=cfg, attempts=5)

    args = _extract_function_call_args(resp)
    if not args or "questions" not in args:
        raise ValueError("Model did not return a function call with questions.")

    questions = args["questions"]
    return _validate_quiz(questions, n_questions)