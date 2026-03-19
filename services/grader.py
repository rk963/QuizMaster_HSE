from typing import List, Dict, Any


def grade_quiz(quiz: List[Dict[str, Any]], user_answers: Dict[int, str]) -> Dict[str, Any]:
    total = len(quiz)
    correct_count = 0
    details = []

    for q in quiz:
        qid = q["id"]
        correct = q["correct_answer"]
        chosen = user_answers.get(qid, None)

        is_correct = (chosen == correct)
        if is_correct:
            correct_count += 1

        details.append({
            "id": qid,
            "question": q["question"],
            "chosen": chosen,
            "correct": correct,
            "is_correct": is_correct,
            "explanation": q.get("explanation", "")
        })

    score_pct = (correct_count / total * 100) if total else 0
    return {"total": total, "correct": correct_count, "score_pct": score_pct, "details": details}