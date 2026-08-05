"""Educational quality gates used by the production pipeline."""

from __future__ import annotations

import json
import re
from typing import Any

from .clients import chat_completion


_WORD = re.compile(r"[A-Za-z][A-Za-z'-]*")


def _json_object(text: str) -> dict[str, Any]:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("Quality reviewer returned no JSON object.")
    value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("Quality review must be an object.")
    return value


def _local_text_checks(question: str, explanation: str) -> list[str]:
    issues: list[str] = []
    words = _WORD.findall(explanation)
    if len(words) < 70:
        issues.append("Explanation is too short to develop the reasoning.")
    if len(words) > 420:
        issues.append("Explanation is longer than a focused high-school lesson.")
    question_terms = {w.lower() for w in _WORD.findall(question) if len(w) >= 5}
    answer_terms = {w.lower() for w in words}
    if question_terms and not question_terms.intersection(answer_terms):
        issues.append("Explanation does not clearly use the question's key concepts.")
    sentences = [part for part in re.split(r"[.!?]+", explanation) if part.strip()]
    if sentences and len(words) / len(sentences) > 27:
        issues.append("Sentence length is likely too dense for the target audience.")
    return issues


def review_explanation(
    client: Any,
    *,
    model: str,
    question: str,
    explanation: str,
    grade: int,
    subject: str,
    max_repairs: int = 1,
) -> dict[str, Any]:
    """Audit, repair once if needed, and return a transparent gate result."""
    current = " ".join(explanation.split())
    rounds: list[dict[str, Any]] = []
    for attempt in range(max_repairs + 1):
        local_issues = _local_text_checks(question, current)
        prompt = f"""Act as an independent educational quality reviewer.

Subject: {subject or "General"}
Grade: {grade}
Question: {question}
Explanation: {current}

Score each dimension from 1 (unsafe/poor) to 4 (excellent):
accuracy, completeness, logical_flow, grade_fit, and clarity.
List concrete issues. Pass only if accuracy is 4 and every other score is at
least 3. If it fails, provide a corrected explanation of 120-260 words.
Do not add citations or facts you cannot verify.

Return JSON only:
{{"scores": {{"accuracy": 1, "completeness": 1, "logical_flow": 1,
"grade_fit": 1, "clarity": 1}}, "issues": ["..."], "pass": false,
"revised_explanation": "..."}}"""
        try:
            review = _json_object(chat_completion(client, model, prompt))
            scores = {
                key: max(1, min(4, int(review.get("scores", {}).get(key, 1))))
                for key in ("accuracy", "completeness", "logical_flow", "grade_fit", "clarity")
            }
            issues = [str(item).strip() for item in review.get("issues", []) if str(item).strip()]
            issues.extend(item for item in local_issues if item not in issues)
            passed = bool(review.get("pass")) and scores["accuracy"] == 4 and min(scores.values()) >= 3 and not local_issues
            revised = " ".join(str(review.get("revised_explanation", "")).split())
        except Exception as exc:
            scores = {"accuracy": 1, "completeness": 2, "logical_flow": 2, "grade_fit": 2, "clarity": 2}
            issues = local_issues + [f"Automated review failed: {type(exc).__name__}"]
            passed, revised = False, ""
        action = "accepted" if passed else ("revised" if revised and attempt < max_repairs else "blocked")
        rounds.append({"round": attempt + 1, "scores": scores, "issues": issues, "pass": passed, "action": action})
        if passed:
            break
        if action == "revised":
            current = revised
            continue
        break

    passed = bool(rounds and rounds[-1]["pass"])
    return {
        "checker_name": "pedagogical_quality_gate_v2",
        "pass": passed,
        "final_explanation": current,
        "was_revised": current != " ".join(explanation.split()),
        "rounds": rounds,
        "total_rounds": len(rounds),
        "issues": rounds[-1]["issues"] if rounds else ["No review was run."],
        "scores": rounds[-1]["scores"] if rounds else {},
    }


def local_explanation_review(question: str, explanation: str, grade: int, subject: str = "") -> dict[str, Any]:
    """Offline structural check for demos and unit tests; it does not claim fact checking."""
    issues = _local_text_checks(question, explanation)
    score = max(0.0, 1.0 - 0.2 * len(issues))
    return {
        "checker_name": "structural_quality_gate",
        "pass": not issues,
        "overall_score": round(score, 3),
        "issues": issues,
        "grade": grade,
        "subject": subject,
        "scope": "structure_only",
    }
