"""Fast, transparent educational quality gates for lesson text."""

from __future__ import annotations

import json
import re
import time
from typing import Any

from .clients import chat_completion
from .markdown_render import preserve_markdown


_WORD = re.compile(r"[A-Za-z][A-Za-z'-]*")
_DIMENSIONS = ("accuracy", "completeness", "logical_flow", "grade_fit", "clarity")


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


def _rounded_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000.0, 2)


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
    """Run a cheap local screen, then one compact semantic rubric and repair if needed."""
    total_started = time.perf_counter()
    current = preserve_markdown(explanation)
    original = current
    rounds: list[dict[str, Any]] = []
    local_latency = 0.0
    model_latency = 0.0
    model_calls = 0

    for attempt in range(max_repairs + 1):
        local_started = time.perf_counter()
        local_issues = _local_text_checks(question, current)
        local_ms = _rounded_ms(local_started)
        local_latency += local_ms
        prompt = f"""Independent high-school explanation audit.
Subject: {subject or "General"}; Grade: {grade}
Question: {question}
Explanation: {current}

Score 1-4: accuracy, completeness, logical_flow, grade_fit, clarity.
Pass only when accuracy=4 and every other score>=3. List only concrete issues.
If failing, return a corrected 120-260 word explanation. Preserve useful Markdown.
Format math only as `$...$` or `$$...$$`. Do not add citations.
JSON only:
{{"scores": {{"accuracy": 1, "completeness": 1, "logical_flow": 1,
"grade_fit": 1, "clarity": 1}}, "issues": ["..."], "pass": false,
"revised_explanation": "..."}}"""
        model_started = time.perf_counter()
        model_calls += 1
        try:
            review = _json_object(chat_completion(client, model, prompt))
            model_ms = _rounded_ms(model_started)
            scores = {
                key: max(1, min(4, int(review.get("scores", {}).get(key, 1))))
                for key in _DIMENSIONS
            }
            issues = [str(item).strip() for item in review.get("issues", []) if str(item).strip()]
            issues.extend(item for item in local_issues if item not in issues)
            passed = bool(review.get("pass")) and scores["accuracy"] == 4 and min(scores.values()) >= 3 and not local_issues
            revised = preserve_markdown(review.get("revised_explanation", ""))
        except Exception as exc:
            model_ms = _rounded_ms(model_started)
            scores = {"accuracy": 1, "completeness": 2, "logical_flow": 2, "grade_fit": 2, "clarity": 2}
            issues = local_issues + [f"Automated review failed: {type(exc).__name__}"]
            passed, revised = False, ""
        model_latency += model_ms
        action = "accepted" if passed else ("revised" if revised and attempt < max_repairs else "blocked")
        rounds.append({
            "round": attempt + 1,
            "scores": scores,
            "issues": issues,
            "pass": passed,
            "action": action,
            "latency_ms": {"local": local_ms, "semantic": model_ms},
        })
        if passed:
            break
        if action == "revised":
            current = revised
            continue
        break

    passed = bool(rounds and rounds[-1]["pass"])
    final_scores = rounds[-1]["scores"] if rounds else {}
    overall = sum(final_scores.values()) / (4.0 * len(_DIMENSIONS)) if final_scores else 0.0
    total_ms = _rounded_ms(total_started)
    return {
        "checker_name": "pedagogical_quality_gate_v3",
        "mode": "local_then_semantic_cascade",
        "pass": passed,
        "overall_score": round(overall, 4),
        "final_explanation": current,
        "was_revised": current != original,
        "rounds": rounds,
        "total_rounds": len(rounds),
        "issues": rounds[-1]["issues"] if rounds else ["No review was run."],
        "scores": final_scores,
        "metrics": {
            "total_latency_ms": total_ms,
            "local_latency_ms": round(local_latency, 2),
            "semantic_latency_ms": round(model_latency, 2),
            "model_calls": model_calls,
            "repairs": int(current != original),
        },
        "method_comparison": [
            {"method": "Local structural screen", "trained": False, "latency_ms": round(local_latency, 2), "coverage": "length, focus, relevance, readability"},
            {"method": "LLM rubric review", "trained": False, "latency_ms": round(model_latency, 2), "coverage": "accuracy, completeness, logic, grade fit, clarity"},
        ],
        "trained_model": {
            "used": False,
            "reason": "Saved text classifiers predict error categories, not whether an explanation is correct; they remain evaluation baselines.",
        },
    }


def local_explanation_review(question: str, explanation: str, grade: int, subject: str = "") -> dict[str, Any]:
    """Offline millisecond structural screen; it does not claim fact checking."""
    started = time.perf_counter()
    issues = _local_text_checks(question, explanation)
    score = max(0.0, 1.0 - 0.2 * len(issues))
    latency_ms = _rounded_ms(started)
    return {
        "checker_name": "structural_quality_gate_v2",
        "mode": "local_only",
        "pass": not issues,
        "overall_score": round(score, 3),
        "issues": issues,
        "grade": grade,
        "subject": subject,
        "scope": "structure_only",
        "metrics": {"total_latency_ms": latency_ms, "model_calls": 0},
        "trained_model": {"used": False},
    }
