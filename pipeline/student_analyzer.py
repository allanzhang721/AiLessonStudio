"""
student_analyzer.py — Student weakness diagnostics from quiz interaction signals.

This module is intentionally lightweight so it can run without extra dependencies.
It aggregates quiz attempts at concept level and returns a ranked weakness report
with interpretable sub-scores and remediation suggestions.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from statistics import median
from typing import Any, TypedDict


class QuizAttempt(TypedDict, total=False):
    question_id: str
    question_text: str
    concept_tags: list[str]
    correct: bool
    response_time_seconds: float
    confidence: int


_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "do",
    "does",
    "for",
    "from",
    "how",
    "if",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "was",
    "what",
    "when",
    "where",
    "which",
    "why",
    "with",
}


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _normalize_text(text: str) -> list[str]:
    tokens = re.findall(r"[a-zA-Z]{3,}", text.lower())
    return [t for t in tokens if t not in _STOPWORDS]


def infer_concept_tags(
    question_text: str,
    explanation_text: str = "",
    subject: str = "",
    *,
    max_tags: int = 2,
) -> list[str]:
    """Infer coarse concept tags from question and explanation text.

    The function uses simple keyword frequency heuristics so it is deterministic,
    fast, and dependency-free for Streamlit inference.
    """
    tokens = _normalize_text(question_text)
    tokens.extend(_normalize_text(explanation_text)[:25])

    counts: dict[str, int] = defaultdict(int)
    for token in tokens:
        counts[token] += 1

    ranked = sorted(counts.items(), key=lambda x: (x[1], len(x[0])), reverse=True)
    tags = [word for word, _ in ranked[:max_tags] if word]

    if subject.strip():
        subject_tag = re.sub(r"\s+", "_", subject.strip().lower())
        if subject_tag and subject_tag not in tags:
            tags.append(subject_tag)

    if not tags:
        return ["general"]
    return tags[:max_tags]


def _weakness_type(*, accuracy: float, time_norm: float, conf_gap: float, repeat_error: float) -> str:
    if accuracy < 0.5 and conf_gap >= 0.45:
        return "misconception_risk"
    if accuracy < 0.65 and time_norm >= 0.6:
        return "missing_prerequisite"
    if repeat_error >= 0.35:
        return "recurring_error_pattern"
    if accuracy < 0.75 and conf_gap < 0.2:
        return "low_confidence"
    return "moderate_gap"


def _recommend_actions(concept: str, weakness_type: str) -> list[str]:
    if weakness_type == "misconception_risk":
        return [
            f"Add one counterexample frame for '{concept}' that contrasts a common wrong idea.",
            "Insert a targeted concept-check question immediately after the counterexample.",
        ]
    if weakness_type == "missing_prerequisite":
        return [
            f"Add a prerequisite recap before the main '{concept}' explanation.",
            "Slow down narration and split one complex frame into two simpler steps.",
        ]
    if weakness_type == "recurring_error_pattern":
        return [
            "Provide two spaced follow-up practice questions on the same concept.",
            "Add explicit 'why this option is wrong' feedback in quiz review.",
        ]
    if weakness_type == "low_confidence":
        return [
            "Use a simpler wording pass and reduce unnecessary visual clutter.",
            "Add one scaffolded hint before asking the same concept question again.",
        ]
    return [
        "Keep current lesson flow but add one reinforcement question.",
    ]


def analyze_student_weakness(
    attempts: list[QuizAttempt],
    *,
    checker2_result: dict[str, Any] | None = None,
    top_k: int = 3,
) -> dict[str, Any]:
    """Return a concept-level weakness report for one quiz session."""
    if not attempts:
        return {
            "status": "no_attempts",
            "overall_accuracy": 0.0,
            "overall_avg_response_seconds": 0.0,
            "concepts": [],
            "top_weak_concepts": [],
        }

    # Global response-time normalization baseline.
    response_values = [max(1.0, float(a.get("response_time_seconds", 1.0))) for a in attempts]
    time_ref = max(1.0, float(median(response_values)))

    content_risk = 0.0
    if checker2_result and isinstance(checker2_result, dict) and not checker2_result.get("error"):
        overall_score = float(checker2_result.get("overall_score", 0.0))
        per_frame = checker2_result.get("per_frame", [])
        failed_count = len([f for f in per_frame if not f.get("pass", False)]) if isinstance(per_frame, list) else 0
        frame_ratio = failed_count / max(1, len(per_frame)) if isinstance(per_frame, list) else 0.0
        content_risk = _clamp01(0.6 * (1.0 - overall_score) + 0.4 * frame_ratio)

    concept_buckets: dict[str, list[QuizAttempt]] = defaultdict(list)
    for attempt in attempts:
        tags = attempt.get("concept_tags") or ["general"]
        for tag in tags:
            concept_buckets[str(tag)].append(attempt)

    concept_rows: list[dict[str, Any]] = []
    for concept, items in concept_buckets.items():
        total = len(items)
        correct_count = sum(1 for item in items if bool(item.get("correct", False)))
        wrong_count = total - correct_count
        accuracy = correct_count / max(1, total)

        avg_time = sum(max(1.0, float(item.get("response_time_seconds", 1.0))) for item in items) / max(1, total)
        time_norm = _clamp01(math.log1p(avg_time / time_ref) / math.log1p(3.0))

        wrong_items = [item for item in items if not bool(item.get("correct", False))]
        if wrong_items:
            wrong_conf = sum(int(item.get("confidence", 3)) for item in wrong_items) / len(wrong_items)
        else:
            wrong_conf = 1.0
        conf_gap = _clamp01((wrong_conf - 1.0) / 4.0)

        repeat_error = _clamp01(max(0, wrong_count - 1) / max(1, total))
        # Weighted weakness score in [0,1].
        weakness_score = _clamp01(
            0.42 * (1.0 - accuracy)
            + 0.18 * time_norm
            + 0.18 * conf_gap
            + 0.12 * repeat_error
            + 0.10 * content_risk
        )
        weakness_type = _weakness_type(
            accuracy=accuracy,
            time_norm=time_norm,
            conf_gap=conf_gap,
            repeat_error=repeat_error,
        )

        concept_rows.append(
            {
                "concept": concept,
                "attempts": total,
                "accuracy": round(accuracy, 4),
                "avg_response_seconds": round(avg_time, 2),
                "confidence_gap": round(conf_gap, 4),
                "repeat_error": round(repeat_error, 4),
                "content_risk": round(content_risk, 4),
                "weakness_score": round(weakness_score, 4),
                "weakness_type": weakness_type,
                "recommended_actions": _recommend_actions(concept, weakness_type),
            }
        )

    concept_rows.sort(key=lambda x: x["weakness_score"], reverse=True)
    top_weak = concept_rows[: max(1, top_k)]

    overall_accuracy = sum(1 for item in attempts if bool(item.get("correct", False))) / max(1, len(attempts))
    overall_avg_time = sum(response_values) / max(1, len(response_values))

    return {
        "status": "ok",
        "overall_accuracy": round(overall_accuracy, 4),
        "overall_avg_response_seconds": round(overall_avg_time, 2),
        "content_risk": round(content_risk, 4),
        "concepts": concept_rows,
        "top_weak_concepts": top_weak,
    }
