"""Student-facing lesson generation helpers."""

from __future__ import annotations

import json
import re
from typing import Any

from .clients import chat_completion


def _extract_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    fence = chr(96) * 3
    if cleaned.startswith(fence):
        cleaned = re.sub(r"^.{3}(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*.{3}$", "", cleaned)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("The model did not return a JSON lesson.")
        value = json.loads(cleaned[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("The lesson response must be a JSON object.")
    return value


def _clean_text(value: Any, maximum: int = 5000) -> str:
    return " ".join(str(value or "").strip().split())[:maximum]


def _normalize_quiz(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    result: list[dict[str, Any]] = []
    for item in value[:5]:
        if not isinstance(item, dict):
            continue
        choices = item.get("choices")
        if not isinstance(choices, list) or len(choices) != 4:
            continue
        answer = str(item.get("answer", "")).strip().upper()
        if answer not in {"A", "B", "C", "D"}:
            continue
        result.append({
            "question": _clean_text(item.get("question"), 500),
            "choices": [_clean_text(choice, 300) for choice in choices],
            "answer": answer,
            "explanation": _clean_text(item.get("explanation"), 500),
            "concept": _clean_text(item.get("concept"), 100) or "core concept",
        })
    return result


def normalize_lesson_bundle(value: dict[str, Any]) -> dict[str, Any]:
    explanation = _clean_text(value.get("explanation"))
    if len(explanation.split()) < 60:
        raise ValueError("The generated explanation was too short to teach the concept.")
    quiz = _normalize_quiz(value.get("quiz"))
    if len(quiz) < 3:
        raise ValueError("The model did not return at least three valid quiz questions.")
    ideas = value.get("key_ideas") if isinstance(value.get("key_ideas"), list) else []
    return {
        "title": _clean_text(value.get("title"), 140) or "Visual lesson",
        "learning_objective": _clean_text(value.get("learning_objective"), 400),
        "explanation": explanation,
        "key_ideas": [_clean_text(item, 400) for item in ideas[:4] if _clean_text(item)],
        "worked_example": _clean_text(value.get("worked_example"), 1400),
        "common_mistake": _clean_text(value.get("common_mistake"), 600),
        "quick_check": _clean_text(value.get("quick_check"), 500),

        "quiz": quiz,
    }


def generate_lesson_bundle(client: Any, *, model: str, question: str, subject: str, grade: int, language: str = "English") -> dict[str, Any]:
    """Create the complete teaching text and quiz with a strict JSON contract."""
    prompt = f"""You are a careful high-school teacher and curriculum designer.
Create a lesson that directly answers the student's question.

Subject: {subject or "General"}
Grade: {grade}
Language: {language}
Question: {question}

Return one valid JSON object only with: title, learning_objective, explanation,
key_ideas, worked_example, common_mistake, quick_check, and quiz. The quiz must
contain exactly 5 objects. Each quiz object must have question, choices (exactly
four strings), answer (A, B, C, or D), explanation, and concept.

The explanation must be 120-260 words, accurate, logically complete, and direct.
Teach causal steps, define necessary technical terms, state important conditions,
and stay appropriate for Grade {grade}. Do not invent citations, links, statistics,
or named studies. Use {language} except for standard symbols and proper nouns.
Return JSON only."""
    return normalize_lesson_bundle(_extract_json(chat_completion(client, model, prompt)))


def lesson_bundle_to_quiz_markdown(bundle: dict[str, Any]) -> str:
    blocks: list[str] = []
    for index, item in enumerate(bundle.get("quiz", []), start=1):
        choices = item["choices"]
        blocks.append("\n".join([
            f"{index}. {item['question']}",
            f"A) {choices[0]}",
            f"B) {choices[1]}",
            f"C) {choices[2]}",
            f"D) {choices[3]}",
            f"Correct Answer: {item['answer']}",
            f"Explanation: {item['explanation']}",
        ]))
    return "\n\n".join(blocks)


def curated_resources(subject: str, question: str) -> list[dict[str, str]]:
    """Return stable, non-invented starting points instead of model-made URLs."""
    text = f"{subject} {question}".lower()
    resources = [
        {"name": "Khan Academy", "url": "https://www.khanacademy.org/", "description": "Free lessons and practice organized by school subject."},
        {"name": "CK-12", "url": "https://www.ck12.org/student/", "description": "Free high-school digital textbooks, simulations, and practice."},
    ]
    if any(word in text for word in ("physics", "force", "motion", "energy", "circuit")):
        resources += [
            {"name": "The Physics Classroom", "url": "https://www.physicsclassroom.com/", "description": "Clear high-school physics tutorials and interactive exercises."},
            {"name": "PhET", "url": "https://phet.colorado.edu/", "description": "Research-based interactive science and mathematics simulations."},
        ]
    elif any(word in text for word in ("biology", "cell", "ecology", "genetic")):
        resources += [
            {"name": "OpenStax Biology 2e", "url": "https://openstax.org/details/books/biology-2e", "description": "A peer-reviewed, openly licensed biology textbook."},
            {"name": "HHMI BioInteractive", "url": "https://www.biointeractive.org/", "description": "Classroom-ready biology interactives, videos, and data activities."},
        ]
    elif any(word in text for word in ("math", "algebra", "geometry", "calculus", "function")):
        resources += [
            {"name": "Desmos", "url": "https://www.desmos.com/", "description": "Interactive graphing tools for mathematical relationships."},
            {"name": "GeoGebra", "url": "https://www.geogebra.org/", "description": "Interactive geometry, algebra, and calculus visualizations."},
        ]
    else:
        resources.append({"name": "OpenStax", "url": "https://openstax.org/subjects", "description": "Free peer-reviewed textbooks across major high-school subjects."})
    return resources
