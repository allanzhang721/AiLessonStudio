"""Student-facing lesson generation and grounded research helpers."""

from __future__ import annotations

import json
import re
from typing import Any
from urllib.parse import quote_plus

from .clients import chat_completion
from .markdown_render import preserve_markdown


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


def _clean_list(value: Any, *, limit: int = 6, maximum: int = 500) -> list[str]:
    if not isinstance(value, list):
        return []
    return [cleaned for item in value[:limit] if (cleaned := _clean_text(item, maximum))]


def _normalize_misconceptions(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    result: list[dict[str, str]] = []
    for item in value[:4]:
        if not isinstance(item, dict):
            continue
        confusion = _clean_text(item.get("confusion"), 350)
        correction = _clean_text(item.get("correction"), 500)
        if confusion and correction:
            result.append({
                "confusion": confusion,
                "correction": correction,
                "memory_tip": _clean_text(item.get("memory_tip"), 300),
            })
    return result


def _normalize_related_topics(value: Any, fallback: Any = None) -> list[dict[str, str]]:
    """Normalize a small concept neighborhood without trusting model URLs."""
    result: list[dict[str, str]] = []
    if isinstance(value, list):
        for item in value[:6]:
            if not isinstance(item, dict):
                continue
            topic = _clean_text(item.get("topic"), 100)
            if topic:
                result.append({
                    "topic": topic,
                    "relationship": _clean_text(item.get("relationship"), 80) or "related concept",
                    "why_useful": _clean_text(item.get("why_useful"), 300),
                })
    if not result and isinstance(fallback, list):
        for item in fallback[:5]:
            topic = _clean_text(item, 100)
            if topic:
                result.append({"topic": topic, "relationship": "related concept", "why_useful": "Builds a broader understanding of the current lesson."})
    return result

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
    explanation = preserve_markdown(value.get("explanation"))
    if len(explanation.split()) < 60:
        raise ValueError("The generated explanation was too short to teach the concept.")
    quiz = _normalize_quiz(value.get("quiz"))
    related_topics = _normalize_related_topics(value.get("related_topics"), value.get("connections"))
    if len(quiz) < 3:
        raise ValueError("The model did not return at least three valid quiz questions.")
    return {
        "title": _clean_text(value.get("title"), 140) or "Visual lesson",
        "learning_objective": _clean_text(value.get("learning_objective"), 400),
        "explanation": explanation,
        "key_ideas": _clean_list(value.get("key_ideas"), limit=5, maximum=400),
        "worked_example": preserve_markdown(value.get("worked_example"), 1400),
        "common_mistake": _clean_text(value.get("common_mistake"), 600),
        "quick_check": _clean_text(value.get("quick_check"), 500),
        "why_it_matters": _clean_text(value.get("why_it_matters"), 800),
        "prerequisites": _clean_list(value.get("prerequisites"), limit=4, maximum=300),
        "easy_to_confuse": _normalize_misconceptions(value.get("easy_to_confuse")),
        "connections": _clean_list(value.get("connections"), limit=4, maximum=500),
        "related_topics": related_topics,
        "study_path": _clean_list(value.get("study_path"), limit=4, maximum=400),
        "follow_up_questions": _clean_list(value.get("follow_up_questions"), limit=4, maximum=400),
        "quiz": quiz,
    }


def generate_lesson_bundle(client: Any, *, model: str, question: str, subject: str, grade: int, language: str = "English") -> dict[str, Any]:
    """Create a layered teaching lesson and quiz with a strict JSON contract."""
    prompt = f"""You are a careful high-school teacher and curriculum designer.
Create a lesson that directly answers the student's question.

Subject: {subject or "General"}
Grade: {grade}
Language: {language}
Question: {question}

Return one valid JSON object only with these keys:
title, learning_objective, explanation, key_ideas, worked_example,
common_mistake, quick_check, why_it_matters, prerequisites, easy_to_confuse,
connections, related_topics, study_path, follow_up_questions, and quiz.

Requirements:
- explanation: 140-280 words with causal steps, definitions, conditions, and no fluff.
- explanation and worked_example: use short Markdown paragraphs. Put inline math in
  `$...$` and display equations in `$$...$$`. Never use `\\(...\\)`, `\\[...\\]`,
  raw undelimited LaTeX, HTML, or fenced code blocks.
- key_ideas: 4-5 concise strings.
- prerequisites: 2-4 ideas the student should already know.
- easy_to_confuse: 3-4 objects with confusion, correction, and memory_tip.
- connections: 2-4 links to real life, another subject, or a larger concept.
- related_topics: exactly 5 objects with topic (a short concept name), relationship
  (prerequisite, application, contrast, or next step), and why_useful. Do not include URLs.
- study_path: 3-4 concrete actions in the best learning order.
- follow_up_questions: 3-4 curiosity-building questions.
- quiz: exactly 5 objects, each with question, choices (four strings), answer
  (A, B, C, or D), explanation, and concept.

Stay accurate and appropriate for Grade {grade}. Do not invent citations, links,
statistics, books, or named studies; source research is handled separately.
Use {language} except for standard symbols and proper nouns. Return JSON only."""
    return normalize_lesson_bundle(_extract_json(chat_completion(client, model, prompt)))


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _cited_markdown(response: Any) -> tuple[str, list[dict[str, str]]]:
    """Convert Responses API URL annotations into visible Markdown links."""
    fallback_text = preserve_markdown(_field(response, "output_text", ""), 16000)
    text = ""
    annotations: list[Any] = []
    for item in _field(response, "output", []) or []:
        if _field(item, "type") != "message":
            continue
        for part in _field(item, "content", []) or []:
            if _field(part, "type") == "output_text":
                text = str(_field(part, "text", "") or "")
                annotations.extend(_field(part, "annotations", []) or [])
    text = text or fallback_text
    sources: list[dict[str, str]] = []
    seen: set[str] = set()
    replacements: list[tuple[int, int, str]] = []
    for annotation in annotations:
        if _field(annotation, "type") != "url_citation":
            continue
        url = str(_field(annotation, "url", "") or "").strip()
        title = _clean_text(_field(annotation, "title", "Source"), 240) or "Source"
        if not url.startswith(("https://", "http://")):
            continue
        if url not in seen:
            sources.append({"title": title, "url": url})
            seen.add(url)
        start = _field(annotation, "start_index")
        end = _field(annotation, "end_index")
        if isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= len(text):
            label = text[start:end].strip()
            if not label or len(label) > 140:
                label = title
            replacements.append((start, end, f"[{label}]({url})"))
    for start, end, replacement in sorted(replacements, reverse=True):
        text = text[:start] + replacement + text[end:]
    return text.strip(), sources


def research_lesson_sources(
    client: Any,
    *,
    question: str,
    subject: str,
    grade: int,
    language: str = "English",
    model: str = "gpt-5.6",
) -> dict[str, Any]:
    """Run grounded web research and preserve clickable source annotations."""
    if client is None or not hasattr(client, "responses"):
        return {"status": "unavailable", "report_markdown": "", "sources": []}
    if model.startswith("deepseek-"):
        return {
            "status": "unavailable",
            "reason": "selected_text_provider_has_no_grounded_web_search",
            "report_markdown": "",
            "sources": [],
        }
    prompt = f"""Research this high-school lesson question using 3-5 trustworthy sources.
Question: {question}
Subject: {subject}
Grade: {grade}
Language: {language}

Prioritize government agencies, universities, museums, standards bodies,
peer-reviewed publications, and established open textbooks. Avoid SEO blogs,
homework-answer sites, forums, and unsourced summaries.

Write a concise report in this exact teaching structure:
## What reliable sources agree on
A short synthesis with citations.
## Notes from each source
For every source, give its name, what it supports, 2-3 student-friendly key
points, and one limitation or scope note. Cite that source in its own subsection.
## Evidence limits
State uncertainty, simplifications, or what the sources do not establish.
Use Markdown headings and lists. Format math only as `$...$` or `$$...$$`; do not
use raw undelimited LaTeX or fenced code blocks.

Every factual section must use inline citations returned by web search. Do not
invent URLs or bibliography entries."""
    try:
        response = client.responses.create(
            model=model,
            tools=[{"type": "web_search", "search_context_size": "medium"}],
            tool_choice="required",
            include=["web_search_call.action.sources"],
            input=prompt,
        )
        report, sources = _cited_markdown(response)
        if not report or not sources:
            return {"status": "unavailable", "report_markdown": "", "sources": []}
        return {"status": "ready", "report_markdown": report, "sources": sources}
    except Exception:
        return {"status": "unavailable", "report_markdown": "", "sources": []}


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
    """Return stable, non-invented learning-library starting points."""
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


def related_topic_resources(subject: str, topic: str) -> list[dict[str, str]]:
    """Return trustworthy, stable study links for one concept-map node."""
    query = quote_plus(f"{subject} {topic}".strip())
    resources = [{
        "name": "Khan Academy search",
        "url": f"https://www.khanacademy.org/search?page_search_query={query}",
        "description": f"Lessons and practice related to {topic}.",
    }]
    text = f"{subject} {topic}".lower()
    if any(word in text for word in ("physics", "force", "motion", "energy", "circuit", "wave")):
        resources.append({"name": "OpenStax Physics", "url": "https://openstax.org/details/books/physics", "description": "Peer-reviewed high-school physics chapters."})
    elif any(word in text for word in ("biology", "cell", "ecology", "genetic", "evolution")):
        resources.append({"name": "OpenStax Biology 2e", "url": "https://openstax.org/details/books/biology-2e", "description": "Peer-reviewed biology explanations and figures."})
    elif any(word in text for word in ("chemistry", "atom", "molecule", "reaction", "bond", "acid")):
        resources.append({"name": "OpenStax Chemistry 2e", "url": "https://openstax.org/details/books/chemistry-2e", "description": "Peer-reviewed chemistry chapters and examples."})
    elif any(word in text for word in ("math", "algebra", "geometry", "calculus", "function", "probability")):
        resources.append({"name": "GeoGebra resources", "url": f"https://www.geogebra.org/search/{quote_plus(topic)}", "description": "Interactive mathematical models for this concept."})
    elif any(word in text for word in ("computer", "code", "algorithm", "program", "data")):
        resources.append({"name": "MDN learning search", "url": f"https://developer.mozilla.org/en-US/search?q={quote_plus(topic)}", "description": "Reliable computing explanations and examples."})
    else:
        resources.append({"name": "OpenStax subjects", "url": "https://openstax.org/subjects", "description": "Peer-reviewed open textbooks across subjects."})
    return resources


def build_concept_map(bundle: dict[str, Any], subject: str) -> dict[str, Any]:
    """Attach curated sources to model-proposed conceptual relationships."""
    topics = _normalize_related_topics(bundle.get("related_topics"), bundle.get("connections"))
    return {
        "center": _clean_text(bundle.get("title"), 140) or "Current lesson",
        "nodes": [
            {**item, "sources": related_topic_resources(subject, item["topic"])}
            for item in topics[:6]
        ],
    }
