"""
validation.py — Schema validation and plan quality scoring for the L15 pipeline.

Three layers of quality control, all called by planner.py before images are rendered:

  1. validate_plan_schema()     — hard structural check: are all required keys present,
                                   all lists the right length, all types correct?
                                   The image pipeline refuses to run on an invalid plan.

  2. score_plan_specificity()   — heuristic 0→1 score measuring how actionable each
                                   step instruction is:  position words, colour refs,
                                   arrow direction semantics, and one-new-element focus.
                                   Low-scoring plans produce inconsistent images.

  3. score_plan_relevance()     — heuristic 0→1 score checking that plan content uses
                                   vocabulary from the original question/explanation and
                                   doesn't leak off-topic concepts (e.g. coding terms
                                   appearing in a biology question).

The planner calls passes_specificity_gate() and passes_relevance_gate() after each
GPT attempt and issues a targeted refinement prompt if either gate fails.
"""

from __future__ import annotations

from typing import Any
import re


# Words too generic to be useful for relevance scoring.
_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "into", "your", "their",
    "have", "has", "more", "same", "than", "what", "when", "where", "which", "using",
    "need", "does", "because", "about", "they", "them", "then", "each", "only", "will",
    "would", "could", "should", "into", "over", "under", "show", "add", "step", "object",
}

# Terms that indicate the plan accidentally drifted into a programming/CS explanation
# when the subject is not computer science.
_CODING_LEAKAGE_TERMS = {
    "for loop", "while loop", "python", "javascript", "variable", "function",
    "array", "list comprehension", "dictionary", "class", "method", "algorithm", "code",
}


def _is_non_empty_str(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def validate_plan_schema(plan: dict, expected_steps: int = 7) -> tuple[bool, list[str]]:
    """
    Validate minimum schema requirements for plan rendering.
    Returns (is_valid, errors).
    """
    errors: list[str] = []
    if not isinstance(plan, dict):
        return False, ["Plan must be a dict."]

    # All of these top-level keys must be present before the image pipeline will run.
    required_top = [
        "question_id",
        "question_text",
        "canonical_answer",
        "visual_family",
        "render_mode",
        "scene_bible",
        "steps",
        "captions",
        "math_elements",
    ]
    for key in required_top:
        if key not in plan:
            errors.append(f"Missing top-level key: {key}")

    if not _is_non_empty_str(plan.get("question_id", "")):
        errors.append("question_id must be a non-empty string")

    # --- scene_bible validation ---
    # The scene_bible drives visual consistency across all 7 frames.
    scene_bible = plan.get("scene_bible")
    if not isinstance(scene_bible, dict):
        errors.append("scene_bible must be a dict")
        scene_bible = {}

    layout = scene_bible.get("layout")
    if not isinstance(layout, dict):
        errors.append("scene_bible.layout must be a dict")
    else:
        if not isinstance(layout.get("canvas"), str):
            errors.append("scene_bible.layout.canvas must be a string")
        if not isinstance(layout.get("zones"), dict):
            errors.append("scene_bible.layout.zones must be a dict")

    typography = scene_bible.get("typography")
    if typography is not None and not isinstance(typography, dict):
        errors.append("scene_bible.typography must be a dict when present")

    colour_contract = scene_bible.get("colour_contract")
    if colour_contract is not None and not isinstance(colour_contract, dict):
        errors.append("scene_bible.colour_contract must be a dict when present")

    for key in ("allowed_visual_elements", "forbidden_elements"):
        value = scene_bible.get(key)
        if value is not None and not isinstance(value, list):
            errors.append(f"scene_bible.{key} must be a list when present")

    # --- steps validation ---
    # Exactly expected_steps (default 7) steps required; step_ids must be consecutive 1..N.
    steps = plan.get("steps")
    if not isinstance(steps, list):
        errors.append("steps must be a list")
        steps = []
    if len(steps) != expected_steps:
        errors.append(f"steps must contain exactly {expected_steps} items")

    for i, step in enumerate(steps, start=1):
        if not isinstance(step, dict):
            errors.append(f"steps[{i}] must be an object")
            continue
        if int(step.get("step_id", -1)) != i:
            errors.append(f"steps[{i}] step_id must equal {i}")
        if not _is_non_empty_str(step.get("goal", "")):
            errors.append(f"steps[{i}] goal must be non-empty string")
        if not _is_non_empty_str(step.get("delta", "")):
            errors.append(f"steps[{i}] delta must be non-empty string")
        for list_key in ("forbidden", "keep", "add"):
            if not isinstance(step.get(list_key), list):
                errors.append(f"steps[{i}] {list_key} must be a list")

    # --- captions validation ---
    # One narration string per frame, must match steps length.
    captions = plan.get("captions")
    if not isinstance(captions, list):
        errors.append("captions must be a list")
        captions = []
    if len(captions) != expected_steps:
        errors.append(f"captions must contain exactly {expected_steps} items")
    for i, cap in enumerate(captions, start=1):
        if not _is_non_empty_str(cap):
            errors.append(f"captions[{i}] must be non-empty string")

    # --- math_elements validation ---
    # Optional formula tiles; each needs a valid bounding box and introduction step.
    math_elements = plan.get("math_elements")
    if not isinstance(math_elements, list):
        errors.append("math_elements must be a list")
        math_elements = []

    for idx, elem in enumerate(math_elements, start=1):
        if not isinstance(elem, dict):
            errors.append(f"math_elements[{idx}] must be an object")
            continue
        if not _is_non_empty_str(elem.get("id", "")):
            errors.append(f"math_elements[{idx}] id must be non-empty string")
        if not isinstance(elem.get("step_introduced"), int):
            errors.append(f"math_elements[{idx}] step_introduced must be int")
        for coord in ("x1", "y1", "x2", "y2"):
            if not isinstance(elem.get(coord), int):
                errors.append(f"math_elements[{idx}] {coord} must be int")
        if isinstance(elem.get("x1"), int) and isinstance(elem.get("x2"), int) and elem["x1"] >= elem["x2"]:
            errors.append(f"math_elements[{idx}] x1 must be < x2")
        if isinstance(elem.get("y1"), int) and isinstance(elem.get("y2"), int) and elem["y1"] >= elem["y2"]:
            errors.append(f"math_elements[{idx}] y1 must be < y2")

    return len(errors) == 0, errors


# Phrases so generic they indicate the planner produced boilerplate rather
# than a concrete, renderable instruction for this specific question.
_VAGUE_PATTERNS = {
    "add object",
    "show concept",
    "illustrate idea",
    "draw scene",
    "show process",
    "add detail",
    "step",
}


def score_plan_specificity(plan: dict, expected_steps: int = 7) -> tuple[float, list[str]]:
    """
    Heuristic score in [0,1] for step specificity.
    Higher = better positional/visual constraints and less vague language.
    """
    issues: list[str] = []
    steps = plan.get("steps") if isinstance(plan.get("steps"), list) else []
    if len(steps) != expected_steps:
        return 0.0, [f"Expected {expected_steps} steps for specificity scoring."]

    total = 0.0
    max_total = float(expected_steps)
    for i, step in enumerate(steps, start=1):
        step_score = 0.0
        delta = str(step.get("delta", "")).lower()
        add = step.get("add") if isinstance(step.get("add"), list) else []
        # Combine delta + add items into one string for heuristic matching.
        merged = " ".join([delta] + [str(x).lower() for x in add])

        # +0.2 — Non-empty actionable content
        if len(merged.strip()) >= 24:
            step_score += 0.2
        else:
            issues.append(f"Step {i}: instruction text too short.")

        # +0.25 — Positional constraints (coordinates, zones, left/right/top/bottom)
        has_coords = bool(re.search(r"\b(?:x|y)\s*=\s*\d+|\b\d{2,4}\s*[x×]\s*\d{2,4}\b|\(\d+\s*,\s*\d+\)", merged))
        has_layout_words = any(w in merged for w in ["left", "right", "top", "bottom", "zone", "center", "centre"])
        if has_coords or has_layout_words:
            step_score += 0.25
        else:
            issues.append(f"Step {i}: missing clear position/layout constraints.")

        # +0.25 — Visual attributes (color/style/size)
        has_hex = bool(re.search(r"#[0-9a-f]{6}\b", merged))
        has_style_words = any(w in merged for w in ["color", "colour", "bold", "font", "px", "arrow", "label", "rounded", "border"])
        if has_hex or has_style_words:
            step_score += 0.25
        else:
            issues.append(f"Step {i}: missing explicit visual style attributes.")

        # Arrow semantics: when arrows are used, require explicit source->target direction.
        # Unsigned arrows cause the image model to draw reversed or duplicated arrows.
        has_arrow = any(w in merged for w in ["arrow", "arrows", "->", "\u2192", "points to", "toward", "towards"])
        has_directed_relation = bool(
            re.search(r"\bfrom\b.+\bto\b", merged)
            or re.search(r"\bto\b.+\bfrom\b", merged)
            or "->" in merged
            or "\u2192" in merged
            or any(w in merged for w in ["left to right", "right to left", "upward", "downward", "toward", "towards"])
        )
        if has_arrow:
            if has_directed_relation:
                step_score += 0.1    # +0.1 bonus for explicit arrow direction
            else:
                step_score -= 0.12  # penalty: ambiguous arrow will likely be rendered wrong
                issues.append(f"Step {i}: arrow mentioned without explicit source/target direction.")

        # +0.2 — One-new-element intent via add list or explicit phrase
        if add and len(add) > 0:
            step_score += 0.2
        elif "one" in merged and "add" in merged:
            step_score += 0.15
        else:
            issues.append(f"Step {i}: weak one-new-element instruction.")

        # Penalize vague phrases
        if any(v in merged for v in _VAGUE_PATTERNS):
            step_score -= 0.1
            issues.append(f"Step {i}: contains vague phrasing.")

        total += max(0.0, min(1.0, step_score))

    return max(0.0, min(1.0, total / max_total)), issues


def find_arrow_direction_issues(plan: dict, expected_steps: int = 7) -> list[str]:
    """Return hard-fail issues where arrows are used without explicit direction semantics."""
    issues: list[str] = []
    steps = plan.get("steps") if isinstance(plan.get("steps"), list) else []
    if len(steps) != expected_steps:
        return issues

    for i, step in enumerate(steps, start=1):
        if not isinstance(step, dict):
            continue
        delta = str(step.get("delta", "")).lower()
        add = step.get("add") if isinstance(step.get("add"), list) else []
        merged = " ".join([delta] + [str(x).lower() for x in add])
        has_arrow = any(w in merged for w in ["arrow", "arrows", "->", "\u2192", "points to", "toward", "towards"])
        has_directed_relation = bool(
            re.search(r"\bfrom\b.+\bto\b", merged)
            or re.search(r"\bto\b.+\bfrom\b", merged)
            or "->" in merged
            or "\u2192" in merged
            or any(w in merged for w in ["left to right", "right to left", "upward", "downward", "toward", "towards"])
        )
        if has_arrow and not has_directed_relation:
            issues.append(f"Step {i}: arrow direction is ambiguous; specify source->target explicitly.")
    return issues


def passes_specificity_gate(
    plan: dict,
    threshold: float = 0.62,
    expected_steps: int = 7,
    hard_enforce_arrow_direction: bool = False,
) -> tuple[bool, float, list[str]]:
    score, issues = score_plan_specificity(plan, expected_steps=expected_steps)
    if hard_enforce_arrow_direction:
        arrow_issues = find_arrow_direction_issues(plan, expected_steps=expected_steps)
        if arrow_issues:
            issues = issues + arrow_issues
            return False, score, issues
    return score >= threshold, score, issues


def _tokenize_keywords(text: str) -> list[str]:
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9']+", text.lower())
    tokens = [t for t in tokens if len(t) >= 4 and t not in _STOPWORDS]
    # preserve order but dedupe
    seen = set()
    result = []
    for tok in tokens:
        if tok not in seen:
            seen.add(tok)
            result.append(tok)
    return result


def score_plan_relevance(question: str, explanation: str, subject: str, plan: dict) -> tuple[float, list[str]]:
    """
    Heuristic topical relevance score in [0,1].
    Rejects plans that drift away from the user topic, especially coding-topic leakage.
    """
    issues: list[str] = []
    source_text = f"{question} {explanation} {subject}".lower()
    plan_text_parts = [
        str(plan.get("question_text", "")),
        str(plan.get("canonical_answer", "")),
        str(plan.get("visual_family", "")),
    ]
    scene_bible = plan.get("scene_bible", {})
    if isinstance(scene_bible, dict):
        plan_text_parts.append(str(scene_bible.get("style", "")))
        plan_text_parts.append(str(scene_bible.get("layout", "")))
    for step in plan.get("steps", []) if isinstance(plan.get("steps"), list) else []:
        if isinstance(step, dict):
            plan_text_parts.extend([
                str(step.get("goal", "")),
                str(step.get("delta", "")),
                " ".join(str(x) for x in step.get("add", []) if isinstance(step.get("add"), list)),
            ])
    plan_text = " ".join(plan_text_parts).lower()

    source_keywords = _tokenize_keywords(source_text)
    if not source_keywords:
        return 0.5, []

    overlap = [kw for kw in source_keywords[:12] if kw in plan_text]
    overlap_score = min(1.0, len(overlap) / max(4.0, min(8.0, len(source_keywords))))

    leakage_hits = [term for term in _CODING_LEAKAGE_TERMS if term in plan_text]
    subject_lower = subject.lower().strip()
    is_cs_subject = any(term in subject_lower for term in ["computer", "coding", "programming", "python", "cs"])
    leakage_penalty = 0.0
    if leakage_hits and not is_cs_subject:
        leakage_penalty = min(0.7, 0.2 * len(leakage_hits))
        issues.append("Off-topic coding concepts detected: " + ", ".join(leakage_hits))

    if len(overlap) < 2:
        issues.append("Plan has weak keyword overlap with the question/explanation.")

    score = max(0.0, min(1.0, overlap_score - leakage_penalty + 0.15))
    return score, issues


def passes_relevance_gate(question: str, explanation: str, subject: str, plan: dict, threshold: float = 0.45) -> tuple[bool, float, list[str]]:
    score, issues = score_plan_relevance(question, explanation, subject, plan)
    return score >= threshold, score, issues
