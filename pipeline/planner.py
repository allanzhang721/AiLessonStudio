"""
planner.py — Stage 1 of the L15 pipeline: text → structured 7-step storyboard Plan.

Public entry point: question_explanation_grade_to_plan()

Flow when an OpenAI client is provided:
  1. _fetch_pedagogical_brief()  — ask GPT for subject/grade-specific teaching advice
                                   (vocabulary level, instructional approach,
                                   misconceptions to avoid). Falls back to static rules.
  2. Build planner prompt        — injects the pedagogical brief + question + explanation
                                   into a detailed JSON-output prompt.
  3. GPT call (up to 3 retries)  — parse the JSON response.
  4. _normalize_plan()           — coerce loosely-typed GPT output into the strict Plan
                                   schema (fills missing keys, normalises scene_bible,
                                   ensures exactly 7 steps/captions).
  5. validate_plan_schema()      — hard schema check; if invalid, send a repair prompt.
  6. passes_specificity_gate()   — heuristic quality check; if too vague, send a
                                   specificity refinement prompt.
  7. passes_relevance_gate()     — topical relevance check; if off-topic, send a
                                   relevance refinement prompt.

Flow when no client is provided:
  Returns a deterministic _generic_fallback_plan() immediately.

Helper functions:
  _normalize_scene_bible()  — coerce scene_bible sub-fields to expected types
  _infer_visual_family()    — keyword-based subject domain detection
  _generic_fallback_plan()  — deterministic 7-step plan used as both the fallback
                              and the fill-in template for missing GPT fields
  _normalize_plan()         — merge GPT output over fallback, filling gaps
  _slug()                   — URL-safe ID generation from question text
  _caption_to_text()        — thin wrapper around extract_caption_text utility
"""

from __future__ import annotations

import json
import re
import time
import uuid
from typing import Optional

from .clients import chat_completion
from .config import DEFAULT_STEPS, PLANNER_MODEL, OPENAI_TEXT_MODEL
from .utils import extract_caption_text
from .validation import passes_relevance_gate, passes_specificity_gate, validate_plan_schema


def _static_grade_guidance(grade: int) -> str:
    """Static fallback grade guidance used when GPT pedagogical brief is unavailable."""
    if grade <= 4:
        return "Use very simple vocabulary, concrete objects, almost no abstraction, and very short labels."
    if grade <= 8:
        return "Use middle-school vocabulary, concrete visual analogies, limited jargon, and one idea per frame."
    if grade <= 10:
        return "Use early-high-school vocabulary, simple technical terms only when necessary, and clear causal sequencing."
    return "Use high-school vocabulary, precise academic terminology where needed, and visually explicit reasoning steps."


def _fetch_pedagogical_brief(
    client,
    question: str,
    explanation: str,
    subject: str,
    grade: int,
    model: str,
) -> dict:
    """
    Ask GPT for a subject/question/grade-specific pedagogical brief.
    Returns grade_guidance, vocabulary_level, instructional_approach,
    misconceptions_to_avoid, and label_density tailored to the actual topic.
    Falls back to static values if the GPT call fails.
    """
    prompt = (
        "You are an expert curriculum designer.\n\n"
        f"Subject: {subject}\n"
        f"Grade: {grade}\n"
        f"Question: {question}\n"
        f"Explanation: {explanation}\n\n"
        "Return a JSON object with exactly these keys:\n"
        '  "grade_guidance": one concise sentence on vocabulary and abstraction appropriate for this grade and subject\n'
        '  "vocabulary_level": short descriptor (e.g. \"concrete, minimal jargon\")\n'
        '  "instructional_approach": 1-2 sentences on the best visual teaching strategy for THIS specific concept\n'
        '  "misconceptions_to_avoid": list of 2-4 common student misconceptions about this specific topic that visuals must not reinforce\n'
        f'  "label_density": one sentence on how much labeling is appropriate for Grade {grade} learners on this topic\n\n'
        "Be specific to this subject, grade, and question — not generic advice. Return valid JSON only."
    )
    try:
        raw_text = chat_completion(client, model, prompt)
        brief = json.loads(raw_text)
        if (
            isinstance(brief, dict)
            and "grade_guidance" in brief
            and "misconceptions_to_avoid" in brief
        ):
            brief["_brief_prompt"] = prompt
            return brief
    except Exception:
        pass
    return {
        "grade_guidance": _static_grade_guidance(grade),
        "vocabulary_level": "age-appropriate",
        "instructional_approach": "Build up the visual explanation one concrete element at a time.",
        "misconceptions_to_avoid": [
            "Do not overwhelm the learner with too many labels at once.",
            "Do not introduce decorative objects unrelated to the explanation.",
        ],
        "label_density": "Low to moderate. Only label what directly teaches the answer.",
        "_brief_prompt": prompt,
    }


def _normalize_scene_bible(scene_bible: object, fallback_scene_bible: dict) -> dict:
    """
    Coerce GPT-returned scene_bible into the dict structure expected by renderers.
    GPT may return nested fields like layout/typography/colour_contract as strings.
    """
    if not isinstance(scene_bible, dict):
        return dict(fallback_scene_bible)

    fallback = dict(fallback_scene_bible)
    out = dict(scene_bible)

    layout = out.get("layout")
    if isinstance(layout, str):
        out["layout"] = {
            "canvas": fallback.get("layout", {}).get("canvas", "1536x1024 px"),
            "zones": {
                "DIAGRAM ZONE y=0-800": layout,
                "RESERVED ZONE y=800-1024": "Keep empty for Python caption",
            },
        }
    elif not isinstance(layout, dict):
        out["layout"] = fallback.get("layout", {})
    else:
        zones = layout.get("zones")
        if isinstance(zones, str):
            layout["zones"] = {
                "DIAGRAM ZONE y=0-800": zones,
                "RESERVED ZONE y=800-1024": "Keep empty for Python caption",
            }
        elif not isinstance(zones, dict):
            layout["zones"] = fallback.get("layout", {}).get("zones", {})
        if not isinstance(layout.get("canvas"), str):
            layout["canvas"] = fallback.get("layout", {}).get("canvas", "1536x1024 px")
        out["layout"] = layout

    for key in ("typography", "colour_contract", "educational_contract"):
        value = out.get(key)
        if isinstance(value, str):
            out[key] = {"summary": value}
        elif not isinstance(value, dict):
            out[key] = fallback.get(key, {})

    for key in ("allowed_visual_elements", "forbidden_elements"):
        value = out.get(key)
        if isinstance(value, str):
            out[key] = [value]
        elif not isinstance(value, list):
            out[key] = fallback.get(key, [])

    if not isinstance(out.get("style"), str) or not out.get("style", "").strip():
        out["style"] = fallback.get("style", "flat-vector 2D educational illustration, white background")

    return out


def _infer_visual_family(subject: str, question: str, explanation: str) -> str:
    """Keyword-heuristic to classify the question into a visual domain.

    The visual_family value is embedded in the plan and used by prompts.py to
    set an appropriate rendering style for the storyboard (e.g. biology diagrams
    look different from physics force diagrams). Falls back to 'natural_scene'.
    """
    text = f"{subject} {question} {explanation}".lower()
    if "force" in text or "acceleration" in text or "newton" in text:
        return "force_motion"
    if "cell" in text or "organism" in text or "photosynthesis" in text:
        return "cell_biology"
    if "ecosystem" in text or "food web" in text or "ecology" in text:
        return "ecology"
    if "circuit" in text or "voltage" in text or "current" in text:
        return "circuit"
    if "triangle" in text or "angle" in text or "geometry" in text:
        return "geometry"
    return "natural_scene"


def _slug(text: str) -> str:
    """Convert arbitrary text into a short URL-safe identifier (max 40 chars).

    Used to build the question_id / output folder name from the question text.
    """
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower()).strip("_")
    return cleaned[:40] or "q"


def _caption_to_text(caption: object, fallback_text: str) -> str:
    """Normalize caption values from GPT into plain text for rendering."""
    return extract_caption_text(caption, fallback=fallback_text)


def _generic_fallback_plan(
    question: str,
    explanation: str,
    grade: int,
    subject: str,
    question_id: str,
) -> dict:
    """Build a deterministic 7-step fallback plan without calling GPT.

    Used in two situations:
      (a) No OpenAI client is available — returned directly as the result.
      (b) As the base template when normalising a GPT-returned plan — any field
          that GPT omitted or mis-typed is filled from this fallback.

    The steps use generic goals/deltas that work for any subject; the planner
    prompt asks GPT to write subject-specific ones.
    """
    visual_family = _infer_visual_family(subject, question, explanation)
    steps = []
    goals = [
        "Establish base scene",
        "Introduce focus condition",
        "Show first mechanism",
        "Show next mechanism",
        "Highlight relationship",
        "Connect to principle",
        "Conclude answer",
    ]
    deltas = [
        "Draw core objects and context for the concept.",
        "Add one visual condition that sets the main constraint.",
        "Add one process arrow or change that starts the explanation.",
        "Add one next causal change while keeping existing content stable.",
        "Add one comparison or mapping between the key objects.",
        "Add one principle label or blank formula tile if relevant.",
        "Add one concise final callout with the conclusion.",
    ]
    for i in range(DEFAULT_STEPS):
        keep = [f"Step {k + 1}: {deltas[k]}" for k in range(i)]
        steps.append(
            {
                "step_id": i + 1,
                "goal": goals[i],
                "delta": deltas[i],
                "forbidden": [] if i == 0 else ["Do not redraw existing elements."],
                "keep": keep,
                "add": [deltas[i]],
            }
        )

    captions = [
        "We begin with the key objects.",
        "Now we define the main condition.",
        "The first mechanism appears.",
        "The process continues step by step.",
        "This relationship links key parts.",
        "A principle explains this pattern.",
        "Together these steps answer the question.",
    ]

    return {
        "question_id": question_id,
        "question_text": question,
        "canonical_answer": explanation.strip(),
        "visual_family": visual_family,
        "render_mode": "gpt_edit",
        "scene_bible": {
            "style": "flat-vector 2D educational illustration, white background",
            "layout": {
                "canvas": "1536x1024 px",
                "zones": {
                    "DIAGRAM ZONE y=0-800": "All diagram elements",
                    "RESERVED ZONE y=800-1024": "Keep empty for Python caption",
                },
            },
            "typography": {"labels": "bold sans-serif >=46 px"},
            "educational_contract": {
                "audience": f"Grade {grade} {subject} students",
                "grade_guidance": _static_grade_guidance(grade),
                "instructional_goal": "One new idea per frame; keep the reasoning easy to follow.",
                "label_density": "Low to moderate. Only label what directly teaches the answer.",
                "misconceptions_to_avoid": [
                    "Do not overwhelm the learner with too many labels at once.",
                    "Do not introduce unrelated decorative objects.",
                ],
            },
            "colour_contract": {
                "primary": "#2563EB",
                "secondary": "#EA580C",
                "accent": "#16A34A",
                "label_text": "#1F2937",
            },
            "allowed_visual_elements": ["core objects", "arrows", "labels", "callouts"],
            "forbidden_elements": [
                "photorealism",
                "3D perspective",
                "shadows",
                "elements below y=800",
            ],
        },
        "steps": steps,
        "captions": captions,
        "math_elements": [],
        "grade": grade,
        "subject": subject,
    }


def _normalize_plan(raw_plan: dict, fallback: dict) -> dict:
    """Merge a GPT-returned raw plan dict over the fallback, filling every gap.

    Handles cases where GPT:
      - Omits optional keys entirely
      - Returns steps/captions with wrong length
      - Uses alternate field names (e.g. "title" instead of "goal")
      - Returns scene_bible sub-fields as strings instead of dicts

    The result is always a fully-populated Plan dict that passes
    validate_plan_schema() in the happy path.
    """
    plan = dict(raw_plan) if isinstance(raw_plan, dict) else {}
    plan["question_id"] = plan.get("question_id") or fallback["question_id"]
    plan["question_text"] = plan.get("question_text") or fallback["question_text"]
    plan["canonical_answer"] = plan.get("canonical_answer") or fallback["canonical_answer"]
    plan["visual_family"] = plan.get("visual_family") or fallback["visual_family"]
    plan["grade"] = plan.get("grade") if isinstance(plan.get("grade"), int) else fallback.get("grade")
    plan["subject"] = plan.get("subject") if isinstance(plan.get("subject"), str) else fallback.get("subject")
    plan["render_mode"] = "gpt_edit"
    plan["scene_bible"] = _normalize_scene_bible(plan.get("scene_bible"), fallback["scene_bible"])

    steps = plan.get("steps") if isinstance(plan.get("steps"), list) else []
    normalized_steps = []
    for i in range(DEFAULT_STEPS):
        if i < len(steps) and isinstance(steps[i], dict):
            s = steps[i]
            goal = str(s.get("goal") or s.get("title") or fallback["steps"][i]["goal"])
            delta = str(s.get("delta") or s.get("description") or fallback["steps"][i]["delta"])
            forbidden = s.get("forbidden") if isinstance(s.get("forbidden"), list) else []
            keep = s.get("keep") if isinstance(s.get("keep"), list) else []
            add = s.get("add") if isinstance(s.get("add"), list) else [delta]
            normalized_steps.append(
                {
                    "step_id": i + 1,
                    "goal": goal,
                    "delta": delta,
                    "forbidden": forbidden,
                    "keep": keep,
                    "add": add,
                }
            )
        else:
            normalized_steps.append(fallback["steps"][i])

    captions = plan.get("captions") if isinstance(plan.get("captions"), list) else []
    normalized_captions = [
        _caption_to_text(captions[i], fallback["captions"][i]) if i < len(captions) else fallback["captions"][i]
        for i in range(DEFAULT_STEPS)
    ]

    math_elements = plan.get("math_elements") if isinstance(plan.get("math_elements"), list) else fallback.get("math_elements", [])

    plan["steps"] = normalized_steps
    plan["captions"] = normalized_captions
    plan["math_elements"] = math_elements
    return plan


def question_explanation_grade_to_plan(
    question: str,
    explanation: str,
    grade: int,
    subject: str = "",
    question_id: Optional[str] = None,
    client=None,
    model: Optional[str] = None,
) -> dict:
    """
    Stage 1: question + explanation + grade -> structured 7-step storyboard plan.
    Uses GPT when client is available; otherwise returns deterministic fallback plan.
    """
    used_model = model or PLANNER_MODEL
    qid = question_id or f"q_{_slug(question)}_{uuid.uuid4().hex[:6]}"
    inferred_subject = subject.strip() or "General"
    fallback = _generic_fallback_plan(question, explanation, grade, inferred_subject, qid)
    fallback["planner_meta"] = {
        "source": "fallback_no_client",
        "planner_model": used_model,
        "attempts": 0,
        "repaired": False,
        "refined_specificity": False,
        "refined_relevance": False,
        "specificity_score": 0.0,
        "relevance_score": 0.0,
        "validation_errors": [],
        "specificity_issues": [],
        "relevance_issues": [],
        "debug_prompts": {},
    }

    if client is None:
        return fallback

    brief = _fetch_pedagogical_brief(
        client, question, explanation, inferred_subject, grade, used_model
    )
    # Enrich the fallback's educational_contract with the GPT-generated brief
    fallback["scene_bible"]["educational_contract"] = {
        "audience": f"Grade {grade} {inferred_subject} students",
        "grade_guidance": brief["grade_guidance"],
        "vocabulary_level": brief.get("vocabulary_level", ""),
        "instructional_approach": brief.get("instructional_approach", ""),
        "misconceptions_to_avoid": brief.get("misconceptions_to_avoid", []),
        "label_density": brief.get("label_density", ""),
    }

    misconceptions_str = "; ".join(str(m) for m in brief.get("misconceptions_to_avoid", []))
    prompt = f"""
You are a storyboard planner for educational diagrams.
Input:
- Question: {question}
- Explanation: {explanation}
- Grade: {grade}
- Subject: {inferred_subject}

Return strict JSON with keys:
question_id, question_text, canonical_answer, visual_family, render_mode,
grade, subject,
scene_bible, steps, captions, math_elements.

Pedagogical brief (generated for this specific question, grade, subject):
- grade_guidance: {brief["grade_guidance"]}
- vocabulary_level: {brief.get("vocabulary_level", "")}
- instructional_approach: {brief.get("instructional_approach", "")}
- misconceptions_to_avoid: {misconceptions_str}
- label_density: {brief.get("label_density", "")}

Rules:
- Exactly 7 steps.
- Each step adds one new visual element.
- Each step includes: step_id, goal, delta, forbidden(list), keep(list), add(list).
- The visual content must be directly relevant to the question and explanation.
- Choose concrete subject-appropriate objects. Example: for force/inertia use carts, blocks, arrows, masses, or similar physics objects.
- Do not use decorative or unrelated objects.
- Follow the pedagogical brief above: match the vocabulary level, instructional approach, and avoid the listed misconceptions.
- Design it like a strong teacher-designed board explanation: follow the instructional_approach above.
- Each frame must communicate ONE teachable idea at a glance in under 5 seconds.
- Avoid cognitive overload: limited text, limited labels, strong spatial organization, no clutter.
- Do not occlude important existing objects; add new objects in open space with clear separation.
- Arrows must have clear direction and purpose. Do not draw duplicate arrows for the same relation.
- If using arrows, specify explicit source->target direction in step text (for example: "arrow from applied force label -> cart, left to right").
- Keep bottom y=800-1024 empty for Python caption.
- Use white background, flat vector 2D style.
- scene_bible should include concrete layout guidance, object design cues, and color assignments.
- scene_bible must include an educational_contract that reflects the pedagogical brief above
  (audience, grade_guidance, vocabulary_level, instructional_approach, label_density, misconceptions_to_avoid).
- Prefer explicit coordinates, left/right placement, and size relationships in step add lists.
- If formulas/symbols are needed, include math_elements entries and tell image model
  to draw blank formula tile backgrounds only.
- Captions are the teacher's spoken narration for that frame. Write each caption as if a teacher is explaining
  this specific moment to a student out loud: 2-4 natural sentences that describe what is shown, why it matters,
  and what the student should notice or conclude from this frame.
- Captions must match the vocabulary_level above. Target 200-400 characters. Do not use bullet points or labels;
  write in flowing educational prose as spoken narration.
- Output valid JSON only.
""".strip()

    raw = None
    planner_attempts = 0
    last_error = ""
    for attempt in range(1, 4):
        planner_attempts = attempt
        try:
            raw_text = chat_completion(client, used_model, prompt)
            raw = json.loads(raw_text)
            break
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(0.6 * attempt)

    if raw is None:
        fallback["planner_meta"] = {
            "source": "fallback_planner_error",
            "planner_model": used_model,
            "attempts": planner_attempts,
            "repaired": False,
            "refined_specificity": False,
            "refined_relevance": False,
            "specificity_score": 0.0,
            "relevance_score": 0.0,
            "validation_errors": [last_error] if last_error else [],
            "specificity_issues": [],
            "relevance_issues": [],
            "debug_prompts": {"planner_prompt": prompt},
        }
        return fallback

    normalized = _normalize_plan(raw, fallback)
    is_valid, errors = validate_plan_schema(normalized, expected_steps=DEFAULT_STEPS)
    repaired = False
    refined_specificity = False
    refined_relevance = False
    specificity_score = 0.0
    relevance_score = 0.0
    specificity_issues: list[str] = []
    relevance_issues: list[str] = []
    debug_prompts: dict[str, str] = {
        "pedagogical_brief": json.dumps(
            {k: v for k, v in brief.items() if k != "_brief_prompt"}, ensure_ascii=False
        ),
        "pedagogical_brief_prompt": brief.get("_brief_prompt", ""),
        "planner_prompt": prompt,
    }

    if not is_valid:
        repair_prompt = f"""
You previously returned an invalid storyboard JSON.
Fix the JSON so it satisfies ALL constraints below and return JSON only.

Validation errors:
{json.dumps(errors, ensure_ascii=False, indent=2)}

Original candidate JSON:
{json.dumps(raw, ensure_ascii=False)}

Hard constraints:
- Exactly 7 steps and 7 captions.
- Each step includes: step_id, goal, delta, forbidden(list), keep(list), add(list).
- step_id must be 1..7 in order.
- Keep bottom y=800-1024 empty for Python caption.
- Keep render_mode="gpt_edit".
- Top-level keys must be:
  question_id, question_text, canonical_answer, visual_family, render_mode,
  scene_bible, steps, captions, math_elements.
""".strip()
        try:
            debug_prompts["repair_prompt"] = repair_prompt
            repair_text = chat_completion(client, used_model, repair_prompt)
            repaired_raw = json.loads(repair_text)
            repaired_normalized = _normalize_plan(repaired_raw, fallback)
            repaired_valid, repaired_errors = validate_plan_schema(repaired_normalized, expected_steps=DEFAULT_STEPS)
            if repaired_valid:
                normalized = repaired_normalized
                is_valid = True
                errors = []
                repaired = True
            else:
                errors = repaired_errors
        except Exception as exc:
            errors.append(f"repair_failed: {type(exc).__name__}: {exc}")

    # Specificity gate: reject vague plans and ask one refinement pass.
    if is_valid:
        passes_gate, specificity_score, specificity_issues = passes_specificity_gate(
            normalized,
            threshold=0.62,
            expected_steps=DEFAULT_STEPS,
            hard_enforce_arrow_direction=True,
        )
        if not passes_gate:
            refine_prompt = f"""
Your JSON plan is structurally valid but too vague for deterministic rendering.
Refine ALL 7 steps to increase specificity and keep JSON schema unchanged.

Specificity score: {specificity_score:.3f} (required >= 0.620)
Specificity issues:
{json.dumps(specificity_issues[:40], ensure_ascii=False, indent=2)}

Current JSON:
{json.dumps(normalized, ensure_ascii=False)}

Refinement requirements:
- Keep top-level keys unchanged.
- Keep exactly 7 steps and 7 captions.
- Each step must include explicit positional constraints (zones, coordinates, left/right/top/bottom).
- Each step must include explicit visual constraints (color/style/font/size/shape).
- If a step uses arrows, include exact arrow direction and source/target objects to prevent flipped meaning.
- Avoid vague phrases like "show concept" or "add detail".
- Each step's ADD list should be concrete and actionable.
- Keep the teaching level appropriate for Grade {grade} students.
- Reduce clutter and maintain one teachable idea per frame.
- Captions must read as teacher-spoken narration: 2-4 natural sentences explaining what is shown, why it matters,
  and what the student should notice from this frame. 200-400 characters, flowing prose, no bullet points.
- Return JSON only.
""".strip()
            try:
                debug_prompts["specificity_refine_prompt"] = refine_prompt
                refine_text = chat_completion(client, used_model, refine_prompt)
                refined_raw = json.loads(refine_text)
                refined_norm = _normalize_plan(refined_raw, fallback)
                refined_valid, refined_errors = validate_plan_schema(refined_norm, expected_steps=DEFAULT_STEPS)
                if refined_valid:
                    refined_pass, refined_score, refined_issues = passes_specificity_gate(
                        refined_norm,
                        threshold=0.62,
                        expected_steps=DEFAULT_STEPS,
                        hard_enforce_arrow_direction=True,
                    )
                    if refined_pass:
                        normalized = refined_norm
                        specificity_score = refined_score
                        specificity_issues = []
                        refined_specificity = True
                    else:
                        specificity_score = refined_score
                        specificity_issues = refined_issues
                else:
                    errors.extend(refined_errors)
            except Exception as exc:
                specificity_issues.append(f"specificity_refine_failed: {type(exc).__name__}: {exc}")

    # Relevance gate: reject plans that drift away from the actual topic.
    if is_valid:
        passes_relevance, relevance_score, relevance_issues = passes_relevance_gate(
            question,
            explanation,
            inferred_subject,
            normalized,
            threshold=0.45,
        )
        if not passes_relevance:
            relevance_prompt = f"""
Your JSON plan is off-topic or insufficiently aligned with the user's question.
Refine it so every visual element directly explains the actual question and explanation.

Question: {question}
Explanation: {explanation}
Subject: {inferred_subject}
Current relevance score: {relevance_score:.3f} (required >= 0.450)
Issues:
{json.dumps(relevance_issues[:40], ensure_ascii=False, indent=2)}

Current JSON:
{json.dumps(normalized, ensure_ascii=False)}

Requirements:
- Keep JSON schema unchanged.
- Keep exactly 7 steps and 7 captions.
- Use only subject-relevant objects and labels.
- Remove unrelated concepts such as coding/programming terms unless the subject is computer science.
- Make the scene, steps, and labels directly explain the provided question.
- Keep the visual explanation appropriate for Grade {grade} learners.
- Return JSON only.
""".strip()
            try:
                debug_prompts["relevance_refine_prompt"] = relevance_prompt
                relevance_text = chat_completion(client, used_model, relevance_prompt)
                relevance_raw = json.loads(relevance_text)
                relevance_norm = _normalize_plan(relevance_raw, fallback)
                relevance_valid, relevance_errors = validate_plan_schema(relevance_norm, expected_steps=DEFAULT_STEPS)
                if relevance_valid:
                    rel_pass, rel_score, rel_issues = passes_relevance_gate(
                        question,
                        explanation,
                        inferred_subject,
                        relevance_norm,
                        threshold=0.45,
                    )
                    if rel_pass:
                        normalized = relevance_norm
                        relevance_score = rel_score
                        relevance_issues = []
                        refined_relevance = True
                    else:
                        relevance_score = rel_score
                        relevance_issues = rel_issues
                else:
                    errors.extend(relevance_errors)
            except Exception as exc:
                relevance_issues.append(f"relevance_refine_failed: {type(exc).__name__}: {exc}")

    if not is_valid:
        fallback["planner_meta"] = {
            "source": "fallback_validation_failed",
            "planner_model": used_model,
            "attempts": planner_attempts,
            "repaired": repaired,
            "refined_specificity": refined_specificity,
            "refined_relevance": refined_relevance,
            "specificity_score": specificity_score,
            "relevance_score": relevance_score,
            "validation_errors": errors,
            "specificity_issues": specificity_issues,
            "relevance_issues": relevance_issues,
            "debug_prompts": debug_prompts,
        }
        return fallback

    # Final gate check; fallback if still too vague after refinement attempt.
    passes_gate, specificity_score, specificity_issues = passes_specificity_gate(
        normalized,
        threshold=0.62,
        expected_steps=DEFAULT_STEPS,
        hard_enforce_arrow_direction=True,
    )
    if not passes_gate:
        normalized["planner_meta"] = {
            "source": "openai_specificity_warned",
            "planner_model": used_model,
            "attempts": planner_attempts,
            "repaired": repaired,
            "refined_specificity": refined_specificity,
            "refined_relevance": refined_relevance,
            "specificity_score": specificity_score,
            "relevance_score": relevance_score,
            "validation_errors": [],
            "specificity_issues": specificity_issues,
            "relevance_issues": relevance_issues,
            "debug_prompts": debug_prompts,
        }
        return normalized

    passes_relevance, relevance_score, relevance_issues = passes_relevance_gate(
        question,
        explanation,
        inferred_subject,
        normalized,
        threshold=0.45,
    )
    if not passes_relevance:
        normalized["planner_meta"] = {
            "source": "openai_relevance_warned",
            "planner_model": used_model,
            "attempts": planner_attempts,
            "repaired": repaired,
            "refined_specificity": refined_specificity,
            "refined_relevance": refined_relevance,
            "specificity_score": specificity_score,
            "relevance_score": relevance_score,
            "validation_errors": [],
            "specificity_issues": specificity_issues,
            "relevance_issues": relevance_issues,
            "debug_prompts": debug_prompts,
        }
        return normalized

    normalized["planner_meta"] = {
        "source": "openai",
        "planner_model": used_model,
        "attempts": planner_attempts,
        "repaired": repaired,
        "refined_specificity": refined_specificity,
        "refined_relevance": refined_relevance,
        "specificity_score": specificity_score,
        "relevance_score": relevance_score,
        "validation_errors": [],
        "specificity_issues": [],
        "relevance_issues": [],
        "debug_prompts": debug_prompts,
    }
    return normalized
