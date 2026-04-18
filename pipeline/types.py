"""
types.py — Shared TypedDict schemas for the L15 pipeline data model.

Three core structures flow through the whole pipeline:
  StepSpec      — one entry in the 7-step storyboard (what to draw on that frame)
  MathElement   — an optional formula/symbol tile to overlay on a frame
  Plan          — the full storyboard document produced by planner.py and consumed
                  by image_pipeline.py and video_pipeline.py
"""

from __future__ import annotations

from typing import TypedDict, List, Dict, Any


class StepSpec(TypedDict, total=False):
    """Specification for a single storyboard frame.

    Fields:
        step_id   — 1-based index of this frame (1..7)
        goal      — high-level teaching objective for this frame
                    e.g. "Introduce the predator-prey relationship"
        delta     — the one new visual change made relative to the previous frame
                    e.g. "Add a bold red arrow from wolf to deer"
        forbidden — list of elements that must NOT appear or be modified on this frame
        keep      — list of elements from previous frames that must be preserved exactly
        add       — list of concrete instructions for what to draw (mirrors delta but
                    may be multi-item for positioning clarity)
    """
    step_id: int
    goal: str
    delta: str
    forbidden: List[str]
    keep: List[str]
    add: List[str]


class MathElement(TypedDict, total=False):
    """An optional formula or symbol tile to be composited onto frames.

    The image model draws a blank coloured rectangle as a placeholder; Python
    then renders the actual display_text on top using Pillow so the formula is
    always crisp and correctly positioned.

    Fields:
        id              — unique identifier referenced in step add/keep lists
        step_introduced — the frame number on which this element first appears
                          (it remains visible on all subsequent frames)
        display_text    — the formula or symbol string to render, e.g. "F = ma"
        x1, y1, x2, y2 — bounding box in image pixels (must satisfy x1<x2, y1<y2)
        style           — rendering style, currently only "formula_tile" is supported
        bg              — fill colour of the tile background, default "#FEF9C3"
        border          — outline colour of the tile, default "#CA8A04"
    """
    id: str
    step_introduced: int
    display_text: str
    x1: int
    y1: int
    x2: int
    y2: int
    style: str
    bg: str
    border: str


class Plan(TypedDict, total=False):
    """The full storyboard document that travels through all pipeline stages.

    Produced by planner.question_explanation_grade_to_plan() and consumed
    sequentially by image_pipeline.plan_to_images() and
    video_pipeline.images_to_video().

    Fields:
        question_id      — URL-safe slug + UUID suffix used as the output folder name
        question_text    — the original educational question
        canonical_answer — the correct explanation that the video must illustrate
        visual_family    — broad visual category inferred from subject/question
                           e.g. "ecology", "force_motion", "circuit"
        render_mode      — always "gpt_edit" for the inpainting-based pipeline
        scene_bible      — style rules dict: layout zones, typography, colour contract,
                           educational contract, allowed/forbidden elements
        steps            — ordered list of 7 StepSpec objects
        captions         — 7 narration strings, one per frame (teacher-spoken prose)
        math_elements    — optional list of MathElement overlays
    """
    question_id: str
    question_text: str
    canonical_answer: str
    visual_family: str
    render_mode: str
    scene_bible: Dict[str, Any]
    steps: List[StepSpec]
    captions: List[str]
    math_elements: List[MathElement]
