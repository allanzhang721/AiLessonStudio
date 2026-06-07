"""
pipeline/__init__.py — Public API surface for the L15 educational video pipeline.

Import any of the four main entry points directly from the package:

    from pipeline import run_pipeline            # full end-to-end orchestrator
    from pipeline import question_explanation_grade_to_plan  # Stage 1: text → plan
    from pipeline import plan_to_images          # Stage 2: plan → PNG frames
    from pipeline import images_to_video         # Stage 3: frames → MP4
    from pipeline import validate_plan_schema    # schema checker (used internally)
    from pipeline import checker1_predict        # Checker 1: error-type classifier
    from pipeline import checker1_loop           # Checker 1: generate → check → fix loop

Provider helpers:
    from pipeline import build_text_client, build_image_client, build_tts_client
    from pipeline import available_text_providers, available_image_providers, available_video_providers

Pipeline stages in order:
  0. checker.py        — Checker 1 DistilBERT error-type classifier + LLM repair loop
  1. planner.py        — question + explanation + grade → 7-step storyboard Plan dict
  2. image_pipeline.py — Plan dict → 7 PNG frames (GPT image generate/edit per step)
  3. video_pipeline.py — PNG frames + optional TTS audio → GIF + MP4
"""

from .checker import checker1_predict, checker1_loop, build_checker_input_text
from .frame_checker import checker2_validate_frames
from .student_analyzer import analyze_student_weakness, infer_concept_tags
from .clients import build_text_client, build_image_client, build_video_client, build_tts_client
from .api_keys import available_text_providers, available_image_providers, available_video_providers
from .planner import question_explanation_grade_to_plan
from .image_pipeline import plan_to_images
from .video_pipeline import images_to_video
from .pipeline import run_pipeline
from .validation import validate_plan_schema

__all__ = [
    "checker1_predict",
    "checker1_loop",
    "build_checker_input_text",
    "checker2_validate_frames",
    "analyze_student_weakness",
    "infer_concept_tags",
    "build_text_client",
    "build_image_client",
    "build_video_client",
    "build_tts_client",
    "available_text_providers",
    "available_image_providers",
    "available_video_providers",
    "question_explanation_grade_to_plan",
    "plan_to_images",
    "images_to_video",
    "run_pipeline",
    "validate_plan_schema",
]
