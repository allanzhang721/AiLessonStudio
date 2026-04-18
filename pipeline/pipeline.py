"""
pipeline.py — End-to-end orchestrator for the L15 educational video pipeline.

Public entry point: run_pipeline()

Calls all three stages in order and records per-stage timing:
  Stage 1 — planner.question_explanation_grade_to_plan()
               question + explanation + grade → Plan dict + plan.json
  Stage 2 — image_pipeline.plan_to_images()
               Plan → 7 PNG frames (GPT image API or local placeholders)
  Stage 3a — make_gif()
               7 frames → storyboard.gif  (preview / Streamlit thumbnail)
  Stage 3b — video_pipeline.synthesize_clean_voiceover()
               Plan captions → TTS narration MP3 (optional, needs API key)
  Stage 3c — video_pipeline.images_to_video()
               7 frames (+ optional narration) → storyboard.mp4

Output layout (under output_root/<question_id>/):
  plan.json          — structured storyboard plan
  storyboard.gif     — animated GIF preview
  storyboard.mp4     — final video (with voiceover if TTS succeeded)
  frames/            — final captioned PNG frames
  frames_raw/        — raw API frames without caption band
  prompts/           — all prompt texts sent to the models
  voiceover_script.txt / voiceover_clean.mp3  — narration files (if generated)
  run_manifest.json  — artifact paths + timing for every stage

Provider selection:
  text_provider   — "openai" or "deepseek" (for planner + checker repair)
  image_provider  — "openai" or "wanx"     (for frame generation)
  video_provider  — "sora" or "wanx"       (for single-video, passed through)

Clients are built via clients.py which reads keys from api_keys.txt / env.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

from .checker import checker1_loop
from .clients import build_text_client, build_image_client, build_tts_client
from .config import (
    DEFAULT_ROOT,
    OPENAI_TEXT_MODEL,
    DEEPSEEK_TEXT_MODEL,
    OPENAI_IMAGE_MODEL,
    WANX_IMAGE_MODEL,
)
from .image_pipeline import plan_to_images
from .planner import question_explanation_grade_to_plan
from .utils import ensure_dir, make_gif, save_json, save_text
from .video_pipeline import build_narration_script, estimate_video_fps, images_to_video, synthesize_clean_voiceover


def _model_for_text_provider(provider: str) -> str:
    """Return the model name string for the given text provider."""
    if provider == "deepseek":
        return DEEPSEEK_TEXT_MODEL
    return OPENAI_TEXT_MODEL


def _model_for_image_provider(provider: str) -> str:
    """Return the model name string for the given image provider."""
    if provider == "wanx":
        return WANX_IMAGE_MODEL
    return OPENAI_IMAGE_MODEL


def run_pipeline(
    question: str,
    explanation: str,
    grade: int,
    subject: str = "",
    question_id: Optional[str] = None,
    output_root: Path = DEFAULT_ROOT,
    run_openai: bool = True,
    run_checker: bool = True,
    checker_max_rounds: int = 3,
    checker_confidence_threshold: float = 0.5,
    text_provider: str = "openai",
    image_provider: str = "openai",
) -> dict:
    """
    End-to-end pipeline:
      0) explanation quality check (Checker 1 DistilBERT) + LLM repair loop
      1) question/explanation/grade -> plan
      2) plan -> 7 frame images
      3) images -> GIF + MP4 video
    """
    t0 = time.time()

    # Build per-stage clients via the centralised client factory
    text_client = build_text_client(text_provider) if run_openai else None
    image_client = build_image_client(image_provider) if run_openai else None
    tts_client = build_tts_client() if run_openai else None

    text_model = _model_for_text_provider(text_provider)
    image_model = _model_for_image_provider(image_provider)

    stage_times: dict[str, float] = {}

    # --- Stage 0: Checker 1 quality gate ---
    checker_result = None
    if run_checker:
        t_checker = time.time()
        try:
            checker_result = checker1_loop(
                client=text_client,
                question=question,
                explanation=explanation,
                grade=grade,
                subject=subject,
                max_rounds=checker_max_rounds,
                confidence_threshold=checker_confidence_threshold,
                model=text_model,
            )
            if checker_result["was_revised"]:
                explanation = checker_result["final_explanation"]
        except Exception as exc:
            checker_result = {"error": str(exc), "was_revised": False, "rounds": [], "total_rounds": 0, "final_explanation": explanation}
        stage_times["checker_seconds"] = round(time.time() - t_checker, 3)

    t_plan = time.time()
    plan = question_explanation_grade_to_plan(
        question=question,
        explanation=explanation,
        grade=grade,
        subject=subject,
        question_id=question_id,
        client=text_client,
        model=text_model,
    )
    stage_times["plan_seconds"] = round(time.time() - t_plan, 3)

    out_dir = ensure_dir(Path(output_root) / plan["question_id"])
    save_json(plan, out_dir / "plan.json")

    prompt_dir = ensure_dir(out_dir / "prompts")
    planner_debug_prompts = plan.get("planner_meta", {}).get("debug_prompts", {})
    if isinstance(planner_debug_prompts, dict):
        for name, text in planner_debug_prompts.items():
            if isinstance(text, str) and text.strip():
                save_text(text, prompt_dir / f"{name}.txt")

    t_images = time.time()
    frames = plan_to_images(plan=plan, out_dir=out_dir, client=image_client, image_model=image_model)
    stage_times["images_seconds"] = round(time.time() - t_images, 3)

    t_gif = time.time()
    gif_path = make_gif(frames, out_dir / "storyboard.gif", fps=1.0)
    stage_times["gif_seconds"] = round(time.time() - t_gif, 3)

    t_voice = time.time()
    narration_script = build_narration_script(plan)
    voiceover_path = synthesize_clean_voiceover(tts_client, plan, out_dir)
    stage_times["voiceover_seconds"] = round(time.time() - t_voice, 3)

    t_video = time.time()
    video_fps = 1.0
    if voiceover_path is not None:
        video_fps = estimate_video_fps(len(frames), narration_script)
    video_path = images_to_video(frames, out_dir / "storyboard.mp4", fps=video_fps, audio_path=voiceover_path)
    stage_times["video_seconds"] = round(time.time() - t_video, 3)

    total_seconds = round(time.time() - t0, 3)
    manifest = {
        "question_id": plan.get("question_id"),
        "used_openai": text_client is not None or image_client is not None,
        "text_provider": text_provider,
        "image_provider": image_provider,
        "checker_result": checker_result,
        "planner_meta": plan.get("planner_meta", {}),
        "render_meta": plan.get("render_meta", {}),
        "stage_times": stage_times,
        "total_seconds": total_seconds,
        "artifacts": {
            "plan_json": str(out_dir / "plan.json"),
            "gif": str(gif_path),
            "video": str(video_path),
            "frames_dir": str(out_dir / "frames"),
            "prompts_dir": str(prompt_dir),
            "voiceover": str(voiceover_path) if voiceover_path is not None else None,
            "voiceover_script": str(out_dir / "voiceover_script.txt") if (out_dir / "voiceover_script.txt").exists() else None,
        },
    }
    save_json(manifest, out_dir / "run_manifest.json")

    return {
        "plan": plan,
        "out_dir": out_dir,
        "frames": frames,
        "gif_path": gif_path,
        "video_path": video_path,
        "voiceover_path": voiceover_path,
        "used_openai": text_client is not None or image_client is not None,
        "text_provider": text_provider,
        "image_provider": image_provider,
        "checker_result": checker_result,
        "manifest_path": out_dir / "run_manifest.json",
        "stage_times": stage_times,
        "total_seconds": total_seconds,
    }
