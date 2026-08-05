"""
frame_checker.py — Checker 2: quality validation for generated storyboard frames.

This checker is intentionally pluggable:
- "heuristic" mode works out-of-the-box and does not require training artifacts.
- "trained" mode is a future extension point for a self-trained model.

Public API:
  checker2_validate_frames(...)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from PIL import Image, ImageFilter, ImageStat


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _score_range(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    return _clamp01((value - low) / (high - low))


def _frame_metrics(frame_path: Path) -> dict[str, float]:
    with Image.open(frame_path) as img:
        rgb = img.convert("RGB")
        gray = rgb.convert("L")

        width, height = rgb.size
        stat = ImageStat.Stat(gray)
        mean = float(stat.mean[0])
        std = float(stat.stddev[0])

        # Edge variance acts as a lightweight sharpness proxy.
        edge = gray.filter(ImageFilter.FIND_EDGES)
        edge_std = float(ImageStat.Stat(edge).stddev[0])

        # Ensure caption band (bottom area) is not fully blank.
        band_top = int(height * 0.78)
        caption_band = gray.crop((0, band_top, width, height))
        caption_std = float(ImageStat.Stat(caption_band).stddev[0])

    return {
        "width": float(width),
        "height": float(height),
        "gray_mean": mean,
        "gray_std": std,
        "edge_std": edge_std,
        "caption_band_std": caption_std,
    }


def _heuristic_frame_quality_score(metrics: dict[str, float]) -> tuple[float, list[str]]:
    issues: list[str] = []

    width = metrics["width"]
    height = metrics["height"]
    gray_mean = metrics["gray_mean"]
    gray_std = metrics["gray_std"]
    edge_std = metrics["edge_std"]
    caption_band_std = metrics["caption_band_std"]

    s_res = 1.0 if width >= 1200 and height >= 800 else 0.4
    s_exposure = min(_score_range(gray_mean, 18.0, 70.0), _score_range(255.0 - gray_mean, 18.0, 70.0))
    s_contrast = _score_range(gray_std, 18.0, 60.0)
    s_edges = _score_range(edge_std, 14.0, 46.0)
    s_caption = _score_range(caption_band_std, 8.0, 28.0)

    score = (
        0.20 * s_res
        + 0.20 * s_exposure
        + 0.25 * s_contrast
        + 0.25 * s_edges
        + 0.10 * s_caption
    )

    if s_res < 0.8:
        issues.append("low_resolution")
    if s_exposure < 0.45:
        issues.append("poor_exposure")
    if s_contrast < 0.45:
        issues.append("low_contrast")
    if s_edges < 0.45:
        issues.append("blurry_or_low_detail")
    if s_caption < 0.4:
        issues.append("caption_band_low_readability")

    return round(_clamp01(score), 4), issues


def _validate_with_trained_model(
    frame_paths: list[Path],
    *,
    model_path: Optional[Path],
    threshold: float,
) -> dict[str, Any]:
    # Placeholder contract for future self-trained model integration.
    # Keep response schema compatible with heuristic mode.
    return {
        "checker_name": "checker2_frame_quality_v1",
        "mode": "trained",
        "pass": False,
        "overall_score": 0.0,
        "threshold": threshold,
        "error": (
            "Trained Checker 2 backend is not implemented yet. "
            "Provide your model integration in _validate_with_trained_model()."
        ),
        "model_path": str(model_path) if model_path is not None else None,
        "per_frame": [],
        "failed_steps": list(range(1, len(frame_paths) + 1)),
    }


def checker2_validate_frames(
    frame_paths: list[Path],
    *,
    threshold: float = 0.58,
    backend: str = "heuristic",
    model_path: Optional[Path] = None,
) -> dict[str, Any]:
    """Validate generated frames with Checker 2.

    Args:
        frame_paths: Ordered list of rendered frame image paths.
        threshold: Per-frame and overall pass threshold in [0, 1].
        backend: "heuristic" (default) or "trained" (future model).
        model_path: Optional path to a trained model artifact for future use.
    """
    if not frame_paths:
        return {
            "checker_name": "checker2_frame_quality_v1",
            "mode": backend,
            "pass": False,
            "overall_score": 0.0,
            "threshold": threshold,
            "error": "No frames were provided to Checker 2.",
            "per_frame": [],
            "failed_steps": [],
        }

    if backend == "trained":
        return _validate_with_trained_model(
            frame_paths,
            model_path=model_path,
            threshold=threshold,
        )

    if backend != "heuristic":
        return {
            "checker_name": "checker2_frame_quality_v1",
            "mode": backend,
            "pass": False,
            "overall_score": 0.0,
            "threshold": threshold,
            "error": f"Unsupported Checker 2 backend: {backend}",
            "per_frame": [],
            "failed_steps": [],
        }

    per_frame: list[dict[str, Any]] = []
    for idx, path in enumerate(frame_paths, start=1):
        metrics = _frame_metrics(path)
        score, issues = _heuristic_frame_quality_score(metrics)
        per_frame.append(
            {
                "step_id": idx,
                "path": str(path),
                "score": score,
                "pass": score >= threshold,
                "issues": issues,
                "metrics": {
                    "gray_mean": round(metrics["gray_mean"], 2),
                    "gray_std": round(metrics["gray_std"], 2),
                    "edge_std": round(metrics["edge_std"], 2),
                    "caption_band_std": round(metrics["caption_band_std"], 2),
                    "width": int(metrics["width"]),
                    "height": int(metrics["height"]),
                },
            }
        )

    avg = sum(item["score"] for item in per_frame) / len(per_frame)
    failed_steps = [item["step_id"] for item in per_frame if not item["pass"]]
    result = {
        "checker_name": "checker2_frame_quality_v1",
        "mode": "heuristic",
        "pass": (avg >= threshold) and (len(failed_steps) == 0),
        "overall_score": round(avg, 4),
        "threshold": threshold,
        "per_frame": per_frame,
        "failed_steps": failed_steps,
    }
    return result


def _extract_json_object(text: str) -> dict[str, Any]:
    import json

    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("Vision reviewer returned no JSON object.")
    value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("Vision review must be an object.")
    return value


def _contact_sheet_data_url(frame_paths: list[Path]) -> str:
    import base64
    import io

    thumbs: list[Image.Image] = []
    for path in frame_paths:
        with Image.open(path) as img:
            thumb = img.convert("RGB")
            thumb.thumbnail((512, 342))
            thumbs.append(thumb.copy())
    width = max(image.width for image in thumbs)
    height = max(image.height for image in thumbs)
    sheet = Image.new("RGB", (width * len(thumbs), height), "white")
    for index, image in enumerate(thumbs):
        sheet.paste(image, (index * width, 0))
    buffer = io.BytesIO()
    sheet.save(buffer, format="JPEG", quality=82)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def checker2_validate_lesson_frames(
    frame_paths: list[Path],
    *,
    plan: Optional[dict[str, Any]] = None,
    client: Any = None,
    model: str = "gpt-5.6-terra",
    threshold: float = 0.65,
) -> dict[str, Any]:
    """Combine render-health metrics with a semantic vision audit."""
    technical = checker2_validate_frames(frame_paths, threshold=0.42)
    if client is None or not plan:
        technical["checker_name"] = "visual_quality_gate_v2"
        technical["mode"] = "technical_only"
        technical["scope"] = "No semantic/factual vision audit was available."
        technical["pass"] = bool(technical.get("pass"))
        return technical

    captions = plan.get("captions", [])
    prompt = f"""Review this contact sheet of ordered educational storyboard frames.

Question: {plan.get("question_text", "")}
Correct explanation: {plan.get("canonical_answer", "")}
Grade: {plan.get("grade", "")}
Frame captions: {captions}

Score 1-4 for semantic_accuracy, teaching_alignment, sequence_continuity,
legibility, and visual_load. List failed frame numbers and concrete issues.
Pass only if semantic_accuracy is 4 and every other score is at least 3.
Return JSON only:
{{"scores": {{"semantic_accuracy": 1, "teaching_alignment": 1,
"sequence_continuity": 1, "legibility": 1, "visual_load": 1}},
"failed_steps": [1], "issues": ["..."], "pass": false}}"""
    try:
        response = client.responses.create(
            model=model,
            input=[{"role": "user", "content": [
                {"type": "input_text", "text": prompt},
                {"type": "input_image", "image_url": _contact_sheet_data_url(frame_paths)},
            ]}],
        )
        semantic = _extract_json_object(response.output_text)
        scores = {
            key: max(1, min(4, int(semantic.get("scores", {}).get(key, 1))))
            for key in ("semantic_accuracy", "teaching_alignment", "sequence_continuity", "legibility", "visual_load")
        }
        semantic_pass = bool(semantic.get("pass")) and scores["semantic_accuracy"] == 4 and min(scores.values()) >= 3
        failed = sorted({int(item) for item in semantic.get("failed_steps", []) if str(item).isdigit()})
        issues = [str(item).strip() for item in semantic.get("issues", []) if str(item).strip()]
    except Exception as exc:
        scores = {}
        semantic_pass = False
        failed = list(range(1, len(frame_paths) + 1))
        issues = [f"Semantic review failed: {type(exc).__name__}"]

    tech_score = float(technical.get("overall_score", 0.0))
    semantic_score = (sum(scores.values()) / (4 * len(scores))) if scores else 0.0
    overall = round(0.35 * tech_score + 0.65 * semantic_score, 4)
    return {
        "checker_name": "visual_quality_gate_v2",
        "mode": "technical_plus_semantic",
        "pass": bool(technical.get("pass")) and semantic_pass and overall >= threshold,
        "overall_score": overall,
        "threshold": threshold,
        "technical": technical,
        "semantic_scores": scores,
        "issues": issues,
        "failed_steps": failed,
        "per_frame": technical.get("per_frame", []),
    }
