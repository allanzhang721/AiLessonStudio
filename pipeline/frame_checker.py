"""Checker 2: fast local render checks followed by one semantic vision audit."""
from __future__ import annotations

import base64
import io
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional

from PIL import Image, ImageFilter, ImageStat


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _score_range(value: float, low: float, high: float) -> float:
    return _clamp01((value - low) / (high - low)) if high > low else 0.0


def _frame_metrics(frame_path: Path) -> dict[str, float]:
    with Image.open(frame_path) as img:
        rgb = img.convert("RGB")
        gray = rgb.convert("L")
        width, height = rgb.size
        stat = ImageStat.Stat(gray)
        edge_std = float(ImageStat.Stat(gray.filter(ImageFilter.FIND_EDGES)).stddev[0])
        band = gray.crop((0, int(height * 0.78), width, height))
        return {
            "width": float(width), "height": float(height),
            "gray_mean": float(stat.mean[0]), "gray_std": float(stat.stddev[0]),
            "edge_std": edge_std,
            "caption_band_std": float(ImageStat.Stat(band).stddev[0]),
        }


def _heuristic_frame_quality_score(metrics: dict[str, float]) -> tuple[float, list[str]]:
    width, height = metrics["width"], metrics["height"]
    mean = metrics["gray_mean"]
    scores = {
        "low_resolution": 1.0 if width >= 1200 and height >= 800 else 0.4,
        "poor_exposure": min(_score_range(mean, 18, 70), _score_range(255 - mean, 18, 70)),
        "low_contrast": _score_range(metrics["gray_std"], 18, 60),
        "blurry_or_low_detail": _score_range(metrics["edge_std"], 14, 46),
        "caption_band_low_readability": _score_range(metrics["caption_band_std"], 8, 28),
    }
    weights = (0.20, 0.20, 0.25, 0.25, 0.10)
    score = sum(weight * value for weight, value in zip(weights, scores.values()))
    cutoffs = (0.8, 0.45, 0.45, 0.45, 0.4)
    issues = [name for (name, value), cutoff in zip(scores.items(), cutoffs) if value < cutoff]
    return round(_clamp01(score), 4), issues


def _validate_with_trained_model(frame_paths: list[Path], *, model_path: Optional[Path], threshold: float) -> dict[str, Any]:
    return {
        "mode": "trained", "pass": False, "overall_score": 0.0, "threshold": threshold,
        "error": "Saved research probes are benchmarks, not a production render-quality model.",
        "model_path": str(model_path) if model_path else None,
        "per_frame": [], "failed_steps": list(range(1, len(frame_paths) + 1)),
    }


def checker2_validate_frames(frame_paths: list[Path], *, threshold: float = 0.58,
                             backend: str = "heuristic", model_path: Optional[Path] = None) -> dict[str, Any]:
    """Validate render health in parallel without loading heavyweight ML libraries."""
    total_started = time.perf_counter()
    if not frame_paths:
        return {"checker_name": "checker2_frame_quality_v3", "mode": backend, "pass": False,
                "overall_score": 0.0, "threshold": threshold, "error": "No frames were provided.",
                "per_frame": [], "failed_steps": [],
                "metrics": {"total_latency_ms": 0.0, "frames_checked": 0, "parallel_workers": 0, "model_calls": 0}}
    requested_backend, fallback_reason = backend, None
    if backend == "trained":
        trained = _validate_with_trained_model(frame_paths, model_path=model_path, threshold=threshold)
        if not trained.get("error"):
            return trained
        backend, fallback_reason = "heuristic", trained["error"]
    if backend != "heuristic":
        return {"checker_name": "checker2_frame_quality_v3", "mode": backend, "pass": False,
                "overall_score": 0.0, "threshold": threshold, "error": f"Unsupported backend: {backend}",
                "per_frame": [], "failed_steps": [],
                "metrics": {"total_latency_ms": 0.0, "frames_checked": 0, "parallel_workers": 0, "model_calls": 0}}

    workers = min(4, len(frame_paths))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        measured = list(pool.map(_frame_metrics, frame_paths))
    per_frame = []
    for index, (path, metrics) in enumerate(zip(frame_paths, measured), 1):
        score, issues = _heuristic_frame_quality_score(metrics)
        per_frame.append({"step_id": index, "path": str(path), "score": score, "pass": score >= threshold,
                          "issues": issues, "metrics": {key: round(value, 2) for key, value in metrics.items()}})
    average = sum(item["score"] for item in per_frame) / len(per_frame)
    failed = [item["step_id"] for item in per_frame if not item["pass"]]
    elapsed = round((time.perf_counter() - total_started) * 1000, 2)
    result = {
        "checker_name": "checker2_frame_quality_v3",
        "mode": "heuristic_fallback" if fallback_reason else "parallel_heuristic",
        "pass": average >= threshold and not failed, "overall_score": round(average, 4),
        "threshold": threshold, "per_frame": per_frame, "failed_steps": failed,
        "metrics": {"total_latency_ms": elapsed, "frames_checked": len(frame_paths),
                    "parallel_workers": workers, "model_calls": 0},
        "method_comparison": [{"method": "Parallel pixel diagnostics", "trained": False,
                               "score": round(average, 4), "latency_ms": elapsed,
                               "role": "Resolution, exposure, contrast, detail, caption band"}],
        "trained_model_requested": requested_backend == "trained", "trained_model_used": False,
    }
    if fallback_reason:
        result["fallback_reason"] = fallback_reason
    return result


def _extract_json_object(text: str) -> dict[str, Any]:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("Vision reviewer returned no JSON object.")
    result = json.loads(text[start:end + 1])
    if not isinstance(result, dict):
        raise ValueError("Vision review must be an object.")
    return result


def _contact_sheet_data_url(frame_paths: list[Path]) -> str:
    thumbs = []
    for path in frame_paths:
        with Image.open(path) as img:
            thumb = img.convert("RGB")
            thumb.thumbnail((512, 342))
            thumbs.append(thumb.copy())
    width, height = max(i.width for i in thumbs), max(i.height for i in thumbs)
    sheet = Image.new("RGB", (width * len(thumbs), height), "white")
    for index, image in enumerate(thumbs):
        sheet.paste(image, (index * width, 0))
    buffer = io.BytesIO()
    sheet.save(buffer, format="JPEG", quality=82)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def checker2_validate_lesson_frames(frame_paths: list[Path], *, plan: Optional[dict[str, Any]] = None,
                                    client: Any = None, model: str = "gpt-5.6-terra",
                                    threshold: float = 0.65, backend: str = "heuristic",
                                    model_path: Optional[Path] = None) -> dict[str, Any]:
    """Run a local-first cascade; poor renders never incur a vision-model call."""
    total_started = time.perf_counter()
    technical = checker2_validate_frames(frame_paths, threshold=0.42, backend=backend, model_path=model_path)
    technical_ms = float(technical.get("metrics", {}).get("total_latency_ms", 0.0))
    comparison = list(technical.get("method_comparison", []))
    if not technical.get("pass"):
        return {"checker_name": "visual_quality_gate_v3", "mode": "technical_block", "pass": False,
                "overall_score": technical.get("overall_score", 0.0), "threshold": threshold,
                "technical": technical, "issues": ["Semantic review skipped because local render checks failed."],
                "failed_steps": technical.get("failed_steps", []), "per_frame": technical.get("per_frame", []),
                "decision_path": ["parallel_local_checks", "blocked_before_semantic_call"],
                "metrics": {"total_latency_ms": round((time.perf_counter() - total_started) * 1000, 2),
                            "technical_latency_ms": technical_ms, "semantic_latency_ms": 0.0, "model_calls": 0},
                "method_comparison": comparison, "trained_model_used": False}
    if client is None or not plan:
        technical.update({"checker_name": "visual_quality_gate_v3", "mode": "technical_only",
                          "scope": "No semantic/factual vision audit was available.",
                          "decision_path": ["parallel_local_checks", "technical_only"]})
        return technical

    prompt = f'''Review these ordered educational storyboard frames.
Question: {plan.get("question_text", "")}
Correct explanation: {plan.get("canonical_answer", "")}
Grade: {plan.get("grade", "")}
Frame captions: {plan.get("captions", [])}
Score 1-4: semantic_accuracy, teaching_alignment, sequence_continuity, legibility, visual_load.
Pass only if semantic_accuracy=4 and all others>=3. JSON only:
{{"scores": {{"semantic_accuracy": 1, "teaching_alignment": 1, "sequence_continuity": 1, "legibility": 1, "visual_load": 1}}, "failed_steps": [1], "issues": ["..."], "pass": false}}'''
    semantic_started = time.perf_counter()
    try:
        response = client.responses.create(model=model, input=[{"role": "user", "content": [
            {"type": "input_text", "text": prompt},
            {"type": "input_image", "image_url": _contact_sheet_data_url(frame_paths)}]}])
        semantic = _extract_json_object(response.output_text)
        names = ("semantic_accuracy", "teaching_alignment", "sequence_continuity", "legibility", "visual_load")
        scores = {key: max(1, min(4, int(semantic.get("scores", {}).get(key, 1)))) for key in names}
        semantic_pass = bool(semantic.get("pass")) and scores["semantic_accuracy"] == 4 and min(scores.values()) >= 3
        failed = sorted({int(item) for item in semantic.get("failed_steps", []) if str(item).isdigit()})
        issues = [str(item).strip() for item in semantic.get("issues", []) if str(item).strip()]
    except Exception as exc:
        scores, semantic_pass = {}, False
        failed, issues = list(range(1, len(frame_paths) + 1)), [f"Semantic review failed: {type(exc).__name__}"]
    semantic_ms = round((time.perf_counter() - semantic_started) * 1000, 2)
    tech_score = float(technical.get("overall_score", 0.0))
    semantic_score = sum(scores.values()) / (4 * len(scores)) if scores else 0.0
    overall = round(0.35 * tech_score + 0.65 * semantic_score, 4)
    return {"checker_name": "visual_quality_gate_v3", "mode": "technical_plus_semantic",
            "pass": bool(technical.get("pass")) and semantic_pass and overall >= threshold,
            "overall_score": overall, "threshold": threshold, "technical": technical,
            "semantic_scores": scores, "issues": issues, "failed_steps": failed,
            "per_frame": technical.get("per_frame", []),
            "decision_path": ["parallel_local_checks", "single_semantic_contact_sheet"],
            "metrics": {"total_latency_ms": round((time.perf_counter() - total_started) * 1000, 2),
                        "technical_latency_ms": technical_ms, "semantic_latency_ms": semantic_ms, "model_calls": 1},
            "method_comparison": comparison + [{"method": "Vision semantic audit", "trained": False,
                                                "score": round(semantic_score, 4), "latency_ms": semantic_ms,
                                                "role": "Accuracy, alignment, continuity, legibility"}],
            "trained_model_used": False}
