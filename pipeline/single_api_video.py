"""
single_api_video.py — Alternative video path using OpenAI's Sora video model.

Unlike the storyboard pipeline (image_pipeline.py + video_pipeline.py), which
assembles a slideshow from 7 individually generated PNG frames, this module sends
a single rich prompt to Sora to produce one continuous animated video clip.

Public entry point:
  generate_single_video_from_run_dir(run_dir, …)
    Reads an existing plan.json + first frame from a pipeline run directory,
    calls Sora, burns captions, optionally synthesizes TTS narration, and muxes
    everything into a final MP4 inside run_dir/single_api_video/.

CLI usage:
  python single_api_video.py --run_dir output/my_run --model sora-2 --seconds 8

Key internal steps:
  _prepare_anchor_frame()         — resize/letterbox the first PNG frame to Sora's
                                    required resolution (e.g. 1280×720 for sora-2)
  build_single_video_prompt()     — build a rich natural-language prompt that describes
                                    the full 7-step timeline, visual style, and labels
  _upload_anchor_file()           — upload the anchor image to OpenAI Files API
  Sora API call                   — client.videos.generate() with the anchor image file
  _burn_captions()                — Python/PIL: overlay synchronised caption band onto
                                    each video frame (does not require ffmpeg)
  _synthesize_voiceover_with_fallback() — try OpenAI TTS first, then macOS `say` fallback
  _extend_video_to_duration()     — pad the video with a frozen last frame if the TTS
                                    audio is longer than the generated video clip
  _mux_voiceover()                — ffmpeg: attach narration as the only audio track

Supported Sora models and their allowed sizes:
  sora-2:     1280x720, 720x1280
  sora-2-pro: 1280x720, 720x1280, 1792x1024, 1024x1792

Output files (inside run_dir/single_api_video/):
  single_api_video.mp4                         — raw Sora video
  single_api_video_captioned.mp4               — with burned caption band
  single_api_video_captioned_with_voiceover.mp4 — final with audio
  single_video_job.json / single_video_result.json — API request/response metadata
  voiceover_script.txt / voiceover_clean.mp3   — narration files
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import textwrap
from pathlib import Path
from typing import Any, Optional

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None

from pipeline.api_keys import get_key
from pipeline.config import TTS_MODEL, TTS_VOICE
from pipeline.utils import ensure_dir, extract_caption_text, save_json, save_text


# Sora API requires exact size strings; map to (width, height) tuples.
VALID_SIZES = {
    "1280x720": (1280, 720),
    "720x1280": (720, 1280),
    "1792x1024": (1792, 1024),
    "1024x1792": (1024, 1792),
}

# Per-model allowed size sets — used to auto-correct invalid size selections.
MODEL_ALLOWED_SIZES = {
    "sora-2": {"1280x720", "720x1280"},
    "sora-2-pro": {"1280x720", "720x1280", "1792x1024", "1024x1792"},
}

VALID_SECONDS = {4, 8, 12}          # Sora only supports these clip durations.
CAPTION_BAND_H = 144                 # Pixel height of the caption band added below each frame.


def _path_relative_to_run(path: Optional[Path], run_path: Path) -> Optional[str]:
    """Return run-relative path text when possible, else fallback to absolute text."""
    if path is None:
        return None

    target = Path(path)
    run_abs = run_path.resolve()
    target_abs = target.resolve()
    try:
        return str(target_abs.relative_to(run_abs))
    except Exception:
        return str(target)


def _portable_result(result: dict[str, Any], run_path: Path) -> dict[str, Any]:
    """Return a copy of result with run-scoped relative paths for persisted metadata."""
    portable = dict(result)
    portable["run_dir"] = "."
    for key in (
        "anchor_frame",
        "prepared_anchor_frame",
        "prompt_path",
        "video_path",
        "silent_video_path",
        "captioned_video_path",
        "voiceover_path",
    ):
        value = portable.get(key)
        if isinstance(value, str) and value.strip():
            portable[key] = _path_relative_to_run(Path(value), run_path)
    return portable


def _build_client() -> OpenAI:
    """Instantiate an OpenAI client, raising clearly if prerequisites are missing."""
    if OpenAI is None:
        raise RuntimeError("openai package is not installed in this environment")
    key = get_key("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set in api_keys.txt or environment")
    return OpenAI(api_key=key)


def _ffmpeg_executable() -> Optional[str]:
    """Return the path to an ffmpeg binary, or None if not available.

    Checks PATH first, then tries imageio_ffmpeg as a fallback for
    environments where ffmpeg was installed as a Python package.
    """
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg  # type: ignore

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _load_plan(run_dir: Path) -> dict:
    """Load and parse plan.json from an existing pipeline run directory."""
    plan_path = run_dir / "plan.json"
    if not plan_path.exists():
        raise FileNotFoundError(f"Missing plan file: {plan_path}")
    return json.loads(plan_path.read_text(encoding="utf-8"))


def _find_anchor_frame(run_dir: Path) -> Path:
    """Return the path to step_01.png, checking frames_raw/ before frames/."""
    candidates = [
        run_dir / "frames_raw" / "step_01.png",
        run_dir / "frames" / "step_01.png",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find first frame in {run_dir}")


def _prepare_anchor_frame(source_path: Path, size: str, out_path: Path) -> Path:
    """Letterbox/resize the first pipeline frame to Sora's required canvas size.

    Uses ImageOps.contain (scale to fit, no crop) then centres on a white canvas
    so no diagram content is cropped when the frame aspect ratio doesn't match.
    """
    if size not in VALID_SIZES:
        raise ValueError(f"Unsupported size {size}. Choose one of: {', '.join(VALID_SIZES)}")

    target_w, target_h = VALID_SIZES[size]
    base = Image.open(source_path).convert("RGB")
    fitted = ImageOps.contain(base, (target_w, target_h), Image.Resampling.LANCZOS)

    canvas = Image.new("RGB", (target_w, target_h), "white")
    offset_x = (target_w - fitted.width) // 2
    offset_y = (target_h - fitted.height) // 2
    canvas.paste(fitted, (offset_x, offset_y))

    ensure_dir(out_path.parent)
    canvas.save(out_path, format="PNG")
    return out_path


def _resolve_size_for_model(model: str, size: str) -> str:
    allowed = MODEL_ALLOWED_SIZES.get(model)
    if not allowed:
        return size
    if size in allowed:
        return size
    if size.endswith("x1024"):
        return "1280x720"
    if size.startswith("1024x"):
        return "720x1280"
    raise ValueError(f"size {size} is not supported for model {model}. Allowed sizes: {sorted(allowed)}")


def _collect_exact_labels(plan: dict) -> list[str]:
    """Extract quoted label strings from step add lists and special caption keywords.

    These are injected into the Sora prompt as "exact labels to preserve" so
    the model maintains consistent on-screen text spelling across the clip.
    """
    labels: set[str] = set()
    for step in plan.get("steps", []):
        if not isinstance(step, dict):
            continue
        for item in step.get("add", []):
            if not isinstance(item, str):
                continue
            for match in re.findall(r"'([^']+)'", item):
                cleaned = match.strip()
                if cleaned:
                    labels.add(cleaned)
    for caption in plan.get("captions", []):
        text = extract_caption_text(caption, fallback="")
        if "trophic cascade" in text.lower():
            labels.add("Trophic cascade")
    return sorted(labels)


def _format_timeline(plan: dict, seconds: int) -> str:
    """Format the 7-step plan into a timestamped storyboard timeline string.

    Divides the clip duration equally among steps (beat = seconds / 7) and
    produces one bullet per step describing start/end time, goal, key visual
    change, and spoken caption. Injected verbatim into the Sora prompt.
    """
    steps = plan.get("steps", [])
    if not steps:
        return ""
    beat = seconds / max(1, len(steps))
    lines: list[str] = []
    for idx, step in enumerate(steps):
        start = round(idx * beat, 1)
        end = round(min(seconds, (idx + 1) * beat), 1)
        goal = str(step.get("goal", "")).strip()
        delta = str(step.get("delta", "")).strip()
        add_items = [str(item).strip() for item in step.get("add", []) if str(item).strip()]
        add_summary = "; ".join(add_items[:3])
        caption = extract_caption_text(
            plan.get("captions", [""] * len(steps))[idx] if idx < len(plan.get("captions", [])) else "",
            fallback="",
        )
        lines.append(
            f"- {start:.1f}s to {end:.1f}s: {goal} {delta} Key visible change: {add_summary} Spoken meaning: {caption}"
        )
    return "\n".join(lines)


def build_single_video_prompt(plan: dict, seconds: int) -> str:
    scene_bible = plan.get("scene_bible", {}) if isinstance(plan.get("scene_bible"), dict) else {}
    educational = scene_bible.get("educational_contract", {}) if isinstance(scene_bible.get("educational_contract"), dict) else {}
    labels = _collect_exact_labels(plan)
    labels_text = ", ".join(labels) if labels else "No extra labels beyond the reference image."

    return f"""
Create one continuous educational explainer video with no cuts.

Topic: {plan.get('question_text', '')}
Canonical explanation: {plan.get('canonical_answer', '')}
Audience: {educational.get('audience', '')}
Visual style: {scene_bible.get('style', 'flat-vector 2D educational illustration, white background')}
Layout guidance: {scene_bible.get('layout_guidance', '')}
Object design cues: {scene_bible.get('object_design_cues', '')}
Instructional approach: {educational.get('instructional_approach', '')}

Use the input reference image as the exact opening frame and visual anchor.
Preserve the same species identities, relative placement, flat-vector board style, white background, and educational diagram look.
Do not turn this into cinematic realism.
Do not use scene cuts, montage edits, or camera jumps.
Use only subtle motion: gentle reveal, morph, highlight, or fade transitions that keep the diagram readable.
Maintain clean spelling for on-screen text. Exact labels to preserve when shown: {labels_text}
Keep text legible and stable. Do not invent extra species, extra labels, extra arrows, or decorative background elements.
No background music, no ambient sound, and no sound effects.

Storyboard timeline for the single continuous clip:
{_format_timeline(plan, seconds)}

The video should feel like one teacher-controlled board explanation where each change smoothly builds on the previous state.
By the end, the viewer should clearly understand that removing one species can trigger a trophic cascade affecting many other species and ecosystem stability.
""".strip()


def _upload_anchor_file(client: OpenAI, image_path: Path) -> str:
    with image_path.open("rb") as file_handle:
        uploaded = client.files.create(file=file_handle, purpose="user_data")
    return uploaded.id


def _to_plain_dict(obj: Any) -> dict:
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if isinstance(obj, dict):
        return obj
    return {"value": str(obj)}


def _download_video_bytes(content: Any) -> bytes:
    if isinstance(content, (bytes, bytearray)):
        return bytes(content)
    if hasattr(content, "read"):
        return content.read()
    if hasattr(content, "content"):
        return bytes(content.content)
    if hasattr(content, "iter_bytes"):
        return b"".join(content.iter_bytes())
    if hasattr(content, "text"):
        return str(content.text).encode("utf-8")
    raise TypeError(f"Unsupported download content type: {type(content)!r}")


def _mux_voiceover(video_path: Path, audio_path: Path, out_path: Path) -> Path:
    ffmpeg_exe = _ffmpeg_executable()
    if ffmpeg_exe is None:
        return video_path

    cmd = [
        ffmpeg_exe,
        "-y",
        "-i",
        str(video_path),
        "-i",
        str(audio_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        str(out_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return out_path if out_path.exists() and out_path.stat().st_size > 0 else video_path


def _media_duration_seconds(path: Path) -> float:
    ffmpeg_exe = _ffmpeg_executable()
    if ffmpeg_exe is None:
        return 0.0

    cmd = [ffmpeg_exe, "-i", str(path), "-f", "null", "-"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    text = (proc.stderr or "") + "\n" + (proc.stdout or "")
    match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", text)
    if not match:
        return 0.0
    hh = int(match.group(1))
    mm = int(match.group(2))
    ss = float(match.group(3))
    return hh * 3600 + mm * 60 + ss


def _extend_video_to_duration(video_path: Path, min_seconds: float, out_path: Path) -> Path:
    current = _media_duration_seconds(video_path)
    delta = max(0.0, min_seconds - current)
    if delta <= 0.05:
        return video_path

    ffmpeg_exe = _ffmpeg_executable()
    if ffmpeg_exe is not None:
        cmd = [
            ffmpeg_exe,
            "-y",
            "-i",
            str(video_path),
            "-vf",
            f"tpad=stop_mode=clone:stop_duration={delta:.3f}",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-an",
            str(out_path),
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            if out_path.exists() and out_path.stat().st_size > 0:
                return out_path
        except Exception:
            pass

    # Fallback: extend by repeating the final frame in Python.
    ensure_dir(out_path.parent)
    reader = imageio.get_reader(str(video_path))
    meta = reader.get_meta_data()
    fps = float(meta.get("fps", 24.0) or 24.0)
    writer = imageio.get_writer(str(out_path), fps=fps, codec="libx264", quality=8)

    last_frame = None
    try:
        for frame in reader:
            last_frame = frame
            writer.append_data(frame)

        if last_frame is None:
            return video_path

        extra_frames = max(1, int(round(delta * fps)))
        for _ in range(extra_frames):
            writer.append_data(last_frame)
    finally:
        writer.close()
        reader.close()

    return out_path if out_path.exists() and out_path.stat().st_size > 0 else video_path


def _build_narration_from_plan(plan: dict) -> str:
    captions = _video_caption_texts(plan, shorten_for_video=True)
    return "\n".join(captions)


def _shorten_caption_text(text: str, max_chars: int = 120) -> str:
    clean = " ".join(text.split()).strip()
    if not clean:
        return ""

    # Prefer one clear sentence for video pacing.
    first_sentence = re.split(r"(?<=[.!?])\s+", clean)[0].strip()
    if first_sentence:
        clean = first_sentence

    if len(clean) <= max_chars:
        return clean

    clipped = clean[: max_chars - 1].rsplit(" ", 1)[0].strip()
    if not clipped:
        clipped = clean[: max_chars - 1].strip()
    return clipped + "…"


def _video_caption_texts(plan: dict, shorten_for_video: bool) -> list[str]:
    raw = plan.get("captions") if isinstance(plan.get("captions"), list) else []
    out: list[str] = []
    for cap in raw:
        text = extract_caption_text(cap, fallback="").strip()
        if not text:
            continue
        out.append(_shorten_caption_text(text) if shorten_for_video else text)
    return out


def _synthesize_voiceover_api_content_only(client: OpenAI, plan: dict, out_dir: Path) -> Optional[Path]:
    script = _build_narration_from_plan(plan)
    if not script.strip():
        return None

    script_path = out_dir / "voiceover_script.txt"
    script_path.write_text(script, encoding="utf-8")
    audio_path = out_dir / "voiceover_clean.mp3"

    instructions = (
        "Speak in a clear educational tone with steady pacing and crisp articulation. "
        "No background music, no ambient sounds, no sound effects, narration voice only."
    )
    try:
        response = client.audio.speech.create(
            model=TTS_MODEL,
            voice=TTS_VOICE,
            input=script,
            instructions=instructions,
            format="mp3",
        )
        if hasattr(response, "stream_to_file"):
            response.stream_to_file(str(audio_path))
        elif hasattr(response, "read"):
            audio_path.write_bytes(response.read())
        elif hasattr(response, "content"):
            audio_path.write_bytes(response.content)
        else:
            return None
        return audio_path if audio_path.exists() and audio_path.stat().st_size > 0 else None
    except Exception:
        return None


def _local_macos_voiceover(plan: dict, out_dir: Path) -> Optional[Path]:
    if shutil.which("say") is None:
        return None

    script = _build_narration_from_plan(plan)
    if not script:
        return None

    script_path = out_dir / "voiceover_script.txt"
    script_path.write_text(script, encoding="utf-8")

    aiff_path = out_dir / "voiceover_local.aiff"
    mp3_path = out_dir / "voiceover_clean.mp3"

    say_cmd = ["say", "-v", "Samantha", "-f", str(script_path), "-o", str(aiff_path)]
    subprocess.run(say_cmd, check=True, capture_output=True, text=True)

    ffmpeg_exe = _ffmpeg_executable()
    if ffmpeg_exe is None:
        return aiff_path if aiff_path.exists() else None

    convert_cmd = [
        ffmpeg_exe,
        "-y",
        "-i",
        str(aiff_path),
        "-codec:a",
        "libmp3lame",
        "-q:a",
        "2",
        str(mp3_path),
    ]
    subprocess.run(convert_cmd, check=True, capture_output=True, text=True)
    return mp3_path if mp3_path.exists() and mp3_path.stat().st_size > 0 else None


def _synthesize_voiceover_with_fallback(plan: dict, out_dir: Path) -> Optional[Path]:
    client = None
    try:
        client = _build_client()
    except Exception:
        client = None

    if client is not None:
        voice = _synthesize_voiceover_api_content_only(client, plan, out_dir)
        if voice is not None:
            return voice

    try:
        return _local_macos_voiceover(plan, out_dir)
    except Exception:
        return None


def _wrap_caption_lines(text: str, width: int = 64) -> list[str]:
    normalized = " ".join(text.split())
    if not normalized:
        return []
    return textwrap.wrap(normalized, width=width)


def _paginate_caption(caption: str, width: int = 64, max_lines_per_page: int = 2) -> list[str]:
    lines = _wrap_caption_lines(caption, width=width)
    if not lines:
        return []
    pages: list[str] = []
    for i in range(0, len(lines), max_lines_per_page):
        pages.append("\n".join(lines[i:i + max_lines_per_page]))
    return pages


def _escape_filter_path(path: Path) -> str:
    text = str(path)
    text = text.replace("\\", r"\\")
    text = text.replace(":", r"\:")
    text = text.replace("'", r"\'")
    return text


def _load_caption_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
        "DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for name in candidates:
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _build_caption_timeline(plan: dict, seconds: int, shorten_for_video: bool = True) -> list[tuple[float, float, str]]:
    captions = _video_caption_texts(plan, shorten_for_video=shorten_for_video)
    if not captions:
        return []

    beat = seconds / max(1, len(captions))
    timeline: list[tuple[float, float, str]] = []
    for i, caption in enumerate(captions):
        pages = _paginate_caption(caption, width=64, max_lines_per_page=2)
        if not pages:
            continue
        start = float(round(i * beat, 3))
        end = float(round(min(seconds, (i + 1) * beat), 3))
        span = max(0.05, end - start)
        page_span = span / len(pages)
        for p_idx, page_text in enumerate(pages):
            page_start = float(round(start + p_idx * page_span, 3))
            page_end = float(round(end if p_idx == len(pages) - 1 else start + (p_idx + 1) * page_span, 3))
            timeline.append((page_start, page_end, page_text))
    return timeline


def _burn_captions(video_path: Path, plan: dict, seconds: int, out_path: Path, shorten_for_video: bool = True) -> Path:
    timeline = _build_caption_timeline(plan, seconds=seconds, shorten_for_video=shorten_for_video)
    if not timeline:
        return video_path

    ensure_dir(out_path.parent)
    reader = imageio.get_reader(str(video_path))
    meta = reader.get_meta_data()
    fps = float(meta.get("fps", 24.0) or 24.0)

    writer = imageio.get_writer(str(out_path), fps=fps, codec="libx264", quality=8)
    font = _load_caption_font(30)

    try:
        active_idx = 0
        for frame_idx, frame in enumerate(reader):
            t = frame_idx / fps

            while active_idx + 1 < len(timeline) and t > timeline[active_idx][1]:
                active_idx += 1

            caption_text = ""
            if active_idx < len(timeline):
                start, end, text = timeline[active_idx]
                if start <= t <= end:
                    caption_text = text

            img = Image.fromarray(frame).convert("RGB")
            w, h = img.size

            canvas = Image.new("RGB", (w, h + CAPTION_BAND_H), "#000000")
            canvas.paste(img, (0, 0))
            draw = ImageDraw.Draw(canvas)
            draw.line([(0, h), (w, h)], fill="#1F2937", width=2)

            if caption_text:
                bbox = draw.multiline_textbbox((0, 0), caption_text, font=font, spacing=8, align="center")
                tw = bbox[2] - bbox[0]
                th = bbox[3] - bbox[1]
                x = max(20, (w - tw) // 2)
                y = h + max(12, (CAPTION_BAND_H - th) // 2)
                draw.rounded_rectangle(
                    (x - 16, y - 10, x + tw + 16, y + th + 12),
                    radius=12,
                    fill="#111827",
                    outline="#374151",
                    width=1,
                )
                draw.multiline_text((x, y), caption_text, font=font, fill="#FFFFFF", spacing=8, align="center")

            writer.append_data(np.asarray(canvas))
    finally:
        writer.close()
        reader.close()

    return out_path if out_path.exists() and out_path.stat().st_size > 0 else video_path


def _postprocess_single_video(
    *,
    run_path: Path,
    plan: dict,
    base_video_path: Path,
    seconds: int,
    add_voiceover: bool,
    add_captions: bool,
    shorten_for_video: bool,
) -> dict:
    if not base_video_path.exists():
        raise FileNotFoundError(f"Base video not found for post-processing: {base_video_path}")

    out_dir = ensure_dir(run_path / "single_api_video")
    captioned_video_path = base_video_path
    if add_captions:
        captioned_video_path = _burn_captions(
            base_video_path,
            plan,
            seconds=seconds,
            out_path=out_dir / "single_api_video_captioned.mp4",
            shorten_for_video=shorten_for_video,
        )

    final_video_path = captioned_video_path
    voiceover_path: Optional[Path] = None
    if add_voiceover:
        voiceover_path = _synthesize_voiceover_with_fallback(plan, out_dir)
        if voiceover_path is None:
            raise RuntimeError("Voiceover was requested but TTS generation failed")
        extended_video_path = _extend_video_to_duration(
            captioned_video_path,
            min_seconds=_media_duration_seconds(voiceover_path),
            out_path=out_dir / "single_api_video_captioned_extended.mp4",
        )
        final_video_path = _mux_voiceover(
            extended_video_path,
            voiceover_path,
            out_dir / "single_api_video_captioned_with_voiceover.mp4",
        )

    return {
        "video_path": str(final_video_path),
        "silent_video_path": str(base_video_path),
        "captioned_video_path": str(captioned_video_path),
        "voiceover_path": str(voiceover_path) if voiceover_path is not None else None,
    }


def postprocess_existing_single_video_from_run_dir(
    run_dir: Path | str,
    seconds: int = 12,
    add_voiceover: bool = True,
    add_captions: bool = True,
    shorten_for_video: bool = True,
) -> dict:
    """Post-process an already downloaded single_api_video.mp4 with captions and optional voiceover."""
    if seconds not in VALID_SECONDS:
        raise ValueError(f"seconds must be one of {sorted(VALID_SECONDS)}")

    run_path = Path(run_dir)
    if not run_path.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_path}")

    plan = _load_plan(run_path)
    out_dir = ensure_dir(run_path / "single_api_video")
    base_video_path = out_dir / "single_api_video.mp4"
    processed = _postprocess_single_video(
        run_path=run_path,
        plan=plan,
        base_video_path=base_video_path,
        seconds=seconds,
        add_voiceover=add_voiceover,
        add_captions=add_captions,
        shorten_for_video=shorten_for_video,
    )

    result = {
        "run_dir": str(run_path),
        "question_id": plan.get("question_id"),
        "job_id": None,
        "status": "postprocessed_existing_video",
        "model": None,
        "seconds": seconds,
        "size": None,
        "anchor_frame": str(_find_anchor_frame(run_path)),
        "prepared_anchor_frame": None,
        "prompt_path": str(out_dir / "single_video_prompt.txt") if (out_dir / "single_video_prompt.txt").exists() else None,
        "video_path": processed["video_path"],
        "silent_video_path": processed["silent_video_path"],
        "captioned_video_path": processed["captioned_video_path"],
        "voiceover_path": processed["voiceover_path"],
    }
    save_json(_portable_result(result, run_path), out_dir / "single_video_result.json")
    return result


def generate_single_video_from_run_dir(
    run_dir: Path | str,
    model: str = "sora-2",
    seconds: int = 12,
    size: str = "1792x1024",
    add_voiceover: bool = False,
    add_captions: bool = True,
    shorten_for_video: bool = True,
) -> dict:
    if seconds not in VALID_SECONDS:
        raise ValueError(f"seconds must be one of {sorted(VALID_SECONDS)}")
    if size not in VALID_SIZES:
        raise ValueError(f"size must be one of {sorted(VALID_SIZES)}")
    size = _resolve_size_for_model(model, size)

    run_path = Path(run_dir)
    if not run_path.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_path}")

    client = _build_client()
    plan = _load_plan(run_path)
    anchor_source = _find_anchor_frame(run_path)

    out_dir = ensure_dir(run_path / "single_api_video")
    anchor_path = _prepare_anchor_frame(anchor_source, size=size, out_path=out_dir / f"anchor_{size}.png")
    prompt = build_single_video_prompt(plan, seconds=seconds)
    save_text(prompt, out_dir / "single_video_prompt.txt")

    video = client.videos.create_and_poll(
        model=model,
        prompt=prompt,
        seconds=str(seconds),
        size=size,
        input_reference=anchor_path,
        poll_interval_ms=10000,
    )
    save_json(_to_plain_dict(video), out_dir / "single_video_job.json")

    if getattr(video, "status", None) != "completed":
        raise RuntimeError(f"Video generation failed with status={getattr(video, 'status', None)}")

    video_bytes = _download_video_bytes(client.videos.download_content(video.id))
    silent_video_path = out_dir / "single_api_video.mp4"
    silent_video_path.write_bytes(video_bytes)
    processed = _postprocess_single_video(
        run_path=run_path,
        plan=plan,
        base_video_path=silent_video_path,
        seconds=seconds,
        add_voiceover=add_voiceover,
        add_captions=add_captions,
        shorten_for_video=shorten_for_video,
    )

    result = {
        "run_dir": str(run_path),
        "question_id": plan.get("question_id"),
        "job_id": getattr(video, "id", None),
        "status": getattr(video, "status", None),
        "model": model,
        "seconds": seconds,
        "size": size,
        "anchor_frame": str(anchor_source),
        "prepared_anchor_frame": str(anchor_path),
        "prompt_path": str(out_dir / "single_video_prompt.txt"),
        "video_path": processed["video_path"],
        "silent_video_path": processed["silent_video_path"],
        "captioned_video_path": processed["captioned_video_path"],
        "voiceover_path": processed["voiceover_path"],
    }
    save_json(_portable_result(result, run_path), out_dir / "single_video_result.json")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate one API video from an existing storyboard run directory.")
    parser.add_argument("run_dir", type=Path, help="Existing storyboard output directory containing plan.json and frames_raw/")
    parser.add_argument("--model", default="sora-2", choices=["sora-2", "sora-2-pro"], help="Video model")
    parser.add_argument("--seconds", type=int, default=12, choices=sorted(VALID_SECONDS), help="Video duration in seconds")
    parser.add_argument("--size", default="1792x1024", choices=sorted(VALID_SIZES), help="Video resolution")
    parser.add_argument("--add-voiceover", action="store_true", help="Generate narration and mux it onto the returned video")
    parser.add_argument("--no-captions", action="store_true", help="Disable burned-in caption overlay")
    args = parser.parse_args()

    result = generate_single_video_from_run_dir(
        run_dir=args.run_dir,
        model=args.model,
        seconds=args.seconds,
        size=args.size,
        add_voiceover=args.add_voiceover,
        add_captions=not args.no_captions,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()