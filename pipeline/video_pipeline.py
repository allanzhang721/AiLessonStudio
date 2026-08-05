"""High-quality storyboard video composition with AI narration.

This is the only video path used by the application: generated lesson images
are animated with subtle camera motion and crossfades, then muxed with a clean
AI-generated narration track.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from typing import Iterable, Optional, Sequence

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageFilter

from .config import TTS_MODEL, TTS_VOICE
from .utils import ensure_dir, extract_caption_text


VIDEO_WIDTH = 1920
VIDEO_HEIGHT = 1080
VIDEO_FPS = 30
TRANSITION_SECONDS = 0.5
MIN_SCENE_SECONDS = 3.2


def _caption_text(caption: object) -> str:
    return extract_caption_text(caption, fallback="")


def build_narration_script(plan: dict) -> str:
    """Build a natural, speech-first narration from the storyboard captions."""
    captions = plan.get("captions") if isinstance(plan.get("captions"), list) else []
    texts = [_caption_text(caption).strip() for caption in captions]
    texts = [text for text in texts if text]
    if not texts:
        return ""

    transitions = ["First,", "Next,", "Then,", "Now,", "After that,", "Notice that,", "Finally,"]
    lines: list[str] = []
    for index, text in enumerate(texts):
        clean = text.rstrip(" .")
        lead = transitions[min(index, len(transitions) - 1)]
        lines.append(f"{lead} {clean}.")
    return " ".join(lines)


def synthesize_clean_voiceover(
    client,
    plan: dict,
    out_dir: Path,
    *,
    voice: str = TTS_VOICE,
) -> Optional[Path]:
    """Generate lossless, speech-only narration suitable for final AAC muxing."""
    if client is None:
        return None

    script = build_narration_script(plan)
    if not script:
        return None
    if len(script) > 4096:
        raise ValueError("Narration exceeds the speech API's 4096-character input limit.")

    ensure_dir(out_dir)
    (out_dir / "voiceover_script.txt").write_text(script, encoding="utf-8")
    audio_path = out_dir / "voiceover_clean.wav"
    instructions = (
        "Use a warm, confident high-school teacher voice. Speak naturally at a measured pace, "
        "with short pauses between ideas and gentle emphasis on key scientific or mathematical terms. "
        "Use crisp pronunciation. Narration only: no music, ambience, or sound effects."
    )
    speech_args = {
        "model": TTS_MODEL,
        "voice": voice,
        "input": script,
        "instructions": instructions,
        "response_format": "wav",
        "speed": 0.96,
    }
    try:
        response = client.audio.speech.create(**speech_args)
        if hasattr(response, "stream_to_file"):
            response.stream_to_file(str(audio_path))
        elif hasattr(response, "read"):
            audio_path.write_bytes(response.read())
        elif hasattr(response, "content"):
            audio_path.write_bytes(response.content)
        else:
            raise RuntimeError("Speech API returned an unsupported response object.")
    except Exception as exc:
        raise RuntimeError(f"Narration generation failed: {exc}") from exc

    if not audio_path.exists() or audio_path.stat().st_size < 44:
        raise RuntimeError("Narration generation returned an empty WAV file.")
    return audio_path


def _ffmpeg_executable() -> Optional[str]:
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def media_duration_seconds(path: Path) -> Optional[float]:
    """Measure media duration with ffmpeg without requiring ffprobe."""
    executable = _ffmpeg_executable()
    if executable is None or not path.exists():
        return None
    process = subprocess.run(
        [executable, "-hide_banner", "-i", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", process.stderr)
    if not match:
        return None
    hours, minutes, seconds = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def calculate_scene_durations(plan: dict, total_seconds: Optional[float] = None) -> list[float]:
    """Allocate narration time across scenes using caption word counts."""
    captions = plan.get("captions") if isinstance(plan.get("captions"), list) else []
    weights = [max(5, len(_caption_text(caption).split())) for caption in captions]
    if not weights:
        return []
    if total_seconds is None:
        total_seconds = max(len(weights) * MIN_SCENE_SECONDS, sum(weights) / 2.35 + 1.2)
    target = max(float(total_seconds) + 0.65, len(weights) * MIN_SCENE_SECONDS)
    base = [MIN_SCENE_SECONDS] * len(weights)
    remaining = max(0.0, target - sum(base))
    weight_total = float(sum(weights))
    return [round(floor + remaining * weight / weight_total, 3) for floor, weight in zip(base, weights)]


def _fit_storyboard_frame(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Preserve the full teaching frame over a softly blurred 16:9 background."""
    width, height = size
    source = image.convert("RGB")

    background_scale = max(width / source.width, height / source.height)
    background_size = (int(source.width * background_scale), int(source.height * background_scale))
    background = source.resize(background_size, Image.Resampling.LANCZOS)
    left = max(0, (background.width - width) // 2)
    top = max(0, (background.height - height) // 2)
    background = background.crop((left, top, left + width, top + height)).filter(ImageFilter.GaussianBlur(22))

    foreground_scale = min((width - 64) / source.width, (height - 44) / source.height)
    foreground_size = (int(source.width * foreground_scale), int(source.height * foreground_scale))
    foreground = source.resize(foreground_size, Image.Resampling.LANCZOS)
    x = (width - foreground.width) // 2
    y = (height - foreground.height) // 2
    background.paste(foreground, (x, y))
    return background


def _motion_frame(image: Image.Image, progress: float, direction: int) -> np.ndarray:
    """Apply a restrained Ken Burns move while keeping captions readable."""
    width, height = image.size
    zoom = 1.0 + 0.018 * max(0.0, min(1.0, progress))
    scaled = image.resize((int(width * zoom), int(height * zoom)), Image.Resampling.LANCZOS)
    max_x = max(0, scaled.width - width)
    max_y = max(0, scaled.height - height)
    horizontal = progress if direction > 0 else 1.0 - progress
    left = int(max_x * horizontal)
    top = int(max_y * 0.5)
    return np.asarray(scaled.crop((left, top, left + width, top + height)), dtype=np.uint8)


def _write_motion_video(
    frame_paths: Sequence[Path],
    out_path: Path,
    scene_durations: Sequence[float],
    *,
    fps: int,
    size: tuple[int, int],
    transition_seconds: float,
) -> Path:
    if not frame_paths:
        raise ValueError("No lesson frames were provided for video composition.")
    if len(frame_paths) != len(scene_durations):
        raise ValueError("Scene duration count must match the lesson frame count.")

    ensure_dir(out_path.parent)
    prepared: list[Image.Image] = []
    for path in frame_paths:
        with Image.open(path) as source:
            prepared.append(_fit_storyboard_frame(source, size))
    transition_frames = max(1, int(round(transition_seconds * fps)))
    writer = imageio.get_writer(
        str(out_path),
        fps=fps,
        codec="libx264",
        quality=None,
        macro_block_size=None,
        ffmpeg_params=[
            "-preset", "medium",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
        ],
    )
    try:
        for index, (image, seconds) in enumerate(zip(prepared, scene_durations)):
            scene_frames = max(transition_frames + 1, int(round(seconds * fps)))
            stable_frames = scene_frames - transition_frames if index < len(prepared) - 1 else scene_frames
            direction = 1 if index % 2 == 0 else -1
            for frame_index in range(stable_frames):
                progress = frame_index / max(1, scene_frames - 1)
                writer.append_data(_motion_frame(image, progress, direction))
            if index < len(prepared) - 1:
                next_image = prepared[index + 1]
                next_direction = 1 if (index + 1) % 2 == 0 else -1
                for transition_index in range(transition_frames):
                    alpha = (transition_index + 1) / (transition_frames + 1)
                    current = _motion_frame(image, (stable_frames + transition_index) / max(1, scene_frames - 1), direction)
                    upcoming = _motion_frame(next_image, 0.0, next_direction)
                    blended = np.rint(current.astype(np.float32) * (1.0 - alpha) + upcoming.astype(np.float32) * alpha)
                    writer.append_data(blended.astype(np.uint8))
    finally:
        writer.close()
        for image in prepared:
            image.close()

    if not out_path.exists() or out_path.stat().st_size == 0:
        raise RuntimeError("Video encoder did not produce an MP4 file.")
    return out_path


def _mux_voiceover(video_path: Path, audio_path: Path, out_path: Path) -> Path:
    """Normalize narration and mux it into a browser-optimized MP4."""
    executable = _ffmpeg_executable()
    if executable is None:
        raise RuntimeError("ffmpeg is required to add narration to the lesson video.")
    video_duration = media_duration_seconds(video_path)
    if video_duration is None:
        raise RuntimeError("Could not measure the composed video duration before audio muxing.")
    command = [
        executable,
        "-y",
        "-i", str(video_path),
        "-i", str(audio_path),
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-af", "loudnorm=I=-16:TP=-1.5:LRA=11,apad",
        "-t", f"{video_duration:.3f}",
        "-movflags", "+faststart",
        str(out_path),
    ]
    process = subprocess.run(command, capture_output=True, text=True, check=False)
    if process.returncode != 0 or not out_path.exists() or out_path.stat().st_size == 0:
        fallback = list(command)
        filter_index = fallback.index("loudnorm=I=-16:TP=-1.5:LRA=11,apad")
        fallback[filter_index] = "apad"
        process = subprocess.run(fallback, capture_output=True, text=True, check=False)

        if process.returncode != 0 or not out_path.exists() or out_path.stat().st_size == 0:
            detail = process.stderr.strip().splitlines()[-1] if process.stderr.strip() else "unknown ffmpeg error"
            raise RuntimeError(f"Could not add narration to the video: {detail}")
    return out_path


def estimate_video_fps(frame_count: int, script: str) -> float:
    """Compatibility helper retained for callers of the former slideshow API."""
    if frame_count <= 0:
        return 1.0
    target_seconds = max(frame_count * MIN_SCENE_SECONDS, max(1, len(script.split())) / 2.35)
    return frame_count / target_seconds


def images_to_video(
    frame_paths: Iterable[Path],
    out_path: Path,
    fps: float = VIDEO_FPS,
    audio_path: Optional[Path] = None,
    *,
    plan: Optional[dict] = None,
    scene_durations: Optional[Sequence[float]] = None,
    resolution: tuple[int, int] = (VIDEO_WIDTH, VIDEO_HEIGHT),
) -> Path:
    """Compose high-quality lesson frames and optional narration into one MP4."""
    frames = [Path(path) for path in frame_paths]
    measured_audio = media_duration_seconds(audio_path) if audio_path is not None else None
    if scene_durations is None:
        if plan is not None:
            durations = calculate_scene_durations(plan, measured_audio)
        else:
            fallback = max(MIN_SCENE_SECONDS, 1.0 / max(float(fps), 0.1))
            durations = [fallback] * len(frames)
    else:
        durations = [float(value) for value in scene_durations]

    video_only_path = out_path.with_name(f"{out_path.stem}_video_only{out_path.suffix}") if audio_path else out_path
    made_path = _write_motion_video(
        frames,
        video_only_path,
        durations,
        fps=VIDEO_FPS,
        size=resolution,
        transition_seconds=TRANSITION_SECONDS,
    )
    if audio_path is None:
        return made_path
    return _mux_voiceover(made_path, audio_path, out_path)
