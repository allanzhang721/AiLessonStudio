"""
video_pipeline.py — Stage 3 of the L15 pipeline: frames → MP4 with narration.

Public entry points:
  images_to_video()          — assemble PNG frames into an MP4, optionally muxing audio
  build_narration_script()   — concatenate 7 captions into a numbered voiceover script
  synthesize_clean_voiceover() — call OpenAI TTS to generate narration-only MP3
  estimate_video_fps()       — heuristic: compute fps so slideshow ≈ audio length

Internal helpers:
  _mux_voiceover()           — ffmpeg subprocess: combine silent video + narration audio
  _caption_text()            — thin wrapper around extract_caption_text

Audio pipeline:
  1. build_narration_script() assembles the full spoken text from all 7 captions.
  2. synthesize_clean_voiceover() calls OpenAI TTS (gpt-4o-mini-tts, voice=alloy)
     with explicit "no music, no sound effects" instructions to get a clean MP3.
  3. estimate_video_fps() calculates the frame rate so the 7-frame slideshow
     roughly matches the narration length  (words / 2.6 words-per-second, capped
     to 0.15–1.0 fps).
  4. images_to_video() first builds a silent MP4, then calls _mux_voiceover()
     via ffmpeg to attach the narration as the single audio track.

If no OpenAI client is available, synthesize_clean_voiceover returns None and
images_to_video outputs a silent video at 1.0 fps.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Iterable, Optional

from .config import TTS_MODEL, TTS_VOICE
from .utils import extract_caption_text, make_mp4


def _caption_text(caption: object) -> str:
    return extract_caption_text(caption, fallback="")


def build_narration_script(plan: dict) -> str:
    """Concatenate all 7 captions into a numbered narration script.

    Output format:
        Step 1. <caption text>
        Step 2. <caption text>
        ...

    Used both for TTS synthesis and for estimating target video duration.
    """
    captions = plan.get("captions") if isinstance(plan.get("captions"), list) else []
    lines: list[str] = []
    for i, cap in enumerate(captions, start=1):
        text = _caption_text(cap)
        if text:
            lines.append(f"Step {i}. {text}")
    return "\n".join(lines)


def synthesize_clean_voiceover(client, plan: dict, out_dir: Path) -> Optional[Path]:
    """Generate clear narration-only audio; no background music/effects."""
    if client is None:
        return None

    script = build_narration_script(plan)
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


def _mux_voiceover(video_path: Path, audio_path: Path, out_path: Path) -> Path:
    """Mux narration as the only audio track. Returns out_path on success, else video_path."""
    if shutil.which("ffmpeg") is None:
        return video_path

    cmd = [
        "ffmpeg",
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
        "-shortest",
        str(out_path),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return out_path if out_path.exists() and out_path.stat().st_size > 0 else video_path
    except Exception:
        return video_path


def estimate_video_fps(frame_count: int, script: str) -> float:
    """Heuristic fps so the frame slideshow timeline roughly matches narration length.

    Assumes ~2.6 spoken words per second (relaxed educational pacing).
    Minimum total video duration is 14 seconds regardless of word count.
    Result is clamped to [0.15, 1.0] fps because frames are static images —
    faster than 1 fps would flash too quickly; slower than 0.15 would stall.
    """
    if frame_count <= 0:
        return 1.0
    words = max(1, len(script.split()))
    target_seconds = max(14.0, words / 2.6)
    fps = frame_count / target_seconds
    return max(0.15, min(1.0, fps))


def images_to_video(
    frame_paths: Iterable[Path],
    out_path: Path,
    fps: float = 1.0,
    audio_path: Optional[Path] = None,
) -> Path:
    """Stage 3: PNG frame images → MP4 video.

    If audio_path is provided, writes a silent intermediate MP4 first, then calls
    _mux_voiceover() to attach the narration as the only audio track.
    Returns the path to the final video file.
    """
    video_only_path = out_path.with_name(f"{out_path.stem}_video_only{out_path.suffix}") if audio_path else out_path
    made_path = make_mp4(frame_paths, video_only_path, fps=fps)
    if audio_path is None:
        return made_path
    return _mux_voiceover(made_path, audio_path, out_path)
