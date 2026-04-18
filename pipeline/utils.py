"""
utils.py — Shared helper functions used across the L15 pipeline.

Responsibilities:
  - File I/O:        ensure_dir, save_json, save_text
  - Caption parsing: extract_caption_text (normalises nested GPT dicts/lists into
                     a plain string)
  - Image rendering: add_bottom_caption, overlay_formula_tile,
                     overlay_plan_math_elements, make_placeholder_frame
  - Text layout:     _fit_text_in_box, _wrap_text_by_px (PIL helpers)
  - Font loading:    _load_font (tries several system paths, falls back to PIL default)
  - Video assembly:  make_gif, make_mp4 (via imageio)

Nothing in this file calls OpenAI; it is pure local computation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .config import IMG_H, IMG_W, MIN_BOTTOM_BAND


# ---------------------------------------------------------------------------
# File I/O helpers
# ---------------------------------------------------------------------------

def ensure_dir(path: Path) -> Path:
    """Create path (and all parents) if it doesn't exist. Returns path unchanged."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(obj: dict, path: Path) -> None:
    """Serialize obj to pretty-printed UTF-8 JSON at path, creating directories as needed."""
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def save_text(text: str, path: Path) -> None:
    """Write a UTF-8 text file, creating directories as needed."""
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# Caption normalisation
# ---------------------------------------------------------------------------

def extract_caption_text(caption: object, fallback: str = "") -> str:
    """Normalize caption payloads from model outputs into plain narration text.

    GPT sometimes returns captions as plain strings, sometimes as dicts with
    keys like "narration", "spoken_text", "voiceover", etc., and sometimes as
    nested lists. This function walks any structure and returns the first
    meaningful text it finds, falling back to `fallback` if nothing is found.
    """
    if isinstance(caption, str):
        text = caption.strip()
        return text if text else fallback

    # Keys to look for first, in order of preference.
    preferred_keys = (
        "narration",
        "spoken_text",
        "voiceover",
        "script",
        "text",
        "caption",
        "explanation",
        "description",
        "summary",
        "content",
        "sentence",
        "value",
    )
    # Keys that hold metadata rather than narration text — skip these.
    ignored_keys = {
        "id",
        "step_id",
        "index",
        "frame",
        "order",
        "duration",
        "timestamp",
        "start",
        "end",
        "confidence",
        "score",
        "meta",
        "metadata",
        "type",
    }

    def _walk(value: Any) -> str:
        if isinstance(value, str):
            return value.strip()

        if isinstance(value, dict):
            # Try preferred keys first for deterministic extraction.
            for key in preferred_keys:
                if key in value:
                    found = _walk(value.get(key))
                    if found:
                        return found
            # Fall back to any non-ignored key.
            for key, item in value.items():
                if str(key).lower() in ignored_keys:
                    continue
                found = _walk(item)
                if found:
                    return found
            return ""

        if isinstance(value, list):
            for item in value:
                found = _walk(item)
                if found:
                    return found
            return ""

        return ""

    text = _walk(caption)
    if text:
        return text
    return fallback


# ---------------------------------------------------------------------------
# Font loading
# ---------------------------------------------------------------------------

def _load_font(size: int) -> ImageFont.ImageFont:
    """Try common system font paths; fall back to PIL's built-in default."""
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


# Pre-load fonts at module import time to avoid repeated disk access.
FONT_CAPTION = _load_font(40)
FONT_STEP = _load_font(30)
FONT_PLACEHOLDER_TITLE = _load_font(64)
FONT_PLACEHOLDER_BODY = _load_font(40)


# ---------------------------------------------------------------------------
# Text layout helpers
# ---------------------------------------------------------------------------

def _fit_text_in_box(
    draw: ImageDraw.ImageDraw,
    text: str,
    max_w: int,
    max_h: int,
    start_size: int = 52,
    min_size: int = 20,
) -> tuple[str, ImageFont.ImageFont, int, int]:
    """
    Fit text into a bounded rectangle by decreasing font size and wrapping.
    Returns (wrapped_text, font, text_width, text_height).

    Steps down in 2-pt increments from start_size to min_size until the wrapped
    text bounding box is within (max_w × max_h). Returns the best fit found.
    """
    best_wrapped = text
    best_font = _load_font(min_size)
    best_w = 0
    best_h = 0

    for size in range(start_size, min_size - 1, -2):
        font = _load_font(size)
        wrapped = _wrap_text_by_px(text, font, max_w, draw)
        bbox = draw.multiline_textbbox((0, 0), wrapped, font=font, spacing=8)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        best_wrapped, best_font, best_w, best_h = wrapped, font, tw, th
        if tw <= max_w and th <= max_h:
            return wrapped, font, tw, th

    return best_wrapped, best_font, best_w, best_h


def _wrap_text_by_px(text: str, font: ImageFont.ImageFont, max_width: int, draw: ImageDraw.ImageDraw) -> str:
    """Word-wrap text so no line exceeds max_width pixels given font."""
    words = text.split()
    if not words:
        return ""
    lines = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        bbox = draw.textbbox((0, 0), candidate, font=font)
        if (bbox[2] - bbox[0]) <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Frame compositing
# ---------------------------------------------------------------------------

def add_bottom_caption(img: Image.Image, caption: str, step_id: int, total_steps: int = 7) -> Image.Image:
    """Append a fixed-height caption band to the bottom of a frame image.

    The band consists of:
      - A thin grey separator line
      - A blue accent bar (6 px)
      - The narration caption text (auto-wrapped and auto-sized)
      - A step counter pill in the bottom-right corner (e.g. "3 / 7")

    Using a fixed band_h (MIN_BOTTOM_BAND) for every frame prevents shape
    mismatches when frames are assembled into a GIF or MP4.
    """
    base = img.convert("RGB")
    width, height = base.size

    pad_x = 48
    # Fixed band height to ensure all frames have the same canvas size.
    accent_h = 6
    top_pad = 28
    bottom_pad = 24
    band_h = MIN_BOTTOM_BAND

    probe = Image.new("RGB", (width, 10), "white")
    probe_draw = ImageDraw.Draw(probe)
    caption_max_w = width - pad_x * 2
    caption_max_h = max(40, band_h - accent_h - top_pad - bottom_pad)
    wrapped, caption_font, _, _ = _fit_text_in_box(
        probe_draw,
        caption,
        max_w=caption_max_w,
        max_h=caption_max_h,
        start_size=40,
        min_size=24,
    )

    canvas = Image.new("RGB", (width, height + band_h), "#F8FAFC")
    canvas.paste(base, (0, 0))
    draw = ImageDraw.Draw(canvas)

    # Thin top separator line
    draw.line([(0, height), (width, height)], fill="#CBD5E1", width=2)
    # Coloured accent bar across full width
    draw.rectangle([(0, height), (width, height + accent_h)], fill="#2563EB")

    # Frame counter pill – subtle, bottom-right corner
    counter_text = f"{step_id} / {total_steps}"
    counter_font = _load_font(26)
    cb = draw.textbbox((0, 0), counter_text, font=counter_font)
    cw, ch = cb[2] - cb[0], cb[3] - cb[1]
    pill_x1 = width - cw - 32
    pill_y1 = height + band_h - ch - 18
    pill_x2 = width - 12
    pill_y2 = height + band_h - 10
    draw.rounded_rectangle((pill_x1 - 10, pill_y1 - 6, pill_x2, pill_y2 + 4),
                            radius=10, fill="#E2E8F0", outline="#94A3B8", width=1)
    draw.text((pill_x1 - 4, pill_y1 - 2), counter_text, font=counter_font, fill="#475569")

    # Caption narration text
    text_y = height + accent_h + top_pad
    draw.multiline_text((pad_x, text_y), wrapped, font=caption_font,
                        fill="#1E293B", spacing=14, align="left")
    return canvas


def overlay_formula_tile(
    img: Image.Image,
    display_text: str,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    bg: str = "#FEF9C3",
    border: str = "#CA8A04",
) -> Image.Image:
    """Draw a rounded-rectangle formula tile with centred auto-fitted text.

    The image model is instructed to leave a blank-coloured rectangle at the
    formula tile coordinates; this function then renders the actual LaTeX-style
    display_text on top with Pillow so the typography is always crisp and
    correctly placed, regardless of what the model drew.
    """
    img = img.copy()
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle((x1, y1, x2, y2), radius=14, fill=bg, outline=border, width=3)

    pad_x = 18
    pad_y = 14
    max_w = max(20, (x2 - x1) - 2 * pad_x)
    max_h = max(20, (y2 - y1) - 2 * pad_y)
    wrapped, font, tw, th = _fit_text_in_box(
        draw,
        display_text,
        max_w=max_w,
        max_h=max_h,
        start_size=52,
        min_size=20,
    )

    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    draw.multiline_text((cx - tw // 2, cy - th // 2), wrapped, font=font, fill="#1F2937", spacing=8, align="center")
    return img


def overlay_plan_math_elements(img: Image.Image, plan: dict, step_id: int) -> Image.Image:
    """Apply all math_elements whose step_introduced <= step_id to img.

    Called after the raw frame image is generated or loaded. Elements introduced
    in earlier steps remain visible on all subsequent frames (cumulative overlay).
    """
    for elem in plan.get("math_elements", []):
        if int(elem.get("step_introduced", 999)) <= step_id:
            if elem.get("style", "formula_tile") == "formula_tile":
                img = overlay_formula_tile(
                    img,
                    display_text=str(elem.get("display_text", "")),
                    x1=int(elem.get("x1", 568)),
                    y1=int(elem.get("y1", 24)),
                    x2=int(elem.get("x2", 968)),
                    y2=int(elem.get("y2", 112)),
                    bg=str(elem.get("bg", "#FEF9C3")),
                    border=str(elem.get("border", "#CA8A04")),
                )
    return img


def make_gif(frame_paths: Iterable[Path], gif_path: Path, fps: float = 1.0) -> Path:
    """Assemble PNG frame files into an animated GIF.

    All frames are normalised to the shape of the first frame before writing
    to prevent imageio shape mismatch errors. FPS is clamped to >=0.1.
    """
    ensure_dir(gif_path.parent)
    frames = [imageio.imread(str(p)) for p in frame_paths]
    if not frames:
        raise ValueError("No frames provided for GIF generation")

    target_h, target_w = frames[0].shape[:2]
    normalized_frames = []
    for frame in frames:
        if frame.shape[:2] != (target_h, target_w):
            resized = Image.fromarray(frame).resize((target_w, target_h), Image.Resampling.LANCZOS)
            normalized_frames.append(np.asarray(resized))
        else:
            normalized_frames.append(frame)

    duration = 1.0 / max(fps, 0.1)
    imageio.mimsave(str(gif_path), normalized_frames, duration=duration)
    return gif_path


def make_mp4(frame_paths: Iterable[Path], mp4_path: Path, fps: float = 1.0) -> Path:
    """Assemble PNG frame files into an H.264 MP4 video via imageio/libx264.

    Frames are resized to the first frame's shape to ensure a consistent
    canvas size. The writer is always closed in a finally block to avoid
    partial/corrupt files.
    """
    ensure_dir(mp4_path.parent)
    writer = imageio.get_writer(str(mp4_path), fps=max(fps, 0.1), codec="libx264", quality=8)
    try:
        target_shape = None
        for path in frame_paths:
            frame = imageio.imread(str(path))
            if target_shape is None:
                target_shape = frame.shape[:2]
            elif frame.shape[:2] != target_shape:
                target_h, target_w = target_shape
                frame = np.asarray(
                    Image.fromarray(frame).resize((target_w, target_h), Image.Resampling.LANCZOS)
                )
            writer.append_data(frame)
    finally:
        writer.close()
    return mp4_path


def make_placeholder_frame(step_id: int, title: str, body: str) -> Image.Image:
    """Create a simple placeholder PNG when no OpenAI client is available.

    The placeholder shows the step number, goal title, and delta body text on
    a light grey card so the rest of the pipeline (caption band, GIF/MP4
    assembly) can still be tested without calling the image API.
    """
    img = Image.new("RGB", (IMG_W, IMG_H), "white")
    draw = ImageDraw.Draw(img)

    draw.rounded_rectangle((60, 60, IMG_W - 60, IMG_H - 60), radius=24, outline="#cbd5e1", width=4, fill="#f8fafc")
    draw.text((100, 110), f"Storyboard Frame {step_id}", font=FONT_PLACEHOLDER_TITLE, fill="#0f172a")

    wrapped_title = _wrap_text_by_px(title, FONT_PLACEHOLDER_BODY, IMG_W - 200, draw)
    wrapped_body = _wrap_text_by_px(body, FONT_PLACEHOLDER_BODY, IMG_W - 200, draw)
    draw.multiline_text((100, 260), wrapped_title, font=FONT_PLACEHOLDER_BODY, fill="#1e293b", spacing=10)
    draw.multiline_text((100, 430), wrapped_body, font=FONT_PLACEHOLDER_BODY, fill="#334155", spacing=10)

    draw.text((100, IMG_H - 120), "(OpenAI image API unavailable: rendered local placeholder)", font=_load_font(30), fill="#64748b")
    return img
