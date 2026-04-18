"""
image_pipeline.py — Stage 2 of the L15 pipeline: Plan → 7 PNG frame images.

Public entry point: plan_to_images(plan, out_dir, client)

For each of the 7 steps the function:
  1. Validates the plan schema (raises ValueError on failure).
  2. Generates or edits a PNG frame:
       - Step 1: calls client.images.generate() with build_first_frame_prompt()
       - Steps 2–7: calls client.images.edit() (inpainting) with build_edit_prompt(),
         using the previous step's raw PNG as the base image.
       - No client: produces a local placeholder via make_placeholder_frame()
  3. Overlays any math_elements (formula tiles) active at this step.
  4. Appends the caption band via add_bottom_caption() and saves to frames/.
  5. Records per-step metadata into plan['render_meta'].

Both API call wrappers (_generate_first_frame_openai, _edit_next_frame_openai)
retry up to 3 times with exponential back-off before raising RuntimeError.

Output files:
  out_dir/frames_raw/step_NN.png  — raw image from API (no caption band)
  out_dir/frames/step_NN.png      — final frame with caption band
  out_dir/prompts/step_NN_*.txt   — the prompt text that was sent to the API
"""

from __future__ import annotations

import base64
import time
from pathlib import Path
from typing import List, Optional

from PIL import Image

from .config import IMAGE_MODEL, OPENAI_IMAGE_MODEL
from .prompts import build_edit_prompt, build_first_frame_prompt
from .utils import (
    add_bottom_caption,
    ensure_dir,
    extract_caption_text,
    make_placeholder_frame,
    overlay_plan_math_elements,
    save_text,
)
from .validation import validate_plan_schema


def _caption_text(caption: object) -> str:
    return extract_caption_text(caption, fallback="")


def _extract_b64_image(response) -> bytes:
    """Extract raw PNG bytes from an OpenAI images.generate / images.edit response."""
    if getattr(response, "data", None):
        item = response.data[0]
        if getattr(item, "b64_json", None):
            return base64.b64decode(item.b64_json)
    raise ValueError("Could not find image bytes in response")


def _generate_first_frame_openai(client, prompt: str, out_path: Path, size: str = "1536x1024", image_model: str = OPENAI_IMAGE_MODEL) -> Path:
    """Call images.generate for frame 1 with up to 3 retries on failure."""
    last_error = ""
    for attempt in range(1, 4):
        try:
            response = client.images.generate(
                model=image_model,
                prompt=prompt,
                size=size,
                quality="medium",
                output_format="png",
            )
            out_path.write_bytes(_extract_b64_image(response))
            return out_path
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(0.7 * attempt)
    raise RuntimeError(f"First-frame image generation failed after retries: {last_error}")


def _edit_next_frame_openai(client, prev_path: Path, prompt: str, out_path: Path, size: str = "1536x1024", image_model: str = OPENAI_IMAGE_MODEL) -> Path:
    """Call images.edit (inpainting) for frames 2–7 with up to 3 retries on failure.

    Opens prev_path as the source image for the inpaint operation, which keeps
    layout and existing elements stable across frames.
    """
    last_error = ""
    for attempt in range(1, 4):
        try:
            with open(prev_path, "rb") as fh:
                response = client.images.edit(model=image_model, image=fh, prompt=prompt, size=size)
            out_path.write_bytes(_extract_b64_image(response))
            return out_path
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(0.7 * attempt)
    raise RuntimeError(f"Inpainting image generation failed after retries: {last_error}")


def plan_to_images(plan: dict, out_dir: Path, client=None, image_model: str = OPENAI_IMAGE_MODEL) -> List[Path]:
    """
    Stage 2: plan -> 7 output frame images with captions.
    Uses GPT image generation if client is provided; otherwise makes local placeholders.
    """
    valid, errors = validate_plan_schema(plan, expected_steps=7)
    if not valid:
        raise ValueError("Invalid plan for rendering: " + " | ".join(errors[:8]))

    if len(plan.get("steps", [])) != len(plan.get("captions", [])):
        raise ValueError("steps/captions length mismatch. Expected equal lengths before rendering.")

    raw_dir = ensure_dir(out_dir / "frames_raw")
    final_dir = ensure_dir(out_dir / "frames")
    prompt_dir = ensure_dir(out_dir / "prompts")

    frames: List[Path] = []
    prev_raw: Optional[Path] = None

    render_meta = {
        "image_model": image_model if client is not None else "placeholder",
        "mode": "openai" if client is not None else "placeholder",
        "steps": [],
    }

    for step, caption in zip(plan["steps"], plan["captions"]):
        sid = int(step["step_id"])
        raw_path = raw_dir / f"step_{sid:02d}.png"
        final_path = final_dir / f"step_{sid:02d}.png"

        if client is not None:
            if sid == 1:
                prompt = build_first_frame_prompt(plan)
                save_text(prompt, prompt_dir / f"step_{sid:02d}_generate_prompt.txt")
                _generate_first_frame_openai(client, prompt, raw_path, image_model=image_model)
            else:
                prompt = build_edit_prompt(plan, sid)
                assert prev_raw is not None
                save_text(prompt, prompt_dir / f"step_{sid:02d}_edit_prompt.txt")
                _edit_next_frame_openai(client, prev_raw, prompt, raw_path, image_model=image_model)

            img = Image.open(raw_path).convert("RGB")
            render_meta["steps"].append({
                "step_id": sid,
                "status": "openai_ok",
                "raw_path": str(raw_path),
                "prompt_path": str(prompt_dir / (f"step_{sid:02d}_generate_prompt.txt" if sid == 1 else f"step_{sid:02d}_edit_prompt.txt")),
            })
        else:
            img = make_placeholder_frame(
                sid,
                title=str(step.get("goal", f"Step {sid}")),
                body=str(step.get("delta", "")),
            )
            img.save(raw_path)
            render_meta["steps"].append({"step_id": sid, "status": "placeholder", "raw_path": str(raw_path)})

        img = overlay_plan_math_elements(img, plan, sid)
        framed = add_bottom_caption(img, _caption_text(caption), sid,
                                    total_steps=len(plan["steps"]))
        framed.save(final_path)
        frames.append(final_path)
        prev_raw = raw_path

    plan["render_meta"] = render_meta
    return frames
