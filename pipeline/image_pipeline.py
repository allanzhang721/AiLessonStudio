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




def _find_wanx_image_url(value) -> Optional[str]:
    if isinstance(value, dict):
        for key in ("image", "url", "file_url"):
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate.startswith(("https://", "http://")):
                return candidate
        for child in value.values():
            found = _find_wanx_image_url(child)
            if found:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find_wanx_image_url(child)
            if found:
                return found
    return None


def _wanx_image(
    client: dict,
    prompt: str,
    out_path: Path,
    *,
    image_model: str,
    previous_image: Optional[Path] = None,
) -> Path:
    """Generate or edit one frame with Alibaba Model Studio's async Wan API."""
    import requests

    endpoint = "https://dashscope-intl.aliyuncs.com/api/v1/services/aigc/image-generation/generation"
    content: list[dict[str, str]] = [{"text": prompt}]
    if previous_image is not None:
        mime = "image/png"
        encoded = base64.b64encode(previous_image.read_bytes()).decode("ascii")
        content.append({"image": f"data:{mime};base64,{encoded}"})
    payload = {
        "model": image_model,
        "input": {"messages": [{"role": "user", "content": content}]},
        "parameters": {
            "size": "1536*1024",
            "n": 1,
            "watermark": False,
            "prompt_extend": True,
        },
    }
    headers = {
        "Authorization": f"Bearer {client['api_key']}",
        "Content-Type": "application/json",
        "X-DashScope-Async": "enable",
    }
    response = requests.post(endpoint, headers=headers, json=payload, timeout=90)
    response.raise_for_status()
    data = response.json()
    task_id = data.get("output", {}).get("task_id")
    image_url = _find_wanx_image_url(data)
    if task_id and not image_url:
        task_url = f"https://dashscope-intl.aliyuncs.com/api/v1/tasks/{task_id}"
        for _ in range(90):
            time.sleep(2)
            polled = requests.get(task_url, headers={"Authorization": headers["Authorization"]}, timeout=30)
            polled.raise_for_status()
            data = polled.json()
            status = str(data.get("output", {}).get("task_status", "")).upper()
            if status == "SUCCEEDED":
                image_url = _find_wanx_image_url(data)
                break
            if status in {"FAILED", "CANCELED", "UNKNOWN"}:
                message = data.get("message") or data.get("output", {}).get("message") or status
                raise RuntimeError(f"Wan image task failed: {message}")
        else:
            raise TimeoutError("Wan image task did not finish within three minutes.")
    if not image_url:
        raise RuntimeError(f"Wan response did not contain an image URL: {data.get('message', 'unknown response')}")
    image_response = requests.get(image_url, timeout=90)
    image_response.raise_for_status()
    out_path.write_bytes(image_response.content)
    with Image.open(out_path) as image:
        image.verify()
    return out_path


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

    provider_mode = (
        "wanx" if isinstance(client, dict) and client.get("provider") == "wanx"
        else "openai" if client is not None
        else "placeholder"
    )
    render_meta = {
        "image_model": image_model if client is not None else "placeholder",
        "mode": provider_mode,
        "steps": [],
    }

    for step, caption in zip(plan["steps"], plan["captions"]):
        sid = int(step["step_id"])
        raw_path = raw_dir / f"step_{sid:02d}.png"
        final_path = final_dir / f"step_{sid:02d}.png"

        if client is not None:
            if sid == 1:
                prompt = build_first_frame_prompt(plan)
                prompt_path = prompt_dir / f"step_{sid:02d}_generate_prompt.txt"
            else:
                prompt = build_edit_prompt(plan, sid)
                prompt_path = prompt_dir / f"step_{sid:02d}_edit_prompt.txt"
            save_text(prompt, prompt_path)

            if provider_mode == "wanx":
                _wanx_image(
                    client,
                    prompt,
                    raw_path,
                    image_model=image_model,
                    previous_image=prev_raw if sid > 1 else None,
                )
            elif sid == 1:
                _generate_first_frame_openai(client, prompt, raw_path, image_model=image_model)
            else:
                assert prev_raw is not None
                _edit_next_frame_openai(client, prev_raw, prompt, raw_path, image_model=image_model)

            img = Image.open(raw_path).convert("RGB")
            render_meta["steps"].append({
                "step_id": sid,
                "status": f"{provider_mode}_ok",
                "raw_path": str(raw_path),
                "prompt_path": str(prompt_path),
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
