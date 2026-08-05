"""Shared configuration for the Explain It! lesson pipeline."""

from __future__ import annotations

from pathlib import Path

# Educational image canvas. The lower band is reserved for deterministic captions.
IMG_W = 1536
IMG_H = 1024
MIN_BOTTOM_BAND = 320

# OpenAI defaults
OPENAI_TEXT_MODEL = "gpt-5.6-terra"
OPENAI_IMAGE_MODEL = "gpt-image-2"
OPENAI_TTS_MODEL = "gpt-4o-mini-tts"
OPENAI_TTS_VOICE = "marin"

# DeepSeek text
DEEPSEEK_TEXT_MODEL = "deepseek-v4-flash"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

# Alibaba Model Studio images
WANX_IMAGE_MODEL = "wan2.7-image"
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/api/v1"

# Default aliases used by the provider-neutral pipeline.
PLANNER_MODEL = OPENAI_TEXT_MODEL
IMAGE_MODEL = OPENAI_IMAGE_MODEL
TTS_MODEL = OPENAI_TTS_MODEL
TTS_VOICE = OPENAI_TTS_VOICE

# Generated run artifacts stay under an ignored directory.
DEFAULT_ROOT = Path("output")
DEFAULT_STEPS = 7