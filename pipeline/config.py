"""
config.py — Global constants for the L15 educational video pipeline.

All tunable values live here so the rest of the codebase never hard-codes
model names, image sizes, or directory paths directly.
"""

from __future__ import annotations

from pathlib import Path

# --- Canvas dimensions ---
# Every generated frame is 1536 × 1024 px (landscape, ~1.5:1 ratio).
# The bottom 320 px are reserved as the caption band; diagram content
# must stay above y = IMG_H - MIN_BOTTOM_BAND = 704 px.
IMG_W = 1536
IMG_H = 1024
MIN_BOTTOM_BAND = 320   # height of the white caption band added below each frame

# --- Provider-specific model names ---
# OpenAI
OPENAI_TEXT_MODEL = "gpt-4o"       # text model: planning, explanations, repair
OPENAI_IMAGE_MODEL = "gpt-image-1" # image model: frame generation / inpainting
OPENAI_TTS_MODEL = "gpt-4o-mini-tts"
OPENAI_TTS_VOICE = "alloy"
OPENAI_VIDEO_MODEL = "sora-2"

# DeepSeek
DEEPSEEK_TEXT_MODEL = "deepseek-chat"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

# Alibaba DashScope / Wanx (万象)
WANX_IMAGE_MODEL = "wanx-v1"
WANX_VIDEO_MODEL = "wanx-v1"
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/api/v1"

# Legacy aliases used throughout codebase (default = OpenAI)
PLANNER_MODEL = OPENAI_TEXT_MODEL
IMAGE_MODEL = OPENAI_IMAGE_MODEL
TTS_MODEL = OPENAI_TTS_MODEL
TTS_VOICE = OPENAI_TTS_VOICE

# --- Checker 1 (DistilBERT error-type classifier) ---
CHECKER1_CKPT_DIR = Path(__file__).resolve().parent.parent / "checker1" / "model" / "distilbert_error_type_ckpt" / "checkpoint-360"
CHECKER1_MAX_LEN = 256             # tokenizer max_length used during training
CHECKER1_LABELS = [                # alphabetical order from LabelEncoder
    "ConceptError",
    "GradeMismatch",
    "LogicalGap",
    "MisleadingAnalogy",
    "MissingCondition",
]

# --- Output / pipeline defaults ---
DEFAULT_ROOT = Path("l15_output")  # base directory where run output folders are written
DEFAULT_STEPS = 7                  # number of storyboard frames per video
