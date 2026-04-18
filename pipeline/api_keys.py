"""
api_keys.py — Load API keys from api_keys.txt (project root) and/or environment.

Reads the file once at import time. Environment variables take precedence
over file values so CI / shell workflows still work unmodified.

Public helpers:
  get_key(name)       — return the key string or ""
  available_text_providers()  — which LLM providers have a key configured
  available_image_providers() — which image providers have a key configured
  available_video_providers() — which video providers have a key configured
"""

from __future__ import annotations

import os
from pathlib import Path

_KEY_FILE = Path(__file__).resolve().parent.parent / "api_keys.txt"

_keys: dict[str, str] = {}


def _load() -> None:
    """Parse api_keys.txt once.  Tolerates missing file / blank values."""
    if not _KEY_FILE.exists():
        return
    for line in _KEY_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        name, _, value = line.partition("=")
        name = name.strip()
        value = value.strip().strip('"').strip("'")
        if name and value:
            _keys[name] = value


_load()


def get_key(name: str) -> str:
    """Return an API key by name.  Environment overrides file."""
    return os.environ.get(name, "") or _keys.get(name, "")


def has_key(name: str) -> bool:
    return bool(get_key(name))


# -----------------------------------------------------------------
# Provider availability helpers
# -----------------------------------------------------------------

def available_text_providers() -> list[str]:
    """Return list of LLM provider names that have a valid API key."""
    providers = []
    if has_key("OPENAI_API_KEY"):
        providers.append("openai")
    if has_key("DEEPSEEK_API_KEY"):
        providers.append("deepseek")
    return providers


def available_image_providers() -> list[str]:
    """Return list of image-gen provider names that have a valid API key."""
    providers = []
    if has_key("OPENAI_API_KEY"):
        providers.append("openai")       # gpt-image-1
    if has_key("DASHSCOPE_API_KEY"):
        providers.append("wanx")         # Alibaba Wanx
    return providers


def available_video_providers() -> list[str]:
    """Return list of video-gen provider names that have a valid API key."""
    providers = []
    if has_key("OPENAI_API_KEY"):
        providers.append("sora")         # OpenAI Sora
    if has_key("DASHSCOPE_API_KEY"):
        providers.append("wanx")         # Alibaba Wanx video
    return providers
