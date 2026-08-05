"""
clients.py — Build API clients for different providers.

Centralises client construction for text, image, and narration APIs.

Supported providers:
  text:   "openai", "deepseek"
  image:  "openai", "wanx"
  tts:    "openai"
"""

from __future__ import annotations

from typing import Optional

from .api_keys import get_key
from .config import DEEPSEEK_BASE_URL


def build_text_client(provider: str = "openai", api_key: Optional[str] = None):
    """Return an OpenAI-compatible client for text generation.

    Both OpenAI and DeepSeek use the same ``openai.OpenAI`` SDK — DeepSeek
    just swaps the ``base_url`` and ``api_key``.
    """
    try:
        from openai import OpenAI
    except ImportError:
        return None

    if provider == "openai":
        key = api_key or get_key("OPENAI_API_KEY")
        if not key:
            return None
        return OpenAI(api_key=key)

    if provider == "deepseek":
        key = api_key or get_key("DEEPSEEK_API_KEY")
        if not key:
            return None
        return OpenAI(api_key=key, base_url=DEEPSEEK_BASE_URL)

    return None


def build_image_client(provider: str = "openai", api_key: Optional[str] = None):
    """Return a client for image generation.

    OpenAI: standard OpenAI client (images.generate / images.edit).
    Wanx:   returns a dict stub with the DashScope key — the actual HTTP
            calls are made inside image_pipeline.py since DashScope uses
            a REST API rather than the openai SDK.
    """
    try:
        from openai import OpenAI
    except ImportError:
        OpenAI = None  # type: ignore[assignment]

    if provider == "openai":
        key = api_key or get_key("OPENAI_API_KEY")
        if not key or OpenAI is None:
            return None
        return OpenAI(api_key=key)

    if provider == "wanx":
        key = api_key or get_key("DASHSCOPE_API_KEY")
        if not key:
            return None
        # Return a lightweight wrapper so callers can detect "wanx" provider
        return {"provider": "wanx", "api_key": key}

    return None



def build_tts_client(api_key: Optional[str] = None):
    """Return an OpenAI client for TTS. Only OpenAI supported currently."""
    try:
        from openai import OpenAI
    except ImportError:
        return None

    key = api_key or get_key("OPENAI_API_KEY")
    if not key:
        return None
    return OpenAI(api_key=key)


def chat_completion(client, model: str, prompt: str) -> str:
    """Send a text prompt and return the response text.

    OpenAI's native API uses ``client.responses.create()``, but DeepSeek
    (and other OpenAI-compatible providers) only support the older
    ``client.chat.completions.create()`` endpoint.

    This helper tries ``responses.create`` first; if the client doesn't
    have that method or returns a 404, it falls back to
    ``chat.completions.create``.
    """
    # Try the OpenAI Responses API first (available on openai >= 1.66)
    if not model.startswith("deepseek-") and hasattr(client, "responses"):
        try:
            resp = client.responses.create(model=model, input=prompt)
            return resp.output_text
        except Exception as exc:
            # 404 means the provider doesn't support this endpoint (e.g. DeepSeek)
            if "404" not in str(exc):
                raise

    # Fallback: Chat Completions API (universally supported)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content
