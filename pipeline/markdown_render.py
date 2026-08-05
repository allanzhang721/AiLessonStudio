"""Safe normalization for model-authored Markdown and KaTeX content."""
from __future__ import annotations

import re
from typing import Any

_MATH_FENCE = re.compile(r"```(?:latex|tex|math)\s*(.*?)```", re.IGNORECASE | re.DOTALL)
_DISPLAY_PARENS = re.compile(r"\\\[(.*?)\\\]", re.DOTALL)
_INLINE_PARENS = re.compile(r"\\\((.*?)\\\)", re.DOTALL)
_EQUATION_ENV = re.compile(r"\\begin\{(?:equation\*?|displaymath)\}(.*?)\\end\{(?:equation\*?|displaymath)\}", re.DOTALL)


def preserve_markdown(value: Any, maximum: int = 5000) -> str:
    """Keep paragraph/list structure while collapsing repeated blank lines."""
    raw = str(value or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    lines = [line.rstrip() for line in raw.split("\n")]
    cleaned: list[str] = []
    blank = False
    for line in lines:
        if not line.strip():
            if cleaned and not blank:
                cleaned.append("")
            blank = True
        else:
            cleaned.append(line.strip())
            blank = False
    return "\n".join(cleaned).strip()[:maximum]


def streamlit_markdown(value: Any) -> str:
    """Convert common model LaTeX wrappers to Streamlit/KaTeX delimiters."""
    text = preserve_markdown(value, maximum=16000)
    text = _MATH_FENCE.sub(lambda match: f"\n$$\n{match.group(1).strip()}\n$$\n", text)
    text = _EQUATION_ENV.sub(lambda match: f"\n$$\n{match.group(1).strip()}\n$$\n", text)
    text = _DISPLAY_PARENS.sub(lambda match: f"\n$$\n{match.group(1).strip()}\n$$\n", text)
    text = _INLINE_PARENS.sub(lambda match: f"${match.group(1).strip()}$", text)
    return preserve_markdown(text, maximum=16000)
