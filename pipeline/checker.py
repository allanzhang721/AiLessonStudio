"""
checker.py — Checker 1: DistilBERT error-type classifier for generated explanations.

Loads a fine-tuned DistilBERT model (5-class) that classifies an explanation
into one of five pedagogical error types. The model was trained on explanations
already identified as *Inconsistent* — it answers "what kind of error is this?"

Label mapping (alphabetical, matching sklearn.preprocessing.LabelEncoder):
  0: ConceptError       — Applies the wrong principle or definition
  1: GradeMismatch      — Uses concepts beyond the target grade level
  2: LogicalGap         — Jumps from premise to conclusion without mechanism
  3: MisleadingAnalogy  — Uses a convincing but incorrect analogy
  4: MissingCondition   — Omits key assumptions or limiting conditions

Public API:
  load_checker1()                  — load model + tokenizer (cached)
  checker1_predict(...)            — classify a single explanation
  build_checker_input_text(...)    — format the 4-field input string
  gpt_fix_explanation(...)         — ask GPT to fix an explanation given the error label
  checker1_loop(...)               — generate → check → fix loop (up to max_rounds)
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

from .clients import chat_completion
from .config import CHECKER1_CKPT_DIR, CHECKER1_MAX_LEN, CHECKER1_LABELS, PLANNER_MODEL


# ---------------------------------------------------------------------------
# Input formatting (must match training pipeline in L9 notebook)
# ---------------------------------------------------------------------------

def build_checker_input_text(
    grade: int,
    question: str,
    explanation: str,
    subject: str = "",
) -> str:
    """Build the concatenated input string expected by Checker 1.

    Format matches ``build_input_text()`` from the training notebook::

        Subject: Physics
        Grade: 9
        Question: Why does ...?
        Explanation: Because ...
    """
    subj_line = f"Subject: {subject}\n" if subject else ""
    return (
        f"{subj_line}"
        f"Grade: {grade}\n"
        f"Question: {question}\n"
        f"Explanation: {explanation}"
    )


# ---------------------------------------------------------------------------
# Model loading (lazy, cached)
# ---------------------------------------------------------------------------

_checker1_cache: dict = {}


def load_checker1(checkpoint_dir: Optional[Path] = None):
    """Load the Checker 1 DistilBERT model and tokenizer.

    Returns (model, tokenizer).  Results are cached so repeated calls are free.
    Requires ``transformers`` and ``torch``.
    """
    ckpt = str(checkpoint_dir or CHECKER1_CKPT_DIR)
    if ckpt in _checker1_cache:
        return _checker1_cache[ckpt]

    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch

    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModelForSequenceClassification.from_pretrained(ckpt)
    model.eval()

    _checker1_cache[ckpt] = (model, tokenizer)
    return model, tokenizer


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def checker1_predict(
    grade: int,
    question: str,
    explanation: str,
    subject: str = "",
    checkpoint_dir: Optional[Path] = None,
) -> dict:
    """Run Checker 1 on a single explanation and return the result.

    Returns a dict with:
      - ``label``       : str — predicted error-type name
      - ``label_id``    : int — numeric class index
      - ``confidence``  : float — softmax probability for the predicted class
      - ``probabilities``: dict[str, float] — softmax probs for all 5 classes
    """
    import torch

    model, tokenizer = load_checker1(checkpoint_dir)
    text = build_checker_input_text(grade, question, explanation, subject)
    enc = tokenizer(text, truncation=True, max_length=CHECKER1_MAX_LEN, return_tensors="pt")

    with torch.no_grad():
        logits = model(**enc).logits                       # (1, 5)
        probs = torch.softmax(logits, dim=-1)[0]           # (5,)

    pred_id = int(torch.argmax(probs).item())
    pred_label = CHECKER1_LABELS[pred_id]
    confidence = float(probs[pred_id].item())

    return {
        "label": pred_label,
        "label_id": pred_id,
        "confidence": confidence,
        "probabilities": {CHECKER1_LABELS[i]: round(float(probs[i].item()), 4) for i in range(len(CHECKER1_LABELS))},
    }


# ---------------------------------------------------------------------------
# GPT-based explanation repair
# ---------------------------------------------------------------------------

_REPAIR_INSTRUCTIONS: dict[str, str] = {
    "ConceptError": (
        "The explanation applies the wrong scientific principle, definition, or causal mechanism. "
        "Identify the conceptual mistake and rewrite the explanation using the correct principle."
    ),
    "GradeMismatch": (
        "The explanation uses terminology, equations, or concepts beyond the target grade level. "
        "Rewrite it using vocabulary and reasoning appropriate for the specified grade."
    ),
    "LogicalGap": (
        "The explanation jumps from a premise to a conclusion without providing the necessary "
        "intermediate causal steps. Fill in the missing mechanism so the reasoning is complete."
    ),
    "MisleadingAnalogy": (
        "The explanation uses an analogy that maps the wrong underlying mechanism. "
        "Remove or replace the analogy with one that correctly represents the concept."
    ),
    "MissingCondition": (
        "The explanation omits key assumptions or limiting conditions required for the claim to hold. "
        "Add the necessary conditions and qualifications."
    ),
}


def gpt_fix_explanation(
    client,
    question: str,
    explanation: str,
    grade: int,
    subject: str,
    error_label: str,
    model: Optional[str] = None,
) -> str:
    """Ask GPT to fix an explanation based on the detected error type.

    Returns the corrected explanation as a plain string.
    """
    used_model = model or PLANNER_MODEL
    repair_detail = _REPAIR_INSTRUCTIONS.get(error_label, "Fix any pedagogical errors.")
    prompt = (
        "You are an expert teacher fixing a flawed student explanation.\n\n"
        f"Subject: {subject or 'General'}\n"
        f"Grade: {grade}\n"
        f"Question: {question}\n\n"
        f"Original explanation (contains a {error_label} error):\n"
        f"{explanation}\n\n"
        f"Error diagnosis: {repair_detail}\n\n"
        "Requirements:\n"
        "- Fix the identified error while preserving the overall structure and length.\n"
        "- Keep the explanation accurate, grade-appropriate, and fluent.\n"
        "- Do NOT add disclaimers, meta-commentary, or bullet points.\n"
        "- Return ONLY the corrected explanation text."
    )
    fixed = chat_completion(client, used_model, prompt).strip()
    if not fixed:
        return explanation  # safety fallback
    return " ".join(fixed.split())


# ---------------------------------------------------------------------------
# Generate → check → fix loop
# ---------------------------------------------------------------------------

def checker1_loop(
    client,
    question: str,
    explanation: str,
    grade: int,
    subject: str = "",
    max_rounds: int = 3,
    confidence_threshold: float = 0.5,
    checkpoint_dir: Optional[Path] = None,
    model: Optional[str] = None,
) -> dict:
    """Run the Checker 1 loop: predict error type, fix via GPT, re-check.

    The loop runs the DistilBERT classifier on the explanation.  Since the
    model was trained *only* on bad explanations, a low-confidence prediction
    (below ``confidence_threshold``) is interpreted as "the explanation is
    likely fine".  If confidence is high, GPT is asked to fix the detected
    error, and the checker re-evaluates the revised explanation.

    Returns a dict with:
      - ``final_explanation``: str — the (possibly revised) explanation
      - ``rounds``: list[dict] — per-round checker results + actions taken
      - ``was_revised``: bool — whether the explanation was changed at all
      - ``total_rounds``: int — number of checker rounds executed
    """
    current = explanation
    rounds: list[dict] = []

    for i in range(max_rounds):
        result = checker1_predict(
            grade=grade,
            question=question,
            explanation=current,
            subject=subject,
            checkpoint_dir=checkpoint_dir,
        )
        round_info: dict = {
            "round": i + 1,
            "checker_result": result,
            "action": "none",
        }

        if result["confidence"] < confidence_threshold:
            round_info["action"] = "accepted"
            rounds.append(round_info)
            break

        # High confidence → fix via GPT
        if client is None:
            round_info["action"] = "flagged_no_client"
            rounds.append(round_info)
            break

        fixed = gpt_fix_explanation(
            client=client,
            question=question,
            explanation=current,
            grade=grade,
            subject=subject,
            error_label=result["label"],
            model=model,
        )
        round_info["action"] = "revised"
        round_info["revised_explanation"] = fixed
        rounds.append(round_info)
        current = fixed
    else:
        # exhausted max_rounds — accept whatever we have
        pass

    return {
        "final_explanation": current,
        "rounds": rounds,
        "was_revised": current != explanation,
        "total_rounds": len(rounds),
    }
