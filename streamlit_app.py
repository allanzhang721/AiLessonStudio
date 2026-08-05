"""
streamlit_app.py â€” Interactive web UI for the L15 educational video pipeline.

Run with:
    streamlit run streamlit_app.py

Features:
  1. New run form â€” user enters a question, subject, and grade.
       - "API mode": calls _generate_explanation() (GPT) to draft an explanation,
         then run_pipeline() to generate the full storyboard + video.
       - "Demo mode": loads a saved run's canonical_answer as the explanation,
         then runs the pipeline without calling GPT for explanation generation.

  2. Past runs browser â€” scans DEFAULT_OUTPUT_ROOT for subdirectories with plan.json.
       - Displays question, grade, subject, and canonical answer for each run.
       - Shows storyboard GIF/MP4 and Sora single video if available.
       - "Generate single video" button calls generate_single_video_from_run_dir()
         from single_api_video.py on the selected run.

Key data type:
  RunEntry (dataclass) â€” holds parsed metadata + file paths for one pipeline run.

Key helpers:
  discover_runs()          â€” scans output root, yields RunEntry for every plan.json dir
  discover_saved_demos()   â€” filters to curated/demo runs with playable media
  _generate_explanation()  â€” single GPT call to draft a grade-appropriate explanation
  _run_explanation_generation() / _run_demo_explanation_from_saved() â€” Streamlit callbacks
  _inject_styles()         â€” injects custom CSS for the hero banner, cards, chips, etc.
  _ensure_state_defaults() â€” initialises st.session_state keys with safe defaults
"""

from __future__ import annotations

import io
import json
import importlib
import importlib.util
import html
import os
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Optional

import streamlit as st

from pipeline.config import PLANNER_MODEL, OPENAI_TEXT_MODEL, DEEPSEEK_TEXT_MODEL
from pipeline.checker import checker1_predict
from pipeline.student_analyzer import analyze_student_weakness, infer_concept_tags
from pipeline.clients import build_text_client, chat_completion
from pipeline.api_keys import available_text_providers, available_image_providers, available_video_providers
from pipeline.pipeline import run_pipeline
from single_api_video import generate_single_video_from_run_dir


APP_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = Path("output")


def _resolve_output_root(value: str | Path | None) -> Path:
    """Resolve output root from app input and support project-relative paths."""
    raw = Path(str(value).strip()) if value is not None else DEFAULT_OUTPUT_ROOT
    if str(raw).strip() == "":
        raw = DEFAULT_OUTPUT_ROOT
    return raw if raw.is_absolute() else (APP_ROOT / raw)


def _display_path(path: Path) -> str:
    """Prefer project-relative path display to avoid machine-specific prefixes."""
    try:
        return str(path.resolve().relative_to(APP_ROOT.resolve()))
    except Exception:
        return str(path)


@dataclass
class RunEntry:
    run_dir: Path
    question_text: str
    canonical_answer: str
    grade: Optional[int]
    subject: str
    frames_dir: Path
    storyboard_video: Optional[Path]
    single_video: Optional[Path]


def _load_run_entry(run_dir: Path) -> Optional[RunEntry]:
    """Parse a pipeline output directory into a RunEntry.

    Returns None if the directory is missing a valid plan.json (so the caller
    can silently skip broken / incomplete run folders).
    Checks for storyboard.mp4 and the three possible single_api_video filenames
    in priority order to find the best available video for display.
    """
    plan_path = run_dir / "plan.json"
    if not plan_path.exists():
        return None

    try:
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    question_text = str(plan.get("question_text", "")).strip()
    canonical_answer = str(plan.get("canonical_answer", "")).strip()
    grade = plan.get("grade") if isinstance(plan.get("grade"), int) else None
    subject = str(plan.get("subject", "")).strip()

    frames_dir = run_dir / "frames"
    storyboard_video = run_dir / "storyboard.mp4"
    if not storyboard_video.exists():
        storyboard_video = None

    single_candidates = [
        run_dir / "single_api_video" / "single_api_video_captioned_with_voiceover.mp4",
        run_dir / "single_api_video" / "single_api_video_captioned.mp4",
        run_dir / "single_api_video" / "single_api_video.mp4",
    ]
    single_video = next((p for p in single_candidates if p.exists()), None)

    return RunEntry(
        run_dir=run_dir,
        question_text=question_text,
        canonical_answer=canonical_answer,
        grade=grade,
        subject=subject,
        frames_dir=frames_dir,
        storyboard_video=storyboard_video,
        single_video=single_video,
    )


def discover_runs(output_root: Path) -> list[RunEntry]:
    """Return a sorted list of RunEntry objects for every valid run directory."""
    if not output_root.exists():
        return []

    runs: list[RunEntry] = []
    for child in sorted(output_root.iterdir()):
        if not child.is_dir():
            continue
        entry = _load_run_entry(child)
        if entry is not None:
            runs.append(entry)
    return runs


def discover_saved_demos(output_root: Path) -> list[RunEntry]:
    """Return runs suitable for display in the demo selector.

    Priority order:
      1. Folders starting with "good_" or containing "demo" that have playable media.
      2. Any run with playable media that isn't an auto-generated "q_*" folder.
      3. All runs with playable media (last resort).
    """
    runs = discover_runs(output_root)
    if not runs:
        return []

    # Prefer curated demo folders and runs that already have playable media.
    curated = [
        run for run in runs
        if (
            run.run_dir.name.startswith("good_")
            or "demo" in run.run_dir.name.lower()
        )
        and (run.storyboard_video is not None or run.single_video is not None)
    ]
    if curated:
        return curated

    # Fallback: any run with playable media, excluding obvious auto-generated q_* folders when possible.
    media_runs = [
        run for run in runs
        if (run.storyboard_video is not None or run.single_video is not None)
    ]
    non_auto = [run for run in media_runs if not run.run_dir.name.startswith("q_")]
    if non_auto:
        return non_auto
    return media_runs


def _demo_label(run: RunEntry) -> str:
    grade_label = f"Grade {run.grade}" if run.grade is not None else "Grade ?"
    subject_label = run.subject or "General"
    question_label = run.question_text.strip() or run.run_dir.name
    if len(question_label) > 90:
        question_label = question_label[:87].rstrip() + "..."
    return f"{run.run_dir.name} | {grade_label} | {subject_label} | {question_label}"


def _ensure_state_defaults() -> None:
    """Initialise all required st.session_state keys with safe defaults.

    Called once at app startup. Uses setdefault-style logic so existing state
    (e.g. from a previous interaction in the same session) is never overwritten.
    """
    defaults = {
        "output_root_input": str(DEFAULT_OUTPUT_ROOT),
        "workflow_mode": "API mode",
        "question_input": "",
        "subject_input": "",
        "grade_input": 11,
        "generated_explanation": "",
        "explanation_signature": "",
        "active_run_dir": None,
        "saved_demo_choice": None,
        "checker_result": None,
        "checker2_result": None,
        "relevant_sources": "",
        "generated_quiz": "",
        "analyzer_result": None,
        "quiz_attempt_history": [],
        "language": "English",
        "text_provider": "openai",
        "image_provider": "openai",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _load_demo_inputs(demo: RunEntry) -> None:
    """Populate the question/subject/grade form fields from a saved demo RunEntry."""
    st.session_state["question_input"] = demo.question_text
    st.session_state["subject_input"] = demo.subject
    st.session_state["grade_input"] = demo.grade or st.session_state.get("grade_input", 11)
    st.session_state["generated_explanation"] = ""
    st.session_state["explanation_signature"] = ""


def _sync_demo_selection(selected_label: str, demo: RunEntry) -> None:
    """Load a demo into the form only if the selection changed since last render.

    Uses a marker string (label + folder name) stored in session_state to avoid
    re-loading inputs on every Streamlit rerun while the same demo is selected.
    """
    marker = f"{selected_label}::{demo.run_dir.name}"
    if st.session_state.get("_last_demo_marker") == marker:
        return
    _load_demo_inputs(demo)
    st.session_state["_last_demo_marker"] = marker


def _build_client(provider: str = "openai") -> Any:
    """Build an LLM client for the given provider via the centralised client factory."""
    client = build_text_client(provider)
    if client is None:
        raise RuntimeError(f"Cannot build client for provider '{provider}'. Check api_keys.txt or environment.")
    return client


def _model_for_provider(provider: str) -> str:
    """Return the model name for the given text provider."""
    if provider == "deepseek":
        return DEEPSEEK_TEXT_MODEL
    return OPENAI_TEXT_MODEL


def _generate_explanation(question: str, subject: str, grade: int, provider: str = "openai", language: str = "English") -> str:
    """Call an LLM to draft a concise, grade-appropriate explanation for the question."""
    client = _build_client(provider)
    model = _model_for_provider(provider)
    subject_line = subject or "General"
    lang_instruction = f" Write the explanation in {language}." if language != "English" else ""
    prompt = (
        "You are an expert teacher creating the core explanation for an educational visual storyboard.\n\n"
        f"Subject: {subject_line}\n"
        f"Grade: {grade}\n"
        f"Question: {question}\n\n"
        "Write one concise but instructionally strong explanation that directly answers the question. "
        "It should be accurate, grade-appropriate, and easy to translate into a 7-step visual teaching sequence. "
        f"Return only the explanation text, with no bullets, labels, or surrounding commentary.{lang_instruction}"
    )
    explanation = chat_completion(client, model, prompt).strip()
    if not explanation:
        raise RuntimeError("Explanation generation returned empty text")
    return " ".join(explanation.split())


def _generate_quiz(question: str, explanation: str, subject: str, grade: int, provider: str = "openai", language: str = "English") -> str:
    """Generate quiz questions based on the explanation."""
    client = _build_client(provider)
    model = _model_for_provider(provider)
    lang_instruction = f" Write in {language}." if language != "English" else ""
    prompt = (
        "Based on this educational explanation, generate 5 quiz questions to test student understanding.\n\n"
        f"Subject: {subject or 'General'}\nGrade: {grade}\n"
        f"Original Question: {question}\n\n"
        f"Explanation:\n{explanation}\n\n"
        "Return ONLY in this exact plain-text template and order:\n\n"
        "1. <question text>\n"
        "A) <choice A>\n"
        "B) <choice B>\n"
        "C) <choice C>\n"
        "D) <choice D>\n"
        "Correct Answer: <A|B|C|D>\n"
        "Explanation: <one short sentence>\n\n"
        "2. <question text>\n"
        "A) ...\n"
        "B) ...\n"
        "C) ...\n"
        "D) ...\n"
        "Correct Answer: <A|B|C|D>\n"
        "Explanation: ...\n\n"
        "Repeat through question 5. Do not use markdown tables. Do not omit question text."
        f"{lang_instruction}"
    )
    return chat_completion(client, model, prompt).strip()


def _estimate_cost(text_provider: str, image_provider: str, has_explanation: bool) -> dict:
    """Estimate API cost for the current pipeline run."""
    # Approximate pricing (USD)
    pricing = {
        "openai": {"text_per_1k": 0.005, "image_per_frame": 0.08, "est_text_tokens": 4000},
        "deepseek": {"text_per_1k": 0.0014, "image_per_frame": 0.0, "est_text_tokens": 4000},
        "wanx": {"text_per_1k": 0.0, "image_per_frame": 0.02, "est_text_tokens": 0},
    }
    tp = pricing.get(text_provider, pricing["openai"])
    ip = pricing.get(image_provider, pricing["openai"])
    n_frames = 7
    expl_cost = 0.0 if has_explanation else (tp["est_text_tokens"] / 1000 * tp["text_per_1k"])
    plan_cost = tp["est_text_tokens"] / 1000 * tp["text_per_1k"]
    img_cost = n_frames * ip["image_per_frame"]
    total = expl_cost + plan_cost + img_cost
    return {"explanation": expl_cost, "planning": plan_cost, "images": img_cost, "total": total}


def _make_frames_zip(frames_dir: Path) -> bytes:
    """Create a ZIP archive of all step frames."""
    buf = io.BytesIO()
    frame_paths = sorted(frames_dir.glob("step_*.png"))
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for fp in frame_paths:
            zf.write(fp, fp.name)
    return buf.getvalue()


import re as _re

def _parse_quiz(quiz_text: str) -> list[dict]:
    """Parse quiz markdown into structured questions.

    Returns a list of dicts: {question, choices: [(label, text)], answer, explanation}
    """
    def _strip_md(text: str) -> str:
        s = text.strip()
        s = _re.sub(r"^[-*\s]+", "", s)
        s = _re.sub(r"\*\*", "", s)
        return s.strip()

    def _push_current(buf: list[dict], current: dict[str, Any]) -> dict[str, Any]:
        if current.get("question") and current.get("choices"):
            if not current.get("answer") and current["choices"]:
                current["answer"] = current["choices"][0][0]
            buf.append(
                {
                    "question": str(current.get("question", "")).strip(),
                    "choices": list(current.get("choices", [])),
                    "answer": str(current.get("answer", "")).strip().upper(),
                    "explanation": str(current.get("explanation", "")).strip(),
                }
            )
        return {"question": "", "choices": [], "answer": "", "explanation": ""}

    lines = [line.rstrip() for line in quiz_text.strip().splitlines() if line.strip()]
    questions: list[dict] = []
    current: dict[str, Any] = {"question": "", "choices": [], "answer": "", "explanation": ""}
    expect_question_line = False

    q_heade×OwÒÚ$z{-®éÜj×ævWB‡6VÆV7FVEöFVÖõöÆ&VÂÐ¢VÆ–bFVÖõöÖöFS Ð¢7BæW'&÷"‚$æò6fVBFVÖ÷2f÷VæBâ"Ð¢&WGW&àÐ Ð¢–bFVÖõöÖöFS Ð¢–bÆVâ†FVÖõöÆöö·W’â Ð¢7Bç6VÆV7F&÷‚‚%6fVBFVÖ÷2"Â÷F–öç3ÖÆ—7B†FVÖõöÆöö·Wæ¶W—2‚’’Â¶W“Ò'6fVEöFVÖõö6†ö–6R"Ð¢6VÆV7FVEöFVÖõöÆ&VÂÒ7G"‡7Bç6W76–öå÷7FFRævWB‚'6fVEöFVÖõö6†ö–6R"’Ð¢6VÆV7FVEöFVÖòÒFVÖõöÆöö·WævWB‡6VÆV7FVEöFVÖõöÆ&VÂÐ Ð¢–b6VÆV7FVEöFVÖó Ð¢÷7–æ5öFVÖõ÷6VÆV7F–öâ‡6VÆV7FVEöFVÖõöÆ&VÂÂ6VÆV7FVEöFVÖòÐ¢7Bæ6F–öâ‚"¢¥VW7F–öâ‡&VBÖöæÇ’’¢¢"Ð¢7Bæ–æfò‡6VÆV7FVEöFVÖòçVW7F–öå÷FW‡BÐ Ð¢–b7Bæ'WGFöâ‚.)kbÆöBFVÖò"ÂG—SÒ'&–Ö'’"ÂW6Uö6öçF–æW%÷v–GFƒÕG'VR“ Ð¢÷'VåöFVÖõöW‡ÆæF–öåög&öÕ÷6fVB‡6VÆV7FVEöFVÖòÐ¢÷6†÷u÷6fVEöFVÖò‡6VÆV7FVEöFVÖòÐ¢7Bç&W'Vâ‚Ð¢–b7Bæ'WGFöâ‚$6ÆV""ÂW6Uö6öçF–æW%÷v–GFƒÕG'VR“ Ð¢7Bç6W76–öå÷7FFU²&7F—fU÷'VåöF—"%ÒÒæöæPÐ¢7Bç6W76–öå÷7FFU²&vVæW&FVEöW‡ÆæF–öâ%ÒÒ" Ð¢7Bç6W76–öå÷7FFU²&W‡ÆæF–öå÷6–væGW&R%ÒÒ" Ð¢7Bç6W76–öå÷7FFU²&6†V6¶W%÷&W7VÇB%ÒÒæöæPÐ¢7Bç6W76–öå÷7FFU²&6†V6¶W#%÷&W7VÇB%ÒÒæöæPÐ¢7Bç6W76–öå÷7FFU²'&VÆWfçE÷6÷W&6W2%ÒÒ" Ð¢7Bç6W76–öå÷7FFU²&vVæW&FVE÷V—¢%ÒÒ" Ð¢7Bç6W76–öå÷7FFU²'V—¥÷7V&Ö—GFVB%ÒÒfÇ6PÐ¢7Bç6W76–öå÷7FFU²&æÇ—¦W%÷&W7VÇB%ÒÒæöæPÐ¢7Bç6W76–öå÷7FFU²'V—¥öGFV×Eö†—7F÷'’%ÒÒµÐÐ¢7Bç&W'Vâ‚Ð¢VÇ6S Ð¢2’ÖöFR6öçG&öÇ0Ð¢7Bæ6F–öâ‚"¢¥VW7F–öâ¢¢"Ð¢7BçFW‡Eö&V‚%VW7F–öâ"Â¶W“Ò'VW7F–öåö–çWB"Â†V–v‡CÓSÀÐ¢Æ6V†öÆFW#Ò%7FR÷"G—RF†RVW7F–öâ†W&Râââ"ÂÆ&VÅ÷f—6–&–Æ—G“Ò&6öÆÆ6VB"Ð Ð¢7BçFW‡Eö–çWB‚%7V&¦V7B"Â¶W“Ò'7V&¦V7Eö–çWB"ÂÆ6V†öÆFW#Ò&RærâV6öÆöw’"Ð¢7Bç6Æ–FW"‚$w&FR"ÂÖ–å÷fÇVSÓrÂÖ…÷fÇVSÓ"Â¶W“Ò&w&FUö–çWB"Â7FWÓÐ¢7Bç6VÆV7F&÷‚‚$ÆæwVvR"Â÷F–öç3Õ²$VævÆ—6‚"Â.KŠÞihr"Â$W7;öÂ"Â$g&ì:v—2"Â$FWWG66‚"Â.iz^iÊÎŠ©â"Â.ÙYÎ«ZÞÉkB%ÒÂ¶W“Ò&ÆæwVvR"Ð Ð¢FW‡E÷&÷f–FW'2Òf–Æ&ÆU÷FW‡E÷&÷f–FW'2‚Ð¢GöÖÒ²&÷Væ’#¢$÷Vä’„uB’"Â&FVW6VV²#¢$FVW6VV²'ÐÐ¢–bFW‡E÷&÷f–FW'3 Ð¢GöÆ&VÇ2Ò·GöÖævWB‡Â’f÷"–âFW‡E÷&÷f–FW'5ÐÐ¢Gö–G‚ÒFW‡E÷&÷f–FW'2æ–æFW‚‡7Bç6W76–öå÷7FFRævWB‚'FW‡E÷&÷f–FW""Â&÷Væ’"’’–b7Bç6W76–öå÷7FFRævWB‚'FW‡E÷&÷f–FW""Â&÷Væ’"’–âFW‡E÷&÷f–FW'2VÇ6R Ð¢6VÅ÷GÒ7Bç6VÆV7F&÷‚‚%FW‡B&÷f–FW""Â÷F–öç3×GöÆ&VÇ2Â–æFWƒ×Gö–G‚Ð¢7Bç6W76–öå÷7FFU²'FW‡E÷&÷f–FW"%ÒÒFW‡E÷&÷f–FW'5·GöÆ&VÇ2æ–æFW‚‡6VÅ÷G•ÐÐ Ð¢–ÖvU÷&÷f–FW'2Òf–Æ&ÆUö–ÖvU÷&÷f–FW'2‚Ð¢—öÖÒ²&÷Væ’#¢$÷Vä’†wBÖ–ÖvRÓ’"Â'vç‚#¢%vç‚ŽKˆ~‹’'ÐÐ¢–b–ÖvU÷&÷f–FW'3 Ð¢—öÆ&VÇ2Ò¶—öÖævWB‡Â’f÷"–â–ÖvU÷&÷f–FW'5ÐÐ¢—ö–G‚Ò–ÖvU÷&÷f–FW'2æ–æFW‚‡7Bç6W76–öå÷7FFRævWB‚&–ÖvU÷&÷f–FW""Â&÷Væ’"’’–b7Bç6W76–öå÷7FFRævWB‚&–ÖvU÷&÷f–FW""Â&÷Væ’"’–â–ÖvU÷&÷f–FW'2VÇ6R Ð¢6VÅö—Ò7Bç6VÆV7F&÷‚‚$–ÖvR&÷f–FW""Â÷F–öç3Ö—öÆ&VÇ2Â–æFWƒÖ—ö–G‚Ð¢7Bç6W76–öå÷7FFU²&–ÖvU÷&÷f–FW"%ÒÒ–ÖvU÷&÷f–FW'5¶—öÆ&VÇ2æ–æFW‚‡6VÅö—•ÐÐ Ð¢26÷7BW7F–ÖFPÐ¢FW‡E÷&÷bÒ7G"‡7Bç6W76–öå÷7FFRævWB‚'FW‡E÷&÷f–FW""Â&÷Væ’"’Ð¢–Öu÷&÷bÒ7G"‡7Bç6W76–öå÷7FFRævWB‚&–ÖvU÷&÷f–FW""Â&÷Væ’"’Ð¢†5öW‡ÂÒ&ööÂ‡7Bç6W76–öå÷7FFRævWB‚&vVæW&FVEöW‡ÆæF–öâ"Â""’Ð¢6÷7BÒöW7F–ÖFUö6÷7B‡FW‡E÷&÷bÂ–Öu÷&÷bÂ†5öW‡ÂÐ¢6÷7E÷'G2ÒµÐÐ¢–b6÷7E²&W‡ÆæF–öâ%Òâ Ð¢6÷7E÷'G2æVæB†b$W‡ÂâG¶6÷7E²vW‡ÆæF–öâuÓ¢ã6gÒ"Ð¢6÷7E÷'G2æVæB†b%ÆââG¶6÷7E²wÆææ–æruÓ¢ã6gÒ"Ð¢6÷7E÷'G2æVæB†b$–ÖrâG¶6÷7E²v–ÖvW2uÓ¢ã&gÒ"Ð¢7Bæ6F–öâ†b/	ù+W7Bâ¢¢G¶6÷7E²wF÷FÂuÓ¢ã&gÒ¢¢‡²r²ræ¦ö–â†6÷7E÷'G2—Ò’"Ð Ð¢7BæF—f–FW"‚Ð Ð¢†5÷&÷f–FW'2Ò&ööÂ†f–Æ&ÆU÷FW‡E÷&÷f–FW'2‚’Ð¢W‡ÆæF–öå÷&VG’Ò&ööÂ‡7Bç6W76–öå÷7FFRævWB‚&vVæW&FVEöW‡ÆæF–öâ"Â""’Ð Ð¢–b7Bæ'WGFöâ‚$vVæW&FRW‡ÆæF–öâ"ÂW6Uö6öçF–æW%÷v–GFƒÕG'VRÂF—6&ÆVCÖæ÷B†5÷&÷f–FW'2“ Ð¢÷'VåöW‡ÆæF–öåövVæW&F–öâ‚Ð Ð¢6åövVåöÖVF–ÒW‡ÆæF–öå÷&VG’æB†5÷&÷f–FW'0Ð¢–b7Bæ'WGFöâ‚$vVæW&FR–ÖvW2bf–FVò"ÂG—SÒ'&–Ö'’"ÂW6Uö6öçF–æW%÷v–GFƒÕG'VRÂF—6&ÆVCÖæ÷B6åövVåöÖVF–“ Ð¢÷'VåövVæW&F–öâ‡'Våö÷Væ“ÕG'VRÂÖ¶U÷6–ævÆU÷f–FVóÕG'VRÐ Ð¢–b7Bæ'WGFöâ‚$6ÆV""ÂW6Uö6öçF–æW%÷v–GFƒÕG'VR“ Ð¢7Bç6W76–öå÷7FFU²&7F—fU÷'VåöF—"%ÒÒæöæPÐ¢7Bç6W76–öå÷7FFU²&vVæW&FVEöW‡ÆæF–öâ%ÒÒ" Ð¢7Bç6W76–öå÷7FFU²&W‡ÆæF–öå÷6–væGW&R%ÒÒ" Ð¢7Bç6W76–öå÷7FFU²&6†V6¶W%÷&W7VÇB%ÒÒæöæPÐ¢7Bç6W76–öå÷7FFU²&6†V6¶W#%÷&W7VÇB%ÒÒæöæPÐ¢7Bç6W76–öå÷7FFU²'&VÆWfçE÷6÷W&6W2%ÒÒ" Ð¢7Bç6W76–öå÷7FFU²&vVæW&FVE÷V—¢%ÒÒ" Ð¢7Bç6W76–öå÷7FFU²'V—¥÷7V&Ö—GFVB%ÒÒfÇ6PÐ¢7Bç6W76–öå÷7FFU²&æÇ—¦W%÷&W7VÇB%ÒÒæöæPÐ¢7Bç6W76–öå÷7FFU²'V—¥öGFV×Eö†—7F÷'’%ÒÒµÐÐ¢7Bç&W'Vâ‚Ð Ð¢7BæF—f–FW"‚Ð¢7BæÖ&¶F÷vâ€Ð¢sÆF—b7G–ÆSÒ&6öÆ÷#¢3“F6#ƒ¶föçB×6—¦S£ãs'&VÓ·FW‡BÖÆ–vã¦6VçFW#²#âpÐ¢t'V–ÇB'’Ç7G&öæsä¦–†–ær$4õ3Â÷7G&öæsãÂöF—cârÀÐ¢Vç6fUöÆÆ÷uö‡FÖÃÕG'VRÀÐ¢Ð Ð¢2)H)HÔ”â$T)H)H Ð¢7BæÖ&¶F÷vâ€Ð¢sÆF—b6Æ73Ò&†W&ò#ãÆƒåf—7VÄÆW76öâ“ÂöƒâpÐ¢sÇåGW&âç’VW7F–öâ–çFò6Æ77&ööÒ×&VG’f—7VÂÆW76öâãÂ÷ãÂöF—cârÀÐ¢Vç6fUöÆÆ÷uö‡FÖÃÕG'VRÀÐ¢Ð Ð¢F%öÆW76öâÂF%÷V—¢ÂF%÷&W6÷W&6W2ÂF%öFWF–Ç2Ò7BçF'2€Ð¢²/	ù9bÆW76öâ"Â/	ù9ÒV—¢"Â/	ù9¢&W6÷W&6W2"Â.(KžûˆòFWF–Ç2%ÐÐ¢Ð Ð¢2)H)HD#¢ÆW76öâ)H)H Ð¢v—F‚F%öÆW76öã Ð¢W‡ÆæF–öå÷FW‡BÒ7G"‡7Bç6W76–öå÷7FFRævWB‚&vVæW&FVEöW‡ÆæF–öâ"Â""’’ç7G&—‚Ð¢7F—fU÷'VåöF—"Ò7Bç6W76–öå÷7FFRævWB‚&7F—fU÷'VåöF—""Ð Ð¢2)H)H7FW¢W‡ÆæF–öâ)H)H Ð¢7BæÖ&¶F÷vâ‚"222227FW(	BW‡ÆæF–öâ"Ð¢–bW‡ÆæF–öå÷FW‡C Ð¢&÷f–FW%öæÖRÒ7Bç6W76–öå÷7FFRævWB‚'FW‡E÷&÷f–FW""Â&÷Væ’"Ð¢&FvUö6Ç2Ò&FVW6VV²"–b&÷f–FW%öæÖRÓÒ&FVW6VV²"VÇ6R&÷Væ’ Ð¢&FvUöÆ&VÂÒ$FVW6VV²"–b&÷f–FW%öæÖRÓÒ&FVW6VV²"VÇ6R$÷Vä’ Ð¢7BæÖ&¶F÷vâ€Ð¢bsÆF—b6Æ73Ò&W‡ÂÖÆ&VÂ#ävVæW&FVBW‡ÆæF–öâpÐ¢bsÇ7â6Æ73Ò'&÷f–FW"Ö&FvR¶&FvUö6Ç7Ò#ç¶&FvUöÆ&VÇÓÂ÷7ããÂöF—câpÐ¢bsÆF—b6Æ73Ò&W‡Â×æVÂ#ç¶‡FÖÂæW66R†W‡ÆæF–öå÷FW‡B—ÓÂöF—cârÀÐ¢Vç6fUöÆÆ÷uö‡FÖÃÕG'VRÀÐ¢Ð¢VÇ6S Ð¢7BæÖ&¶F÷vâ€Ð¢sÆF—b6Æ73Ò&W‡Â×æVÂ#ãÇ7â6Æ73Ò'Æ6V†öÆFW"#âpÐ¢uW6RF†R6–FV&"Fò6Æ–6²Ç7G&öæsävVæW&FRW‡ÆæF–öãÂ÷7G&öæsâf—'7BãÂ÷7ããÂöF—cârÀÐ¢Vç6fUöÆÆ÷uö‡FÖÃÕG'VRÀÐ¢Ð Ð¢2)H)H7FW#¢–ÖvW2bf–FVò†öæÇ’6†÷vâgFW"vVæW&F–öâ’)H)H Ð¢–bW‡ÆæF–öå÷FW‡C Ð¢7BæF—f–FW"‚Ð¢7BæÖ&¶F÷vâ‚"222227FW"(	B–ÖvW2bf–FVò"Ð¢–b7F—fU÷'VåöF—# Ð¢'VâÒöÆöE÷'VåöVçG'’…F‚†7F—fU÷'VåöF—"’Ð¢–b'Vâ—2æöæS Ð¢7BæW'&÷"‚%F†RvVæW&FVB'Vâ6÷VÆBæ÷B&RÆöFVBg&öÒF—6²â"Ð¢VÇ6S Ð¢÷&VæFW%÷'Vå÷7VÖÖ'’‡'VâÐ¢ÆVgBÂ&–v‡BÒ7Bæ6öÇVÖç2…³ÂÒÐ¢v—F‚ÆVgC Ð¢6†÷uög&ÖW2‡'Vâæg&ÖW5öF—"Ð¢v—F‚&–v‡C Ð¢6†÷u÷f–FV÷2‡'Vâç7F÷'–&ö&E÷f–FVòÂ'Vâç6–ævÆU÷f–FVòÐ¢VÇ6S Ð¢7BæÖ&¶F÷vâ€Ð¢sÆF—b6Æ73Ò&W‡Â×æVÂ#ãÇ7â6Æ73Ò'Æ6V†öÆFW"#âpÐ¢t6Æ–6²Ç7G&öæsävVæW&FR–ÖvW2f×²f–FVóÂ÷7G&öæsâ–âF†R6–FV&"Fò7&VFRF†Rf—7VÂÆW76öâãÂ÷7ããÂöF—cârÀÐ¢Vç6fUöÆÆ÷uö‡FÖÃÕG'VRÀÐ¢Ð Ð¢2)H)HD#¢V—¢)H)H Ð¢v—F‚F%÷V—£ Ð¢V—¥÷FW‡BÒ7G"‡7Bç6W76–öå÷7FFRævWB‚&vVæW&FVE÷V—¢"Â""’’ç7G&—‚Ð¢–bV—¥÷FW‡C Ð¢÷&VæFW%ö–çFW&7F—fU÷V—¢€Ð¢V—¥÷FW‡BÀÐ¢7V&¦V7C×7G"‡7Bç6W76–öå÷7FFRævWB‚'7V&¦V7Eö–çWB"Â""’’ç7G&—‚’ÀÐ¢W‡ÆæF–öå÷FW‡C×7G"‡7Bç6W76–öå÷7FFRævWB‚&vVæW&FVEöW‡ÆæF–öâ"Â""’’ç7G&—‚’ÀÐ¢6†V6¶W#%÷&W7VÇC×7Bç6W76–öå÷7FFRævWB‚&6†V6¶W#%÷&W7VÇB"’ÀÐ¢Ð¢VÇ6S Ð¢7Bæ–æfò‚$æòV—¢vVæW&FVB–WBâvVæW&FRâW‡ÆæF–öâf—'7BFò7&VFRV—¢VW7F–öç2â"Ð Ð¢2)H)HD#¢&W6÷W&6W2)H)H Ð¢v—F‚F%÷&W6÷W&6W3 Ð¢6÷W&6W5÷FW‡BÒ7G"‡7Bç6W76–öå÷7FFRævWB‚'&VÆWfçE÷6÷W&6W2"Â""’’ç7G&—‚Ð¢–b6÷W&6W5÷FW‡C Ð¢7BæÖ&¶F÷vâ‚"¢¥&VÆWfçB6÷W&6W2…vV'6—FW2Â–÷UGV&RÂFW‡F&öö·2’¢¢"Ð¢7BæÖ&¶F÷vâ‡6÷W&6W5÷FW‡BÐ¢VÇ6S Ð¢7Bæ–æfò‚$æò6÷W&6W2vVæW&FVB–WBâvVæW&FRâW‡ÆæF–öâFò6VR&VÆWfçBÆV&æ–ær&W6÷W&6W2â"Ð Ð¢7F—fU÷'VåöF—"Ò7Bç6W76–öå÷7FFRævWB‚&7F—fU÷'VåöF—""Ð¢–b7F—fU÷'VåöF—# Ð¢'VâÒöÆöE÷'VåöVçG'’…F‚†7F—fU÷'VåöF—"’Ð¢–b'Vâ—2æ÷BæöæS Ð¢7BæF—f–FW"‚Ð¢7BæÖ&¶F÷vâ‚"¢¤F÷væÆöG2¢¢"Ð¢FÅö6öÇ2Ò7Bæ6öÇVÖç2ƒBÐ¢v—F‚FÅö6öÇ5³Ó Ð¢–b'Vâæg&ÖW5öF—"æW†—7G2‚’æBÆ—7B‡'Vâæg&ÖW5öF—"ævÆö"‚'7FWò¢çær"’“ Ð¢7BæF÷væÆöEö'WGFöâ€Ð¢/	ù:bg&ÖW2…¤•’"ÀÐ¢FFÕöÖ¶Uög&ÖW5÷¦—‡'Vâæg&ÖW5öF—"’ÀÐ¢f–ÆUöæÖSÒ&ÆW76öåög&ÖW2ç¦—"ÀÐ¢Ö–ÖSÒ&Æ–6F–öâ÷¦—"ÀÐ¢W6Uö6öçF–æW%÷v–GFƒÕG'VRÀÐ¢Ð¢v—F‚FÅö6öÇ5³Ó Ð¢f–E÷F‚Ò'Vâç7F÷'–&ö&E÷f–FVò÷"'Vâç6–ævÆU÷f–FVðÐ¢–bf–E÷F‚æBf–E÷F‚æW†—7G2‚“ Ð¢7BæF÷væÆöEö'WGFöâ€Ð¢/	øêÂf–FVò„ÕB’"ÀÐ¢FF×f–E÷F‚ç&VEö'—FW2‚’ÀÐ¢f–ÆUöæÖSÒ&ÆW76öå÷f–FVòæ×B"ÀÐ¢Ö–ÖSÒ'f–FVòö×B"ÀÐ¢W6Uö6öçF–æW%÷v–GFƒÕG'VRÀÐ¢Ð¢v—F‚FÅö6öÇ5³%Ó Ð¢W‡ÂÒ7G"‡7Bç6W76–öå÷7FFRævWB‚&vVæW&FVEöW‡ÆæF–öâ"Â""’’ç7G&—‚Ð¢–bW‡Ã Ð¢7BæF÷væÆöEö'WGFöâ€Ð¢/	ù8BW‡ÆæF–öâ…E…B’"ÀÐ¢FFÖW‡ÂÀÐ¢f–ÆUöæÖSÒ&W‡ÆæF–öâçG‡B"ÀÐ¢Ö–ÖSÒ'FW‡B÷Æ–â"ÀÐ¢W6Uö6öçF–æW%÷v–GFƒÕG'VRÀÐ¢Ð¢v—F‚FÅö6öÇ5³5Ó Ð¢V—¥öFÂÒ7G"‡7Bç6W76–öå÷7FFRævWB‚&vVæW&FVE÷V—¢"Â""’’ç7G&—‚Ð¢–bV—¥öFÃ Ð¢7BæF÷væÆöEö'WGFöâ€Ð¢/	ù9ÒV—¢„ÔB’"ÀÐ¢FF×V—¥öFÂÀÐ¢f–ÆUöæÖSÒ'V—¢æÖB"ÀÐ¢Ö–ÖSÒ'FW‡BöÖ&¶F÷vâ"ÀÐ¢W6Uö6öçF–æW%÷v–GFƒÕG'VRÀÐ¢Ð Ð¢2)H)HD#¢FWF–Ç2)H)H Ð¢v—F‚F%öFWF–Ç3 Ð¢6†V6¶W%÷&W7VÇBÒ7Bç6W76–öå÷7FFRævWB‚&6†V6¶W%÷&W7VÇB"Ð¢–b6†V6¶W%÷&W7VÇBæB—6–ç7Fæ6R†6†V6¶W%÷&W7VÇBÂF–7B’æB6†V6¶W%÷&W7VÇBævWB‚'&÷VæG2"“ Ð¢7BæÖ&¶F÷vâ‚"¢¤6†V6¶W"&W7VÇG2„F—7F–Ä$U%BW'&÷"ÕG—R6Æ76–f–W"’¢¢"Ð¢–b6†V6¶W%÷&W7VÇBævWB‚'v5÷&Wf—6VB"“ Ð¢7Bç7V66W72†b$W‡ÆæF–öâv2&Wf—6VBgFW"¶6†V6¶W%÷&W7VÇE²wF÷FÅ÷&÷VæG2u×Ò6†V6¶W"&÷VæB‡2’â"Ð¢VÇ6S Ð¢7Bæ–æfò‚$W‡ÆæF–öâ66WFVB'’6†V6¶W"â"Ð¢f÷"&æB–â6†V6¶W%÷&W7VÇE²'&÷VæG2%Ó Ð¢7"Ò&æBævWB‚&6†V6¶W%÷&W7VÇB"Â·ÒÐ¢7BæÖ&¶F÷vâ†b"¢¥&÷VæB·&æE²w&÷VæBu×Ò¢£¢¶7"ævWB‚vÆ&VÂrÂsòr—Ò†6öæf–FVæ6R¶7"ævWB‚v6öæf–FVæ6RrÂ“¢ã6gÒ’(	B7F–öã¢·&æE²v7F–öâu×Ò"Ð¢–b7"ævWB‚'&ö&&–Æ—F–W2"“ Ð¢7Bæ§6öâ†7%²'&ö&&–Æ—F–W2%ÒÐ¢VÇ6S Ð¢7Bæ–æfò‚$æò6†V6¶W"&W7VÇG2f–Æ&ÆRâ'VâF†R—VÆ–æRFò6VRW'&÷"×G—R6Æ76–f–6F–öââ"Ð Ð¢7BæF—f–FW"‚Ð¢6†V6¶W#%÷&W7VÇBÒ7Bç6W76–öå÷7FFRævWB‚&6†V6¶W#%÷&W7VÇB"Ð¢–b6†V6¶W#%÷&W7VÇBæB—6–ç7Fæ6R†6†V6¶W#%÷&W7VÇBÂF–7B“ Ð¢7BæÖ&¶F÷vâ‚"¢¤6†V6¶W""&W7VÇG2„g&ÖRVÆ—G’fÆ–FF÷"’¢¢"Ð¢–b6†V6¶W#%÷&W7VÇBævWB‚&W'&÷""“ Ð¢7BæW'&÷"†b$6†V6¶W""W'&÷#¢¶6†V6¶W#%÷&W7VÇE²vW'&÷"u×Ò"Ð¢VÇ6S Ð¢76VBÒ&ööÂ†6†V6¶W#%÷&W7VÇBævWB‚'72"ÂfÇ6R’Ð¢66÷&RÒfÆöB†6†V6¶W#%÷&W7VÇBævWB‚&÷fW&ÆÅ÷66÷&R"Âã’Ð¢F‡&W6†öÆBÒfÆöB†6†V6¶W#%÷&W7VÇBævWB‚'F‡&W6†öÆB"Âã’Ð¢ÖöFRÒ7G"†6†V6¶W#%÷&W7VÇBævWB‚&ÖöFR"Â&†WW&—7F–2"’Ð¢–b76VC Ð¢7Bç7V66W72†b$6†V6¶W""76VB†ÖöFS×¶ÖöFWÒÂ66÷&S×·66÷&S¢ã6gÒÂF‡&W6†öÆC×·F‡&W6†öÆC¢ã&gÒ’â"Ð¢VÇ6S Ð¢f–ÆVE÷7FW2Ò6†V6¶W#%÷&W7VÇBævWB‚&f–ÆVE÷7FW2"ÂµÒÐ¢7Bçv&æ–ær€Ð¢b$6†V6¶W""fÆvvVBg&ÖRVÆ—G’†ÖöFS×¶ÖöFWÒÂ66÷&S×·66÷&S¢ã6gÒÂ Ð¢b'F‡&W6†öÆC×·F‡&W6†öÆC¢ã&gÒÂf–ÆVB7FW3×¶f–ÆVE÷7FW7Ò’â Ð¢Ð Ð¢W%ög&ÖRÒ6†V6¶W#%÷&W7VÇBævWB‚'W%ög&ÖR"ÂµÒÐ¢–b—6–ç7Fæ6R‡W%ög&ÖRÂÆ—7B’æBW%ög&ÖS Ð¢f÷"—FVÒ–âW%ög&ÖS Ð¢6–BÒ—FVÒævWB‚'7FWö–B"Â#ò"Ð¢—FVÕ÷66÷&RÒfÆöB†—FVÒævWB‚'66÷&R"Âã’Ð¢—FVÕ÷72Ò&ööÂ†—FVÒævWB‚'72"ÂfÇ6R’Ð¢—77VW2Ò—FVÒævWB‚&—77VW2"ÂµÒÐ¢7BæÖ&¶F÷vâ€Ð¢b"¢¥7FW·6–GÒ¢£¢²u52r–b—FVÕ÷72VÇ6Rtd”ÂwÒ Ð¢b"‡66÷&R¶—FVÕ÷66÷&S¢ã6gÒ’Â—77VW3¢¶—77VW2÷"væöæRwÒ Ð¢Ð¢VÇ6S Ð¢7Bæ–æfò‚$æò6†V6¶W""&W7VÇG2f–Æ&ÆRâ'Vâ–ÖvRvVæW&F–öâFòfÆ–FFRg&ÖRVÆ—G’â"Ð Ð¢7BæF—f–FW"‚Ð¢æÇ—¦W%÷&W7VÇBÒ7Bç6W76–öå÷7FFRævWB‚&æÇ—¦W%÷&W7VÇB"Ð¢–bæÇ—¦W%÷&W7VÇBæB—6–ç7Fæ6R†æÇ—¦W%÷&W7VÇBÂF–7B’æBæÇ—¦W%÷&W7VÇBævWB‚'7FGW2"’ÓÒ&ö²# Ð¢7BæÖ&¶F÷vâ‚"¢¥7GVFVçBvV¶æW72æÇ—¦W"¢¢"Ð¢7Bæ§6öâ†æÇ—¦W%÷&W7VÇBÐ¢VÇ6S Ð¢7Bæ–æfò‚$æòæÇ—¦W"&W7VÇG2f–Æ&ÆRâ6ö×ÆWFRæB7V&Ö—BV—¢Fò6VRF–væ÷7F–72â"Ð Ð¢7F—fU÷'VåöF—"Ò7Bç6W76–öå÷7FFRævWB‚&7F—fU÷'VåöF—""Ð Ð Ð¦g&öÒ÷c"–×÷'BÖ–àÐ Ð Ð¦–bõöæÖUõòÓÒ%õöÖ–åõò# Ð¢Ö–â‚