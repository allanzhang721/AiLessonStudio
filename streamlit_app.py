"""
streamlit_app.py — Interactive web UI for the L15 educational video pipeline.

Run with:
    streamlit run streamlit_app.py

Features:
  1. New run form — user enters a question, subject, and grade.
       - "API mode": calls _generate_explanation() (GPT) to draft an explanation,
         then run_pipeline() to generate the full storyboard + video.
       - "Demo mode": loads a saved run's canonical_answer as the explanation,
         then runs the pipeline without calling GPT for explanation generation.

  2. Past runs browser — scans DEFAULT_OUTPUT_ROOT for subdirectories with plan.json.
       - Displays question, grade, subject, and canonical answer for each run.
       - Shows storyboard GIF/MP4 and Sora single video if available.
       - "Generate single video" button calls generate_single_video_from_run_dir()
         from single_api_video.py on the selected run.

Key data type:
  RunEntry (dataclass) — holds parsed metadata + file paths for one pipeline run.

Key helpers:
  discover_runs()          — scans output root, yields RunEntry for every plan.json dir
  discover_saved_demos()   — filters to curated/demo runs with playable media
  _generate_explanation()  — single GPT call to draft a grade-appropriate explanation
  _run_explanation_generation() / _run_demo_explanation_from_saved() — Streamlit callbacks
  _inject_styles()         — injects custom CSS for the hero banner, cards, chips, etc.
  _ensure_state_defaults() — initialises st.session_state keys with safe defaults
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
from pipeline.single_api_video import generate_single_video_from_run_dir


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

    q_header_re = _re.compile(r"^\s*(?:\*\*)?(?:Q(?:uestion)?\s*\d+|\d+)[\).:\-\s]*(.*)$", _re.IGNORECASE)
    choice_re = _re.compile(r"^\s*[-*\s]*([A-D])[\).\]:-]\s*(.+)$", _re.IGNORECASE)
    answer_re = _re.compile(r"^\s*[-*\s]*(?:correct\s*answer|answer|correct)\s*[:\-]?\s*([A-D])\b", _re.IGNORECASE)
    explain_re = _re.compile(r"^\s*[-*\s]*(?:explanation|reason)\s*[:\-]?\s*(.*)$", _re.IGNORECASE)

    for raw_line in lines:
        line = _strip_md(raw_line)
        if not line:
            continue

        qh = q_header_re.match(line)
        if qh:
            current = _push_current(questions, current)
            q_text = _strip_md(qh.group(1))
            if q_text:
                current["question"] = q_text
                expect_question_line = False
            else:
                expect_question_line = True
            continue

        ch = choice_re.match(line)
        if ch:
            if not current.get("question"):
                # If choices appear before explicit question marker, start a fallback question.
                current["question"] = "Question text missing from model output"
            current["choices"].append((ch.group(1).upper(), _strip_md(ch.group(2))))
            continue

        ans = answer_re.match(line)
        if ans:
            current["answer"] = ans.group(1).upper()
            continue

        exp = explain_re.match(line)
        if exp:
            tail = _strip_md(exp.group(1))
            if tail:
                current["explanation"] = (str(current.get("explanation", "")) + " " + tail).strip()
            continue

        if expect_question_line and not current.get("question"):
            current["question"] = line
            expect_question_line = False
            continue

        if line.endswith("?") and not current.get("question"):
            current["question"] = line
            continue

        if current.get("answer"):
            current["explanation"] = (str(current.get("explanation", "")) + " " + line).strip()

    _push_current(questions, current)
    return questions


def _build_quiz_attempts(
    questions: list[dict],
    *,
    subject: str,
    explanation_text: str,
    submitted_at: float,
) -> list[dict[str, Any]]:
    attempts: list[dict[str, Any]] = []
    for i, q in enumerate(questions):
        selected = st.session_state.get(f"quiz_q_{i}", "")
        selected_label = selected[0] if selected else ""
        correct_label = str(q.get("answer", "")).strip().upper()
        start_key = f"quiz_start_{i}"
        start_time = float(st.session_state.get(start_key, submitted_at))
        elapsed = max(1.0, submitted_at - start_time)
        confidence = int(st.session_state.get(f"quiz_conf_{i}", 3))
        attempts.append(
            {
                "question_id": f"quiz_q_{i+1}",
                "question_text": str(q.get("question", "")),
                "concept_tags": infer_concept_tags(
                    question_text=str(q.get("question", "")),
                    explanation_text=explanation_text,
                    subject=subject,
                    max_tags=2,
                ),
                "correct": selected_label == correct_label,
                "response_time_seconds": elapsed,
                "confidence": confidence,
            }
        )
    return attempts


def _render_weakness_report(analyzer_result: dict[str, Any]) -> None:
    if not analyzer_result or analyzer_result.get("status") != "ok":
        return

    st.markdown("### Student Weakness Analyzer")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Overall Accuracy", f"{float(analyzer_result.get('overall_accuracy', 0.0)) * 100:.1f}%")
    with c2:
        st.metric("Avg Response Time", f"{float(analyzer_result.get('overall_avg_response_seconds', 0.0)):.1f}s")
    with c3:
        st.metric("Content Risk (Checker 2)", f"{float(analyzer_result.get('content_risk', 0.0)):.2f}")

    top = analyzer_result.get("top_weak_concepts", [])
    if isinstance(top, list) and top:
        st.markdown("**Top Weak Concepts**")
        for idx, row in enumerate(top, start=1):
            concept = str(row.get("concept", "general"))
            score = float(row.get("weakness_score", 0.0))
            wtype = str(row.get("weakness_type", "moderate_gap"))
            st.markdown(
                f"**{idx}. {concept}** | weakness={score:.3f} | type={wtype} | "
                f"accuracy={float(row.get('accuracy', 0.0)) * 100:.1f}%"
            )
            actions = row.get("recommended_actions", [])
            if isinstance(actions, list):
                for action in actions[:2]:
                    st.write(f"- {action}")


def _render_interactive_quiz(quiz_text: str, *, subject: str, explanation_text: str, checker2_result: dict[str, Any] | None) -> None:
    """Render quiz as interactive radio buttons with check answers."""
    questions = _parse_quiz(quiz_text)
    if not questions:
        # Fallback: just render as markdown
        st.markdown(quiz_text)
        return

    st.markdown("**Quiz — Test Your Understanding**")

    quiz_signature = f"{len(questions)}::{hash(quiz_text)}"
    if st.session_state.get("_quiz_signature") != quiz_signature:
        st.session_state["_quiz_signature"] = quiz_signature
        st.session_state["quiz_submitted"] = False
        st.session_state["analyzer_result"] = None
        st.session_state["quiz_attempt_history"] = []
        for i in range(len(questions)):
            st.session_state.pop(f"quiz_start_{i}", None)
            st.session_state.pop(f"quiz_conf_{i}", None)
            st.session_state.pop(f"quiz_q_{i}", None)

    # Initialize quiz state
    if "quiz_submitted" not in st.session_state:
        st.session_state["quiz_submitted"] = False

    now = time.time()

    for i, q in enumerate(questions):
        st.markdown(f"**{i + 1}. {q['question']}**")
        start_key = f"quiz_start_{i}"
        if start_key not in st.session_state:
            st.session_state[start_key] = now

        options = [f"{label}) {text}" for label, text in q["choices"]]
        selected = st.radio(
            f"Q{i + 1}",
            options=options,
            index=None,
            key=f"quiz_q_{i}",
            label_visibility="collapsed",
        )
        st.slider(
            "Confidence (1=guess, 5=very sure)",
            min_value=1,
            max_value=5,
            value=int(st.session_state.get(f"quiz_conf_{i}", 3)),
            key=f"quiz_conf_{i}",
        )

        # Show result if submitted
        if st.session_state.get("quiz_submitted"):
            selected_label = selected[0] if selected else ""
            if selected_label == q["answer"]:
                st.success(f"✅ Correct! {q['explanation']}")
            else:
                st.error(f"❌ The correct answer is **{q['answer']}**. {q['explanation']}")
        st.markdown("---")

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Check Answers", type="primary", use_container_width=True, key="quiz_check"):
            st.session_state["quiz_submitted"] = True
            submitted_at = time.time()
            attempts = _build_quiz_attempts(
                questions,
                subject=subject,
                explanation_text=explanation_text,
                submitted_at=submitted_at,
            )
            st.session_state["quiz_attempt_history"] = attempts
            analyzer_result = analyze_student_weakness(
                attempts,
                checker2_result=checker2_result,
                top_k=3,
            )
            st.session_state["analyzer_result"] = analyzer_result

            active_run_dir = st.session_state.get("active_run_dir")
            if active_run_dir and isinstance(analyzer_result, dict):
                try:
                    run_path = Path(active_run_dir)
                    run_path.joinpath("student_analyzer.json").write_text(
                        json.dumps(analyzer_result, indent=2, ensure_ascii=False),
                        encoding="utf-8",
                    )
                except Exception:
                    pass
            st.rerun()
    with col2:
        if st.button("Reset Quiz", use_container_width=True, key="quiz_reset"):
            st.session_state["quiz_submitted"] = False
            st.session_state["analyzer_result"] = None
            st.session_state["quiz_attempt_history"] = []
            for i in range(len(questions)):
                if f"quiz_q_{i}" in st.session_state:
                    del st.session_state[f"quiz_q_{i}"]
                if f"quiz_conf_{i}" in st.session_state:
                    del st.session_state[f"quiz_conf_{i}"]
                if f"quiz_start_{i}" in st.session_state:
                    del st.session_state[f"quiz_start_{i}"]
            st.rerun()

    if st.session_state.get("quiz_submitted"):
        correct = 0
        for i, q in enumerate(questions):
            sel = st.session_state.get(f"quiz_q_{i}", "")
            if sel and sel[0] == q["answer"]:
                correct += 1
        st.markdown(f"### Score: {correct} / {len(questions)}")

    analyzer_result = st.session_state.get("analyzer_result")
    if analyzer_result and isinstance(analyzer_result, dict):
        st.divider()
        _render_weakness_report(analyzer_result)


def _generate_sources(question: str, subject: str, grade: int, provider: str = "openai") -> str:
    """Ask the LLM to suggest relevant learning sources for the question."""
    client = _build_client(provider)
    model = _model_for_provider(provider)
    prompt = (
        "You are an expert educator. A student asked the following question:\n\n"
        f"Subject: {subject or 'General'}\n"
        f"Grade: {grade}\n"
        f"Question: {question}\n\n"
        "Suggest 5-8 high-quality, real learning resources the student can use to study this topic further. "
        "Include a mix of:\n"
        "- Websites (e.g. Khan Academy, Wikipedia, official educational sites)\n"
        "- YouTube channels or specific video series\n"
        "- Textbooks or online courses\n\n"
        "For each resource, provide:\n"
        "1. The name/title\n"
        "2. The URL (if applicable — use real, well-known URLs only)\n"
        "3. A one-sentence description of why it's helpful\n\n"
        "Format each as a bullet point. Only return the list, no intro or outro."
    )
    return chat_completion(client, model, prompt).strip()


def _input_signature(question: str, subject: str, grade: int) -> str:
    return f"{question.strip().lower()}||{subject.strip().lower()}||{grade}"


def _set_explanation_state(explanation: str, question: str, subject: str, grade: int) -> None:
    st.session_state["generated_explanation"] = explanation
    st.session_state["explanation_signature"] = _input_signature(question, subject, grade)


def _run_explanation_generation() -> None:
    question = str(st.session_state.get("question_input", "")).strip()
    subject = str(st.session_state.get("subject_input", "")).strip()
    grade = int(st.session_state.get("grade_input", 11))
    text_provider = str(st.session_state.get("text_provider", "openai"))

    if not question:
        st.error("Question is required before generating explanation.")
        return
    text_providers = available_text_providers()
    if not text_providers:
        st.error("No API key configured for any text provider. Add keys to api_keys.txt.")
        return

    try:
        with st.status("Generating explanation, sources & quiz", expanded=True) as status:
            language = str(st.session_state.get("language", "English"))
            status.write("Generating explanation...")
            explanation = _generate_explanation(question=question, subject=subject, grade=grade, provider=text_provider, language=language)
            _set_explanation_state(explanation, question, subject, grade)
            status.write("Finding relevant sources...")
            sources = _generate_sources(question=question, subject=subject, grade=grade, provider=text_provider)
            st.session_state["relevant_sources"] = sources
            status.write("Generating quiz questions...")
            quiz = _generate_quiz(question=question, explanation=explanation, subject=subject, grade=grade, provider=text_provider, language=language)
            st.session_state["generated_quiz"] = quiz
            # Save sources if a run dir is active
            run_dir = st.session_state.get("active_run_dir")
            if run_dir:
                run_path = Path(run_dir)
                run_path.joinpath("sources.md").write_text(sources, encoding="utf-8")
                run_path.joinpath("quiz.md").write_text(quiz, encoding="utf-8")
            status.update(label="Explanation, sources & quiz ready", state="complete", expanded=False)
        st.rerun()
    except Exception as exc:
        st.error(f"Explanation generation failed: {exc}")


def _run_demo_explanation_from_saved(demo: RunEntry) -> None:
    question = str(st.session_state.get("question_input", "")).strip()
    subject = str(st.session_state.get("subject_input", "")).strip()
    grade = int(st.session_state.get("grade_input", 11))

    explanation = (demo.canonical_answer or "").strip()
    if not explanation:
        explanation = (
            f"This concept can be explained step by step for Grade {grade} by showing the key elements, "
            "their relationships, and the causal changes that answer the question."
        )
    _set_explanation_state(explanation, question, subject, grade)


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        /* ── Base ── */
        .stApp {
            background: #f0f4f8;
            color: #0f172a;
            font-family: "Inter", "SF Pro Display", "Segoe UI", sans-serif;
        }
        .block-container { padding-top: 1rem; max-width: 1260px; }

        /* ── Hero banner ── */
        .hero {
            border-radius: 16px;
            padding: 1.25rem 1.6rem 1.4rem;
            background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 45%, #2563eb 100%);
            color: #f8fafc;
            margin-bottom: 1rem;
            box-shadow: 0 12px 28px rgba(2, 6, 23, 0.25);
        }
        .hero h1 { margin: 0; font-size: 1.75rem; font-weight: 800; letter-spacing: -0.3px; }
        .hero p  { margin: 0.3rem 0 0; font-size: 0.92rem; color: #bfdbfe; }

        /* ── Cards ── */
        .cfg-card {
            background: #fff;
            border: 1px solid #e2e8f0;
            border-radius: 14px;
            padding: 1rem 1.1rem 0.8rem;
            box-shadow: 0 4px 14px rgba(15, 23, 42, 0.05);
            margin-bottom: 0.8rem;
        }
        .cfg-title {
            font-size: 0.72rem; text-transform: uppercase; letter-spacing: .09em;
            color: #64748b; font-weight: 800; margin-bottom: 0.55rem;
        }

        /* ── Explanation panel ── */
        .expl-panel {
            background: #fff; border: 1px solid #93c5fd; border-radius: 12px;
            padding: 0.9rem 1rem; font-size: 0.97rem; line-height: 1.65;
            color: #0f172a; box-shadow: 0 6px 16px rgba(15, 23, 42, 0.06);
            margin-bottom: 0.7rem;
        }
        .expl-panel .placeholder { color: #94a3b8; font-style: italic; }
        .expl-label {
            font-size: 0.72rem; text-transform: uppercase; letter-spacing: .09em;
            color: #64748b; font-weight: 800; margin-bottom: 0.25rem;
        }

        /* ── Provider pill badges ── */
        .provider-badge {
            display: inline-block; font-size: 0.68rem; font-weight: 700;
            letter-spacing: .04em; padding: 0.18rem 0.52rem; border-radius: 999px;
            margin-left: 0.4rem; vertical-align: middle;
        }
        .provider-badge.openai  { background: #dcfce7; color: #166534; border: 1px solid #86efac; }
        .provider-badge.deepseek { background: #e0e7ff; color: #3730a3; border: 1px solid #a5b4fc; }

        /* ── Meta cards ── */
        .meta-card {
            background: #fff; border: 1px solid #e2e8f0; border-radius: 12px;
            padding: 0.7rem 0.85rem; box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
        }
        .meta-title { color: #64748b; font-size: 0.7rem; font-weight: 700; text-transform: uppercase; letter-spacing: .06em; }
        .meta-value { color: #0f172a; font-size: 1.05rem; font-weight: 800; margin-top: 0.1rem; }

        /* ── Buttons ── */
        .stButton > button { border-radius: 10px; font-weight: 700; border: 1px solid #cbd5e1; }

        /* ── Tabs ── */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0;
            border-bottom: 2px solid #e2e8f0;
        }
        .stTabs [data-baseweb="tab"] {
            flex: 1;
            justify-content: center;
            padding: 0.75rem 1rem;
            font-size: 1rem;
            font-weight: 700;
            color: #64748b;
            border-bottom: 3px solid transparent;
            background: transparent;
        }
        .stTabs [aria-selected="true"] {
            color: #1e3a8a;
            border-bottom: 3px solid #2563eb;
            background: transparent;
        }
        .stTabs [data-baseweb="tab"]:hover {
            color: #1e3a8a;
            background: #f0f4f8;
        }

        /* ── Section labels ── */
        .section-label {
            font-size: 0.72rem; text-transform: uppercase; letter-spacing: .08em;
            color: #64748b; font-weight: 700; margin-bottom: .3rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def show_frames(frames_dir: Path) -> None:
    frame_paths = sorted(frames_dir.glob("step_*.png"))
    if not frame_paths:
        st.info("No frames found in this run.")
        return

    st.markdown("**Frames**")
    frame_key = f"frame_index::{frames_dir}"
    if frame_key not in st.session_state:
        st.session_state[frame_key] = 0

    current_idx = max(0, min(int(st.session_state[frame_key]), len(frame_paths) - 1))
    st.session_state[frame_key] = current_idx

    nav1, nav2, nav3 = st.columns([1, 3, 1])
    with nav1:
        if st.button("◀", key=f"prev::{frames_dir}", use_container_width=True, disabled=current_idx == 0):
            st.session_state[frame_key] = max(0, current_idx - 1)
            st.rerun()
    with nav2:
        selected = st.slider(
            "Frame",
            min_value=1,
            max_value=len(frame_paths),
            value=current_idx + 1,
            label_visibility="collapsed",
        )
        if selected - 1 != current_idx:
            st.session_state[frame_key] = selected - 1
            current_idx = selected - 1
    with nav3:
        if st.button(
            "▶",
            key=f"next::{frames_dir}",
            use_container_width=True,
            disabled=current_idx >= len(frame_paths) - 1,
        ):
            st.session_state[frame_key] = min(len(frame_paths) - 1, current_idx + 1)
            st.rerun()

    current_frame = frame_paths[st.session_state[frame_key]]
    st.image(str(current_frame), caption=f"Step {current_idx + 1} / {len(frame_paths)}", use_container_width=True)


def show_videos(storyboard_video: Optional[Path], single_video: Optional[Path]) -> None:
    st.markdown("**Videos**")
    tab_story, tab_single = st.tabs(["Storyboard", "Single Video"])

    with tab_story:
        if storyboard_video is not None:
            st.video(str(storyboard_video))
        else:
            st.info("Storyboard video not found.")

    with tab_single:
        if single_video is not None:
            st.video(str(single_video))
        else:
            st.info("Single video output not found.")


def _render_run_summary(run: RunEntry) -> None:
    frame_count = len(sorted(run.frames_dir.glob("step_*.png"))) if run.frames_dir.exists() else 0

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            '<div class="meta-card"><div class="meta-title">Grade</div><div class="meta-value">'
            + str(run.grade if run.grade is not None else "N/A")
            + "</div></div>",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            '<div class="meta-card"><div class="meta-title">Subject</div><div class="meta-value">'
            + (run.subject or "N/A")
            + "</div></div>",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            '<div class="meta-card"><div class="meta-title">Frames</div><div class="meta-value">'
            + str(frame_count)
            + "</div></div>",
            unsafe_allow_html=True,
        )

    st.markdown("### Generated Question")
    st.write(run.question_text)


def _run_generation(*, run_openai: bool, make_single_video: bool) -> None:
    question = str(st.session_state.get("question_input", "")).strip()
    subject = str(st.session_state.get("subject_input", "")).strip()
    grade = int(st.session_state.get("grade_input", 11))
    output_root = _resolve_output_root(st.session_state.get("output_root_input", str(DEFAULT_OUTPUT_ROOT)))
    explanation = str(st.session_state.get("generated_explanation", "")).strip()
    current_signature = _input_signature(question, subject, grade)
    saved_signature = str(st.session_state.get("explanation_signature", ""))
    text_provider = str(st.session_state.get("text_provider", "openai"))
    image_provider = str(st.session_state.get("image_provider", "openai"))

    if not question:
        st.error("Question is required.")
        return
    if not explanation:
        st.error("Generate explanation first, then generate images and videos.")
        return
    if current_signature != saved_signature:
        st.error("Inputs changed after explanation generation. Please regenerate explanation first.")
        return
    if run_openai and not available_text_providers():
        st.error("No API key configured. Add keys to api_keys.txt or set environment variables.")
        return

    try:
        with st.status("Running generation", expanded=True) as status:
            status.write("Running Checker 1 (DistilBERT error-type classifier) on explanation...")
            pipeline_result = run_pipeline(
                question=question,
                explanation=explanation,
                grade=grade,
                subject=subject,
                output_root=output_root,
                run_openai=run_openai,
                run_checker=True,
                text_provider=text_provider,
                image_provider=image_provider,
            )

            # Show checker results
            checker_result = pipeline_result.get("checker_result")
            if checker_result and isinstance(checker_result, dict):
                rounds = checker_result.get("rounds", [])
                if checker_result.get("was_revised"):
                    status.write(f"Checker revised the explanation after {checker_result['total_rounds']} round(s).")
                    st.session_state["generated_explanation"] = checker_result["final_explanation"]
                elif rounds:
                    last = rounds[-1].get("checker_result", {})
                    status.write(f"Checker accepted explanation (confidence {last.get('confidence', 0):.2f}).")
                st.session_state["checker_result"] = checker_result

            checker2_result = pipeline_result.get("checker2_result")
            if checker2_result and isinstance(checker2_result, dict):
                st.session_state["checker2_result"] = checker2_result
                if checker2_result.get("error"):
                    status.write(f"Checker 2 failed: {checker2_result['error']}")
                else:
                    status.write(
                        "Checker 2 frame quality "
                        f"score {checker2_result.get('overall_score', 0):.3f} "
                        f"(threshold {checker2_result.get('threshold', 0):.2f})."
                    )

            status.write("Generating storyboard plan, frames, GIF, and storyboard video...")
            run_dir = Path(pipeline_result["out_dir"])
            st.session_state["active_run_dir"] = str(run_dir)
            # Save sources to run dir if available
            saved_sources = str(st.session_state.get("relevant_sources", "")).strip()
            if saved_sources:
                run_dir.joinpath("sources.md").write_text(saved_sources, encoding="utf-8")

            if make_single_video:
                status.write("Generating single API video from the storyboard run...")
                generate_single_video_from_run_dir(run_dir=run_dir)

            status.update(label="Generation complete", state="complete", expanded=False)
    except Exception as exc:
        st.error(f"Media generation failed: {exc}")


def _run_full_lesson() -> None:
    """Single-click: generate explanation → checker → plan → images → video."""
    question = str(st.session_state.get("question_input", "")).strip()
    subject = str(st.session_state.get("subject_input", "")).strip()
    grade = int(st.session_state.get("grade_input", 11))
    text_provider = str(st.session_state.get("text_provider", "openai"))
    image_provider = str(st.session_state.get("image_provider", "openai"))
    output_root = _resolve_output_root(st.session_state.get("output_root_input", str(DEFAULT_OUTPUT_ROOT)))

    if not question:
        st.error("Question is required.")
        return
    if not available_text_providers():
        st.error("No API key configured. Add keys to api_keys.txt.")
        return

    try:
        with st.status("Generating full lesson...", expanded=True) as status:
            language = str(st.session_state.get("language", "English"))
            # Step 1: Generate explanation
            status.write("Step 1/5 — Generating explanation...")
            explanation = _generate_explanation(
                question=question, subject=subject, grade=grade, provider=text_provider, language=language,
            )
            _set_explanation_state(explanation, question, subject, grade)
            status.write(f"Explanation ready ({len(explanation.split())} words)")

            # Generate sources + quiz
            status.write("Step 2/5 — Finding sources & generating quiz...")
            sources = _generate_sources(question=question, subject=subject, grade=grade, provider=text_provider)
            st.session_state["relevant_sources"] = sources
            quiz = _generate_quiz(question=question, explanation=explanation, subject=subject, grade=grade, provider=text_provider, language=language)
            st.session_state["generated_quiz"] = quiz

            # Steps 3-5: Run full pipeline (checker + plan + images + video)
            # (sources.md saved to run_dir after pipeline creates it)
            status.write("Step 3/5 — Running Checker 1 (DistilBERT error-type classifier)...")
            pipeline_result = run_pipeline(
                question=question,
                explanation=explanation,
                grade=grade,
                subject=subject,
                output_root=output_root,
                run_openai=True,
                run_checker=True,
                text_provider=text_provider,
                image_provider=image_provider,
            )

            # Checker results
            checker_result = pipeline_result.get("checker_result")
            if checker_result and isinstance(checker_result, dict):
                rounds = checker_result.get("rounds", [])
                if checker_result.get("was_revised"):
                    status.write(f"Checker revised explanation after {checker_result['total_rounds']} round(s).")
                    st.session_state["generated_explanation"] = checker_result["final_explanation"]
                elif rounds:
                    last = rounds[-1].get("checker_result", {})
                    status.write(f"Checker accepted (confidence {last.get('confidence', 0):.2f}).")
                st.session_state["checker_result"] = checker_result

            checker2_result = pipeline_result.get("checker2_result")
            if checker2_result and isinstance(checker2_result, dict):
                st.session_state["checker2_result"] = checker2_result
                if checker2_result.get("error"):
                    status.write(f"Checker 2 failed: {checker2_result['error']}")
                else:
                    status.write(
                        "Checker 2 frame quality "
                        f"score {checker2_result.get('overall_score', 0):.3f} "
                        f"(threshold {checker2_result.get('threshold', 0):.2f})."
                    )

            status.write("Step 4/5 — Plan + frames + GIF generated.")
            run_dir = Path(pipeline_result["out_dir"])
            st.session_state["active_run_dir"] = str(run_dir)
            # Save sources + quiz to run dir
            if sources:
                run_dir.joinpath("sources.md").write_text(sources, encoding="utf-8")
            if quiz:
                run_dir.joinpath("quiz.md").write_text(quiz, encoding="utf-8")

            status.write("Step 5/5 — Storyboard video assembled.")
            timing = pipeline_result.get("stage_times", {})
            total = pipeline_result.get("total_seconds", 0)
            status.write(f"Done in {total:.1f}s (plan {timing.get('plan_seconds', 0):.1f}s, images {timing.get('images_seconds', 0):.1f}s)")
            status.update(label="Lesson generation complete", state="complete", expanded=False)
    except Exception as exc:
        st.error(f"Generation failed: {exc}")


def _show_saved_demo(selected_demo: Optional[RunEntry]) -> None:
    if selected_demo is None:
        st.error("No demo selected.")
        return

    st.session_state["active_run_dir"] = str(selected_demo.run_dir)
    # Load saved sources and quiz if available
    sources_path = selected_demo.run_dir / "sources.md"
    if sources_path.exists():
        st.session_state["relevant_sources"] = sources_path.read_text(encoding="utf-8").strip()
    else:
        st.session_state["relevant_sources"] = ""
    quiz_path = selected_demo.run_dir / "quiz.md"
    if quiz_path.exists():
        st.session_state["generated_quiz"] = quiz_path.read_text(encoding="utf-8").strip()
    else:
        st.session_state["generated_quiz"] = ""
    st.success("Loaded saved demo results.")


def main() -> None:
    st.set_page_config(page_title="VisualLesson AI", layout="wide", initial_sidebar_state="expanded")
    _ensure_state_defaults()
    _inject_styles()

    # ── Resolve saved demos early (needed for both sidebar and main) ──
    runs = discover_saved_demos(_resolve_output_root(st.session_state.get("output_root_input", str(DEFAULT_OUTPUT_ROOT))))
    demo_lookup: dict[str, RunEntry] = {}
    selected_demo: Optional[RunEntry] = None
    selected_demo_label: Optional[str] = None

    # ── SIDEBAR ──
    with st.sidebar:
        st.markdown(
            '<div style="font-size:1.25rem;font-weight:800;color:#1e3a8a;margin-bottom:0.1rem;">VisualLesson AI</div>'
            '<div style="font-size:0.75rem;color:#64748b;margin-bottom:1rem;">Turn any question into a visual lesson</div>',
            unsafe_allow_html=True,
        )

        st.radio("Mode", options=["API mode", "Demo mode (no API)"], key="workflow_mode", horizontal=False, label_visibility="collapsed")
        demo_mode = st.session_state.get("workflow_mode") == "Demo mode (no API)"

        st.divider()

        if runs:
            demo_lookup = {_demo_label(run): run for run in runs}
            demo_labels = list(demo_lookup.keys())
            if st.session_state.get("saved_demo_choice") not in demo_labels:
                st.session_state["saved_demo_choice"] = demo_labels[0]
            selected_demo_label = str(st.session_state.get("saved_demo_choice"))
            selected_demo = demo_lookup.get(selected_demo_label)
        elif demo_mode:
            st.error("No saved demos found.")
            return

        if demo_mode:
            if len(demo_lookup) > 1:
                st.selectbox("Saved demos", options=list(demo_lookup.keys()), key="saved_demo_choice")
                selected_demo_label = str(st.session_state.get("saved_demo_choice"))
                selected_demo = demo_lookup.get(selected_demo_label)

            if selected_demo:
                _sync_demo_selection(selected_demo_label, selected_demo)
                st.caption("**Question (read-only)**")
                st.info(selected_demo.question_text)

                if st.button("▶ Load Demo", type="primary", use_container_width=True):
                    _run_demo_explanation_from_saved(selected_demo)
                    _show_saved_demo(selected_demo)
                    st.rerun()
                if st.button("Clear", use_container_width=True):
                    st.session_state["active_run_dir"] = None
                    st.session_state["generated_explanation"] = ""
                    st.session_state["explanation_signature"] = ""
                    st.session_state["checker_result"] = None
                    st.session_state["checker2_result"] = None
                    st.session_state["relevant_sources"] = ""
                    st.session_state["generated_quiz"] = ""
                    st.session_state["quiz_submitted"] = False
                    st.session_state["analyzer_result"] = None
                    st.session_state["quiz_attempt_history"] = []
                    st.rerun()
        else:
            # API mode controls
            st.caption("**Question**")
            st.text_area("Question", key="question_input", height=150,
                          placeholder="Paste or type the question here...", label_visibility="collapsed")

            st.text_input("Subject", key="subject_input", placeholder="e.g. Ecology")
            st.slider("Grade", min_value=7, max_value=12, key="grade_input", step=1)
            st.selectbox("Language", options=["English", "中文", "Español", "Français", "Deutsch", "日本語", "한국어"], key="language")

            text_providers = available_text_providers()
            tp_map = {"openai": "OpenAI (GPT)", "deepseek": "DeepSeek"}
            if text_providers:
                tp_labels = [tp_map.get(p, p) for p in text_providers]
                tp_idx = text_providers.index(st.session_state.get("text_provider", "openai")) if st.session_state.get("text_provider", "openai") in text_providers else 0
                sel_tp = st.selectbox("Text Provider", options=tp_labels, index=tp_idx)
                st.session_state["text_provider"] = text_providers[tp_labels.index(sel_tp)]

            image_providers = available_image_providers()
            ip_map = {"openai": "OpenAI (gpt-image-1)", "wanx": "Wanx (万象)"}
            if image_providers:
                ip_labels = [ip_map.get(p, p) for p in image_providers]
                ip_idx = image_providers.index(st.session_state.get("image_provider", "openai")) if st.session_state.get("image_provider", "openai") in image_providers else 0
                sel_ip = st.selectbox("Image Provider", options=ip_labels, index=ip_idx)
                st.session_state["image_provider"] = image_providers[ip_labels.index(sel_ip)]

            # Cost estimate
            text_prov = str(st.session_state.get("text_provider", "openai"))
            img_prov = str(st.session_state.get("image_provider", "openai"))
            has_expl = bool(st.session_state.get("generated_explanation", ""))
            cost = _estimate_cost(text_prov, img_prov, has_expl)
            cost_parts = []
            if cost["explanation"] > 0:
                cost_parts.append(f"Expl ~${cost['explanation']:.3f}")
            cost_parts.append(f"Plan ~${cost['planning']:.3f}")
            cost_parts.append(f"Img ~${cost['images']:.2f}")
            st.caption(f"💰 Est. **${cost['total']:.2f}** ({' + '.join(cost_parts)})")

            st.divider()

            has_providers = bool(available_text_providers())
            explanation_ready = bool(st.session_state.get("generated_explanation", ""))

            if st.button("Generate Explanation", use_container_width=True, disabled=not has_providers):
                _run_explanation_generation()

            can_gen_media = explanation_ready and has_providers
            if st.button("Generate Images & Video", type="primary", use_container_width=True, disabled=not can_gen_media):
                _run_generation(run_openai=True, make_single_video=True)

            if st.button("Clear", use_container_width=True):
                st.session_state["active_run_dir"] = None
                st.session_state["generated_explanation"] = ""
                st.session_state["explanation_signature"] = ""
                st.session_state["checker_result"] = None
                st.session_state["checker2_result"] = None
                st.session_state["relevant_sources"] = ""
                st.session_state["generated_quiz"] = ""
                st.session_state["quiz_submitted"] = False
                st.session_state["analyzer_result"] = None
                st.session_state["quiz_attempt_history"] = []
                st.rerun()

        st.divider()
        st.markdown(
            '<div style="color:#94a3b8;font-size:0.72rem;text-align:center;">'
            'Built by <strong>Jiaxing BCOS</strong></div>',
            unsafe_allow_html=True,
        )

    # ── MAIN AREA ──
    st.markdown(
        '<div class="hero"><h1>VisualLesson AI</h1>'
        '<p>Turn any question into a classroom-ready visual lesson.</p></div>',
        unsafe_allow_html=True,
    )

    tab_lesson, tab_quiz, tab_resources, tab_details = st.tabs(
        ["📖 Lesson", "📝 Quiz", "📚 Resources", "ℹ️ Details"]
    )

    # ── TAB: Lesson ──
    with tab_lesson:
        explanation_text = str(st.session_state.get("generated_explanation", "")).strip()
        active_run_dir = st.session_state.get("active_run_dir")

        # ── Step 1: Explanation ──
        st.markdown("##### Step 1 — Explanation")
        if explanation_text:
            provider_name = st.session_state.get("text_provider", "openai")
            badge_cls = "deepseek" if provider_name == "deepseek" else "openai"
            badge_label = "DeepSeek" if provider_name == "deepseek" else "OpenAI"
            st.markdown(
                f'<div class="expl-label">Generated Explanation'
                f'<span class="provider-badge {badge_cls}">{badge_label}</span></div>'
                f'<div class="expl-panel">{html.escape(explanation_text)}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="expl-panel"><span class="placeholder">'
                'Use the sidebar to click <strong>Generate Explanation</strong> first.</span></div>',
                unsafe_allow_html=True,
            )

        # ── Step 2: Images & Video (only shown after generation) ──
        if explanation_text:
            st.divider()
            st.markdown("##### Step 2 — Images & Video")
            if active_run_dir:
                run = _load_run_entry(Path(active_run_dir))
                if run is None:
                    st.error("The generated run could not be loaded from disk.")
                else:
                    _render_run_summary(run)
                    left, right = st.columns([1, 1])
                    with left:
                        show_frames(run.frames_dir)
                    with right:
                        show_videos(run.storyboard_video, run.single_video)
            else:
                st.markdown(
                    '<div class="expl-panel"><span class="placeholder">'
                    'Click <strong>Generate Images &amp; Video</strong> in the sidebar to create the visual lesson.</span></div>',
                    unsafe_allow_html=True,
                )

    # ── TAB: Quiz ──
    with tab_quiz:
        quiz_text = str(st.session_state.get("generated_quiz", "")).strip()
        if quiz_text:
            _render_interactive_quiz(
                quiz_text,
                subject=str(st.session_state.get("subject_input", "")).strip(),
                explanation_text=str(st.session_state.get("generated_explanation", "")).strip(),
                checker2_result=st.session_state.get("checker2_result"),
            )
        else:
            st.info("No quiz generated yet. Generate an explanation first to create quiz questions.")

    # ── TAB: Resources ──
    with tab_resources:
        sources_text = str(st.session_state.get("relevant_sources", "")).strip()
        if sources_text:
            st.markdown("**Relevant Sources (Websites, YouTube, Textbooks)**")
            st.markdown(sources_text)
        else:
            st.info("No sources generated yet. Generate an explanation to see relevant learning resources.")

        active_run_dir = st.session_state.get("active_run_dir")
        if active_run_dir:
            run = _load_run_entry(Path(active_run_dir))
            if run is not None:
                st.divider()
                st.markdown("**Downloads**")
                dl_cols = st.columns(4)
                with dl_cols[0]:
                    if run.frames_dir.exists() and list(run.frames_dir.glob("step_*.png")):
                        st.download_button(
                            "📦 Frames (ZIP)",
                            data=_make_frames_zip(run.frames_dir),
                            file_name="lesson_frames.zip",
                            mime="application/zip",
                            use_container_width=True,
                        )
                with dl_cols[1]:
                    vid_path = run.storyboard_video or run.single_video
                    if vid_path and vid_path.exists():
                        st.download_button(
                            "🎬 Video (MP4)",
                            data=vid_path.read_bytes(),
                            file_name="lesson_video.mp4",
                            mime="video/mp4",
                            use_container_width=True,
                        )
                with dl_cols[2]:
                    expl = str(st.session_state.get("generated_explanation", "")).strip()
                    if expl:
                        st.download_button(
                            "📄 Explanation (TXT)",
                            data=expl,
                            file_name="explanation.txt",
                            mime="text/plain",
                            use_container_width=True,
                        )
                with dl_cols[3]:
                    quiz_dl = str(st.session_state.get("generated_quiz", "")).strip()
                    if quiz_dl:
                        st.download_button(
                            "📝 Quiz (MD)",
                            data=quiz_dl,
                            file_name="quiz.md",
                            mime="text/markdown",
                            use_container_width=True,
                        )

    # ── TAB: Details ──
    with tab_details:
        checker_result = st.session_state.get("checker_result")
        if checker_result and isinstance(checker_result, dict) and checker_result.get("rounds"):
            st.markdown("**Checker 1 Results (DistilBERT Error-Type Classifier)**")
            if checker_result.get("was_revised"):
                st.success(f"Explanation was revised after {checker_result['total_rounds']} checker round(s).")
            else:
                st.info("Explanation accepted by checker.")
            for rnd in checker_result["rounds"]:
                cr = rnd.get("checker_result", {})
                st.markdown(f"**Round {rnd['round']}**: {cr.get('label', '?')} (confidence {cr.get('confidence', 0):.3f}) — action: {rnd['action']}")
                if cr.get("probabilities"):
                    st.json(cr["probabilities"])
        else:
            st.info("No checker results available. Run the pipeline to see error-type classification.")

        st.divider()
        checker2_result = st.session_state.get("checker2_result")
        if checker2_result and isinstance(checker2_result, dict):
            st.markdown("**Checker 2 Results (Frame Quality Validator)**")
            if checker2_result.get("error"):
                st.error(f"Checker 2 error: {checker2_result['error']}")
            else:
                passed = bool(checker2_result.get("pass", False))
                score = float(checker2_result.get("overall_score", 0.0))
                threshold = float(checker2_result.get("threshold", 0.0))
                mode = str(checker2_result.get("mode", "heuristic"))
                if passed:
                    st.success(f"Checker 2 passed (mode={mode}, score={score:.3f}, threshold={threshold:.2f}).")
                else:
                    failed_steps = checker2_result.get("failed_steps", [])
                    st.warning(
                        f"Checker 2 flagged frame quality (mode={mode}, score={score:.3f}, "
                        f"threshold={threshold:.2f}, failed steps={failed_steps})."
                    )

                per_frame = checker2_result.get("per_frame", [])
                if isinstance(per_frame, list) and per_frame:
                    for item in per_frame:
                        sid = item.get("step_id", "?")
                        item_score = float(item.get("score", 0.0))
                        item_pass = bool(item.get("pass", False))
                        issues = item.get("issues", [])
                        st.markdown(
                            f"**Step {sid}**: {'PASS' if item_pass else 'FAIL'} "
                            f"(score {item_score:.3f}) | issues: {issues or 'none'}"
                        )
        else:
            st.info("No Checker 2 results available. Run image generation to validate frame quality.")

        st.divider()
        analyzer_result = st.session_state.get("analyzer_result")
        if analyzer_result and isinstance(analyzer_result, dict) and analyzer_result.get("status") == "ok":
            st.markdown("**Student Weakness Analyzer**")
            st.json(analyzer_result)
        else:
            st.info("No analyzer results available. Complete and submit a quiz to see diagnostics.")

        active_run_dir = st.session_state.get("active_run_dir")



if __name__ == "__main__":
    main()