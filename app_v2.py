"""Production Streamlit interface for VisualLesson AI."""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from typing import Any

import streamlit as st

from pipeline.api_keys import get_key
from pipeline.clients import build_text_client
from pipeline.config import OPENAI_TEXT_MODEL
from pipeline.lesson_service import curated_resources, generate_lesson_bundle
from pipeline.pipeline import run_pipeline
from pipeline.quality_gates import review_explanation


DEMO_BUNDLE = {
    "title": "Why removing one species can change a food web",
    "learning_objective": "Explain how a population change can spread through an ecosystem.",
    "explanation": (
        "A food web connects organisms through feeding relationships, so changing one population can affect several others. "
        "Suppose a predator is removed. Its prey may survive in larger numbers and consume more plants or smaller animals. "
        "Those resources can then decline, leaving less food or habitat for other species. The change can continue through several "
        "links; ecologists call this a trophic cascade. The result is not always a simple increase or decrease because organisms "
        "often have more than one food source, and competition, disease, migration, and weather also influence populations. "
        "The strongest effects usually occur when the removed species has many connections or performs a role that few other "
        "species can replace. A food web therefore helps us predict possible directions of change, but field evidence is still "
        "needed to measure how large each effect will be."
    ),
    "key_ideas": [
        "Species are connected by feeding relationships.",
        "A population change can spread through several links.",
        "Highly connected species can have especially large effects.",
        "Food webs predict possibilities, not exact population sizes.",
    ],
    "worked_example": "If wolves decrease, deer may increase, heavier browsing may reduce young trees, and species that depend on those trees may lose habitat.",
    "common_mistake": "A food chain is not the whole food web: most organisms have several feeding relationships.",
    "quick_check": "Why might removing a predator eventually affect plants?",
    "quiz": [
        {"question": "What is a trophic cascade?", "choices": ["A change that spreads through feeding relationships", "A seasonal migration", "A type of cell division", "A weather cycle"], "answer": "A", "explanation": "It is a chain of ecological effects across trophic levels.", "concept": "trophic cascade"},
        {"question": "What may happen first when a predator is removed?", "choices": ["Its prey increases", "All plants disappear", "Weather changes", "Decomposers vanish"], "answer": "A", "explanation": "Reduced predation can allow the prey population to grow.", "concept": "predator-prey"},
        {"question": "Why are exact outcomes difficult to predict?", "choices": ["Food webs contain many interacting factors", "Energy is not conserved", "Species never compete", "All organisms eat one food"], "answer": "A", "explanation": "Multiple food sources and environmental factors alter the response.", "concept": "system complexity"},
        {"question": "Which species can cause a large cascade?", "choices": ["A highly connected species", "Only the smallest species", "Only plants", "Any species equally"], "answer": "A", "explanation": "Many connections or a hard-to-replace role can amplify its effect.", "concept": "ecological role"},
        {"question": "What should support a food-web prediction?", "choices": ["Field evidence", "A single guess", "Color alone", "No measurements"], "answer": "A", "explanation": "Observations and measurements test the predicted effect size.", "concept": "evidence"},
    ],
}


def _init_state() -> None:
    defaults = {
        "bundle": None,
        "gate1": None,
        "pipeline_result": None,
        "quiz_submitted": False,
        "started_at": {},
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _styles() -> None:
    st.markdown(
        """
        <style>
        .stApp { background: #f6f7fb; color: #172033; }
        .block-container { max-width: 1160px; padding-top: 2rem; }
        [data-testid="stSidebar"] { background: #101828; }
        [data-testid="stSidebar"] * { color: #f8fafc; }
        .hero { padding: 2.2rem; border-radius: 24px; color: white;
          background: linear-gradient(125deg,#13213c 0%,#214f8b 58%,#19a6a2 100%);
          box-shadow: 0 18px 50px rgba(20,42,78,.18); margin-bottom: 1.2rem; }
        .hero h1 { font-size: 2.35rem; margin: 0; }
        .hero p { opacity: .9; margin: .5rem 0 0; font-size: 1.05rem; }
        .card { background: white; border: 1px solid #e5e9f2; border-radius: 18px;
          padding: 1.25rem 1.35rem; margin: .55rem 0; box-shadow: 0 5px 20px rgba(23,32,51,.05); }
        .eyebrow { color:#177c78; font-weight:700; text-transform:uppercase; letter-spacing:.08em; font-size:.72rem; }
        .idea { background:#edf8f7; border-left:4px solid #19a6a2; border-radius:10px; padding:.7rem .9rem; margin:.45rem 0; }
        .gate-pass { color:#087d55; font-weight:700; }
        .gate-fail { color:#bd3f32; font-weight:700; }
        div.stButton > button { border-radius: 12px; min-height: 44px; font-weight: 700; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _configured_key() -> str:
    return get_key("OPENAI_API_KEY")


def _create_lesson(question: str, subject: str, grade: int, language: str, api_key: str, create_video: bool) -> None:
    if not question.strip():
        st.error("Enter a question first.")
        return
    effective_key = api_key.strip() or _configured_key()
    if not effective_key:
        st.error("Add an OpenAI API key for this session, or open the demo.")
        return
    client = build_text_client("openai", api_key=effective_key)
    if client is None:
        st.error("The OpenAI client could not be initialized.")
        return

    try:
        with st.status("Building your lesson", expanded=True) as status:
            status.write("Drafting the explanation and quiz...")
            bundle = generate_lesson_bundle(
                client,
                model=OPENAI_TEXT_MODEL,
                question=question,
                subject=subject,
                grade=grade,
                language=language,
            )
            status.write("Gate 1: checking accuracy, logic, and grade fit...")
            gate1 = review_explanation(
                client,
                model=OPENAI_TEXT_MODEL,
                question=question,
                explanation=bundle["explanation"],
                grade=grade,
                subject=subject,
                max_repairs=1,
            )
            bundle["explanation"] = gate1["final_explanation"]
            st.session_state.bundle = bundle
            st.session_state.gate1 = gate1
            st.session_state.pipeline_result = None
            st.session_state.quiz_submitted = False

            if not gate1["pass"]:
                status.update(label="Lesson needs review", state="error")
                st.error("Gate 1 could not verify this explanation. Media generation was stopped before image costs were incurred.")
                return

            if create_video:
                status.write("Planning and illustrating seven teaching steps...")
                run = run_pipeline(
                    question=question,
                    explanation=bundle["explanation"],
                    grade=grade,
                    subject=subject,
                    output_root=Path(tempfile.gettempdir()) / "visual_lesson_ai",
                    run_openai=True,
                    run_checker=False,
                    run_checker2=True,
                    text_provider="openai",
                    image_provider="openai",
                    api_key=effective_key,
                )
                st.session_state.pipeline_result = run
                run_dir = Path(run["out_dir"])
                (run_dir / "lesson.json").write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
            status.update(label="Lesson ready", state="complete", expanded=False)
    except Exception as exc:
        st.error(f"Lesson generation stopped safely: {exc}")


def _show_gate(name: str, result: dict[str, Any] | None) -> None:
    if not result:
        st.caption(f"{name}: not run")
        return
    passed = bool(result.get("pass"))
    score = result.get("overall_score")
    label = "Passed" if passed else "Needs review"
    suffix = f" ? {float(score) * 100:.0f}%" if isinstance(score, (int, float)) else ""
    st.markdown(f"<span class=\"{'gate-pass' if passed else 'gate-fail'}\">{name}: {label}{suffix}</span>", unsafe_allow_html=True)
    issues = result.get("issues", [])
    if issues:
        with st.expander("Review notes"):
            for issue in issues:
                st.write(f"- {issue}")


def _lesson_tab(bundle: dict[str, Any]) -> None:
    st.markdown(f"<div class='card'><div class='eyebrow'>Learning objective</div><h2>{bundle['title']}</h2><p>{bundle['learning_objective']}</p></div>", unsafe_allow_html=True)
    st.markdown("### Explanation")
    st.write(bundle["explanation"])
    left, right = st.columns([1.25, 1])
    with left:
        st.markdown("#### Key ideas")
        for idea in bundle.get("key_ideas", []):
            st.markdown(f"<div class='idea'>{idea}</div>", unsafe_allow_html=True)
        st.markdown("#### Worked example")
        st.info(bundle.get("worked_example", ""))
    with right:
        st.markdown("#### Watch out")
        st.warning(bundle.get("common_mistake", ""))
        st.markdown("#### Quick check")
        st.success(bundle.get("quick_check", ""))

    result = st.session_state.pipeline_result
    if result:
        st.divider()
        st.markdown("### Visual story")
        frames = result.get("frames", [])
        if frames:
            selected = st.select_slider("Teaching step", options=list(range(1, len(frames) + 1)))
            st.image(str(frames[selected - 1]), use_container_width=True)
        video = result.get("video_path")
        if video and Path(video).exists():
            st.markdown("### Narrated lesson video")
            st.video(str(video))


def _quiz_tab(bundle: dict[str, Any]) -> None:
    questions = bundle.get("quiz", [])
    for index, item in enumerate(questions):
        st.markdown(f"#### {index + 1}. {item['question']}")
        labels = ["A", "B", "C", "D"]
        options = [f"{label}. {choice}" for label, choice in zip(labels, item["choices"])]
        if index not in st.session_state.started_at:
            st.session_state.started_at[index] = time.time()
        selected = st.radio("Choose one", options, index=None, key=f"answer_{index}", label_visibility="collapsed")
        if st.session_state.quiz_submitted:
            if selected and selected[0] == item["answer"]:
                st.success(f"Correct. {item['explanation']}")
            else:
                st.error(f"Answer: {item['answer']}. {item['explanation']}")
    if st.button("Check answers", type="primary", use_container_width=True):
        st.session_state.quiz_submitted = True
        st.rerun()
    if st.session_state.quiz_submitted:
        score = sum(
            1 for index, item in enumerate(questions)
            if str(st.session_state.get(f"answer_{index}", ""))[:1] == item["answer"]
        )
        st.metric("Score", f"{score} / {len(questions)}")
        if score < len(questions):
            missed = [item["concept"] for index, item in enumerate(questions) if str(st.session_state.get(f"answer_{index}", ""))[:1] != item["answer"]]
            st.info("Review next: " + ", ".join(dict.fromkeys(missed)))


def _resources_tab(subject: str, question: str, bundle: dict[str, Any]) -> None:
    st.markdown("### Trusted starting points")
    st.caption("These links come from a curated list; the model does not iߞu��$z{-���jם      st.markdown(
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


from app_v2 import main


if __name__ == "__main__":
    main()