"""Production Streamlit interface for VisualLesson AI."""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from typing import Any

import streamlit as st

from pipeline.clients import build_text_client
from pipeline.gate_benchmarks import benchmark_rows
from pipeline.lesson_service import (
    build_concept_map,
    curated_resources,
    generate_lesson_bundle,
    research_lesson_sources,
)
from pipeline.markdown_render import streamlit_markdown
from pipeline.pipeline import run_pipeline
from pipeline.quality_gates import review_explanation


TEXT_PROVIDERS = {
    "OpenAI": {
        "id": "openai",
        "models": {
            "GPT-5.6 Terra - balanced": "gpt-5.6-terra",
            "GPT-5.6 Luna - lower cost": "gpt-5.6-luna",
            "GPT-5.6 Sol - highest quality": "gpt-5.6-sol",
        },
        "key_help": "Create a key at platform.openai.com/api-keys.",
    },
    "DeepSeek": {
        "id": "deepseek",
        "models": {
            "DeepSeek V4 Flash - fast/low cost": "deepseek-v4-flash",
            "DeepSeek V4 Pro - higher quality": "deepseek-v4-pro",
        },
        "key_help": "Create a key at platform.deepseek.com/api_keys.",
    },
}

IMAGE_PROVIDERS = {
    "OpenAI": {
        "id": "openai",
        "models": {"GPT Image 2 - recommended": "gpt-image-2"},
        "key_help": "Uses an OpenAI API key with image-model access.",
    },
    "Alibaba Wan": {
        "id": "wanx",
        "models": {
            "Wan 2.7 Image - faster": "wan2.7-image",
            "Wan 2.7 Image Pro - highest quality": "wan2.7-image-pro",
        },
        "key_help": "Uses an international Alibaba Model Studio key.",
    },
}


_WAITING_TIPS = {
    "draft": [
        "Predict the answer before it appears; comparing your prediction with the lesson makes the idea stick.",
        "Write the question in your own words. If you can simplify the question, you already understand its target.",
        "Recall one fact you already know about this topic without opening notes.",
    ],
    "gate": [
        "Check units as well as numbers. A correct-looking calculation with the wrong unit is still wrong.",
        "Ask: what assumption makes this explanation true? Conditions matter in science and mathematics.",
        "Try to find one possible counterexample. Good explanations survive careful questions.",
    ],
    "research": [
        "A source is useful when it supports a specific claim, not simply because its title sounds relevant.",
        "Compare two sources on the same point. Agreement is evidence; disagreement is a reason to investigate.",
        "Prefer textbooks, universities, government science pages, and original research over anonymous summaries.",
    ],
    "frame": [
        "Turn each picture into one sentence: what changed from the previous frame, and why?",
        "Follow arrows before reading labels; direction often carries the main idea in a diagram.",
        "Pause after each visual step and explain it aloud as if teaching a classmate.",
    ],
    "compose": [
        "Retrieval beats rereading: close your notes and reconstruct the main steps from memory.",
        "Use the finished video once for overview, then replay only the step that caused confusion.",
        "A strong study note contains an idea, an example, and one common mistake—not a copied paragraph.",
    ],
}


def _waiting_tip(stage: str, grade: int, subject: str, step: int = 0) -> str:
    """Return a stable, short study tip suited to the current generation stage."""
    category = "frame" if stage in {"storyboard", "frame", "visual_check", "preview"} else stage
    if category in {"narration", "complete"}:
        category = "compose"
    tips = _WAITING_TIPS.get(category, _WAITING_TIPS["draft"])
    seed = int(grade) + step + sum(ord(char) for char in str(subject))
    return tips[seed % len(tips)]

def _init_state() -> None:
    defaults = {
        "bundle": None,
        "gate1": None,
        "gate2": None,
        "pipeline_result": None,
        "media_requested": False,
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
        [data-testid="stSidebar"] { color: #f8fafc; }
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] h4,
        [data-testid="stSidebar"] [data-testid="stWidgetLabel"] *,
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] * { color: #f8fafc !important; }
        [data-testid="stTextInput"] input,
        [data-testid="stTextArea"] textarea,
        [data-testid="stNumberInput"] input,
        [data-baseweb="select"] input,
        [data-baseweb="select"] > div { color: #172033 !important; }
        [data-testid="stSidebar"] [data-baseweb="select"] * {
          color: #172033 !important;
          -webkit-text-fill-color: #172033 !important;
          opacity: 1 !important;
        }
        [data-testid="stSidebar"] [data-baseweb="select"] svg { fill: #172033 !important; }
        [data-testid="stTextInput"] input::placeholder,
        [data-testid="stTextArea"] textarea::placeholder { color: #667085 !important; opacity: 1; }
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
        .flow-step { background:white; border:1px solid #e5e9f2; border-top:4px solid #19a6a2;
          border-radius:16px; padding:1rem; min-height:142px; box-shadow:0 5px 20px rgba(23,32,51,.05); }
        .flow-step .number { color:#177c78; font-size:.75rem; font-weight:800; letter-spacing:.08em; }
        .flow-step h4 { margin:.35rem 0; color:#172033; }
        .flow-step p { color:#526078; font-size:.9rem; margin:0; }
        div.stButton > button { border-radius: 12px; min-height: 44px; font-weight: 700; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _create_lesson(
    question: str,
    subject: str,
    grade: int,
    language: str,
    *,
    text_provider: str,
    text_model: str,
    text_key: str,
    image_provider: str,
    image_model: str,
    image_key: str,
    create_video: bool,
    narration_key: str,
    narration_voice: str,
    research_enabled: bool,
) -> None:
    if not question.strip():
        st.error("Enter a question first.")
        return
    text_key = text_key.strip()
    image_key = image_key.strip()
    narration_key = narration_key.strip()
    if not text_key:
        st.error("Enter your text API key in Step 1.")
        return
    if create_video and not image_key:
        st.error("Enter your image API key in Step 1, or turn off illustrated MP4.")
        return

    if create_video and not narration_key:
        st.error("Enter an OpenAI narration API key in Step 1, or turn off illustrated MP4.")
        return
    client = build_text_client(text_provider, api_key=text_key)
    if client is None:
        st.error(f"The {text_provider} text client could not be initialized.")
        return

    st.session_state.media_requested = create_video
    st.session_state.gate2 = (
        {"mode": "waiting", "pass": None, "status": "Waiting for generated frames."}
        if create_video
        else {"mode": "not_applicable", "pass": None, "status": "Illustrated video was not requested."}
    )

    try:
        with st.status("Building your lesson", expanded=True) as status:
            progress_bar = st.progress(0.05, text="Starting your lesson...")
            tip_slot = st.empty()

            def update_wait(stage: str, message: str, progress: float, step: int = 0) -> None:
                progress_bar.progress(max(0.0, min(1.0, progress)), text=message)
                tip_slot.info(f"Study tip while you wait: {_waiting_tip(stage, grade, subject, step)}")
                status.write(message)

            update_wait("draft", f"Drafting with {text_provider}: {text_model}...", 0.08)
            bundle = generate_lesson_bundle(
                client,
                model=text_model,
                question=question,
                subject=subject,
                grade=grade,
                language=language,
            )
            update_wait("gate", "Gate 1: checking accuracy, logic, and grade fit...", 0.25)
            gate1 = review_explanation(
                client,
                model=text_model,
                question=question,
                explanation=bundle["explanation"],
                grade=grade,
                subject=subject,
                max_repairs=1,
            )
            bundle["explanation"] = gate1["final_explanation"]
            bundle["research"] = {"status": "disabled", "report_markdown": "", "sources": []}

            bundle["request"] = {
                "question": question,
                "subject": subject,
                "grade": grade,
                "language": language,
            }
            bundle["generation"] = {
                "text_provider": text_provider,
                "text_model": text_model,
                "image_provider": image_provider if create_video else None,
                "image_model": image_model if create_video else None,
                "narration_voice": narration_voice if create_video else None,
                "cited_web_research": research_enabled,
            }
            st.session_state.bundle = bundle
            st.session_state.gate1 = gate1
            st.session_state.pipeline_result = None
            st.session_state.quiz_submitted = False

            if not gate1["pass"]:
                status.update(label="Lesson needs review", state="error")
                st.session_state.gate2 = {"mode": "blocked", "pass": None, "status": "Gate 1 did not pass, so visuals were not generated."}
                st.error("Gate 1 could not verify this explanation. No research or media costs were incurred.")
                return

            if research_enabled:
                update_wait("research", "Searching reliable sources and building cited notes...", 0.34)
                bundle["research"] = research_lesson_sources(
                    client,
                    question=question,
                    subject=subject,
                    grade=grade,
                    language=language,
                    model=text_model,
                )
            if create_video:
                update_wait("storyboard", f"Preparing seven visual steps with {image_provider}: {image_model}...", 0.40)

                def media_progress(stage: str, details: dict) -> None:
                    update_wait(
                        stage,
                        str(details.get("message") or "Building the visual lesson..."),
                        float(details.get("progress") or 0.40),
                        int(details.get("step") or 0),
                    )

                run = run_pipeline(
                    question=question,
                    explanation=bundle["explanation"],
                    grade=grade,
                    subject=subject,
                    output_root=Path(tempfile.gettempdir()) / "visual_lesson_ai",
                    run_openai=True,
                    run_checker=False,
                    run_checker2=True,
                    text_provider=text_provider,
                    image_provider=image_provider,
                    text_model=text_model,
                    image_model=image_model,
                    text_api_key=text_key,
                    image_api_key=image_key,
                    tts_api_key=narration_key,
                    tts_voice=narration_voice,
                    gate2_callback=lambda gate: st.session_state.__setitem__("gate2", gate),
                    progress_callback=media_progress,
                )
                st.session_state.pipeline_result = run
                st.session_state.gate2 = run.get("checker2_result") or st.session_state.gate2
                run_dir = Path(run["out_dir"])
                (run_dir / "lesson.json").write_text(
                    json.dumps(bundle, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            update_wait("complete", "Lesson ready—open a tab and test your understanding.", 1.0)
            status.update(label="Lesson ready", state="complete", expanded=False)
    except Exception as exc:
        gate2_state = st.session_state.get("gate2") or {}
        if gate2_state.get("mode") == "waiting":
            st.session_state.gate2 = {
                "mode": "not_completed",
                "pass": None,
                "status": "Media generation stopped before frames were available for Gate 2.",
            }
        st.error(f"Lesson generation stopped safely: {exc}")


def _regenerate_media(
    bundle: dict[str, Any],
    *,
    text_provider: str,
    text_model: str,
    text_key: str,
    image_provider: str,
    image_model: str,
    image_key: str,
    narration_key: str,
    narration_voice: str,
) -> None:
    """Rebuild images, narration, and MP4 while keeping the approved lesson text."""
    if not image_key.strip() or not narration_key.strip():
        st.error("Image and narration API keys are required to regenerate the visuals.")
        return
    request = bundle.get("request") if isinstance(bundle.get("request"), dict) else {}
    st.session_state.gate2 = {"mode": "waiting", "pass": None, "status": "Regenerating frames for Gate 2."}
    st.session_state.pipeline_result = None
    try:
        with st.status("Regenerating visual lesson", expanded=True) as status:
            regeneration_progress = st.progress(0.10, text="Preparing a fresh storyboard...")
            regeneration_tip = st.empty()
            request_grade = int(request.get("grade") or 10)
            request_subject = str(request.get("subject") or "General")

            def regeneration_update(stage: str, details: dict) -> None:
                progress = float(details.get("progress") or 0.10)
                step = int(details.get("step") or 0)
                message = str(details.get("message") or "Creating a fresh seven-frame storyboard...")
                regeneration_progress.progress(max(0.0, min(1.0, progress)), text=message)
                regeneration_tip.info(f"Study tip while you wait: {_waiting_tip(stage, request_grade, request_subject, step)}")
                status.write(message)

            regeneration_update("storyboard", {"message": "Creating a fresh seven-frame storyboard...", "progress": 0.10})
            run = run_pipeline(
                question=str(request.get("question") or bundle.get("title") or "Lesson topic"),
                explanation=bundle["explanation"],
                grade=int(request.get("grade") or 10),
                subject=str(request.get("subject") or "General"),
                output_root=Path(tempfile.gettempdir()) / "visual_lesson_ai",
                run_openai=True,
                run_checker=False,
                run_checker2=True,
                text_provider=text_provider,
                image_provider=image_provider,
                text_model=text_model,
                image_model=image_model,
                text_api_key=text_key,
                image_api_key=image_key,
                tts_api_key=narration_key,
                tts_voice=narration_voice,
                gate2_callback=lambda gate: st.session_state.__setitem__("gate2", gate),
                progress_callback=regeneration_update,
            )
            st.session_state.pipeline_result = run
            st.session_state.gate2 = run.get("checker2_result") or st.session_state.gate2
            regeneration_update("complete", {"message": "Fresh visual lesson ready.", "progress": 1.0})
            status.update(label="Fresh visual lesson ready", state="complete", expanded=False)
    except Exception as exc:
        current = st.session_state.get("gate2") or {}
        if current.get("mode") == "waiting":
            st.session_state.gate2 = {"mode": "not_completed", "pass": None, "status": "Visual regeneration stopped before Gate 2."}
        st.error(f"Visual regeneration stopped safely: {exc}")


def _dot_escape(value: Any) -> str:
    return str(value or "").replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")


def _dot_label(value: Any, width: int = 23) -> str:
    """Wrap graph labels without cutting words."""
    words = str(value or "Related concept").split()
    lines: list[str] = []
    current: list[str] = []
    for word in words:
        if current and len(" ".join(current + [word])) > width:
            lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(" ".join(current))
    return r"\n".join(_dot_escape(line) for line in lines[:3])


def _concept_map_tab(bundle: dict[str, Any], subject: str) -> None:
    concept_map = build_concept_map(bundle, subject)
    nodes = concept_map.get("nodes", [])
    st.markdown("### Explore the concept neighborhood")
    st.caption("See what comes before this lesson, how it differs from nearby ideas, where it is used, and what to learn next.")
    if not nodes:
        st.info("Regenerate this lesson to create its related-topic network.")
        return

    role_names = {
        "all": "All links",
        "prerequisite": "Foundations",
        "application": "Applications",
        "contrast": "Compare",
        "next step": "Next steps",
        "related concept": "Related",
    }
    available_roles = [
        role for role in concept_map.get("relationship_order", [])
        if any(node.get("relationship_kind") == role for node in nodes)
    ]
    selected_role = st.radio(
        "Learning direction",
        ["all", *available_roles],
        index=0,
        format_func=lambda role: role_names.get(role, str(role).title()),
        horizontal=True,
        key="concept_role_filter",
    )
    visible_nodes = nodes if selected_role in (None, "all") else [
        node for node in nodes if node.get("relationship_kind") == selected_role
    ]

    center = _dot_label(concept_map.get("center", "Current lesson"), 27)
    dot = [
        "digraph ConceptMap {",
        'graph [rankdir=LR, bgcolor="transparent", pad=0.35, nodesep=0.55, ranksep=1.05, splines=curved, outputorder=edgesfirst];',
        'node [shape=box, style="rounded,filled", fontname="Arial", fontsize=12, margin="0.22,0.15", penwidth=1.7];',
        'edge [fontname="Arial", fontsize=9, arrowsize=0.8, penwidth=1.5];',
        f'center [label="{center}", fillcolor="#17345F", color="#17345F", fontcolor="white", penwidth=2.4];',
    ]
    for index, node in enumerate(visible_nodes, start=1):
        label = _dot_label(node.get("topic"))
        relationship = _dot_escape(role_names.get(str(node.get("relationship_kind")), str(node.get("relationship", "related"))))
        color = _dot_escape(node.get("color", "#526078"))
        fill = _dot_escape(node.get("fill", "#EEF2F7"))
        dot.append(f'n{index} [label="{label}", fillcolor="{fill}", color="{color}", fontcolor="#172033"];')
        direction = node.get("direction")
        if direction == "in":
            dot.append(f'n{index} -> center [label="{relationship}", color="{color}", fontcolor="{color}"];')
        elif direction == "side":
            dot.append(f'center -> n{index} [label="{relationship}", color="{color}", fontcolor="{color}", dir=both, arrowtail=none];')
        else:
            dot.append(f'center -> n{index} [label="{relationship}", color="{color}", fontcolor="{color}"];')
    dot.append("}")
    st.graphviz_chart("\n".join(dot), use_container_width=True)
    st.caption("Blue points toward required foundations; green and purple lead toward use and further learning. Filter the map or focus on one connection below.")

    topics = [str(node.get("topic", "Related topic")) for node in nodes]
    focus_key = "concept_focus_topic"
    if st.session_state.get(focus_key) not in topics:
        st.session_state[focus_key] = topics[0]
    controls = st.columns([4, 1])
    if controls[1].button("Next link", use_container_width=True, key="next_concept_link"):
        current_index = topics.index(st.session_state.get(focus_key, topics[0]))
        st.session_state[focus_key] = topics[(current_index + 1) % len(topics)]
    focus_topic = controls[0].selectbox("Focus on one connection", topics, key=focus_key)
    focus_node = next(node for node in nodes if str(node.get("topic")) == focus_topic)

    detail, action = st.columns([3, 2])
    with detail.container(border=True):
        kind = str(focus_node.get("relationship_kind", "related concept"))
        st.caption(role_names.get(kind, kind.title()).upper())
        st.markdown(f"#### {focus_topic}")
        st.write(focus_node.get("why_useful") or "This topic broadens the current lesson.")
        st.info(f"Quick challenge: {focus_node.get('challenge', 'Explain the connection in your own words.')}")
    with action.container(border=True):
        st.markdown("#### Study this topic")
        for source in focus_node.get("sources", []):
            st.link_button(source["name"], source["url"], use_container_width=True)
        mastery_key = f"concept_mastered::{concept_map.get('center')}::{focus_topic}"
        st.toggle("I can explain this link", key=mastery_key)

    completed = sum(
        bool(st.session_state.get(f"concept_mastered::{concept_map.get('center')}::{topic}"))
        for topic in topics
    )
    st.progress(completed / max(1, len(topics)), text=f"Connections explained: {completed} of {len(topics)}")

    with st.expander("Browse every connection and source"):
        columns = st.columns(2)
        for index, node in enumerate(nodes):
            with columns[index % 2].container(border=True):
                kind = str(node.get("relationship_kind", "related concept"))
                st.caption(role_names.get(kind, kind.title()).upper())
                st.markdown(f"**{node.get('topic', 'Related topic')}**")
                st.write(node.get("why_useful") or "Builds a broader understanding of this lesson.")
                for source in node.get("sources", []):
                    st.markdown(f"[{source['name']}]({source['url']})")

def _show_gate(name: str, result: dict[str, Any] | None) -> None:
    if not result:
        st.caption(f"{name}: not run")
        return
    if result.get("pass") is None:
        mode = result.get("mode")
        label = {
            "not_applicable": "Not applicable",
            "blocked": "Blocked",
            "not_completed": "Not completed",
        }.get(mode, "Waiting")
        st.markdown(f"**{name}: {label}**")
        st.caption(result.get("status", "Gate 2 will run after visual frames are generated."))
        return
    passed = bool(result.get("pass"))
    score = result.get("overall_score")
    label = "Passed" if passed else "Needs review"
    suffix = f" - {float(score) * 100:.0f}%" if isinstance(score, (int, float)) else ""
    st.markdown(f"<span class=\"{'gate-pass' if passed else 'gate-fail'}\">{name}: {label}{suffix}</span>", unsafe_allow_html=True)
    issues = result.get("issues", [])
    if issues:
        with st.expander("Review notes"):
            for issue in issues:
                st.write(f"- {issue}")


def _lesson_tab(bundle: dict[str, Any]) -> None:
    st.markdown(f"<div class='card'><div class='eyebrow'>Learning objective</div><h2>{bundle['title']}</h2><p>{bundle['learning_objective']}</p></div>", unsafe_allow_html=True)
    overview, example, mistakes, explore = st.tabs(["Understand", "Worked example", "Easy to get wrong", "Connect & study"])

    with overview:
        st.markdown("### Clear explanation")
        st.markdown(streamlit_markdown(bundle["explanation"]))
        if bundle.get("why_it_matters"):
            st.info(f"Why it matters: {bundle['why_it_matters']}")
        st.markdown("#### Key ideas")
        for idea in bundle.get("key_ideas", []):
            st.markdown(f"- {streamlit_markdown(idea)}")
        if bundle.get("prerequisites"):
            with st.expander("Check the foundations first"):
                for item in bundle["prerequisites"]:
                    st.markdown(f"- {item}")

    with example:
        st.markdown("### Worked example")
        st.markdown(streamlit_markdown(bundle.get("worked_example", "")))
        st.markdown("### Try it yourself")
        st.success(streamlit_markdown(bundle.get("quick_check", "")))
        st.caption("Say your reasoning aloud before checking notes; retrieval strengthens memory.")

    with mistakes:
        st.markdown("### Common traps and how to repair them")
        if bundle.get("common_mistake"):
            st.warning(streamlit_markdown(bundle["common_mistake"]))
        for index, item in enumerate(bundle.get("easy_to_confuse", []), start=1):
            with st.expander(f"{index}. {item['confusion']}", expanded=index == 1):
                st.markdown(f"**Correction:** {streamlit_markdown(item['correction'])}")
                if item.get("memory_tip"):
                    st.info(f"Memory tip: {item['memory_tip']}")

    with explore:
        left, right = st.columns(2)
        with left:
            st.markdown("### Useful connections")
            for item in bundle.get("connections", []):
                st.markdown(f"- {item}")
            st.markdown("### Questions to explore next")
            for item in bundle.get("follow_up_questions", []):
                st.markdown(f"- {item}")
        with right:
            st.markdown("### Short study path")
            for index, item in enumerate(bundle.get("study_path", []), start=1):
                st.markdown(f"**{index}.** {item}")

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


def _research_tab(bundle: dict[str, Any]) -> None:
    st.markdown("### Evidence and source notes")
    st.caption("Web-grounded notes are generated separately from the lesson draft. Open the citations and compare what each source actually supports.")
    research = bundle.get("research") if isinstance(bundle.get("research"), dict) else {}
    if research.get("status") != "ready":
        if research.get("reason") == "selected_text_provider_has_no_grounded_web_search":
            st.warning(
                "Your selected text provider does not offer grounded web search through this API. "
                "The lesson still uses the same text API key; use the trusted Learning library links "
                "below for verified starting points."
            )
        else:
            st.warning("Cited research was not available for this lesson. Use the trusted library tab as a starting point and verify important claims with a teacher or textbook.")
        return
    st.markdown(streamlit_markdown(research.get("report_markdown", "")))
    sources = research.get("sources") if isinstance(research.get("sources"), list) else []
    if sources:
        st.divider()
        st.markdown("### Sources used")
        for index, source in enumerate(sources, start=1):
            title = source.get("title", "Source")
            url = source.get("url", "")
            if url.startswith(("https://", "http://")):
                st.markdown(f"{index}. [{title}]({url})")
        st.caption("A citation shows where a claim came from; it does not guarantee that every source is correct or that the lesson covers every viewpoint.")

def _quiz_tab(bundle: dict[str, Any]) -> None:
    questions = bundle.get("quiz", [])
    for index, item in enumerate(questions):
        st.markdown(f"#### {index + 1}. {item['question']}")
        labels = ["A", "B", "C", "D"]
        options = [f"{label}. {streamlit_markdown(choice)}" for label, choice in zip(labels, item["choices"])]
        if index not in st.session_state.started_at:
            st.session_state.started_at[index] = time.time()
        selected = st.radio("Choose one", options, index=None, key=f"answer_{index}", label_visibility="collapsed")
        if st.session_state.quiz_submitted:
            if selected and selected[0] == item["answer"]:
                st.success("Correct. " + streamlit_markdown(item["explanation"]))
            else:
                st.error(f"Answer: {item['answer']}. " + streamlit_markdown(item["explanation"]))
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
    st.caption("These links come from a curated list; the model does not invent resource URLs.")
    for resource in curated_resources(subject, question):
        st.markdown(f"**[{resource['name']}]({resource['url']})**  \n{resource['description']}")
    st.divider()
    st.download_button(
        "Download lesson JSON",
        json.dumps(bundle, ensure_ascii=False, indent=2),
        file_name="visual_lesson.json",
        mime="application/json",
        use_container_width=True,
    )
    result = st.session_state.pipeline_result
    if result and Path(result["video_path"]).exists():
        visual_gate = result.get("checker2_result") or {}
        if visual_gate.get("pass"):
            st.download_button(
                "Download MP4 video",
                Path(result["video_path"]).read_bytes(),
                file_name="visual_lesson.mp4",
                mime="video/mp4",
                use_container_width=True,
            )
        else:
            st.warning("MP4 download is held because Gate 2 marked the visuals as needing review.")


def _benchmark_dashboard() -> None:
    st.markdown("### How the quality system works")
    cols = st.columns(4)
    cards = [
        ("01 · DRAFT", "Explain", "A structured high-school lesson is drafted with the learner's chosen text model."),
        ("02 · GATE 1", "Verify reasoning", "Fast local checks plus one compact semantic rubric; one repair is allowed."),
        ("03 · BUILD", "Ground & illustrate", "Sources, storyboard images, captions, and narration are assembled only after Gate 1."),
        ("04 · GATE 2", "Release safely", "Parallel render checks run first; one contact-sheet audit checks teaching alignment."),
    ]
    for column, (number, title, body) in zip(cols, cards):
        column.markdown(f"<div class='flow-step'><div class='number'>{number}</div><h4>{title}</h4><p>{body}</p></div>", unsafe_allow_html=True)

    st.markdown("### What did our checker tests teach us?")
    st.caption("A quick, friendly view of saved lab results. Higher scores are better; these are not the waiting time for your lesson.")
    gate1, gate2 = st.tabs(["Explanation checks", "Visual matching"])
    short_names = {
        "DistilBERT error classifier": "DistilBERT",
        "BERT error classifier": "BERT",
        "RoBERTa error classifier": "RoBERTa",
        "CLIP cosine threshold": "CLIP similarity",
        "CLIP linear probe": "CLIP + linear layer",
        "CLIP MLP probe": "CLIP + small neural net",
    }
    for tab, gate in ((gate1, "Gate 1"), (gate2, "Gate 2")):
        with tab:
            rows = sorted(benchmark_rows(gate), key=lambda row: row["F1"], reverse=True)
            cards = st.columns(3)
            for rank, (column, row) in enumerate(zip(cards, rows), start=1):
                with column.container(border=True):
                    badge = "Best result" if rank == 1 else ("Trained model" if row["Trained"] == "Yes" else "No extra training")
                    st.caption(badge.upper())
                    st.markdown(f"#### {short_names.get(row['Method'], row['Method'])}")
                    st.metric("F1 score", f"{row['F1'] * 100:.1f}%")
                    st.progress(float(row["F1"]), text="Overall balance")
                    st.caption(f"Accuracy {row['Accuracy'] * 100:.1f}% · Recall {row['Recall'] * 100:.1f}%")
                    if row.get("AUROC") is not None:
                        st.caption(f"AUROC {row['AUROC'] * 100:.1f}%")
            if gate == "Gate 1":
                st.success("Takeaway: RoBERTa was strongest at sorting known error types. For a brand-new lesson, we still use fast local checks plus a compact meaning review because error labels alone cannot prove an explanation is correct.")
            else:
                st.success("Takeaway: the simplest CLIP similarity method won this comparison. That supports our production choice to keep visual checks light, then use one semantic review only when the frames pass basic quality checks.")
            with st.expander("How to read these scores"):
                st.markdown("- **F1** balances missed problems and false alarms.\n- **Accuracy** is the share classified correctly.\n- **Recall** shows how many target examples were found.\n- **AUROC** summarizes ranking quality across thresholds.\n\nThe two tabs use different datasets, so compare methods inside a tab rather than comparing Gate 1 directly with Gate 2.")

def _quality_report() -> None:
    gate1 = st.session_state.gate1 or {}
    pipeline_result = st.session_state.pipeline_result or {}
    gate2 = pipeline_result.get("checker2_result") or st.session_state.get("gate2") or {}
    st.markdown("### Live gate performance")
    for title, result in (("Gate 1 · explanation", gate1), ("Gate 2 · visuals", gate2)):
        st.markdown(f"#### {title}")
        if not result:
            st.caption("Not run for this lesson.")
            continue
        metrics = result.get("metrics", {})
        columns = st.columns(3)
        if result.get("pass") is None:
            decision = "N/A" if result.get("mode") == "not_applicable" else result.get("mode", "waiting").replace("_", " ").title()
        else:
            decision = "Pass" if result.get("pass") else "Review"
        columns[0].metric("Decision", decision)
        score = result.get("overall_score")
        columns[1].metric("Score", f"{float(score) * 100:.0f}%" if isinstance(score, (int, float)) else "—")
        columns[2].metric("Live latency", f"{float(metrics.get('total_latency_ms', 0)):.0f} ms")
        st.caption(f"Model calls: {metrics.get('model_calls', 0)} · Path: {' → '.join(result.get('decision_path', [])) or result.get('mode', 'unknown')}")
        methods = result.get("method_comparison", [])
        if methods:
            st.dataframe(methods, hide_index=True, use_container_width=True)
        with st.expander(f"{title} technical details"):
            st.json(result)
    stage_times = pipeline_result.get("stage_times", {})
    if stage_times:
        st.markdown("#### End-to-end stage time")
        st.bar_chart([{"Stage": key.replace("_", " ").title(), "Seconds": value} for key, value in stage_times.items()], x="Stage", y="Seconds")
    st.caption("A passed automated gate reduces risk but is not a substitute for teacher review in high-stakes instruction.")

def main() -> None:
    st.set_page_config(page_title="VisualLesson AI", page_icon=":material/school:", layout="wide")
    _init_state()
    _styles()

    with st.sidebar:
        st.markdown("## Create a lesson")

        with st.container():
            st.markdown("### 1. Connect your models")
            provider_label = st.selectbox(
                "Text provider",
                list(TEXT_PROVIDERS),
                help="Choose the model provider that will write and check the lesson.",
            )
            provider_config = TEXT_PROVIDERS[provider_label]
            text_model_label = st.selectbox("Text model", list(provider_config["models"]))
            text_model = provider_config["models"][text_model_label]
            api_key = st.text_input(
                "Text API key",
                type="password",
                key=f"visitor_api_key_{provider_config['id']}",
                help=provider_config["key_help"],
                placeholder=f"Paste your {provider_label} key - session only",
            )
            text_label = provider_label
            text_config = provider_config
            text_key = api_key
            research_enabled = st.toggle(
                "Add cited web research",
                value=True,
                help="Uses this same text API key. Grounded citations appear when the selected provider supports web search.",
            )

            image_label = "OpenAI"
            image_config = IMAGE_PROVIDERS[image_label]
            image_model_label = next(iter(image_config["models"]))
            image_model = image_config["models"][image_model_label]
            image_key = ""
            narration_key = ""
            narration_voice = "marin"
            provider_supports_video = provider_label == "OpenAI"

            if provider_supports_video:
                create_video = st.toggle(
                    "Create illustrated MP4",
                    value=True,
                    key="create_video_openai",
                    help="The text key is reused for seven images and OpenAI narration.",
                )
                image_key = api_key if create_video else ""
                narration_key = api_key if create_video else ""
            else:
                st.warning(
                    f"{provider_label} is text-only in this app and cannot generate images or narration. "
                    "You can still create the written lesson, or optionally connect OpenAI for the visual video."
                )
                add_image_api = st.toggle(
                    "Add image API (optional)",
                    value=False,
                    key=f"add_image_api_{provider_config['id']}",
                    help="Adds a separate provider only for images, visual checking, and narration.",
                )
                create_video = add_image_api
                if add_image_api:
                    image_label = st.selectbox(
                        "Image provider",
                        ["OpenAI"],
                        help="OpenAI currently supplies both the lesson images and narration required for the MP4.",
                    )
                    image_config = IMAGE_PROVIDERS[image_label]
                    image_api_key = st.text_input(
                        "Image API key",
                        type="password",
                        key=f"visitor_image_api_key_{provider_config['id']}_{image_config['id']}",
                        help="A separate OpenAI key with image-model access. It is also used for narration and the visual semantic check.",
                        placeholder="Paste your OpenAI image key - session only",
                    )
                    image_key = image_api_key
                    narration_key = image_api_key

            if create_video:
                image_model_label = st.selectbox("Image model", list(image_config["models"]))
                image_model = image_config["models"][image_model_label]
                if provider_supports_video:
                    st.caption("Images and narration reuse the OpenAI text key entered above.")
                else:
                    st.caption("The optional OpenAI key is used only for images, visual checking, and narration; DeepSeek still writes the lesson.")
                st.warning("Seven image calls may incur noticeable cost and take several minutes.")
                narration_voice = st.selectbox(
                    "Teaching voice",
                    ["marin", "cedar", "coral", "sage", "alloy"],
                    help="Marin and cedar are recommended for the clearest narration.",
                )
                st.caption("The narration voice is AI-generated, not a human recording.")
            keys_ready = bool(api_key.strip()) and (not create_video or bool(image_key.strip()))
            if keys_ready:
                st.success("API setup ready")
            elif api_key.strip() and create_video:
                st.info("Enter the optional image API key, or turn off the image API option.")
            else:
                st.info("Enter your text API key to unlock generation.")

            st.divider()
            st.markdown("### 2. Describe the lesson")
            question = st.text_area(
                "Student question",
                placeholder="Why does increasing mass require more force for the same acceleration?",
                height=130,
            )
            subject = st.selectbox(
                "Subject",
                ["Physics", "Biology", "Chemistry", "Mathematics", "Computer Science", "Other"],
            )
            grade = st.slider("Grade", 9, 12, 10)
            language = st.selectbox(
                "Language",
                ["English", "Chinese", "Spanish", "French", "German", "Japanese", "Korean"],
            )
            if st.button("Create lesson", type="primary", use_container_width=True, disabled=not keys_ready):
                _create_lesson(
                    question,
                    subject,
                    grade,
                    language,
                    text_provider=text_config["id"],
                    text_model=text_model,
                    text_key=text_key,
                    image_provider=image_config["id"],
                    image_model=image_model,
                    image_key=image_key,
                    create_video=create_video,
                    narration_key=narration_key,
                    narration_voice=narration_voice,
                    research_enabled=research_enabled,
                )
            st.caption("Your API keys stay in this Streamlit session. They are never written to files, logs, or environment variables.")

    st.markdown("<div class='hero'><h1>VisualLesson AI</h1><p>Clear explanations, visual stories, and feedback built for high-school learners.</p></div>", unsafe_allow_html=True)
    bundle = st.session_state.bundle
    if not bundle:
        st.info("Connect your API providers and ask a question in the sidebar to create a lesson.")
        metrics = st.columns(4)
        metrics[0].metric("Quality gates", "2", "local-first")
        metrics[1].metric("Gate 1 dimensions", "5", "one compact audit")
        metrics[2].metric("Visual checks", "5 × 7", "parallel")
        metrics[3].metric("Heavy local ML", "0", "fast startup")
        _benchmark_dashboard()
        return

    gate_col1, gate_col2 = st.columns(2)
    with gate_col1:
        _show_gate("Gate 1 - explanation", st.session_state.gate1)
    with gate_col2:
        result = st.session_state.pipeline_result
        gate2_result = (result.get("checker2_result") if result else None) or st.session_state.get("gate2")
        _show_gate("Gate 2 - visuals", gate2_result)

    with st.expander("Regenerate this lesson"):
        st.caption("Full regeneration creates a new explanation and, when enabled, seven new images. Visual-only regeneration keeps the approved explanation but creates seven new images, narration, and MP4.")
        saved_request = bundle.get("request") if isinstance(bundle.get("request"), dict) else {}
        regenerate_full, regenerate_visuals = st.columns(2)
        if regenerate_full.button("Regenerate full lesson", use_container_width=True, disabled=not keys_ready):
            _create_lesson(
                str(saved_request.get("question") or question),
                str(saved_request.get("subject") or subject),
                int(saved_request.get("grade") or grade),
                str(saved_request.get("language") or language),
                text_provider=text_config["id"], text_model=text_model, text_key=text_key,
                image_provider=image_config["id"], image_model=image_model, image_key=image_key,
                create_video=create_video, narration_key=narration_key, narration_voice=narration_voice,
                research_enabled=research_enabled,
            )
            bundle = st.session_state.bundle
        visual_ready = create_video and bool(text_key.strip()) and bool(image_key.strip()) and bool(narration_key.strip())
        if regenerate_visuals.button("Regenerate visuals only", use_container_width=True, disabled=not visual_ready):
            _regenerate_media(
                bundle,
                text_provider=text_config["id"], text_model=text_model, text_key=text_key,
                image_provider=image_config["id"], image_model=image_model, image_key=image_key,
                narration_key=narration_key, narration_voice=narration_voice,
            )
        if not create_video:
            st.info("Turn on Create illustrated MP4 in the sidebar to enable visual-only regeneration.")

    lesson, concept_map, evidence, quiz, resources, quality = st.tabs([
        "Lesson", "Concept map", "Evidence & sources", "Practice", "Learning library", "Quality report"
    ])
    with lesson:
        _lesson_tab(bundle)
    with concept_map:
        lesson_subject = str((bundle.get("request") or {}).get("subject") or subject)
        _concept_map_tab(bundle, lesson_subject)
    with evidence:
        _research_tab(bundle)
    with quiz:
        _quiz_tab(bundle)
    with resources:
        _resources_tab(subject, question, bundle)
    with quality:
        _quality_report()
