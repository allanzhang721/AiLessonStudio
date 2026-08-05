"""Production Streamlit interface for VisualLesson AI."""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from typing import Any

import streamlit as st

from pipeline.clients import build_text_client
from pipeline.lesson_service import curated_resources, generate_lesson_bundle
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
        [data-testid="stTextInput"] input,
        [data-testid="stTextArea"] textarea,
        [data-testid="stNumberInput"] input,
        [data-baseweb="select"] input,
        [data-baseweb="select"] > div { color: #172033 !important; }
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

    try:
        with st.status("Building your lesson", expanded=True) as status:
            status.write(f"Drafting with {text_provider}: {text_model}...")
            bundle = generate_lesson_bundle(
                client,
                model=text_model,
                question=question,
                subject=subject,
                grade=grade,
                language=language,
            )
            status.write("Gate 1: checking accuracy, logic, and grade fit...")
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
            bundle["generation"] = {
                "text_provider": text_provider,
                "text_model": text_model,
                "image_provider": image_provider if create_video else None,
                "image_model": image_model if create_video else None,
                "narration_voice": narration_voice if create_video else None,
            }
            st.session_state.bundle = bundle
            st.session_state.gate1 = gate1
            st.session_state.pipeline_result = None
            st.session_state.quiz_submitted = False

            if not gate1["pass"]:
                status.update(label="Lesson needs review", state="error")
                st.error("Gate 1 could not verify this explanation. Media generation was stopped before image costs were incurred.")
                return

            if create_video:
                status.write(f"Illustrating seven steps with {image_provider}: {image_model}...")
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
                )
                st.session_state.pipeline_result = run
                run_dir = Path(run["out_dir"])
                (run_dir / "lesson.json").write_text(
                    json.dumps(bundle, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
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
    suffix = f" - {float(score) * 100:.0f}%" if isinstance(score, (int, float)) else ""
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


def main() -> None:
    st.set_page_config(page_title="VisualLesson AI", page_icon=":material/school:", layout="wide")
    _init_state()
    _styles()

    with st.sidebar:
        st.markdown("## Create a lesson")

        with st.container():
            st.markdown("### 1. Connect your models")
            text_label = st.selectbox("Text provider", list(TEXT_PROVIDERS))
            text_config = TEXT_PROVIDERS[text_label]
            text_model_label = st.selectbox("Text model", list(text_config["models"]))
            text_model = text_config["models"][text_model_label]
            text_key = st.text_input(
                f"{text_label} text API key",
                type="password",
                key="visitor_text_api_key",
                help=text_config["key_help"],
                placeholder="Paste your key - session only",
            )

            create_video = st.toggle(
                "Create illustrated MP4",
                value=True,
                help="Uses seven image calls. Text-only lessons need no image key.",
            )
            image_label = "OpenAI"
            image_config = IMAGE_PROVIDERS[image_label]
            image_model_label = next(iter(image_config["models"]))
            image_model = image_config["models"][image_model_label]
            image_key = ""
            narration_key = ""
            narration_voice = "marin"
            if create_video:
                image_label = st.selectbox("Image provider", list(IMAGE_PROVIDERS))
                image_config = IMAGE_PROVIDERS[image_label]
                image_model_label = st.selectbox("Image model", list(image_config["models"]))
                image_model = image_config["models"][image_model_label]
                reuse_openai = (
                    text_label == "OpenAI"
                    and image_label == "OpenAI"
                    and st.checkbox("Use the same OpenAI key for images", value=True)
                )
                image_key = text_key if reuse_openai else st.text_input(
                    f"{image_label} image API key",
                    type="password",
                    key="visitor_image_api_key",
                    help=image_config["key_help"],
                    placeholder="Paste your image key - session only",
                )
                st.warning("Seven image calls may incur noticeable cost and take several minutes.")
                st.markdown("#### Narration")
                narration_voice = st.selectbox(
                    "Teaching voice",
                    ["marin", "cedar", "coral", "sage", "alloy"],
                    help="Marin and cedar are recommended for the clearest narration.",
                )
                reusable_openai_key = (
                    text_key if text_label == "OpenAI"
                    else image_key if image_label == "OpenAI"
                    else ""
                )
                if reusable_openai_key:
                    narration_key = reusable_openai_key
                    st.caption("Narration will reuse the OpenAI key already entered above.")
                else:
                    narration_key = st.text_input(
                        "OpenAI narration API key",
                        type="password",
                        key="visitor_narration_api_key",
                        help="Required for high-quality AI narration when text and images use non-OpenAI providers.",
                    )
                st.caption("The narration voice is AI-generated, not a human recording.")

            keys_ready = bool(text_key.strip()) and (
                not create_video
                or (bool(image_key.strip()) and bool(narration_key.strip()))
            )
            if keys_ready:
                st.success("API setup ready")
            else:
                st.info("Enter the required key(s) to unlock generation.")

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
                )
            st.caption("Keys stay in this Streamlit session. They are never written to files, logs, or environment variables.")

    st.markdown("<div class='hero'><h1>VisualLesson AI</h1><p>Clear explanations, visual stories, and feedback built for high-school learners.</p></div>", unsafe_allow_html=True)
    bundle = st.session_state.bundle
    if not bundle:
        st.info("Connect your API providers and ask a question in the sidebar to create a lesson.")
        st.markdown("#### What changed")
        c1, c2, c3 = st.columns(3)
        c1.markdown("<div class='card'><b>Useful Gate 1</b><br>Checks accuracy, logic, completeness, clarity, and grade fit before media spending.</div>", unsafe_allow_html=True)
        c2.markdown("<div class='card'><b>Useful Gate 2</b><br>Checks both render health and whether the visuals teach the stated explanation.</div>", unsafe_allow_html=True)
        c3.markdown("<div class='card'><b>Student feedback</b><br>Turns missed quiz concepts into a focused review list.</div>", unsafe_allow_html=True)
        return

    gate_col1, gate_col2 = st.columns(2)
    with gate_col1:
        _show_gate("Gate 1 - explanation", st.session_state.gate1)
    with gate_col2:
        result = st.session_state.pipeline_result
        _show_gate("Gate 2 - visuals", result.get("checker2_result") if result else None)

    lesson, quiz, resources, quality = st.tabs(["Lesson", "Practice", "Resources", "Quality report"])
    with lesson:
        _lesson_tab(bundle)
    with quiz:
        _quiz_tab(bundle)
    with resources:
        _resources_tab(subject, question, bundle)
    with quality:
        st.markdown("### Explanation audit")
        st.json(st.session_state.gate1 or {})
        if st.session_state.pipeline_result:
            st.markdown("### Visual audit")
            st.json(st.session_state.pipeline_result.get("checker2_result") or {})
        st.caption("A passed automated gate reduces risk but is not a substitute for teacher review in high-stakes instruction.")
