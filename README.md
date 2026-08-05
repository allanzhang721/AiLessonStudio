# VisualLesson AI

VisualLesson AI turns a high-school student's question into a checked explanation, a seven-scene visual lesson, a narrated 1080p MP4, practice questions, and a focused review report.

The application uses one media workflow only:

1. Generate and quality-check the explanation.
2. Plan seven cumulative teaching scenes.
3. Generate or edit one high-quality image per scene.
4. Add deterministic captions and formulas.
5. Generate a clean AI narration track.
6. compose the frames with subtle motion and crossfades.
7. Encode a browser-ready H.264/AAC MP4.

## Public app

https://teachhighschool.streamlit.app/

The hosted app asks each visitor for the required API credentials. Keys remain in the active Streamlit session and are not written to files, logs, or environment variables.

## Provider choices

Text:

- OpenAI: GPT-5.6 Terra, Luna, or Sol
- DeepSeek: DeepSeek V4 Flash or Pro

Images:

- OpenAI: GPT Image 2
- Alibaba Model Studio: Wan 2.7 Image or Image Pro

Narration:

- OpenAI `gpt-4o-mini-tts`
- Selectable voices, with `marin` as the default
- If an OpenAI key was already entered for text or images, the app reuses it for narration

The interface clearly discloses that narration is AI-generated.

## Video quality

- 1920 x 1080 output
- 30 frames per second
- High-quality source image generation and editing
- Full-frame educational diagrams preserved over a blurred 16:9 background
- Subtle Ken Burns motion rather than a static slideshow
- Half-second crossfades between teaching scenes
- Scene timing allocated from measured narration duration and caption density
- Lossless WAV narration source
- EBU-style narration loudness normalization
- H.264 video at CRF 18 and AAC audio at 192 kbps
- `yuv420p` compatibility and fast-start metadata for web playback

## Quality gates

Gate 1 reviews explanation accuracy, completeness, logical flow, grade fit, and clarity before image costs are incurred.

Gate 2 checks rendered image health and whether the visuals teach the approved explanation. MP4 download is held when Gate 2 reports that the visuals need review.

## Local setup

Python 3.12 is recommended.

```bash
python -m venv .venv
```

Activate the environment, then install and run:

```bash
python -m pip install -r requirements.txt
streamlit run streamlit_app.py
```

No server-side API secret is required because visitors enter their own keys in the application.

## Tests

```bash
python -m pytest -q
```

The suite covers explanation gates, schema validation, provider routing, API-only Streamlit rendering, narration arguments, audio-aware scene timing, and real H.264 MP4 encoding.

## Streamlit Community Cloud

- Repository: `allanzhang721/AiLessonStudio`
- Entrypoint: `streamlit_app.py`
- Python: `3.12`

See `DEPLOYMENT.md` for deployment details.

## Main production files

- `streamlit_app.py`: Streamlit Cloud entrypoint
- `app_v2.py`: API-only product interface
- `pipeline/lesson_service.py`: lesson and quiz generation
- `pipeline/quality_gates.py`: explanation review
- `pipeline/planner.py`: seven-scene visual plan
- `pipeline/image_pipeline.py`: high-quality image generation/editing
- `pipeline/frame_checker.py`: visual quality gate
- `pipeline/video_pipeline.py`: narration and 1080p MP4 composition
- `pipeline/pipeline.py`: end-to-end orchestration
