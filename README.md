# VisualLesson AI

**Turn any classroom question into a visual lesson — explanation, storyboard frames, narrated video, quiz, and sources — in one click.**

Built by **Jiaxing BCOS**

---

## ICML-Clean Repository Layer

This repository now includes an ICML-style organization layer that keeps research runs reproducible and easy to audit, while preserving your existing working code.

### Added standardization paths

- `configs/icml/default.yaml` — canonical run settings
- `scripts/icml/run_smoke.sh` — quick smoke validation for Checker 1/2
- `scripts/icml/reproduce_core_results.sh` — core comparison reproduction
- `scripts/analysis/` — analysis utilities (for example paper plot generation)
- `docs/ICML_REPO_STANDARD.md` — structure and contribution conventions
- `docs/ICML_ARTIFACT_CHECKLIST.md` — pre-submission artifact checklist
- `papers/` — manuscript sources (`.tex`, `.bib`)
- `logs/` — experiment and smoke logs
- `notebooks/` — root-level research notebooks
- `Makefile` — one-command install/test/run/reproduce tasks
- `pyproject.toml` — Python project metadata + test/format config

### ICML quickstart commands

```bash
make install
make test
make smoke
make reproduce
```

### Clean logical layout to follow

```
AiLessonStudio/
├── pipeline/            # Core pipeline implementation
├── checker1/            # Text explanation quality research
├── checker2/            # Visual frame quality research
├── tests/               # Automated tests
├── configs/icml/        # Canonical reproducible configs
├── scripts/icml/        # Reproduction and smoke scripts
├── docs/                # Technical + artifact documentation
├── paper_figures/       # Publication figures
└── output/              # Generated artifacts (ignored by git)
```

---

## What it does

| Feature | Description |
|---------|-------------|
| **Explanation generation** | LLM drafts a grade-appropriate explanation for any question |
| **Checker 1 quality gate** | Fine-tuned DistilBERT detects 5 error types and triggers GPT repair |
| **7-frame storyboard** | Planner creates a structured teaching sequence; image pipeline generates/edits 7 PNG frames |
| **Storyboard video** | Frames assembled into GIF + MP4 with optional TTS voiceover |
| **Single API video (Sora)** | Continuous 12-second narrated video generated from anchor frame + plan |
| **Relevant sources** | LLM suggests websites, YouTube channels, and textbooks for the topic |
| **Interactive quiz** | 5 MCQs generated from the explanation; robust parser supports varied LLM formats; radio buttons + Check Answers + score |
| **Student weakness analyzer** | Concept-level diagnostics from quiz correctness, confidence, response time, and Checker 2 frame risk |
| **Downloads** | ZIP frames, MP4 video, TXT explanation, MD quiz |
| **Multi-language** | Explanation and quiz in English, 中文, Español, Français, Deutsch, 日本語, 한국어 |
| **Multi-provider** | Text: OpenAI GPT-4o or DeepSeek; Images: OpenAI gpt-image-1 or Wanx |
| **Demo mode** | Loads saved demo outputs without any API call |

---

## Project Structure

```
AiLessonStudio/
├── streamlit_app.py          # Streamlit web app entry point
├── run_demo.py               # CLI demo runner
├── single_api_video.py       # Single-video generation utility
├── requirements.txt
├── pyproject.toml
├── Makefile
├── README.md
├── PIPELINE.md
├── pipeline/                 # Core pipeline package
├── checker1/                 # Checker 1 research module + data + experiments
├── checker2/                 # Checker 2 research module + data + experiments
├── tests/                    # Automated tests
├── configs/icml/             # Canonical reproducible ICML configs
├── scripts/icml/             # Reproduction/smoke scripts
├── scripts/analysis/         # Analysis helpers (plots, aggregation)
├── docs/                     # Documentation and standards
├── papers/                   # LaTeX manuscript sources
├── paper_figures/            # Final paper figures
├── notebooks/                # Exploratory notebooks
├── logs/                     # Local logs (non-source artifacts)
└── output/                   # Generated media and pipeline outputs
```

---

## Pipeline Overview

```
Question + Subject + Grade
          │
          ▼
   LLM generates Explanation
   (+ Sources + Quiz in same call)
          │
          ▼
  ┌───────────────────────────────┐
  │  Checker 1 (DistilBERT)       │
  │  5 error types:               │
  │  ConceptError · GradeMismatch │
  │  LogicalGap · MisleadingAnalogy│
  │  MissingCondition             │
  └──────────┬────────────────────┘
             │ confidence ≥ 0.5?
      No ────┤──── Yes
      │      │       │
      │      │    GPT repairs explanation
      │      │       │
      └──────┴───────┘
         (up to 3 rounds)
                │
                ▼
       Planner → 7-step JSON plan
                │
                ▼
       Image pipeline → 7 PNG frames
       (generate step 1, edit steps 2–7)
                │
                ▼
       Video pipeline → GIF + MP4
                │
                ▼  (optional)
       Sora single-video → 12 s narrated clip
                │
                ▼
       Quiz submission → Student Weakness Analyzer
```

---

## Prerequisites

- macOS or Linux
- Python 3.11 or 3.12
- Internet connection for API mode
- At least one API key (OpenAI or DeepSeek) for API mode

---

## Setup

### Recommended: conda env `research`

```bash
conda activate research
cd /path/to/AiLessonStudio
pip install -r requirements.txt
```

### Alternative: local venv

```bash
cd /path/to/AiLesssonStudio
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Verify:

```bash
python -c "import openai, streamlit, torch, transformers; print('ok')"
```

---

## Configure API Keys

Create `api_keys.txt` in the project root (one entry per line):

```
openai=sk-...
deepseek=sk-...
```

Quotes are stripped automatically. Environment variables (`OPENAI_API_KEY`, etc.) also work as fallback.

---

## Run the App

```bash
cd /path/to/AiLessonStudio
conda activate research          # or: source .venv/bin/activate
python -m streamlit run streamlit_app.py
```

Open **http://localhost:8501** in your browser.

### Run Checker 2 Research Comparison

To compare the Flickr8k-based Checker 2 CLIP variants on a smaller subset:

```bash
cd /path/to/AiLessonStudio
conda activate research          # or: source .venv/bin/activate
python -m checker2.compare_variants \
        --dataset-dir checker2/Flickr8k \
        --output-dir checker2/experiments/clip_variant_comparison \
        --max-images 1200
```

For a faster smoke test, reduce the subset size:

```bash
python -m checker2.compare_variants --max-images 600
```

### Run Checker 1 Research Comparison

To compare text-model variants for the Checker 1 pedagogical error classifier:

```bash
cd /path/to/AiLessonStudio
conda activate research          # or: source .venv/bin/activate
python -m checker1.compare_variants \
        --data-file checker1/data/L10_data_labelled.csv \
        --output-dir checker1/experiments/variant_comparison \
        --max-samples 2500 \
        --epochs 3
```

For a faster smoke test:

```bash
python -m checker1.compare_variants --max-samples 800 --epochs 1
```

---

## How to Use

### API mode (step-by-step)

1. Make sure `api_keys.txt` is configured
2. Select **API mode** in the sidebar
3. Enter **Question**, **Subject**, **Grade** (7–12), **Language**, and select providers
4. Click **Generate Explanation** → explanation, quiz, and sources are generated
5. Switch to the **📝 Quiz** and **📚 Resources** tabs to review
6. Click **Check Answers** in **📝 Quiz** to see concept-level weakness diagnostics and recommended next actions
7. *(Optional)* Click **Generate Images & Video** → Checker 1 and Checker 2 run, then frames + video are produced
8. View frames and videos in the **📖 Lesson** tab

### Demo mode (no API required)

1. Select **Demo mode (no API)** in the sidebar
2. The curated demo is auto-selected
3. Click **Load Demo** → explanation, frames, video, quiz, and sources are loaded from disk
4. Browse all 4 tabs

---

## App Layout

**Sidebar** — always visible  
Mode toggle · Question input · Settings (subject, grade, language, providers) · Cost estimate · Action buttons

**Main area — 4 tabs**

| Tab | Content |
|-----|---------|
| 📖 **Lesson** | Step 1: explanation · Step 2: run summary + frames viewer + video player |
| 📝 **Quiz** | 5 MCQs with confidence sliders, Check Answers, Reset, score, and weakness diagnostics |
| 📚 **Resources** | Learning sources + download buttons (ZIP, MP4, TXT, MD) |
| ℹ️ **Details** | Checker 1 rounds, Checker 2 frame quality, and full analyzer JSON |

---

## Output Structure

Each pipeline run produces a folder under `output/`:

```
output/<question_id>/
├── plan.json
├── run_manifest.json
├── frames/step_01.png … step_07.png
├── storyboard.mp4
├── storyboard.gif
├── sources.md
├── quiz.md
├── student_analyzer.json
├── voiceover_clean.mp3
└── single_api_video/
    ├── single_api_video.mp4
    ├── single_api_video_captioned.mp4
    └── single_api_video_captioned_with_voiceover.mp4
```

Demo folders must start with `good_` or contain `demo` to be auto-detected by the app.

---

## Cost Estimates (approximate)

| Action | OpenAI | DeepSeek |
|--------|--------|---------|
| Generate Explanation + Quiz + Sources | ~$0.02 | ~$0.006 |
| Generate Images & Video (7 frames) | ~$0.56 | N/A |
| Sora single video | ~$0.50–$2.00 | N/A |

The app displays a live estimate in the sidebar before any generation.

---

## Troubleshooting

### Buttons disabled in API mode

Check that `api_keys.txt` exists and has a valid key, or that `OPENAI_API_KEY` is set:

```bash
python -c "from pipeline.api_keys import available_text_providers; print(available_text_providers())"
```

Expected: `['openai']` or `['openai', 'deepseek']`

### DeepSeek errors

The app uses `chat.completions.create` for DeepSeek (not `responses.create`). The `chat_completion()` helper in `clients.py` handles this automatically.

### Demo not appearing

The app looks for folders starting with `good_` or containing `demo` that have a playable video (`storyboard.mp4` or `single_api_video/*.mp4`). Rename the folder or ensure the video file exists.

### `ModuleNotFoundError`

You are in the wrong Python environment. Activate the correct one and reinstall:

```bash
conda activate research
pip install -r requirements.txt
```

### ffmpeg not found (no audio in video)

```bash
brew install ffmpeg
```

---

## Reset Environment

```bash
conda activate research
pip install --upgrade -r requirements.txt
```

Or with venv:

```bash
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Quick Start

```bash
conda activate research
cd /path/to/L15
# Add your API key
echo "openai=sk-..." > api_keys.txt
# Launch
python -m streamlit run streamlit_app.py
```

---

## Full Technical Reference

See [PIPELINE.md](PIPELINE.md) for:
- Multi-provider architecture
- Checker 1 model architecture and inference math
- Planner prompt contracts
- Specificity and relevance scoring formulas
- Sora video pipeline details
- Failure modes and fallback strategy


- API mode: generates explanation → checks quality with Checker 1 → generates storyboard + single API video.
- Demo mode (no API): loads and plays saved demo outputs from `output/`.

---

## 1. Project Structure

- `streamlit_app.py`: main Streamlit app.
- `requirements.txt`: Python dependencies.
- `pipeline/`: planning, image, checker, and storyboard generation pipeline.
  - `checker.py`: Checker 1 — DistilBERT error-type classifier + GPT repair loop.
  - `planner.py`: Stage 1 — question + explanation + grade → 7-step plan.
  - `image_pipeline.py`: Stage 2 — plan → 7 PNG frames.
  - `video_pipeline.py`: Stage 3 — frames + audio → MP4.
  - `pipeline.py`: end-to-end orchestrator.
  - `config.py`: global constants (models, paths, checker config).
- `checker1/`: trained Checker 1 model, training data, notebooks, and docs.
  - `model/distilbert_error_type_ckpt/checkpoint-360/`: best DistilBERT checkpoint (macro-F1=0.90).
  - `data/`: training CSVs and pickled train/test splits.
  - `notebooks/`: training code (L9/L10 notebooks) and DeepSeek bad-example generation.
  - `docs/`: documentation PDFs.
- `single_api_video.py`: single-video API generation + postprocessing helpers.
- `output/`: generated and saved demo runs.

---

## 2. Pipeline Overview

```
Question + Grade + Subject
        │
        ▼
  GPT generates explanation
        │
        ▼
  ┌─────────────────────────────┐
  │  Checker 1 (DistilBERT)     │
  │  Detects error type:        │
  │  ConceptError, GradeMismatch│
  │  LogicalGap, MisleadingAnalogy│
  │  MissingCondition           │
  └──────────┬──────────────────┘
             │ confidence ≥ 0.5?
     ┌───────┴───────┐
     │ Yes           │ No
     ▼               ▼
  GPT repairs     Accept explanation
  explanation       │
     │               │
     └───────┬───────┘
             ▼
     (re-check up to 3 rounds)
             │
             ▼
   Planner → 7-step storyboard plan
             │
             ▼
   Image pipeline → 7 PNG frames
             │
             ▼
   Video pipeline → GIF + MP4
```

---

## 2. Prerequisites

- macOS or Linux terminal
- Python 3.11+ (3.12 is fine)
- Internet connection for API mode
- OpenAI API key for API mode- PyTorch + Transformers for Checker 1 (installed via requirements.txt)
---

## 3. Setup (Recommended: Local `.venv`)

Run these commands from the project root:

```bash
cd /path/to/L15
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Verify key packages:

```bash
python -c "import openai, streamlit, PIL, imageio; print('ok', openai.__version__)"
```

Expected output starts with `ok`.

---

## 4. Configure API Key (API Mode Only)

Set your API key in the same terminal session before launching:

```bash
export OPENAI_API_KEY="your_api_key_here"
```

Optional (persistent): add to your shell profile (`~/.zshrc`), then restart terminal.

---

## 5. Run the App

Always run Streamlit from the local env:

```bash
cd /path/to/AiLessonStudio
. .venv/bin/activate
streamlit run streamlit_app.py
```

If `streamlit` command is not found, run:

```bash
python -m streamlit run streamlit_app.py
```

---

## 6. How to Use the App

### API mode workflow

1. Select `API mode`.
2. Enter `Question`, `Subject`, `Grade`.
3. Click `Generate Explanation`.
4. Click `Generate Images and Videos`.
   - Checker 1 runs automatically on the explanation.
   - If an error is detected, GPT repairs the explanation before proceeding.
   - Checker results are shown in the expandable "Checker 1 Results" panel.
5. Review generated frames and videos in the lower section.

Notes:

- Sequence is enforced: explanation must be generated first.
- If inputs change, regenerate explanation before video generation.

### Demo mode workflow (no API)

1. Select `Demo mode (no API)`.
2. If multiple demos exist, select one saved demo.
3. Click `Generate Explanation` (uses saved canonical explanation).
4. Click `Generate Images and Videos` (loads saved outputs; no API call).
5. Review frames and videos.

---

## 7. Output Locations

Default output root used by app internals:

- `output/` (relative to project root)

Each run folder contains files like:

- `plan.json`
- `frames/step_*.png`
- `storyboard.mp4`
- `single_api_video/single_api_video*.mp4` (if API single video was generated)

---

## 8. Troubleshooting

### A) `ModuleNotFoundError: No module named 'openai'`

Cause: running app in wrong Python environment.

Fix:

```bash
cd /path/to/AiLessonStudio
. .venv/bin/activate
pip install -r requirements.txt
python -c "import openai; print(openai.__version__)"
```

Then relaunch Streamlit from the same terminal.

### B) API mode button disabled

Check both:

- `openai` package is installed in active env.
- `OPENAI_API_KEY` is set.

Verify:

```bash
python -c "import importlib.util, os; print(importlib.util.find_spec('openai') is not None, bool(os.environ.get('OPENAI_API_KEY')))"
```

### C) Demo shows wrong run

The app prioritizes curated folders (for example names starting with `good_` or containing `demo`) and runs with playable media.

If needed, clean or rename folders under `output/` so demo selection is unambiguous.

### D) No videos displayed

Check whether saved artifacts exist in selected run:

- `storyboard.mp4`
- `single_api_video/single_api_video*.mp4`

---

## 9. Reinstall / Reset Environment

If dependencies become inconsistent:

```bash
cd /path/to/L15
rm -rf .venv
python3 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 10. Quick Start (Copy/Paste)

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
export OPENAI_API_KEY="your_api_key_here"
python -m streamlit run streamlit_app.py
```

Checker 2 research comparison:

```bash
python -m checker2.compare_variants \
        --dataset-dir checker2/Flickr8k \
        --output-dir checker2/experiments/clip_variant_comparison \
        --max-images 1200
```

Checker 1 research comparison:

```bash
python -m checker1.compare_variants \
        --data-file checker1/data/L10_data_labelled.csv \
        --output-dir checker1/experiments/variant_comparison \
        --max-samples 2500 \
        --epochs 3
```
