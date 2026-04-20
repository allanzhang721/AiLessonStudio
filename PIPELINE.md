# VisualLesson AI — Pipeline Reference

Last updated: 2026-04-20  
Built by **Jiaxing BCOS**

---

## 1. High-level Architecture

VisualLesson AI converts a classroom question into a fully narrated visual lesson through a multi-stage pipeline with three output/diagnostic paths:

| Path | Description |
|------|-------------|
| **Storyboard path** | Explanation → Checker 1 quality gate → 7-frame plan → frame generation → GIF + MP4 |
| **Single API video path** | Reuses anchor frame + plan → Sora continuous clip → caption burn → voiceover mux |
| **Diagnostic path** | Quiz attempts + confidence + timing (+ Checker 2 risk) → Student Weakness Analyzer |

### Key source files

| File | Role |
|------|------|
| `streamlit_app.py` | Streamlit web UI (sidebar controls, 4-tab output layout) |
| `pipeline/pipeline.py` | End-to-end storyboard orchestrator |
| `pipeline/checker.py` | Checker 1 — DistilBERT classifier + GPT repair loop |
| `pipeline/planner.py` | Stage A — structured 7-step plan generation |
| `pipeline/image_pipeline.py` | Stage B — plan → 7 PNG frames |
| `pipeline/video_pipeline.py` | Stage C — frames + audio → GIF + MP4 |
| `pipeline/clients.py` | Provider client factory (OpenAI, DeepSeek, Wanx) |
| `pipeline/api_keys.py` | Key loader from `api_keys.txt` with quote stripping |
| `pipeline/config.py` | Global constants — model names, thresholds |
| `pipeline/prompts.py` | Prompt builders |
| `pipeline/student_analyzer.py` | Concept-level weakness scoring + remediation suggestions |
| `pipeline/validation.py` | Specificity + relevance scoring gates |
| `pipeline/utils.py` | Shared utilities |
| `single_api_video.py` | Single-video API generation + caption + voiceover |

---

## 2. Multi-provider Support

### Text providers

| Provider | Model constant | Key name in `api_keys.txt` |
|----------|---------------|---------------------------|
| OpenAI | `gpt-4o` (`OPENAI_TEXT_MODEL`) | `openai` |
| DeepSeek | `deepseek-chat` (`DEEPSEEK_TEXT_MODEL`) | `deepseek` |

DeepSeek uses `chat.completions.create` via the `chat_completion()` helper in `clients.py`, which tries `responses.create` first and falls back on 404.

### Image providers

| Provider | Model | Key name |
|----------|-------|----------|
| OpenAI | `gpt-image-1` (`OPENAI_IMAGE_MODEL`) | `openai` |
| Wanx | `wanx-v1` | `wanx` |

### API key file

Keys are loaded from `api_keys.txt` at project root (one `key=value` pair per line). Quotes are stripped automatically:

```
openai=sk-...
deepseek=sk-...
```

Environment variables are also accepted as a fallback.

---

## 3. Web UI — Streamlit App

### Layout

The app uses a **sidebar + 4-tab** layout:

**Sidebar** — always visible:
- Mode toggle: `API mode` / `Demo mode (no API)`
- **API mode**: question textarea, subject, grade (7–12), language, text provider, image provider, cost estimate, action buttons
- **Demo mode**: read-only question from saved demo, `Load Demo` button
- Creator credit at bottom

**Main area — 4 tabs**:

| Tab | Content |
|-----|---------|
| 📖 Lesson | Step 1: explanation with provider badge → Step 2: run summary + frames + videos (only visible after explanation exists) |
| 📝 Quiz | Interactive 5-question MCQ with confidence sliders, Check Answers, Reset, score, and weakness diagnostics |
| 📚 Resources | Relevant sources (websites, YouTube, textbooks) + Downloads (ZIP frames, MP4 video, TXT explanation, MD quiz) |
| ℹ️ Details | Checker 1 per-round results, Checker 2 frame quality, and analyzer JSON |

### Workflow

**API mode (sequential steps):**

1. Enter question, subject, grade, language, providers
2. Click **Generate Explanation** → calls LLM, generates sources and quiz in one pass
3. (Optional) Click **Generate Images & Video** → runs full pipeline (checker → plan → frames → video)
4. Click **Check Answers** in Quiz tab → computes concept-level weakness report and suggested interventions

**Demo mode:**

1. Select a saved demo (auto-discovers curated `good_*` or `demo*` folders with playable media)
2. Click **Load Demo** → loads saved explanation, sources, quiz, and media

### Languages supported

English, 中文, Español, Français, Deutsch, 日本語, 한국어

### Session state keys

| Key | Description |
|-----|-------------|
| `workflow_mode` | `"API mode"` or `"Demo mode (no API)"` |
| `question_input` | Question text |
| `subject_input` | Subject string |
| `grade_input` | Integer 7–12 |
| `language` | Selected language |
| `text_provider` | `"openai"` or `"deepseek"` |
| `image_provider` | `"openai"` or `"wanx"` |
| `generated_explanation` | Current explanation text |
| `explanation_signature` | Hash of question/subject/grade (guards stale regeneration) |
| `relevant_sources` | Markdown list of sources |
| `generated_quiz` | Markdown quiz (5 MCQ) |
| `quiz_submitted` | Bool — whether Check Answers was clicked |
| `analyzer_result` | Dict — weakness report (top weak concepts + recommendations) |
| `quiz_attempt_history` | List of parsed attempts (correctness, confidence, response time) |
| `checker_result` | Dict with rounds, labels, confidences |
| `checker2_result` | Dict with frame quality scores and failed steps |
| `active_run_dir` | Path string to current output folder |
| `saved_demo_choice` | Selected demo label key |

---

## 4. Cost Estimation

Approximate pricing used for the in-app estimate (not charged here — shown for planning):

| Provider | Text per 1k tokens | Image per frame |
|----------|--------------------|-----------------|
| OpenAI | $0.005 | $0.08 |
| DeepSeek | $0.0014 | — |
| Wanx | — | $0.02 |

Estimated tokens: ~4,000 per call (planning). 7 frames assumed.

---

## 5. Stage 0 — Checker 1 (DistilBERT Quality Gate)

### Purpose

Detects error types in generated explanations before they reach the image pipeline, then optionally repairs them.

### Model

- Architecture: `distilbert-base-uncased` fine-tuned for 5-class error-type classification
- Best checkpoint: `checkpoint-360` (macro-F1 = 0.9019)
- Trained on human-verified flawed explanations from L9/L10 datasets
- Checkpoint path: `checker1_distilbert_error_type_ckpt/checkpoint-360/`

### Input format

```
Subject: Physics
Grade: 9
Question: Why does higher radiant heat flux reduce ignition time?
Explanation: Higher heat flux increases oxygen concentration, so ignition happens sooner.
```

Tokenized with `max_length=256`.

### Error type labels

| ID | Label | Description |
|----|-------|-------------|
| 0 | ConceptError | Wrong principle or definition applied |
| 1 | GradeMismatch | Concept beyond target grade level |
| 2 | LogicalGap | Missing causal mechanism between premise and conclusion |
| 3 | MisleadingAnalogy | Convincing but incorrect analogy |
| 4 | MissingCondition | Omits key assumptions or limiting conditions |

### Inference math

The input is encoded by DistilBERT into hidden states $H \in \mathbb{R}^{T \times 768}$. The pooled CLS token $h \in \mathbb{R}^{768}$ is projected to logits:

$$
z = Wh + b, \quad W \in \mathbb{R}^{5 \times 768},\ b \in \mathbb{R}^5
$$

Softmax gives class probabilities:

$$
p(y=k \mid x) = \frac{e^{z_k}}{\sum_{j=1}^{5} e^{z_j}}
$$

Prediction: $\hat{y} = \arg\max_k\, p(y=k \mid x)$

### Decision logic

```
confidence ≥ threshold (default 0.5)?
    Yes → flag error type → GPT repair with error-specific instructions
          → re-check → loop up to max_rounds (default 3)
    No  → accept explanation (low confidence = no strong error match)
```

The model was trained only on *inconsistent* explanations. Low confidence on all classes implies the explanation does not match any known error pattern.

### Artifacts

- `checker_result` dict: `rounds`, `total_rounds`, `was_revised`, `final_explanation`
- Each round: `checker_result.label`, `checker_result.confidence`, `checker_result.probabilities`, `action`

---

## 6. Stage A — Plan Generation (Text → 7-Step Plan)

### Entrypoint

`question_explanation_grade_to_plan(question, explanation, grade, subject, client, ...)`

### Input

- question text
- canonical explanation (post-checker)
- grade (int)
- subject string

### Output — `plan.json`

Required keys: `question_id`, `question_text`, `canonical_answer`, `visual_family`, `render_mode`, `scene_bible`, `steps` (list of 7), `captions` (list of 7), `math_elements`

### Flow

1. Build deterministic fallback plan template
2. If client available:
   - Generate pedagogical brief
   - Build strict planner prompt
   - Parse JSON output
   - Normalize schema; fill missing fields from fallback
   - Validate schema
   - Run specificity (≥0.62) and relevance (≥0.45) gates
   - Issue repair/refinement prompts if gates fail
3. If no client or final validation fails → return fallback plan

### Artifacts

- `output/<question_id>/plan.json`
- `output/<question_id>/prompts/planner_prompt.txt`

---

## 7. Stage B — Frame Generation (Plan → 7 PNGs)

### Entrypoint

`plan_to_images(plan, out_dir, client)`

### Flow per step

1. Validate plan schema
2. **Step 1**: `images.generate` with first-frame prompt
3. **Steps 2–7**: `images.edit` (inpainting) using prior raw frame as base
4. Overlay formula tiles from `math_elements` if active
5. Add bottom caption band and step counter
6. Save raw + annotated frame

### Retries

Up to 3 attempts with exponential backoff per OpenAI image call.

### Artifacts

- `output/<question_id>/frames_raw/step_XX.png`
- `output/<question_id>/frames/step_XX.png`
- `output/<question_id>/prompts/step_XX_*.txt`

---

## 8. Stage C — Video Assembly

### Entrypoints

`make_gif(...)`, `build_narration_script(...)`, `synthesize_clean_voiceover(...)`, `images_to_video(...)`

### Flow

1. Build GIF from final frames
2. Build narration script from captions
3. Synthesize TTS voiceover (if OpenAI client available)
4. Build MP4 slideshow from frames
5. Mux with ffmpeg if audio exists

### FPS estimation (when voiceover exists)

Words-to-duration:

$$
T = \max\!\left(14,\ \frac{W}{2.6}\right)
$$

Then:

$$
fps = \mathrm{clip}_{[0.15,\,1.0]}\!\left(\frac{F}{T}\right)
$$

where $W$ = word count, $F$ = number of frames. Falls back to `fps = 1.0` if no voiceover.

### Artifacts

- `output/<question_id>/storyboard.gif`
- `output/<question_id>/storyboard.mp4`
- `output/<question_id>/voiceover_script.txt`
- `output/<question_id>/voiceover_clean.mp3`
- `output/<question_id>/run_manifest.json`

---

## 9. Single API Video Path (Sora)

### Entrypoint

`generate_single_video_from_run_dir(run_dir)`

### Flow

1. Load existing `plan.json` and `step_01.png` anchor frame
2. Resize/letterbox anchor to model-supported resolution
3. Build one timeline prompt from all 7 steps
4. Generate continuous clip with Sora (`sora-2`, 1280×720, 12 s)
5. Burn synchronized caption pages (optional)
6. Synthesize voiceover (optional)
7. Extend video if narration is longer than clip
8. Mux narration audio track

### Artifacts under `output/<question_id>/single_api_video/`

- `single_api_video.mp4`
- `single_api_video_captioned.mp4`
- `single_api_video_captioned_with_voiceover.mp4`
- `single_video_prompt.txt`
- `single_video_job.json`
- `single_video_result.json`
- `voiceover_script.txt`
- `voiceover_clean.mp3`

---

## 10. LLM-generated Auxiliary Content

In addition to the explanation, the app generates per-run:

| Content | Function | Saved to |
|---------|----------|----------|
| Relevant sources | `_generate_sources()` — 5–8 curated URLs + descriptions | `sources.md` |
| Quiz | `_generate_quiz()` — 5 MCQ with answer + explanation using strict parseable template | `quiz.md` |

Both are loaded automatically in Demo mode if the files exist in the run directory.

Quiz parsing in the UI is tolerant to common LLM format variations (for example: `1.`, `Question 1:`, `Q1:` headers; `Answer:` / `Correct Answer:`; `Reason:` / `Explanation:`) to avoid missing question text in API mode.

---

## 11. Student Weakness Analyzer (Quiz Diagnostics)

### Entrypoint

`analyze_student_weakness(attempts, checker2_result=..., top_k=3)`

### Inputs

- Quiz attempts from the UI:
  - parsed question text
  - selected answer correctness
  - confidence slider (1–5)
  - per-question response time
- Optional Checker 2 output (`overall_score`, per-frame pass/fail)

### Core outputs

- `overall_accuracy`
- `overall_avg_response_seconds`
- `content_risk` (derived from Checker 2)
- `top_weak_concepts` (ranked)
- `recommended_actions` per weak concept

### Saved artifact

- `output/<question_id>/student_analyzer.json` (written after quiz submission if a run is active)

---

## 12. Scoring Gates

### Specificity score

Per-step score weighted by: actionable content, positional constraints, style constraints, arrow semantics, one-new-element focus.

$$
S = \frac{1}{N} \sum_{i=1}^{N} \mathrm{clip}_{[0,1]}(s_i), \quad N=7, \quad \text{gate: } S \ge 0.62
$$

Notes:
- Arrow with explicit direction adds score
- Arrow without explicit source/target → penalty; can hard-fail with strict mode

### Relevance score

Keyword overlap between (question + explanation + subject) and plan text, minus leakage penalties:

$$
R = \mathrm{clip}_{[0,1]}\!\left(\text{overlap\_score} - p_{leak} + 0.15\right), \quad \text{gate: } R \ge 0.45
$$

$$
p_{leak} = \min(0.7,\ 0.2h)
$$

where $h$ = count of off-topic leakage terms.

---

## 13. Prompt Design Contracts

Each planner prompt is composed of 5 blocks:

1. Teaching context (question, grade, subject, explanation)
2. Canvas / zone specifications
3. Style notation (`scene_bible`)
4. Plan spec (FORBIDDEN / KEEP / ADD ONE instructions)
5. Strict render contract

Absolute rendering constraints:
- Flat 2D only — no photorealism, no 3D perspective, no shadows
- Bottom caption band reserved — do not place content there during generation
- Edit steps: preserve all prior objects, add only the requested new element
- Arrow semantics: explicit source → target required; no duplicate arrows

---

## 14. Failure Modes and Fallback Strategy

| Stage | Failure | Fallback |
|-------|---------|----------|
| Plan | JSON parse fail / schema fail / gate fail | Repair/refine attempts → deterministic fallback plan |
| Image | OpenAI call fails | Retry up to 3× with backoff → placeholder frames (no client) |
| Video mux | ffmpeg unavailable | Skip mux; keep silent video |
| Single video TTS | API fails | macOS `say` fallback |
| Checker | Model load fails | Skip checker; accept explanation |
| Demo load | `sources.md` or `quiz.md` missing | Sources/quiz silently omitted |
| Quiz parsing | LLM format drift | Robust parser fallback; malformed blocks marked instead of dropped |

---

## 15. Run Output Directory Structure

```
output/<question_id>/
├── plan.json
├── run_manifest.json
├── frames/
│   └── step_01.png … step_07.png
├── frames_raw/
│   └── step_01.png … step_07.png
├── storyboard.gif
├── storyboard.mp4
├── voiceover_script.txt
├── voiceover_clean.mp3
├── sources.md
├── quiz.md
├── student_analyzer.json
├── prompts/
│   ├── planner_prompt.txt
│   └── step_XX_*.txt
└── single_api_video/
    ├── single_api_video.mp4
    ├── single_api_video_captioned.mp4
    ├── single_api_video_captioned_with_voiceover.mp4
    ├── single_video_prompt.txt
    ├── single_video_job.json
    ├── single_video_result.json
    ├── voiceover_script.txt
    └── voiceover_clean.mp3
```

---

## 16. Curated Demo Run

**Folder:** `output/good_ecology_foodweb_species_removal_cascade/`

| Metric | Value |
|--------|-------|
| Specificity score | 0.904 |
| Relevance score | 0.950 |
| Plan time | 101.9 s |
| Image time | 300.8 s |
| Total time | 404.4 s |
| Sora model | sora-2 |
| Video size | 1280×720, 12 s |
| Final video | `single_api_video_captioned_with_voiceover.mp4` |

---

## 17. Practical Tuning Levers

| Lever | Impact |
|-------|--------|
| Planner prompt strictness | Step specificity, arrow clarity |
| Specificity threshold (0.62) | Gating plan quality |
| Arrow hard-enforcement | Prevents ambiguous diagrams |
| Caption length | Affects TTS duration and slideshow fps |
| Sora clip seconds | Video length vs. cost |
| Edit-step ADD constraints | Prevents frame drift/occlusion |
| `max_rounds` in checker | Trade-off: correction quality vs. latency |
| `confidence_threshold` in checker | Sensitivity to errors |
| Analyzer `top_k` | Number of weak concepts surfaced to learner/teacher |

---

## 18. Mental Model

```
Curriculum model   →   Rendering model   →   Packaging model
(pedagogical brief      (generate frame 1,     (caption band,
 + 7-step plan)          then edit × 6)         audio, mux,
                                                 manifests)
```

Quality frontier is controlled primarily by Stage A plan specificity and Stage B edit determinism.


