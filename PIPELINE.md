# VisualLesson AI — Pipeline Reference

Last updated: 2026-04-14  
Built by **Jiaxing BCOS**

---

## 1. High-level Architecture

VisualLesson AI converts a classroom question into a fully narrated visual lesson through a multi-stage pipeline with two output paths:

| Path | Description |
|------|-------------|
| **Storyboard path** | Explanation → Checker 1 quality gate → 7-frame plan → frame generation → GIF + MP4 |
| **Single API video path** | Reuses anchor frame + plan → Sora continuous clip → caption burn → voiceover mux |

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
| 📝 Quiz | Interactive 5-question MCQ with radio buttons, Check Answers, Reset, score display |
| 📚 Resources | Relevant sources (websites, YouTube, textbooks) + Downloads (ZIP frames, MP4 video, TXT explanation, MD quiz) |
| ℹ️ Details | Checker 1 per-round results with probabilities |

### Workflow

**API mode (sequential steps):**

1. Enter question, subject, grade, language, providers
2. Click **Generate Explanation** → calls LLM, generates sources and quiz in one pass
3. (Optional) Click **Generate Images & Video** → runs full pipeline (checker → plan → frames → video)

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
| `checker_result` | Dict with rounds, labels, confidences |
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
| Quiz | `_generate_quiz()` — 5 MCQ with answer + explanation | `quiz.md` |

Both are loaded automatically in Demo mode if the files exist in the run directory.

---

## 11. Scoring Gates

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

## 12. Prompt Design Contracts

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

## 13. Failure Modes and Fallback Strategy

| Stage | Failure | Fallback |
|-------|---------|----------|
| Plan | JSON parse fail / schema fail / gate fail | Repair/refine attempts → deterministic fallback plan |
| Image | OpenAI call fails | Retry up to 3× with backoff → placeholder frames (no client) |
| Video mux | ffmpeg unavailable | Skip mux; keep silent video |
| Single video TTS | API fails | macOS `say` fallback |
| Checker | Model load fails | Skip checker; accept explanation |
| Demo load | `sources.md` or `quiz.md` missing | Sources/quiz silently omitted |

---

## 14. Run Output Directory Structure

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

## 15. Curated Demo Run

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

## 16. Practical Tuning Levers

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

---

## 17. Mental Model

```
Curriculum model   →   Rendering model   →   Packaging model
(pedagogical brief      (generate frame 1,     (caption band,
 + 7-step plan)          then edit × 6)         audio, mux,
                                                 manifests)
```

Quality frontier is controlled primarily by Stage A plan specificity and Stage B edit determinism.


## 1) High-level Architecture

This project has two production paths:

1. Storyboard path (explanation → Checker 1 quality gate → 7-frame plan → frame generation/edit → GIF/MP4)
2. Single API video path (Sora continuous clip → optional caption burn → optional voiceover mux)

Main orchestrator:
- pipeline/pipeline.py

Key modules:
- pipeline/checker.py    ← NEW: DistilBERT error-type classifier + GPT repair loop
- pipeline/planner.py
- pipeline/image_pipeline.py
- pipeline/video_pipeline.py
- pipeline/prompts.py
- pipeline/validation.py
- pipeline/utils.py
- single_api_video.py

## 2) End-to-end Storyboard Pipeline

### Stage 0. Checker 1 — Explanation quality gate

Entrypoint:
- checker1_loop(client, question, explanation, grade, subject, max_rounds, confidence_threshold)

Model:
- Fine-tuned DistilBERT (distilbert-base-uncased) for 5-class error-type classification
- Best checkpoint: checkpoint-360 (macro-F1 = 0.9019)
- Trained on human-verified inconsistent explanations from L9/L10 datasets

Input format:
```
Subject: Physics
Grade: 9
Question: Why does higher radiant heat flux reduce ignition time?
Explanation: Higher heat flux increases oxygen concentration, so ignition happens sooner.
```

Tokenized with max_length=256.

Output — one of 5 error-type labels:

| Label ID | Label | Description |
|----------|-------|-------------|
| 0 | ConceptError | Applies the wrong principle or definition |
| 1 | GradeMismatch | Uses concepts beyond the target grade level |
| 2 | LogicalGap | Jumps from premise to conclusion without mechanism |
| 3 | MisleadingAnalogy | Uses a convincing but incorrect analogy |
| 4 | MissingCondition | Omits key assumptions or limiting conditions |

Inference math:

The input is tokenized and encoded by DistilBERT into hidden states $H \in \mathbb{R}^{T \times 768}$.
The pooled first-token representation $h \in \mathbb{R}^{768}$ is mapped to logits:

$$
z = Wh + b, \quad W \in \mathbb{R}^{5 \times 768}, b \in \mathbb{R}^5
$$

Softmax gives class probabilities:

$$
p(y=k \mid x) = \frac{e^{z_k}}{\sum_{j=1}^5 e^{z_j}}
$$

Prediction: $\hat{y} = \arg\max_k p(y=k \mid x)$

Decision logic:
- If confidence below threshold (default 0.5): **accept** (explanation is likely fine)
- If confidence above threshold: **flag** the error type, ask GPT to repair
- GPT repair uses error-type-specific instructions (see `_REPAIR_INSTRUCTIONS` in checker.py)
- Re-check after repair; loop up to `max_rounds` (default 3)

The model was trained only on *Inconsistent* explanations, so low confidence on
any error class implies the explanation does not strongly match any known error pattern.

Training data generation:
- DeepSeek API was used to generate intentionally flawed explanations under controlled prompts
- For GPT-based pipelines, the same prompt templates work with GPT (see checker1/notebooks/)
- Human annotators verified all labels; GPT-generated labels were treated as hypotheses only

Artifacts:
- checker_result dict in run_manifest.json (rounds, labels, confidences, revisions)

### Stage A. Plan generation (text → structured 7-step plan)

Entrypoint:
- question_explanation_grade_to_plan(...)

Input:
- question
- canonical explanation
- grade
- subject

Output:
- plan dict with required keys:
  - question_id, question_text, canonical_answer
  - visual_family, render_mode
  - scene_bible
  - steps (exactly 7)
  - captions (exactly 7)
  - math_elements

Flow:
1. Build fallback deterministic plan template.
2. If OpenAI client exists:
   - Generate pedagogical brief.
   - Build strict planner prompt.
   - Parse JSON output.
   - Normalize schema and fill missing fields from fallback.
   - Validate schema.
   - Run specificity and relevance gates.
   - Optionally issue repair/refinement prompts.
3. If no client or final validation fails: return fallback plan.

Artifacts:
- output/<question_id>/plan.json
- output/<question_id>/prompts/planner_prompt.txt (and related debug prompts)

### Stage B. Frame generation/edit (plan -> 7 PNGs)

Entrypoint:
- plan_to_images(plan, out_dir, client)

Flow per step:
1. Validate plan schema.
2. Step 1: images.generate with first-frame prompt.
3. Steps 2-7: images.edit (inpainting) with prior raw frame as base.
4. Overlay formula tiles from math_elements if active.
5. Add bottom caption band and step counter.
6. Save raw + final frame.

Retries:
- Up to 3 attempts with backoff for each OpenAI image call.

Artifacts:
- output/<question_id>/frames_raw/step_XX.png
- output/<question_id>/frames/step_XX.png
- output/<question_id>/prompts/step_XX_generate_prompt.txt or step_XX_edit_prompt.txt

### Stage C. Video assembly

Entrypoints:
- make_gif(...)
- build_narration_script(...)
- synthesize_clean_voiceover(...)
- images_to_video(...)

Flow:
1. Build GIF from final frames.
2. Build narration script from captions.
3. Try TTS (if OpenAI client available).
4. Build MP4 slideshow.
5. If audio exists, mux with ffmpeg.

Artifacts:
- output/<question_id>/storyboard.gif
- output/<question_id>/storyboard.mp4
- output/<question_id>/voiceover_script.txt
- output/<question_id>/voiceover_clean.mp3 (if successful)
- output/<question_id>/run_manifest.json

## 3) Prompt Design Contracts

Prompt builder composes 5 blocks:
- teaching context
- canvas/zones
- style notation (scene_bible)
- plan spec (FORBIDDEN / KEEP / ADD ONE)
- strict render contract

Absolute constraints include:
- flat 2D, no photorealism
- no 3D perspective/shadows
- keep bottom reserved area clear during generation
- preserve old objects in edit steps
- add only requested new content
- maintain arrow semantics and avoid duplicates

## 4) Math and Scoring Heuristics

## 4.1 Specificity score

Per-step score is weighted by actionable content, positional constraints, style constraints, arrow semantics, and one-new-element focus.

Overall score:

$$
S = \frac{1}{N} \sum_{i=1}^{N} \mathrm{clip}_{[0,1]}(s_i), \quad N=7
$$

Default gate:

$$
S \ge 0.62
$$

Notes:
- Arrow with explicit direction can add score.
- Arrow without explicit source/target causes penalty and can hard-fail with strict mode.

## 4.2 Relevance score

Keyword overlap between source (question + explanation + subject) and plan text is computed, then leakage penalties (for off-topic coding terms) are applied.

$$
R = \mathrm{clip}_{[0,1]}\left(\text{overlap\_score} - p_{leak} + 0.15\right)
$$

with

$$
p_{leak} = \min(0.7, 0.2h)
$$

where $h$ is leakage term count.

Default gate:

$$
R \ge 0.45
$$

## 4.3 Storyboard FPS estimation (when voiceover exists)

Words-to-duration heuristic:

$$
T = \max\left(14, \frac{W}{2.6}\right)
$$

where $W$ is word count.

Then:

$$
fps = \mathrm{clip}_{[0.15,1.0]}\left(\frac{F}{T}\right)
$$

where $F$ is number of frames.

If no voiceover is available, pipeline uses fps = 1.0.

## 5) Single API Video Pipeline (Sora)

Entrypoint:
- generate_single_video_from_run_dir(...)

Flow:
1. Load existing plan.json and step_01 anchor frame.
2. Resize/letterbox anchor to model-supported size.
3. Build one timeline prompt from all 7 steps.
4. Generate continuous video with Sora.
5. Optional: burn synchronized caption pages.
6. Optional: synthesize voiceover.
7. Extend video if narration longer than clip.
8. Mux narration audio track.

Outputs under:
- output/<question_id>/single_api_video/

Common files:
- single_api_video.mp4
- single_api_video_captioned.mp4
- single_api_video_captioned_with_voiceover.mp4
- single_video_prompt.txt
- single_video_job.json
- single_video_result.json
- voiceover_script.txt
- voiceover_clean.mp3

## 6) Current Project Outputs (as of this snapshot)

Detected run directories:
- output/good_ecology_foodweb_species_removal_cascade/
- output/q_why_can_removing_one_species_from_a_food_ec6a8e/

### Curated run summary

Run:
- output/good_ecology_foodweb_species_removal_cascade/

Manifest highlights:
- used_openai: true
- plan_seconds: 101.862
- images_seconds: 300.792
- gif_seconds: 1.156
- voiceover_seconds: 0.0
- video_seconds: 0.537
- total_seconds: 404.362
- planner source: openai_specificity_warned
- specificity_score: 0.9042857142857142
- relevance_score: 0.9500000000000001

Interpretation:
- Image generation/edit dominates runtime.
- Planning and validation quality is strong.
- Storyboard voiceover was not produced in this run, but single-video branch did produce voiceover.

### Single-video result summary

Run file:
- output/good_ecology_foodweb_species_removal_cascade/single_api_video/single_video_result.json

Highlights:
- model: sora-2
- status: completed
- seconds: 12
- size: 1280x720
- final video path: single_api_video_captioned_with_voiceover.mp4

### Fallback run summary

Run:
- output/q_why_can_removing_one_species_from_a_food_ec6a8e/

Highlights:
- planner source: fallback_no_client
- generic 7-step fallback plan used
- useful for offline/debug baseline behavior

## 7) Failure Modes and Fallback Strategy

Planner:
- JSON parse fail, schema fail, or gate fail -> repair/refine attempts -> fallback if still invalid.

Image stage:
- OpenAI call retries up to 3 times.
- If no client, placeholder frames are generated locally.

Video stage:
- If ffmpeg unavailable, mux step is skipped and silent video is kept.
- For single video, if API TTS fails, macOS say fallback can be used.

## 8) Practical Tuning Levers

Fastest quality-impacting knobs:
1. Planner prompt strictness (coordinates, arrow direction semantics).
2. Specificity threshold and arrow hard-enforcement policy.
3. Caption length (affects TTS duration and slideshow fps target).
4. Sora clip seconds and size choice.
5. Edit-step ADD constraints to avoid drift/occlusion.

## 9) Minimal Mental Model

Think of the system as:

1. Curriculum model (pedagogical brief + structured teaching plan)
2. Rendering model (generate first frame, then deterministic inpaint edits)
3. Packaging model (caption band, audio, mux, manifests)

The quality frontier is mostly controlled by Stage A plan specificity and Stage B edit determinism.
