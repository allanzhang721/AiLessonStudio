"""
run_demo.py — Minimal command-line entry point for a single end-to-end pipeline run.

Usage:
    python run_demo.py
    python run_demo.py --text-provider deepseek --image-provider openai

Runs the full pipeline on a hardcoded Newton's Second Law question (Grade 11 Physics)
and prints the paths of all output artifacts plus per-stage timing.

The pipeline now includes Checker 1 (DistilBERT error-type classifier) which
validates the explanation quality before storyboard generation. If the checker
detects an error, the selected LLM provider repairs the explanation automatically.

Requires at least one API key configured in api_keys.txt or as an environment variable.
If no key is set, the pipeline runs in placeholder mode (local placeholder images, no TTS).

Supported providers:
  --text-provider   openai | deepseek   (default: openai)
  --image-provider  openai | wanx       (default: openai)

This script is intended as a quick smoke test and demonstration of run_pipeline().
To run with a custom question or grade, call run_pipeline() directly or use
the Streamlit app (streamlit_app.py).
"""

import argparse
from pathlib import Path

from pipeline import run_pipeline


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the L15 pipeline demo")
    parser.add_argument("--text-provider", default="openai", choices=["openai", "deepseek"])
    parser.add_argument("--image-provider", default="openai", choices=["openai", "wanx"])
    args = parser.parse_args()

    result = run_pipeline(
        question="Why does a heavier object need more force to get the same acceleration?",
        explanation="According to Newton's second law, force equals mass times acceleration. For equal acceleration, larger mass requires larger force.",
        grade=11,
        subject="Physics",
        output_root=Path("l15_output"),
        run_openai=True,
        run_checker=True,
        text_provider=args.text_provider,
        image_provider=args.image_provider,
    )

    print("used_openai:", result["used_openai"])
    print("text_provider:", result["text_provider"])
    print("image_provider:", result["image_provider"])
    print("out_dir:", result["out_dir"])
    print("gif:", result["gif_path"])
    print("video:", result["video_path"])
    print("manifest:", result["manifest_path"])
    print("stage_times:", result["stage_times"])
    print("total_seconds:", result["total_seconds"])

    checker = result.get("checker_result")
    if checker:
        print("checker_revised:", checker.get("was_revised", False))
        print("checker_rounds:", checker.get("total_rounds", 0))
        for rnd in checker.get("rounds", []):
            cr = rnd.get("checker_result", {})
            print(f"  round {rnd['round']}: {cr.get('label')} conf={cr.get('confidence', 0):.3f} action={rnd['action']}")
