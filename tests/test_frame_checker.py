import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from PIL import Image

from pipeline.frame_checker import checker2_validate_frames, checker2_validate_lesson_frames
from pipeline.pipeline import run_pipeline


class FrameCheckerTests(unittest.TestCase):
    def test_checker2_rejects_empty_input(self):
        result = checker2_validate_frames([])
        self.assertFalse(result["pass"])
        self.assertIn("error", result)

    def test_checker2_scores_generated_frames(self):
        with tempfile.TemporaryDirectory() as td:
            frame_paths = []
            for i in range(1, 4):
                path = Path(td) / f"step_{i:02d}.png"
                # Simple non-uniform frame to avoid being fully blank.
                img = Image.new("RGB", (1536, 1024), color=(240, 240, 240))
                for x in range(200, 1300):
                    img.putpixel((x, 200 + i * 10), (20, 20, 20))
                img.save(path)
                frame_paths.append(path)

            result = checker2_validate_frames(frame_paths, threshold=0.2)
            self.assertIn("overall_score", result)
            self.assertEqual(len(result["per_frame"]), 3)
            self.assertEqual(result["metrics"]["model_calls"], 0)
            self.assertEqual(result["metrics"]["parallel_workers"], 3)
            self.assertTrue(result["method_comparison"])

    def test_trained_request_falls_back_without_breaking(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "frame.png"
            Image.new("RGB", (1536, 1024), color=(100, 140, 180)).save(path)
            result = checker2_validate_frames([path], threshold=0.0, backend="trained")
            self.assertEqual(result["mode"], "heuristic_fallback")
            self.assertTrue(result["trained_model_requested"])
            self.assertFalse(result["trained_model_used"])
            self.assertEqual(len(result["per_frame"]), 1)

    def test_bad_render_skips_semantic_model_call(self):
        class NeverCall:
            class Responses:
                def create(self, **kwargs):
                    raise AssertionError("semantic model should not be called")
            responses = Responses()

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "blank.png"
            Image.new("RGB", (320, 200), color="white").save(path)
            result = checker2_validate_lesson_frames(
                [path], plan={"question_text": "Why?"}, client=NeverCall()
            )
            self.assertEqual(result["mode"], "technical_block")
            self.assertEqual(result["metrics"]["model_calls"], 0)

    def test_gate2_result_is_published_before_later_media_failure(self):
        gate_result = {"pass": True, "overall_score": 0.9, "mode": "technical_only"}
        received = []
        progress_events = []
        plan = {
            "question_id": "callback_test",
            "question_text": "Why?",
            "canonical_answer": "Because.",
            "captions": ["Frame"],
            "planner_meta": {},
            "render_meta": {},
        }
        with tempfile.TemporaryDirectory() as td, \
             patch("pipeline.pipeline.question_explanation_grade_to_plan", return_value=plan), \
             patch("pipeline.pipeline.plan_to_images", return_value=[Path(td) / "frame.png"]), \
             patch("pipeline.pipeline.checker2_validate_lesson_frames", return_value=gate_result), \
             patch("pipeline.pipeline.make_gif", return_value=Path(td) / "story.gif"), \
             patch("pipeline.pipeline.synthesize_clean_voiceover", side_effect=RuntimeError("audio failed")):
            with self.assertRaisesRegex(RuntimeError, "audio failed"):
                run_pipeline(
                    question="Why?", explanation="A sufficiently complete explanation.", grade=10,
                    output_root=Path(td), run_openai=False, run_checker=False,
                    gate2_callback=received.append,
                    progress_callback=lambda stage, details: progress_events.append(stage),
                )
        self.assertEqual(received, [gate_result])
        self.assertEqual(progress_events, ["storyboard", "visual_check", "preview", "narration"])

if __name__ == "__main__":
    unittest.main()
