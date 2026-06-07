import tempfile
import unittest
from pathlib import Path

from l15_pipeline.image_pipeline import plan_to_images


class RenderFailFastTests(unittest.TestCase):
    def _base_plan(self):
        steps = []
        for i in range(7):
            steps.append(
                {
                    "step_id": i + 1,
                    "goal": f"Goal {i+1}",
                    "delta": f"Add element {i+1} at left/right zone with #2563EB and 46 px label.",
                    "forbidden": [] if i == 0 else ["Do not redraw existing elements."],
                    "keep": [] if i == 0 else ["Keep prior elements fixed."],
                    "add": [f"Add one item near ({100+i*10}, {200+i*10}) with border #1F2937."],
                }
            )

        return {
            "question_id": "q_render_001",
            "question_text": "Question",
            "canonical_answer": "Answer",
            "visual_family": "natural_scene",
            "render_mode": "gpt_edit",
            "scene_bible": {
                "style": "flat-vector 2D educational illustration",
                "layout": {"canvas": "1536x1024 px", "zones": {"DIAGRAM": "objects", "RESERVED": "empty"}},
                "typography": {"labels": "bold >=46 px"},
                "colour_contract": {"primary": "#2563EB"},
                "allowed_visual_elements": ["objects"],
                "forbidden_elements": ["3D perspective"],
            },
            "steps": steps,
            "captions": [f"Caption {i+1}" for i in range(7)],
            "math_elements": [],
        }

    def test_plan_to_images_raises_for_invalid_schema(self):
        plan = self._base_plan()
        del plan["scene_bible"]
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError):
                plan_to_images(plan, Path(td), client=None)

    def test_plan_to_images_raises_for_steps_caption_mismatch(self):
        plan = self._base_plan()
        plan["captions"] = plan["captions"][:6]
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError):
                plan_to_images(plan, Path(td), client=None)


if __name__ == "__main__":
    unittest.main()
