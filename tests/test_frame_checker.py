import tempfile
import unittest
from pathlib import Path

from PIL import Image

from pipeline.frame_checker import checker2_validate_frames


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


if __name__ == "__main__":
    unittest.main()
