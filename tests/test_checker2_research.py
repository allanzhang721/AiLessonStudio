import csv
import tempfile
import unittest
from pathlib import Path

from checker2.dataset import build_pair_examples, load_flickr8k_captions, prepare_flickr8k_subset
from checker2.metrics import best_threshold, classification_metrics, roc_curve


class Checker2ResearchTests(unittest.TestCase):
    def _make_tiny_flickr8k(self, root: Path) -> tuple[Path, Path]:
        images_dir = root / "Images"
        images_dir.mkdir(parents=True, exist_ok=True)
        captions_file = root / "captions.txt"
        rows = [
            ("img_1.jpg", "a red ball"),
            ("img_1.jpg", "a bright red ball"),
            ("img_2.jpg", "a blue car"),
            ("img_2.jpg", "a small blue car"),
            ("img_3.jpg", "a green tree"),
            ("img_3.jpg", "a tall green tree"),
            ("img_4.jpg", "a yellow house"),
            ("img_4.jpg", "a yellow home"),
        ]
        with captions_file.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["image", "caption"])
            writer.writerows(rows)
        for name in {row[0] for row in rows}:
            (images_dir / name).write_bytes(b"fake")
        return captions_file, images_dir

    def test_load_and_build_pairs(self):
        with tempfile.TemporaryDirectory() as td:
            captions_file, images_dir = self._make_tiny_flickr8k(Path(td))
            grouped = load_flickr8k_captions(captions_file=captions_file, images_dir=images_dir)
            self.assertEqual(len(grouped), 4)
            examples = build_pair_examples(grouped, ["img_1.jpg", "img_2.jpg"], images_dir=images_dir, split_name="train", negatives_per_positive=1, seed=7)
            positives = [example for example in examples if example.label == 1]
            negatives = [example for example in examples if example.label == 0]
            self.assertEqual(len(positives), len(negatives))
            self.assertTrue(all(example.split == "train" for example in examples))

    def test_prepare_subset_writes_summaries(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            captions_file, images_dir = self._make_tiny_flickr8k(root)
            prepared = prepare_flickr8k_subset(
                captions_file=captions_file,
                images_dir=images_dir,
                max_images=4,
                negatives_per_positive=1,
                seed=1,
                output_dir=root / "pairs",
            )
            self.assertIn("summary", prepared)
            self.assertTrue((root / "pairs" / "train_pairs.csv").exists())

    def test_metrics_helpers(self):
        labels = [1, 1, 0, 0]
        scores = [0.9, 0.8, 0.2, 0.1]
        tau, best = best_threshold(labels, scores)
        metrics = classification_metrics(labels, scores, tau)
        roc = roc_curve(labels, scores)
        self.assertGreaterEqual(float(best["f1"]), 0.99)
        self.assertGreaterEqual(float(metrics["accuracy"]), 0.99)
        self.assertGreaterEqual(float(roc["auroc"]), 0.99)


if __name__ == "__main__":
    unittest.main()