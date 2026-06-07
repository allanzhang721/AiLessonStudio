import tempfile
import unittest
from pathlib import Path

from checker1.dataset import build_input_text, prepare_checker1_subset
from checker1.metrics import multiclass_metrics


class Checker1ResearchTests(unittest.TestCase):
    def _write_dataset(self, path: Path) -> None:
        rows = [
            "sample_id,subject,grade,question,explanation,auto_y,label_y\n",
            'Q000_ConceptError_V01,Physics,10,Why?,Bad explanation one,Inconsistent,ConceptError\n',
            'Q000_ConceptError_V02,Physics,10,Why?,Bad explanation two,Inconsistent,ConceptError\n',
            'Q001_LogicalGap_V01,Physics,10,How?,Bad explanation three,Inconsistent,LogicalGap\n',
            'Q001_LogicalGap_V02,Physics,10,How?,Bad explanation four,Inconsistent,LogicalGap\n',
            'Q002_GradeMismatch_V01,Biology,9,What?,Bad explanation five,Inconsistent,GradeMismatch\n',
            'Q003_MissingCondition_V01,Biology,9,What?,Bad explanation six,Inconsistent,MissingCondition\n',
            'Q004_MisleadingAnalogy_V01,Chemistry,11,Explain?,Bad explanation seven,Inconsistent,MisleadingAnalogy\n',
            'Q005_Consistent_V01,Chemistry,11,Explain?,Good explanation,Consistent,Consistent\n',
        ]
        path.write_text("".join(rows), encoding="utf-8")

    def test_prepare_checker1_subset_filters_consistent(self):
        with tempfile.TemporaryDirectory() as td:
            data_file = Path(td) / "mini.csv"
            self._write_dataset(data_file)
            prepared = prepare_checker1_subset(data_file=data_file, max_samples=20, seed=1, output_dir=Path(td) / "splits")
            total_samples = sum(stats["samples"] for stats in prepared["summary"].values())
            self.assertEqual(total_samples, 7)
            self.assertTrue((Path(td) / "splits" / "train_examples.csv").exists())

    def test_build_input_text(self):
        text = build_input_text("Physics", 10, "Why?", "Because")
        self.assertIn("Subject: Physics", text)
        self.assertIn("Grade: 10", text)

    def test_prepare_checker1_subset_with_cp1252_data(self):
        with tempfile.TemporaryDirectory() as td:
            data_file = Path(td) / "latin1.csv"
            rows = [
                "sample_id,subject,grade,question,explanation,auto_y,label_y\n",
                'Q000_ConceptError_V01,Physics,10,Why?,Gravity pulls harder \xb7 wrong,Inconsistent,ConceptError\n',
                'Q001_LogicalGap_V01,Physics,10,How?,Missing step,Inconsistent,LogicalGap\n',
            ]
            data_file.write_bytes("".join(rows).encode("cp1252"))

            prepared = prepare_checker1_subset(data_file=data_file, max_samples=20, seed=1)
            total_samples = sum(stats["samples"] for stats in prepared["summary"].values())

            self.assertEqual(total_samples, 2)

    def test_multiclass_metrics(self):
        metrics = multiclass_metrics([0, 1, 2, 2], [0, 1, 1, 2], 3)
        self.assertGreater(float(metrics["accuracy"]), 0.7)
        self.assertGreater(float(metrics["macro_f1"]), 0.7)


if __name__ == "__main__":
    unittest.main()