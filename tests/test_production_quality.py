import json
import unittest

from pipeline.lesson_service import curated_resources, normalize_lesson_bundle
from pipeline.quality_gates import local_explanation_review, review_explanation


class FakeResponses:
    def __init__(self, payloads):
        self.payloads = iter(payloads)

    def create(self, **kwargs):
        class Response:
            pass
        response = Response()
        response.output_text = json.dumps(next(self.payloads))
        return response


class FakeClient:
    def __init__(self, payloads):
        self.responses = FakeResponses(payloads)


class ProductionQualityTests(unittest.TestCase):
    def _bundle(self):
        return {
            "title": "Forces",
            "learning_objective": "Explain the relationship.",
            "explanation": " ".join(["Force changes motion through acceleration."] * 20),
            "key_ideas": ["Force can change motion."],
            "worked_example": "A cart accelerates when pushed.",
            "common_mistake": "Mass and weight are not identical.",
            "quick_check": "What changes acceleration?",
            "quiz": [
                {"question": f"Question {i}", "choices": ["One", "Two", "Three", "Four"], "answer": "A", "explanation": "One is correct.", "concept": "force"}
                for i in range(5)
            ],
        }

    def test_bundle_normalization(self):
        result = normalize_lesson_bundle(self._bundle())
        self.assertEqual(len(result["quiz"]), 5)
        self.assertGreaterEqual(len(result["explanation"].split()), 60)

    def test_curated_resources_are_real_fixed_urls(self):
        resources = curated_resources("Physics", "Why does force accelerate mass?")
        self.assertTrue(any(item["name"] == "PhET" for item in resources))
        self.assertTrue(all(item["url"].startswith("https://") for item in resources))

    def test_local_gate_rejects_thin_answer(self):
        result = local_explanation_review("Why does gravity act?", "Because it does.", 10, "Physics")
        self.assertFalse(result["pass"])
        self.assertTrue(result["issues"])

    def test_llm_gate_repairs_then_passes(self):
        bad = "This is much too short."
        fixed = " ".join(["Gravity attracts masses and changes their motion through acceleration."] * 15)
        client = FakeClient([
            {"scores": {"accuracy": 2, "completeness": 2, "logical_flow": 2, "grade_fit": 3, "clarity": 3}, "issues": ["Incomplete"], "pass": False, "revised_explanation": fixed},
            {"scores": {"accuracy": 4, "completeness": 4, "logical_flow": 4, "grade_fit": 4, "clarity": 4}, "issues": [], "pass": True, "revised_explanation": ""},
        ])
        result = review_explanation(client, model="test", question="Why does gravity attract mass?", explanation=bad, grade=10, subject="Physics", max_repairs=1)
        self.assertTrue(result["pass"])
        self.assertTrue(result["was_revised"])
        self.assertEqual(result["total_rounds"], 2)


if __name__ == "__main__":
    unittest.main()
