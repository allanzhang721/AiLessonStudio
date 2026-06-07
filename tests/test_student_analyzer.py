import unittest

from pipeline.student_analyzer import analyze_student_weakness, infer_concept_tags


class StudentAnalyzerTests(unittest.TestCase):
    def test_infer_concept_tags_has_fallback(self):
        tags = infer_concept_tags("Why?", "", "")
        self.assertTrue(isinstance(tags, list))
        self.assertGreaterEqual(len(tags), 1)

    def test_analyze_student_weakness_ranks_concepts(self):
        attempts = [
            {
                "question_id": "q1",
                "question_text": "Food web effect",
                "concept_tags": ["food_web"],
                "correct": False,
                "response_time_seconds": 42,
                "confidence": 5,
            },
            {
                "question_id": "q2",
                "question_text": "Food web predator removal",
                "concept_tags": ["food_web"],
                "correct": False,
                "response_time_seconds": 39,
                "confidence": 4,
            },
            {
                "question_id": "q3",
                "question_text": "Photosynthesis basics",
                "concept_tags": ["photosynthesis"],
                "correct": True,
                "response_time_seconds": 14,
                "confidence": 3,
            },
        ]

        checker2_result = {
            "overall_score": 0.61,
            "per_frame": [
                {"pass": True},
                {"pass": False},
                {"pass": True},
            ],
        }
        report = analyze_student_weakness(attempts, checker2_result=checker2_result, top_k=2)
        self.assertEqual(report["status"], "ok")
        self.assertTrue(report["top_weak_concepts"])
        self.assertEqual(report["top_weak_concepts"][0]["concept"], "food_web")


if __name__ == "__main__":
    unittest.main()
