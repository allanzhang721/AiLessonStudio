import json
import unittest

from pipeline.lesson_service import _cited_markdown, curated_resources, normalize_lesson_bundle, research_lesson_sources
from pipeline.quality_gates import local_explanation_review, review_explanation
from pipeline.markdown_render import preserve_markdown, streamlit_markdown


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
            "why_it_matters": "Forces explain motion in transport and engineering.",
            "prerequisites": ["Velocity", "Mass"],
            "easy_to_confuse": [{"confusion": "Force equals motion", "correction": "Net force changes motion.", "memory_tip": "Force means change."}],
            "connections": ["Connects to momentum."],
            "study_path": ["Sketch the forces.", "Apply the equation."],
            "follow_up_questions": ["What happens with drag?"],
            "quiz": [
                {"question": f"Question {i}", "choices": ["One", "Two", "Three", "Four"], "answer": "A", "explanation": "One is correct.", "concept": "force"}
                for i in range(5)
            ],
        }

    def test_markdown_and_common_latex_delimiters_are_preserved(self):
        source = "First paragraph.\n\n- Key idea\n\n\\[F = ma\\] and \\(a = F/m\\)."
        preserved = preserve_markdown(source)
        rendered = streamlit_markdown(preserved)
        self.assertIn("\n\n- Key idea", preserved)
        self.assertIn("$$\nF = ma\n$$", rendered)
        self.assertIn("$a = F/m$", rendered)
    def test_bundle_normalization(self):
        result = normalize_lesson_bundle(self._bundle())
        self.assertEqual(len(result["quiz"]), 5)
        self.assertGreaterEqual(len(result["explanation"].split()), 60)
        self.assertEqual(result["easy_to_confuse"][0]["memory_tip"], "Force means change.")
        self.assertEqual(len(result["study_path"]), 2)

    def test_curated_resources_are_real_fixed_urls(self):
        resources = curated_resources("Physics", "Why does force accelerate mass?")
        self.assertTrue(any(item["name"] == "PhET" for item in resources))
        self.assertTrue(all(item["url"].startswith("https://") for item in resources))

    def test_citation_annotations_become_clickable_markdown(self):
        text = "Gravity changes motion according to NASA."
        start = text.index("NASA")
        response = {
            "output": [{
                "type": "message",
                "content": [{
                    "type": "output_text",
                    "text": text,
                    "annotations": [{
                        "type": "url_citation",
                        "start_index": start,
                        "end_index": start + 4,
                        "url": "https://nasa.gov/gravity",
                        "title": "NASA gravity overview",
                    }],
                }],
            }],
        }
        markdown, sources = _cited_markdown(response)
        self.assertIn("[NASA](https://nasa.gov/gravity)", markdown)
        self.assertEqual(sources[0]["title"], "NASA gravity overview")

    def test_web_research_requires_search_and_returns_sources(self):
        text = "A source note from NASA."
        start = text.index("NASA")

        class Responses:
            kwargs = None

            def create(self, **kwargs):
                self.kwargs = kwargs
                return {
                    "output": [{"type": "message", "content": [{
                        "type": "output_text",
                        "text": text,
                        "annotations": [{"type": "url_citation", "start_index": start, "end_index": start + 4, "url": "https://nasa.gov/", "title": "NASA"}],
                    }]}],
                }

        class Client:
            responses = Responses()

        client = Client()
        result = research_lesson_sources(client, question="What is gravity?", subject="Physics", grade=10)
        self.assertEqual(result["status"], "ready")
        self.assertEqual(client.responses.kwargs["tool_choice"], "required")
        self.assertEqual(client.responses.kwargs["tools"][0]["type"], "web_search")
        self.assertTrue(result["sources"])
    def test_local_gate_rejects_thin_answer(self):
        result = local_explanation_review("Why does gravity act?", "Because it does.", 10, "Physics")
        self.assertFalse(result["pass"])
        self.assertTrue(result["issues"])
        self.assertEqual(result["metrics"]["model_calls"], 0)
        self.assertIn("overall_score", result)

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
        self.assertEqual(result["metrics"]["model_calls"], 2)
        self.assertEqual(len(result["method_comparison"]), 2)
        self.assertFalse(result["trained_model"]["used"])


if __name__ == "__main__":
    unittest.main()
