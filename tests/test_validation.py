import unittest

from pipeline.validation import passes_relevance_gate, passes_specificity_gate, validate_plan_schema


class ValidationTests(unittest.TestCase):
    def _make_valid_plan(self):
        steps = []
        for i in range(7):
            steps.append(
                {
                    "step_id": i + 1,
                    "goal": f"Goal {i+1}",
                    "delta": f"Add one element at left zone with color #2563EB and label size 46 px for step {i+1}.",
                    "forbidden": [] if i == 0 else ["Do not redraw existing elements."],
                    "keep": [] if i == 0 else ["Keep prior elements fixed."],
                    "add": [f"Add one rounded rectangle near x={100+i*10}, y={200+i*10} with border #1F2937."],
                }
            )

        return {
            "question_id": "q_test_001",
            "question_text": "Why does this happen?",
            "canonical_answer": "Because of causal mechanism.",
            "visual_family": "natural_scene",
            "render_mode": "gpt_edit",
            "scene_bible": {
                "style": "flat-vector 2D educational illustration",
                "layout": {"canvas": "1536x1024 px", "zones": {"DIAGRAM y=0-800": "objects", "RESERVED y=800-1024": "empty"}},
                "typography": {"labels": "bold >=46 px"},
                "colour_contract": {"primary": "#2563EB"},
                "allowed_visual_elements": ["objects"],
                "forbidden_elements": ["3D perspective"],
            },
            "steps": steps,
            "captions": [f"Caption {i+1}" for i in range(7)],
            "math_elements": [],
        }

    def test_validate_plan_schema_passes_for_valid_plan(self):
        plan = self._make_valid_plan()
        valid, errors = validate_plan_schema(plan)
        self.assertTrue(valid)
        self.assertEqual(errors, [])

    def test_validate_plan_schema_fails_for_wrong_step_count(self):
        plan = self._make_valid_plan()
        plan["steps"] = plan["steps"][:6]
        valid, errors = validate_plan_schema(plan)
        self.assertFalse(valid)
        self.assertTrue(any("exactly 7" in e for e in errors))

    def test_specificity_gate_flags_vague_plan(self):
        plan = self._make_valid_plan()
        for step in plan["steps"]:
            step["delta"] = "show concept"
            step["add"] = ["add detail"]
        passed, score, issues = passes_specificity_gate(plan, threshold=0.62)
        self.assertFalse(passed)
        self.assertLess(score, 0.62)
        self.assertTrue(len(issues) > 0)

    def test_specificity_gate_hard_fails_ambiguous_arrow_direction(self):
        plan = self._make_valid_plan()
        plan["steps"][2]["delta"] = "Add acceleration arrow near the cart."
        plan["steps"][2]["add"] = ["Add one acceleration arrow beside the cart label."]
        passed, score, issues = passes_specificity_gate(
            plan,
            threshold=0.62,
            hard_enforce_arrow_direction=True,
        )
        self.assertFalse(passed)
        self.assertTrue(score >= 0.0)
        self.assertTrue(any("arrow direction" in issue.lower() for issue in issues))

    def test_relevance_gate_flags_coding_leakage_for_physics(self):
        plan = self._make_valid_plan()
        plan["visual_family"] = "python_intro"
        plan["steps"][0]["goal"] = "Explain a for loop"
        plan["steps"][0]["delta"] = "Show Python code for a for loop over an array."
        plan["steps"][0]["add"] = ["Add Python code snippet with variable and function labels."]
        passed, score, issues = passes_relevance_gate(
            question="Why does a heavier object need more force to get the same acceleration?",
            explanation="Newton's second law says larger mass needs larger force for the same acceleration.",
            subject="Physics",
            plan=plan,
            threshold=0.45,
        )
        self.assertFalse(passed)
        self.assertLess(score, 0.45)
        self.assertTrue(any("coding" in issue.lower() or "keyword overlap" in issue.lower() for issue in issues))


if __name__ == "__main__":
    unittest.main()
