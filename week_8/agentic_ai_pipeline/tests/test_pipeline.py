"""End-to-end tests for the assembled pipeline (Q1–Q5, Q9)."""

import unittest

from agentic_pipeline import Pipeline


class TestPipelineEndToEnd(unittest.TestCase):
    def setUp(self):
        self.pipe = Pipeline()

    def test_calculator_route(self):
        r = self.pipe.run("calculate 2 + 3 * 4", expected_route="calculator")
        self.assertEqual(r.status, "success")
        self.assertEqual(r.response, "2 + 3 * 4 = 14")
        self.assertEqual(r.evaluation["chosen_route"], "calculator")
        self.assertTrue(r.evaluation["correct_route"])
        self.assertTrue(r.evaluation["completed"])

    def test_keywords_route(self):
        r = self.pipe.run(
            "extract keywords from: agents route queries to specialised tools",
            expected_route="keywords")
        self.assertEqual(r.status, "success")
        self.assertTrue(r.response.startswith("Top keywords:"))
        self.assertEqual(r.evaluation["chosen_route"], "keywords")

    def test_general_route(self):
        r = self.pipe.run("hello, what can you do?", expected_route="general")
        self.assertEqual(r.status, "success")
        self.assertEqual(r.evaluation["chosen_route"], "general")

    def test_what_is_question_without_math_goes_general(self):
        # Regression: "what is X?" with no number must be answered by the
        # general responder, not routed to the calculator and failed.
        r = self.pipe.run("what is your name?", expected_route="general")
        self.assertEqual(r.status, "success")
        self.assertEqual(r.evaluation["chosen_route"], "general")

    def test_canonical_happy_path(self):
        r = self.pipe.run("calculate 1 + 1")
        self.assertEqual(
            r.trajectory.path,
            ["analyze", "route", "calculator", "validate", "respond"])

    def test_trajectory_eval_catches_wrong_route(self):
        # Correct answer but asserted against the wrong expected route.
        r = self.pipe.run("calculate 6 * 7", expected_route="keywords")
        self.assertEqual(r.response, "6 * 7 = 42")
        self.assertFalse(r.evaluation["correct_route"])

    def test_to_dict_shape(self):
        r = self.pipe.run("calculate 2 + 2")
        d = r.to_dict()
        for key in ("query", "response", "status", "cost", "evaluation", "trajectory"):
            self.assertIn(key, d)
        self.assertIn("path", d["trajectory"])

    def test_determinism(self):
        a = self.pipe.run("calculate 9 - 4")
        b = self.pipe.run("calculate 9 - 4")
        self.assertEqual(a.response, b.response)
        self.assertEqual(a.trajectory.path, b.trajectory.path)


if __name__ == "__main__":
    unittest.main()
