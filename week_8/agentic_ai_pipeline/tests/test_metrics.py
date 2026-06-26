"""Tests for completion-rate and cost aggregation (Q10)."""

import unittest

from agentic_pipeline import MetricsCollector, Pipeline


class TestMetrics(unittest.TestCase):
    def setUp(self):
        self.pipe = Pipeline()
        self.collector = MetricsCollector()

    def _run(self, query, expected=None):
        return self.pipe.run(query, expected_route=expected, collector=self.collector)

    def test_completion_rate_and_counts(self):
        self._run("calculate 2 + 2")                          # success
        self._run("extract keywords from: agents tools routing")  # success
        self._run("calculate 1 / 0")                          # failure
        self.assertEqual(self.collector.runs, 3)
        self.assertEqual(self.collector.completed, 2)
        self.assertAlmostEqual(self.collector.completion_rate, 2 / 3)

    def test_cost_accumulates(self):
        self._run("calculate 2 + 2")        # calculator cost 1.0
        self._run("hello")                  # general cost 0.5
        self.assertAlmostEqual(self.collector.total_cost, 1.5)
        self.assertAlmostEqual(self.collector.avg_cost, 0.75)

    def test_by_intent_breakdown(self):
        self._run("calculate 2 + 2")
        self._run("calculate 3 + 3")
        self._run("hello there")
        self.assertEqual(self.collector.by_intent["calculate"], 2)
        self.assertEqual(self.collector.by_intent["general"], 1)

    def test_summary_keys(self):
        self._run("calculate 2 + 2")
        s = self.collector.summary()
        for key in ("runs", "completion_rate", "avg_cost", "total_cost",
                    "avg_steps", "by_intent"):
            self.assertIn(key, s)


if __name__ == "__main__":
    unittest.main()
