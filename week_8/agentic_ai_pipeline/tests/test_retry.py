"""Tests for the retry loop / cycle behaviour (Q4, Q8)."""

import unittest

from agentic_pipeline import Pipeline
from agentic_pipeline.tools import CalculatorTool, FlakyTool, default_tools


def _pipeline_with_flaky(fail_times):
    tools = default_tools()
    tools["calculator"] = FlakyTool(CalculatorTool(), fail_times=fail_times)
    return Pipeline(tools=tools)


class TestRetry(unittest.TestCase):
    def test_recovers_within_budget(self):
        pipe = _pipeline_with_flaky(fail_times=1)
        result = pipe.run("calculate 4 + 4", max_retries=2)
        self.assertEqual(result.status, "success")
        self.assertEqual(result.response, "4 + 4 = 8")
        self.assertEqual(result.state.attempts, 2)        # one retry used
        self.assertEqual(result.trajectory.num_retries, 1)

    def test_exhausts_budget_then_fails(self):
        pipe = _pipeline_with_flaky(fail_times=5)           # never succeeds in time
        result = pipe.run("calculate 4 + 4", max_retries=2)
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.state.attempts, 3)          # initial + 2 retries
        self.assertEqual(result.trajectory.num_retries, 2)

    def test_non_recoverable_gives_up_immediately(self):
        pipe = Pipeline()
        # Has a digit (so it routes to the calculator) but is malformed, so the
        # parse error is non-recoverable.
        result = pipe.run("calculate 5 + * 3", max_retries=3)
        self.assertEqual(result.status, "failed")
        # A non-recoverable error must NOT consume retries.
        self.assertEqual(result.state.attempts, 1)
        self.assertEqual(result.trajectory.num_retries, 0)

    def test_division_by_zero_is_not_retried(self):
        pipe = Pipeline()
        result = pipe.run("calculate 5 / 0", max_retries=3)
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.state.attempts, 1)

    def test_exhaustion_records_max_retries_exceeded(self):
        pipe = _pipeline_with_flaky(fail_times=5)
        result = pipe.run("calculate 4 + 4", max_retries=2)
        self.assertEqual(result.status, "failed")
        # The documented MaxRetriesExceeded type is actually recorded, and it
        # surfaces as the final failure reason.
        self.assertTrue(any("retry budget exhausted" in e for e in result.state.errors))
        self.assertIn("retry budget exhausted", result.response)

    def test_retry_increases_cost(self):
        # Each (failed) attempt is charged, so a retried run costs more.
        clean = Pipeline().run("calculate 2 + 2")
        flaky = _pipeline_with_flaky(fail_times=1).run("calculate 2 + 2", max_retries=2)
        self.assertGreater(flaky.cost, clean.cost)


if __name__ == "__main__":
    unittest.main()
