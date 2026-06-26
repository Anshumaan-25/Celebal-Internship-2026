"""Tests for the three tools and their schema contracts (Q3, Q6, Q8)."""

import unittest

from agentic_pipeline.errors import SchemaValidationError, ToolError
from agentic_pipeline.tools import (
    CalculatorTool,
    FlakyTool,
    GeneralResponseTool,
    KeywordExtractionTool,
)


class TestCalculator(unittest.TestCase):
    def setUp(self):
        self.calc = CalculatorTool()

    def test_order_of_operations(self):
        self.assertEqual(self.calc({"expression": "2 + 3 * 4"})["result"], 14)

    def test_parentheses_and_power(self):
        self.assertEqual(self.calc({"expression": "(10 - 4) / 2"})["result"], 3)
        self.assertEqual(self.calc({"expression": "2 ** 5"})["result"], 32)

    def test_float_integer_normalisation(self):
        # 8 / 2 == 4.0 but is reported as the int 4.
        out = self.calc({"expression": "8 / 2"})
        self.assertEqual(out["result"], 4)
        self.assertIsInstance(out["result"], int)

    def test_true_float_preserved(self):
        self.assertAlmostEqual(self.calc({"expression": "10 / 4"})["result"], 2.5)

    def test_division_by_zero_is_non_recoverable(self):
        with self.assertRaises(ToolError) as ctx:
            self.calc({"expression": "1 / 0"})
        self.assertFalse(ctx.exception.recoverable)

    def test_malformed_is_non_recoverable(self):
        with self.assertRaises(ToolError) as ctx:
            self.calc({"expression": "7 + +"})
        self.assertFalse(ctx.exception.recoverable)

    def test_rejects_names_and_calls(self):
        # The safe evaluator must refuse anything that is not pure arithmetic.
        for expr in ("__import__('os')", "len([1])", "x + 1"):
            with self.assertRaises(ToolError):
                self.calc({"expression": expr})

    def test_missing_input_field_raises_schema_error(self):
        with self.assertRaises(SchemaValidationError):
            self.calc({})


class TestKeywordExtractor(unittest.TestCase):
    def setUp(self):
        self.kw = KeywordExtractionTool()

    def test_ranks_by_frequency(self):
        text = "agents route queries. agents call tools. tools return data."
        out = self.kw({"text": text, "top_k": 2})
        # 'agents' and 'tools' each appear twice and should rank first.
        self.assertIn("agents", out["keywords"])
        self.assertIn("tools", out["keywords"])
        self.assertEqual(len(out["keywords"]), 2)

    def test_stopwords_removed(self):
        out = self.kw({"text": "the and of to is a graph", "top_k": 5})
        self.assertEqual(out["keywords"], ["graph"])

    def test_empty_of_keywords_raises(self):
        with self.assertRaises(ToolError):
            self.kw({"text": "the and of to is"})

    def test_default_top_k(self):
        out = self.kw({"text": "one two three four five six seven"})
        self.assertLessEqual(len(out["keywords"]), 5)


class TestGeneralResponse(unittest.TestCase):
    def test_returns_response_field(self):
        out = GeneralResponseTool()({"query": "hello"})
        self.assertIn("response", out)
        self.assertTrue(out["response"])


class TestFlakyTool(unittest.TestCase):
    def test_fails_then_succeeds(self):
        flaky = FlakyTool(CalculatorTool(), fail_times=2)
        # First two calls fail (recoverably), third succeeds.
        for _ in range(2):
            with self.assertRaises(ToolError) as ctx:
                flaky({"expression": "1 + 1"})
            self.assertTrue(ctx.exception.recoverable)
        self.assertEqual(flaky({"expression": "1 + 1"})["result"], 2)


if __name__ == "__main__":
    unittest.main()
