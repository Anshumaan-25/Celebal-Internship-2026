"""Tests for the input-extraction helpers used by the tool nodes."""

import unittest

from agentic_pipeline.nodes import extract_expression, extract_text_for_keywords


class TestExtractExpression(unittest.TestCase):
    def test_pulls_expression_from_question(self):
        self.assertEqual(extract_expression("what is 5 * 6?"), "5 * 6")

    def test_keeps_parentheses(self):
        self.assertEqual(extract_expression("calculate (3+4)*2"), "(3+4)*2")

    def test_translates_caret_to_power(self):
        self.assertEqual(extract_expression("8 ^ 2"), "8 ** 2")

    def test_strips_trailing_operator(self):
        # "50%" would be a syntax error; trailing operator is trimmed to "50".
        self.assertEqual(extract_expression("calculate 50%"), "50")

    def test_no_digits_yields_empty(self):
        self.assertEqual(extract_expression("calculate the (((  "), "")


class TestExtractTextForKeywords(unittest.TestCase):
    def test_splits_after_from_colon(self):
        self.assertEqual(
            extract_text_for_keywords("extract keywords from: alpha beta gamma"),
            "alpha beta gamma")

    def test_earliest_marker_wins(self):
        # The colon is the true boundary; a later " from " must not hijack it.
        self.assertEqual(
            extract_text_for_keywords("extract keywords: tips from experts now"),
            "tips from experts now")

    def test_falls_back_to_stripping_triggers(self):
        out = extract_text_for_keywords("extract keywords graphs and tools")
        self.assertNotIn("keywords", out.lower())
        self.assertIn("graphs", out)


if __name__ == "__main__":
    unittest.main()
