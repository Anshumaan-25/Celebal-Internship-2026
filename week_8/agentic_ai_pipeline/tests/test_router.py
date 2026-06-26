"""Tests for rule-based intent classification and routing (Q3)."""

import unittest

from agentic_pipeline.router import (
    INTENT_CALCULATE,
    INTENT_GENERAL,
    INTENT_KEYWORDS,
    classify,
    route_for_intent,
)


class TestClassify(unittest.TestCase):
    def test_calculate_by_verb(self):
        self.assertEqual(classify("calculate 2 plus 2"), INTENT_CALCULATE)
        self.assertEqual(classify("what is 5 times 6"), INTENT_CALCULATE)

    def test_calculate_by_arithmetic(self):
        self.assertEqual(classify("12 * (3 + 4)"), INTENT_CALCULATE)

    def test_keywords(self):
        self.assertEqual(classify("extract keywords from this text"), INTENT_KEYWORDS)
        self.assertEqual(classify("what are the key terms here"), INTENT_KEYWORDS)

    def test_keywords_wins_over_digits(self):
        # A keyword request that happens to contain arithmetic is still keywords.
        self.assertEqual(classify("extract keywords from 2 + 2 reasons"), INTENT_KEYWORDS)

    def test_general_default(self):
        self.assertEqual(classify("hello there"), INTENT_GENERAL)
        self.assertEqual(classify(""), INTENT_GENERAL)

    def test_calc_words_without_digit_are_general(self):
        # A calc trigger word with nothing to compute must NOT go to the
        # calculator (it would fail). "what is X?" is the common offender.
        self.assertEqual(classify("what is your name"), INTENT_GENERAL)
        self.assertEqual(classify("what is the capital of France"), INTENT_GENERAL)
        self.assertEqual(classify("can you compute the total for me"), INTENT_GENERAL)

    def test_route_mapping(self):
        self.assertEqual(route_for_intent(INTENT_CALCULATE), "calculator")
        self.assertEqual(route_for_intent(INTENT_KEYWORDS), "keywords")
        self.assertEqual(route_for_intent(INTENT_GENERAL), "general")
        self.assertEqual(route_for_intent("nonsense"), "general")


if __name__ == "__main__":
    unittest.main()
