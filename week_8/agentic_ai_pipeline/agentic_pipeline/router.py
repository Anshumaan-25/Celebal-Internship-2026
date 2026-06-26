"""Rule-based intent classification and conditional routing (quiz Q3).

The router is deliberately transparent: a handful of regexes and keyword checks
decide which of three intents a query has, and a fixed mapping turns that intent
into the graph node that handles it. No model, no randomness — the same query
always routes the same way, which is exactly what makes the behaviour testable.
"""

from __future__ import annotations

import re

# Intent constants keep the strings in one place.
INTENT_CALCULATE = "calculate"
INTENT_KEYWORDS = "keywords"
INTENT_GENERAL = "general"

# Maps an intent to the graph node (and tool) that serves it.
INTENT_TO_NODE: dict[str, str] = {
    INTENT_CALCULATE: "calculator",
    INTENT_KEYWORDS: "keywords",
    INTENT_GENERAL: "general",
}

# Signals for the "calculate" intent: explicit verbs or a bare arithmetic
# expression like "12 * (3 + 4)".
_CALC_WORDS = re.compile(r"\b(calculate|compute|evaluate|what\s+is|sum|product)\b", re.I)
_ARITHMETIC = re.compile(r"\d\s*[-+*/%^]\s*\d")

# Signals for the "keywords" intent.
_KEYWORD_WORDS = re.compile(r"\b(keywords?|key\s*terms?|extract|tags?|topics?)\b", re.I)


def classify(query: str) -> str:
    """Return the intent for ``query``: calculate, keywords, or general."""
    text = query.strip()
    if not text:
        # An empty query has no actionable intent; the general responder will
        # produce a sensible "I received nothing" style answer.
        return INTENT_GENERAL

    # Keyword extraction is checked first: a request like "extract keywords from
    # 2 + 2 reasons" is about keywords even though it contains digits.
    if _KEYWORD_WORDS.search(text):
        return INTENT_KEYWORDS

    # Calculate intent requires something to actually compute: either a bare
    # arithmetic expression, or a calc verb *together with* a digit. Gating on a
    # digit stops ordinary questions like "what is your name" (which match the
    # "what is" verb) from being sent to the calculator, where they would fail.
    has_digit = any(ch.isdigit() for ch in text)
    if _ARITHMETIC.search(text) or (_CALC_WORDS.search(text) and has_digit):
        return INTENT_CALCULATE

    return INTENT_GENERAL


def route_for_intent(intent: str) -> str:
    """Map an intent to its handling node, defaulting to the general node."""
    return INTENT_TO_NODE.get(intent, "general")
