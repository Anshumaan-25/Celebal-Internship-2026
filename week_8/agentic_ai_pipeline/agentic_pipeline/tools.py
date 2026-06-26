"""The three tools the agent can route to (quiz Q3, Q6, Q8).

Every tool is a :class:`Tool`: it advertises a JSON schema for its input and
output, validates both on the way through, charges a notional ``cost``, and
raises :class:`ToolError` (never a bare exception) when it cannot do its job.
That uniform contract is what lets the graph treat tools interchangeably and
what lets the retry loop reason about failures.
"""

from __future__ import annotations

import ast
import operator
import re
import time
from typing import Any, Callable

from .errors import ToolError
from .schemas import validate_or_raise


class Tool:
    """Base class: a validated, costed, uniformly-callable capability.

    Subclasses implement :meth:`run`; callers use ``tool(payload)`` which wraps
    ``run`` with input/output schema validation and an optional simulated
    latency (handy for the parallel-vs-sequential demo).
    """

    name: str = "tool"
    description: str = ""
    input_schema: dict[str, Any] = {"type": "object"}
    output_schema: dict[str, Any] = {"type": "object"}
    cost: float = 1.0  # notional units charged per successful call (quiz Q10)

    def __call__(self, payload: dict[str, Any], *, latency: float = 0.0) -> dict[str, Any]:
        validate_or_raise(payload, self.input_schema, what=f"{self.name} input")
        if latency:
            # Purely to make concurrency observable in the parallel demo.
            time.sleep(latency)
        result = self.run(payload)
        validate_or_raise(result, self.output_schema, what=f"{self.name} output")
        return result

    def run(self, payload: dict[str, Any]) -> dict[str, Any]:  # pragma: no cover
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Calculator
# ---------------------------------------------------------------------------

# Only these AST node types are allowed to evaluate, which is what makes the
# calculator safe: arbitrary names, attribute access, and function calls are
# rejected, so this is *not* ``eval`` on untrusted input.
_BIN_OPS: dict[type, Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS: dict[type, Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _safe_eval(node: ast.AST) -> float:
    """Recursively evaluate a parsed arithmetic expression, nothing else."""
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ToolError(f"unsupported constant: {node.value!r}", recoverable=False)
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_safe_eval(node.operand))
    raise ToolError(
        f"unsupported expression element: {type(node).__name__}", recoverable=False
    )


class CalculatorTool(Tool):
    """Evaluate an arithmetic expression safely."""

    name = "calculator"
    description = "Evaluate an arithmetic expression (+ - * / // % ** and parens)."
    cost = 1.0
    input_schema = {
        "type": "object",
        "required": ["expression"],
        "properties": {"expression": {"type": "string", "minLength": 1}},
    }
    output_schema = {
        "type": "object",
        "required": ["result", "expression"],
        "properties": {
            "result": {"type": "number"},
            "expression": {"type": "string"},
        },
    }

    def run(self, payload: dict[str, Any]) -> dict[str, Any]:
        expr = payload["expression"].strip()
        try:
            tree = ast.parse(expr, mode="eval")
        except SyntaxError as exc:
            # Malformed input is *not* recoverable: retrying the same string
            # will fail identically, so tell the loop not to bother.
            raise ToolError(f"could not parse expression {expr!r}: {exc.msg}",
                            recoverable=False) from exc
        try:
            value = _safe_eval(tree)
        except ZeroDivisionError as exc:
            raise ToolError("division by zero", recoverable=False) from exc
        # Normalise 3.0 -> 3 for tidy output while keeping true floats intact.
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        return {"result": value, "expression": expr}


# ---------------------------------------------------------------------------
# Keyword extraction
# ---------------------------------------------------------------------------

_STOPWORDS = frozenset(
    """
    a an and are as at be by for from has have how i in is it its of on or that
    the to was were what when where which who why will with you your we they this
    these those there here do does did can could should would may might must not
    me my our about into over under than then them their if so but al
    """.split()
)
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9'-]*")


class KeywordExtractionTool(Tool):
    """Extract the most salient keywords from a block of text."""

    name = "keyword_extractor"
    description = "Return the top-k frequency-ranked keywords from text."
    cost = 1.5
    input_schema = {
        "type": "object",
        "required": ["text"],
        "properties": {
            "text": {"type": "string", "minLength": 1},
            "top_k": {"type": "integer", "minimum": 1},
        },
    }
    output_schema = {
        "type": "object",
        "required": ["keywords"],
        "properties": {
            "keywords": {"type": "array", "items": {"type": "string"}},
            "scored": {"type": "array"},
        },
    }

    def run(self, payload: dict[str, Any]) -> dict[str, Any]:
        text = payload["text"]
        top_k = int(payload.get("top_k", 5))
        counts: dict[str, int] = {}
        for match in _WORD_RE.finditer(text.lower()):
            word = match.group(0)
            if word in _STOPWORDS or len(word) < 2:
                continue
            counts[word] = counts.get(word, 0) + 1
        if not counts:
            raise ToolError("no extractable keywords found in text", recoverable=False)
        # Rank by frequency, breaking ties alphabetically for determinism.
        ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        top = ranked[:top_k]
        return {
            "keywords": [w for w, _ in top],
            "scored": [{"keyword": w, "count": c} for w, c in top],
        }


# ---------------------------------------------------------------------------
# General response (fallback)
# ---------------------------------------------------------------------------


class GeneralResponseTool(Tool):
    """Rule-based fallback for anything that is not calc or keyword work.

    With no LLM in the loop this is intentionally simple: it recognises a few
    conversational intents and otherwise gives an honest, templated answer.
    The seam is obvious, so swapping in a real model call later is a one-method
    change.
    """

    name = "general_response"
    description = "Produce a templated natural-language response (no LLM)."
    cost = 0.5
    input_schema = {
        "type": "object",
        "required": ["query"],
        "properties": {"query": {"type": "string", "minLength": 1}},
    }
    output_schema = {
        "type": "object",
        "required": ["response"],
        "properties": {"response": {"type": "string"}},
    }

    def run(self, payload: dict[str, Any]) -> dict[str, Any]:
        q = payload["query"].strip()
        low = q.lower()
        if any(g in low for g in ("hello", "hi ", "hey", "good morning", "good evening")):
            text = "Hello! I'm a single-agent pipeline. Ask me to calculate something or to extract keywords."
        elif "help" in low or "what can you do" in low:
            text = (
                "I route each query to one of three tools: a calculator for "
                "arithmetic, a keyword extractor for text, or this general "
                "responder for everything else."
            )
        elif low.endswith("?"):
            text = (
                f"That's a general question: \"{q}\". I don't have an LLM "
                "wired in, so I can't answer open questions — but the routing, "
                "tools, retries, and metrics around me are fully working."
            )
        else:
            text = (
                f"Received: \"{q}\". No calculation or keyword task detected, "
                "so I'm handling it with the general responder."
            )
        return {"response": text}


# ---------------------------------------------------------------------------
# Flaky wrapper — used only to *demonstrate* the retry loop recovering
# ---------------------------------------------------------------------------


class FlakyTool(Tool):
    """Wrap a tool so it fails its first ``fail_times`` calls, then succeeds.

    This exists purely so the retry loop (quiz Q4) can be shown recovering from
    a transient fault deterministically, instead of relying on real network
    flakiness. The failures are flagged ``recoverable=True`` so the loop knows
    retrying is worthwhile.
    """

    def __init__(self, inner: Tool, fail_times: int = 1) -> None:
        self.inner = inner
        self.fail_times = fail_times
        self._calls = 0
        # Mirror the wrapped tool's identity and contract.
        self.name = inner.name
        self.description = f"[flaky x{fail_times}] {inner.description}"
        self.input_schema = inner.input_schema
        self.output_schema = inner.output_schema
        self.cost = inner.cost

    def run(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._calls += 1
        if self._calls <= self.fail_times:
            raise ToolError(
                f"simulated transient failure (attempt {self._calls})",
                recoverable=True,
            )
        return self.inner.run(payload)


# A ready-made registry the rest of the package builds graphs from.
def default_tools() -> dict[str, Tool]:
    """Return a fresh registry of the standard tools, keyed by name."""
    return {
        "calculator": CalculatorTool(),
        "keyword_extractor": KeywordExtractionTool(),
        "general_response": GeneralResponseTool(),
    }
