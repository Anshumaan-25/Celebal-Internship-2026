"""The node functions and edge routers that make up the agent (quiz Q5).

A single agent here *simulates a team*: distinct nodes play distinct roles —
an analyst (classify intent), a dispatcher (route), three specialists (the
tools), a reviewer (validate), and a writer (respond) — even though one process
runs them all. Each ``*_node`` is a plain ``fn(state) -> state`` callable; each
``*_router`` is a ``fn(state) -> str`` used by a conditional edge.
"""

from __future__ import annotations

import re
from typing import Callable

from .errors import MaxRetriesExceeded, SchemaValidationError, ToolError
from .router import classify, route_for_intent
from .state import AgentState
from .tools import Tool

# --- input preparation -----------------------------------------------------

# A run of characters that can make up an arithmetic expression.
_EXPR_RE = re.compile(r"[-+*/%^().\d\s]+")


def extract_expression(query: str) -> str:
    """Pull the arithmetic expression out of a natural-language query.

    "what is 5 * 6?" -> "5 * 6";  "calculate (3+4)*2" -> "(3+4)*2".
    ``^`` is translated to Python's ``**`` so users can type powers naturally.
    A trailing bare operator / separator is trimmed (best-effort), so a fragment
    like "50%" becomes "50" instead of reaching the parser as a syntax error.
    """
    candidates = [m.group(0).strip() for m in _EXPR_RE.finditer(query)]
    # Keep the longest candidate that actually contains a digit.
    best = max((c for c in candidates if any(ch.isdigit() for ch in c)),
               key=len, default="")
    best = best.replace("^", "**").strip()
    # Strip trailing operators/separators (but never leading ones — a leading
    # '-' is a valid unary minus we must preserve, e.g. "-5 + 3").
    return re.sub(r"[\s+\-*/%(.]+$", "", best).strip()


def extract_text_for_keywords(query: str) -> str:
    """Isolate the content a keyword request is *about*.

    Strips a leading instruction ("extract keywords from: ...") so the
    instruction words don't pollute the extracted keywords.
    """
    lowered = query.lower()
    # Split at the EARLIEST-occurring marker, not the highest-priority one, so a
    # "from"/colon buried in the real content can't hijack the split. Ties favour
    # the earlier marker in this list (the more specific " from:").
    best_idx: int | None = None
    best_len = 0
    for marker in (" from:", " from ", ":"):
        idx = lowered.find(marker)
        if idx != -1 and (best_idx is None or idx < best_idx):
            best_idx, best_len = idx, len(marker)
    if best_idx is not None:
        return query[best_idx + best_len:].strip()
    # No explicit content marker; drop the trigger words and keep the rest.
    return re.sub(r"\b(extract|keywords?|key\s*terms?|tags?|topics?)\b", " ",
                  query, flags=re.I).strip()


# --- role nodes ------------------------------------------------------------


def analyze_node(state: AgentState) -> AgentState:
    """Analyst role: classify the query's intent and pick its route."""
    state.intent = classify(state.query)
    state.route = route_for_intent(state.intent)  # the tool node that handles it
    return state


def route_node(state: AgentState) -> AgentState:
    """Dispatcher role: an explicit, named decision point in the graph.

    It performs no mutation — its purpose is to make the routing decision a
    visible step in the trajectory; the choice itself is made by the conditional
    edge's router (:func:`route_by_intent`).
    """
    return state


def make_tool_node(tool_node_name: str, tool_key: str,
                   prepare: Callable[[str], dict],
                   tools: dict[str, Tool]) -> Callable[[AgentState], AgentState]:
    """Build a specialist node that runs one tool with error handling (Q8).

    ``tool_node_name`` is the node's name in the graph (and what the retry loop
    targets); ``tool_key`` indexes into the ``tools`` registry; ``prepare``
    turns the raw query into the tool's JSON payload.
    """

    def node(state: AgentState) -> AgentState:
        state.attempts += 1
        tool = tools[tool_key]
        state.tool_name = tool.name
        # Every call consumes resources — charge cost even on attempts that fail,
        # so retried runs cost more (the lever quiz Q10 asks us to optimise).
        state.metadata["cost"] = state.metadata.get("cost", 0.0) + tool.cost
        # Rebuild the payload each attempt — deterministic, and lets a future
        # version tweak inputs between retries if it wanted to.
        state.tool_input = prepare(state.query)
        try:
            state.tool_output = tool(state.tool_input)
            state.tool_succeeded = True
        except (ToolError, SchemaValidationError) as exc:
            # Catch *expected* tool failures and turn them into routable state
            # rather than letting them crash the graph (quiz Q8, strategy 1).
            state.tool_succeeded = False
            state.tool_output = None
            recoverable = getattr(exc, "recoverable", False)
            state.metadata["last_error_recoverable"] = recoverable
            state.record_error(f"[{tool.name} attempt {state.attempts}] {exc}")
        return state

    node.__name__ = f"{tool_node_name}_node"
    return node


def validate_node(state: AgentState) -> AgentState:
    """Reviewer role: confirm the tool produced usable output.

    If the tool failed *recoverably* but the retry budget is now exhausted, this
    is specifically the :class:`MaxRetriesExceeded` case — record it so the final
    answer distinguishes "ran out of retries" from "could never have worked".
    """
    if state.tool_succeeded:
        return state
    recoverable = state.metadata.get("last_error_recoverable", False)
    if recoverable and state.retries_remaining == 0:
        state.record_error(str(MaxRetriesExceeded(
            f"{state.tool_name} failed after {state.attempts} attempts; "
            "retry budget exhausted")))
    return state


def respond_node(state: AgentState) -> AgentState:
    """Writer role: turn tool output (or a failure) into the final answer."""
    if state.tool_succeeded and state.tool_output is not None:
        state.response = _format_success(state)
        state.status = "success"
    else:
        reason = state.errors[-1] if state.errors else "unknown error"
        state.response = (
            f"Sorry — I couldn't complete that request via the "
            f"{state.tool_name or 'pipeline'}. Reason: {reason}"
        )
        state.status = "failed"
    return state


def _format_success(state: AgentState) -> str:
    """Render a successful tool output as a natural sentence."""
    out = state.tool_output or {}
    if state.tool_name == "calculator":
        return f"{out.get('expression')} = {out.get('result')}"
    if state.tool_name == "keyword_extractor":
        kws = ", ".join(out.get("keywords", []))
        return f"Top keywords: {kws}"
    # general_response and any future tool that returns a 'response' field.
    return out.get("response", str(out))


# --- edge routers ----------------------------------------------------------


def route_by_intent(state: AgentState) -> str:
    """Conditional-edge router after ``route`` — pick the tool node (Q3).

    Returns the node name already resolved in :func:`analyze_node` via
    ``router.INTENT_TO_NODE``, so that mapping is the single source of truth for
    intent->node and can't drift from a second copy here.
    """
    return state.route or "general"


def validate_router(state: AgentState) -> str:
    """Conditional-edge router after ``validate`` — finish or loop back (Q4).

    Returns ``"respond"`` to finish, or the tool node's name to retry it. The
    retry is taken only when the failure was *recoverable* and there is retry
    budget left; otherwise we stop and let ``respond`` answer gracefully.
    """
    if state.tool_succeeded:
        return "respond"
    recoverable = state.metadata.get("last_error_recoverable", False)
    if recoverable and state.retries_remaining > 0:
        return state.route or "respond"  # loop back to the same specialist
    return "respond"
