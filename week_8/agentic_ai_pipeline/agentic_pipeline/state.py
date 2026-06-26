"""The shared, mutable state object that flows through the graph.

This is what makes the pipeline a *stateful* directed graph (quiz Q1): every
node receives the same ``AgentState`` instance, can read what earlier nodes
wrote, and can write fields that later nodes (or a later loop iteration) read.
A plain linear pipeline would just pass a value forward with no memory of how
it got there.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class AgentState:
    """Working memory for a single run through the pipeline.

    Fields are grouped by the lifecycle stage that fills them in:

    * ``query`` is the only required input.
    * ``intent`` / ``route`` are filled by the analyze + route nodes.
    * ``tool_input`` / ``tool_output`` / ``tool_name`` are filled around the
      tool-execution node.
    * ``attempts`` / ``errors`` are the retry-loop bookkeeping.
    * ``response`` / ``status`` are the final outcome.
    """

    # --- input ---------------------------------------------------------------
    query: str

    # --- analysis / routing --------------------------------------------------
    intent: Optional[str] = None          # "calculate" | "keywords" | "general"
    route: Optional[str] = None           # node name the router chose

    # --- tool execution ------------------------------------------------------
    tool_name: Optional[str] = None
    tool_input: dict[str, Any] = field(default_factory=dict)
    tool_output: Optional[dict[str, Any]] = None
    tool_succeeded: bool = False

    # --- retry-loop bookkeeping ---------------------------------------------
    attempts: int = 0                     # how many times the tool has run
    max_retries: int = 2                  # additional attempts after the first
    errors: list[str] = field(default_factory=list)

    # --- outcome -------------------------------------------------------------
    response: Optional[str] = None
    status: str = "pending"               # pending | success | failed

    # --- free-form scratch space --------------------------------------------
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def retries_remaining(self) -> int:
        """How many more attempts the retry loop is still allowed."""
        # The first run is "attempt 1"; retries are the budget on top of it.
        return max(0, (self.max_retries + 1) - self.attempts)

    def record_error(self, message: str) -> None:
        """Append a human-readable error string to the run's error log."""
        self.errors.append(message)

    def snapshot(self) -> dict[str, Any]:
        """A small, copy-safe view of the interesting fields.

        Used by the trajectory logger so each step captures *what the state
        looked like* at that moment without holding a live reference.
        """
        out = self.tool_output
        return {
            "intent": self.intent,
            "route": self.route,
            "tool_name": self.tool_name,
            "attempts": self.attempts,
            "tool_succeeded": self.tool_succeeded,
            "tool_output": dict(out) if isinstance(out, dict) else out,
            "status": self.status,
            "num_errors": len(self.errors),
        }
