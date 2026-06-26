"""Trajectory logging and evaluation (quiz Q9).

A *trajectory* is the ordered list of steps the agent actually took: which node
ran, what the state looked like afterward, whether it errored, how long it took.
Evaluating the trajectory — not just the final answer — is what catches "right
answer, wrong reasons" bugs: a query that reached the correct response but took
the wrong route, or burned three retries it shouldn't have needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class TrajectoryStep:
    """One node execution within a run."""

    index: int
    node: str
    status: str                      # "ok" | "error"
    duration_ms: float
    snapshot: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "node": self.node,
            "status": self.status,
            "duration_ms": round(self.duration_ms, 3),
            "error": self.error,
            "snapshot": self.snapshot,
        }


class Trajectory:
    """An append-only log of the steps taken during one run."""

    def __init__(self) -> None:
        self.steps: list[TrajectoryStep] = []

    def add(
        self,
        node: str,
        status: str,
        duration_ms: float,
        snapshot: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> TrajectoryStep:
        step = TrajectoryStep(
            index=len(self.steps),
            node=node,
            status=status,
            duration_ms=duration_ms,
            snapshot=snapshot or {},
            error=error,
        )
        self.steps.append(step)
        return step

    # --- derived views ----------------------------------------------------
    @property
    def path(self) -> list[str]:
        """The sequence of node names, e.g. ['analyze','route','calculator',...]."""
        return [s.node for s in self.steps]

    @property
    def num_steps(self) -> int:
        return len(self.steps)

    @property
    def had_error(self) -> bool:
        return any(s.status == "error" for s in self.steps)

    @property
    def num_retries(self) -> int:
        """A retry is any execution of a tool node beyond the first."""
        tool_nodes = ("calculator", "keywords", "general")
        seen = 0
        retries = 0
        for s in self.steps:
            if s.node in tool_nodes:
                seen += 1
                if seen > 1:
                    retries += 1
        return retries

    @property
    def total_duration_ms(self) -> float:
        return sum(s.duration_ms for s in self.steps)

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "steps": [s.to_dict() for s in self.steps]}

    def pretty(self) -> str:
        """A compact, human-readable rendering of the path."""
        lines = ["Trajectory:"]
        for s in self.steps:
            mark = "✓" if s.status == "ok" else "✗"
            line = f"  {s.index:>2} {mark} {s.node:<12} ({s.duration_ms:6.2f} ms)"
            if s.error:
                line += f"  ! {s.error}"
            lines.append(line)
        lines.append(f"  path: {' -> '.join(self.path)}")
        return "\n".join(lines)


def evaluate_trajectory(
    trajectory: Trajectory,
    final_status: str,
    expected_route: str | None = None,
) -> dict[str, Any]:
    """Score a trajectory beyond its final output (quiz Q9).

    ``expected_route`` is optional ground truth (the node that *should* have
    handled the query); when supplied we can check the agent didn't merely get
    a plausible answer from the wrong path.
    """
    path = trajectory.path
    completed = final_status == "success" and path[-1:] == ["respond"]

    # Which tool node actually handled the query (last tool seen).
    tool_nodes = ("calculator", "keywords", "general")
    chosen = next((n for n in reversed(path) if n in tool_nodes), None)
    correct_route = None if expected_route is None else (chosen == expected_route)

    # Efficiency: 1.0 for the ideal no-retry/no-error path; each retry and each
    # error chips away at it. Purely a heuristic for at-a-glance comparison.
    efficiency = 1.0 - 0.2 * trajectory.num_retries - 0.1 * (
        sum(1 for s in trajectory.steps if s.status == "error")
    )
    efficiency = max(0.0, round(efficiency, 3))

    return {
        "completed": completed,
        "chosen_route": chosen,
        "expected_route": expected_route,
        "correct_route": correct_route,
        "num_steps": trajectory.num_steps,
        "num_retries": trajectory.num_retries,
        "had_error": trajectory.had_error,
        "efficiency_score": efficiency,
        "path": path,
    }
