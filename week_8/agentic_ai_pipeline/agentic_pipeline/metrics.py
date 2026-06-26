"""Aggregate performance metrics across runs (quiz Q10).

Two headline numbers the assignment asks for:

* **Task completion rate** — the fraction of runs that finished successfully.
* **Cost** — resource units consumed. Here one unit is one tool *call*; because
  every attempt is charged (including retried ones), a run that loops costs more
  than one that succeeds first try. That makes the optimisation levers concrete:
  better routing and fewer needless retries directly lower average cost.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # avoid a runtime import cycle with pipeline.py
    from .pipeline import RunResult


@dataclass
class MetricsCollector:
    """Accumulates per-run results and reports aggregate metrics."""

    runs: int = 0
    completed: int = 0
    total_cost: float = 0.0
    total_steps: int = 0
    total_retries: int = 0
    total_latency_ms: float = 0.0
    by_intent: dict[str, int] = field(default_factory=dict)

    def record(self, result: "RunResult") -> None:
        self.runs += 1
        if result.status == "success":
            self.completed += 1
        self.total_cost += result.cost
        self.total_steps += result.trajectory.num_steps
        self.total_retries += result.trajectory.num_retries
        self.total_latency_ms += result.latency_ms
        intent = result.state.intent or "unknown"
        self.by_intent[intent] = self.by_intent.get(intent, 0) + 1

    # --- headline metrics -------------------------------------------------
    @property
    def completion_rate(self) -> float:
        return self.completed / self.runs if self.runs else 0.0

    @property
    def avg_cost(self) -> float:
        return self.total_cost / self.runs if self.runs else 0.0

    @property
    def avg_steps(self) -> float:
        return self.total_steps / self.runs if self.runs else 0.0

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / self.runs if self.runs else 0.0

    def summary(self) -> dict[str, Any]:
        return {
            "runs": self.runs,
            "completed": self.completed,
            "completion_rate": round(self.completion_rate, 3),
            "avg_cost": round(self.avg_cost, 3),
            "total_cost": round(self.total_cost, 3),
            "avg_steps": round(self.avg_steps, 2),
            "total_retries": self.total_retries,
            "avg_latency_ms": round(self.avg_latency_ms, 3),
            "by_intent": dict(self.by_intent),
        }

    def pretty(self) -> str:
        s = self.summary()
        lines = [
            "Metrics:",
            f"  runs ................ {s['runs']}",
            f"  completed ........... {s['completed']}",
            f"  completion rate ..... {s['completion_rate'] * 100:.1f}%",
            f"  avg cost (units) .... {s['avg_cost']}",
            f"  total cost (units) .. {s['total_cost']}",
            f"  avg steps/run ....... {s['avg_steps']}",
            f"  total retries ....... {s['total_retries']}",
            f"  avg latency ......... {s['avg_latency_ms']} ms",
            f"  by intent ........... {s['by_intent']}",
        ]
        return "\n".join(lines)
