r"""Assemble the nodes and edges into a runnable agent pipeline.

``build_graph`` wires the stateful directed graph; ``Pipeline`` is the friendly
entry point that runs a query through it and returns a :class:`RunResult`
bundling the answer, the trajectory, its evaluation, and the cost.

The graph shape (also drawn in ``docs/architecture.md``)::

    analyze -> route -?-> calculator -+
                     \-> keywords  --+-> validate -?-> respond -> END
                     \-> general   --+                  \--(retry)--^
                                                          (back to the tool node)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional

from .errors import PipelineError
from .graph import END, StateGraph
from .metrics import MetricsCollector
from .nodes import (
    analyze_node,
    extract_expression,
    extract_text_for_keywords,
    make_tool_node,
    respond_node,
    route_by_intent,
    route_node,
    validate_node,
    validate_router,
)
from .state import AgentState
from .tools import Tool, default_tools
from .trajectory import Trajectory, evaluate_trajectory


@dataclass
class RunResult:
    """Everything a caller might want to inspect after one run."""

    query: str
    response: Optional[str]
    status: str
    cost: float
    latency_ms: float
    state: AgentState
    trajectory: Trajectory
    evaluation: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "response": self.response,
            "status": self.status,
            "cost": round(self.cost, 3),
            "latency_ms": round(self.latency_ms, 3),
            "evaluation": self.evaluation,
            "trajectory": self.trajectory.to_dict(),
        }


def build_graph(tools: dict[str, Tool] | None = None) -> StateGraph:
    """Construct and wire the stateful directed graph."""
    registry = tools if tools is not None else default_tools()

    # Each specialist node knows which tool it drives and how to shape its input.
    calculator_node = make_tool_node(
        "calculator", "calculator",
        lambda q: {"expression": extract_expression(q)}, registry)
    keywords_node = make_tool_node(
        "keywords", "keyword_extractor",
        lambda q: {"text": extract_text_for_keywords(q), "top_k": 5}, registry)
    general_node = make_tool_node(
        "general", "general_response",
        lambda q: {"query": q}, registry)

    g = StateGraph()
    g.add_node("analyze", analyze_node)
    g.add_node("route", route_node)
    g.add_node("calculator", calculator_node)
    g.add_node("keywords", keywords_node)
    g.add_node("general", general_node)
    g.add_node("validate", validate_node)
    g.add_node("respond", respond_node)

    g.set_entry_point("analyze")
    g.add_edge("analyze", "route")
    # Conditional routing by intent (quiz Q3). route_by_intent returns the node
    # name already resolved from intent in analyze_node, so this mapping is a
    # plain identity over the tool nodes (the allowed routing targets).
    g.add_conditional_edges("route", route_by_intent, {
        "calculator": "calculator",
        "keywords": "keywords",
        "general": "general",
    })
    # Every specialist reports to the reviewer.
    g.add_edge("calculator", "validate")
    g.add_edge("keywords", "validate")
    g.add_edge("general", "validate")
    # The retry loop / cycle (quiz Q4): validate either finishes or loops back
    # to whichever specialist just ran.
    g.add_conditional_edges("validate", validate_router, {
        "respond": "respond",
        "calculator": "calculator",
        "keywords": "keywords",
        "general": "general",
    })
    g.add_edge("respond", END)
    g.validate()
    return g


class Pipeline:
    """A built graph plus convenience methods for running queries."""

    def __init__(self, tools: dict[str, Tool] | None = None) -> None:
        self.tools = tools if tools is not None else default_tools()
        self.graph = build_graph(self.tools)

    def run(
        self,
        query: str,
        *,
        max_retries: int = 2,
        expected_route: str | None = None,
        collector: MetricsCollector | None = None,
    ) -> RunResult:
        """Run one query end-to-end and return a :class:`RunResult`."""
        state = AgentState(query=query, max_retries=max_retries)
        trajectory = Trajectory()
        start = time.perf_counter()
        try:
            state, trajectory = self.graph.invoke(state, trajectory=trajectory)
        except PipelineError as exc:
            # A structural/engine failure (not a tool failure — those are caught
            # inside the nodes). Degrade gracefully rather than crash (quiz Q8).
            state.status = "failed"
            state.record_error(str(exc))
            state.response = f"Pipeline error: {exc}"
        latency_ms = (time.perf_counter() - start) * 1000.0

        evaluation = evaluate_trajectory(trajectory, state.status, expected_route)
        cost = float(state.metadata.get("cost", 0.0))
        result = RunResult(
            query=query,
            response=state.response,
            status=state.status,
            cost=cost,
            latency_ms=latency_ms,
            state=state,
            trajectory=trajectory,
            evaluation=evaluation,
        )
        if collector is not None:
            collector.record(result)
        return result
