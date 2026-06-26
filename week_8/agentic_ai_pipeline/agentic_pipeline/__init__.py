"""A single-agent pipeline built as a stateful directed graph.

Public API::

    from agentic_pipeline import Pipeline, MetricsCollector

    pipe = Pipeline()
    result = pipe.run("calculate 2 + 3 * 4")
    print(result.response)          # "2 + 3 * 4 = 14"
    print(result.trajectory.pretty())
    print(result.evaluation)

This package is pure standard-library Python and ships no model: routing,
tools, retries, trajectory evaluation, and metrics are all rule-based and
deterministic. See ``docs/architecture.md`` for how each piece maps onto the
Week 8 quiz concepts.
"""

from __future__ import annotations

from .errors import (
    GraphError,
    MaxRetriesExceeded,
    PipelineError,
    SchemaValidationError,
    ToolError,
)
from .graph import END, StateGraph
from .metrics import MetricsCollector
from .pipeline import Pipeline, RunResult, build_graph
from .state import AgentState
from .tools import (
    CalculatorTool,
    FlakyTool,
    GeneralResponseTool,
    KeywordExtractionTool,
    Tool,
    default_tools,
)
from .trajectory import Trajectory, TrajectoryStep, evaluate_trajectory

__version__ = "1.0.0"

__all__ = [
    "Pipeline",
    "RunResult",
    "build_graph",
    "StateGraph",
    "END",
    "AgentState",
    "MetricsCollector",
    "Trajectory",
    "TrajectoryStep",
    "evaluate_trajectory",
    "Tool",
    "CalculatorTool",
    "KeywordExtractionTool",
    "GeneralResponseTool",
    "FlakyTool",
    "default_tools",
    "PipelineError",
    "GraphError",
    "ToolError",
    "SchemaValidationError",
    "MaxRetriesExceeded",
    "__version__",
]
