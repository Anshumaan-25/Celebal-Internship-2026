#!/usr/bin/env python3
"""A guided tour of the pipeline that exercises every quiz concept.

Run from the project root::

    python examples/demo.py

It walks through: conditional routing to all three tools, a retry loop that
*recovers* from a transient fault, a failure that gracefully *gives up*,
sequential-vs-parallel tool calls, full trajectory evaluation, and aggregate
completion-rate / cost metrics.
"""

from __future__ import annotations

import os
import sys

# Make the package importable when run as ``python examples/demo.py``.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentic_pipeline import MetricsCollector, Pipeline  # noqa: E402
from agentic_pipeline.parallel import compare  # noqa: E402
from agentic_pipeline.pipeline import build_graph  # noqa: E402
from agentic_pipeline.state import AgentState  # noqa: E402
from agentic_pipeline.tools import (  # noqa: E402
    CalculatorTool,
    FlakyTool,
    KeywordExtractionTool,
    default_tools,
)
from agentic_pipeline.trajectory import Trajectory, evaluate_trajectory  # noqa: E402


def banner(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def part_1_routing() -> None:
    banner("1. Conditional routing — one agent, three specialists (Q2, Q3, Q5)")
    pipe = Pipeline()
    queries = [
        ("calculate 2 + 3 * 4", "calculator"),
        ("extract keywords from: nodes and edges define a stateful agent graph", "keywords"),
        ("hi there, what can you do?", "general"),
    ]
    for query, expected in queries:
        result = pipe.run(query, expected_route=expected)
        print(f"\n> {query}")
        print(f"  routed to : {result.evaluation['chosen_route']} "
              f"(expected {expected}, correct={result.evaluation['correct_route']})")
        print(f"  response  : {result.response}")
        print(result.trajectory.pretty())


def part_2_retry_recovers() -> None:
    banner("2. Retry loop that RECOVERS from a transient fault (Q4, Q8)")
    # Swap in a calculator that fails its first two calls, then succeeds.
    tools = default_tools()
    tools["calculator"] = FlakyTool(CalculatorTool(), fail_times=2)
    pipe = Pipeline(tools=tools)
    result = pipe.run("calculate 100 / 5", max_retries=3)
    print(f"\n> calculate 100 / 5   (calculator is flaky: fails twice first)")
    print(f"  response : {result.response}")
    print(f"  attempts : {result.state.attempts}   retries: "
          f"{result.trajectory.num_retries}   cost: {result.cost}")
    print(result.trajectory.pretty())


def part_3_gives_up() -> None:
    banner("3. Non-recoverable failure that GIVES UP gracefully (Q8)")
    pipe = Pipeline()
    result = pipe.run("calculate 8 / 0", max_retries=2)  # mathematically undefined
    print(f"\n> calculate 8 / 0    (undefined — retrying can't help)")
    print(f"  status   : {result.status}")
    print(f"  response : {result.response}")
    print(f"  attempts : {result.state.attempts} (no wasted retries on a "
          f"non-recoverable error)")
    print(result.trajectory.pretty())


def part_4_sequential_vs_parallel() -> None:
    banner("4. Sequential vs parallel tool calls (Q7)")
    extractor = KeywordExtractionTool()
    texts = [
        "stateful directed graphs support branching looping and decisions",
        "conditional routing sends each query to the most appropriate tool",
        "retry loops repeat a failed step until it succeeds or budget runs out",
        "trajectory evaluation inspects the whole path not just the final answer",
    ]
    tasks = [(extractor, {"text": t, "top_k": 3}) for t in texts]
    timing = compare(tasks, latency=0.02)  # 20ms simulated work per call
    print(f"\n  {timing['num_tasks']} independent extractions, "
          f"{timing['latency_per_call_ms']:.0f}ms each:")
    print(f"  sequential : {timing['sequential_ms']} ms")
    print(f"  parallel   : {timing['parallel_ms']} ms")
    print(f"  speedup    : {timing['speedup']}x  "
          f"(independent calls overlap; dependent ones could not)")


def part_5_trajectory_eval() -> None:
    banner("5. Trajectory evaluation — judging the path, not just the answer (Q9)")
    pipe = Pipeline()
    # Right answer, but we assert the WRONG expected route to show the evaluator
    # catching a route mismatch even when the response looks fine.
    result = pipe.run("calculate 6 * 7", expected_route="keywords")
    ev = result.evaluation
    print(f"\n> calculate 6 * 7  (we pretend the 'correct' route was 'keywords')")
    print(f"  response       : {result.response}  (looks correct in isolation)")
    print(f"  chosen_route   : {ev['chosen_route']}")
    print(f"  expected_route : {ev['expected_route']}")
    print(f"  correct_route  : {ev['correct_route']}  <-- caught by trajectory eval")
    print(f"  efficiency     : {ev['efficiency_score']}  steps={ev['num_steps']} "
          f"retries={ev['num_retries']}")


def part_6_metrics() -> None:
    banner("6. Aggregate metrics — completion rate & cost (Q10)")
    pipe = Pipeline()
    collector = MetricsCollector()
    batch = [
        ("calculate 2 + 2", "calculator"),
        ("what is 9 / 3", "calculator"),
        ("extract keywords from: agents tools routing metrics", "keywords"),
        ("hello", "general"),
        ("calculate 5 / 0", "calculator"),     # division by zero -> failure
        ("calculate 6 + * 2", "calculator"),   # malformed (has digit) -> failure
    ]
    for query, expected in batch:
        pipe.run(query, expected_route=expected, collector=collector)
    print()
    print(collector.pretty())
    print("\n  (completion rate < 100% because two queries are intentionally "
          "unanswerable;\n   cost reflects every tool call including failed ones.)")


def part_7_low_level_graph() -> None:
    banner("7. The graph engine is reusable on its own (Q1)")
    # Build the graph directly and drive a single state through it, showing the
    # state object accumulating information at each node.
    graph = build_graph(default_tools())
    state = AgentState(query="calculate 3 + 4")
    traj = Trajectory()
    final, traj = graph.invoke(state, traj)
    print(f"\n  final state snapshot: {final.snapshot()}")
    print(f"  evaluation: {evaluate_trajectory(traj, final.status, 'calculator')}")


def main() -> int:
    part_1_routing()
    part_2_retry_recovers()
    part_3_gives_up()
    part_4_sequential_vs_parallel()
    part_5_trajectory_eval()
    part_6_metrics()
    part_7_low_level_graph()
    banner("Demo complete — every quiz concept (Q1–Q10) exercised above.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
