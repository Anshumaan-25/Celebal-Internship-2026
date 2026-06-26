#!/usr/bin/env python3
"""Command-line entry point for the agentic pipeline.

Examples
--------
    python main.py "calculate 2 + 3 * 4"
    python main.py "extract keywords from: agent pipelines route queries to tools"
    python main.py "hello there"
    python main.py --json "what is 10 / 4"
    python main.py --batch          # run a small batch and print aggregate metrics
"""

from __future__ import annotations

import argparse
import json
import sys

from agentic_pipeline import MetricsCollector, Pipeline

# A small, representative batch used by --batch (and handy for a quick demo).
_BATCH = [
    ("calculate 2 + 3 * 4", "calculator"),
    ("what is (10 - 4) / 2", "calculator"),
    ("extract keywords from: stateful directed graphs route queries to tools", "keywords"),
    ("hello, what can you do?", "general"),
    ("compute 8 ^ 2 + 1", "calculator"),
]


def _run_one(pipe: Pipeline, query: str, *, max_retries: int, as_json: bool,
             show_trajectory: bool, quiet: bool) -> None:
    result = pipe.run(query, max_retries=max_retries)
    if as_json:
        print(json.dumps(result.to_dict(), indent=2))
        return
    if quiet:
        print(result.response)
        return
    print(f"Query     : {result.query}")
    print(f"Intent    : {result.state.intent}  ->  route: {result.evaluation['chosen_route']}")
    print(f"Response  : {result.response}")
    print(f"Status    : {result.status}   cost: {result.cost}   "
          f"latency: {result.latency_ms:.2f} ms")
    if show_trajectory:
        print(result.trajectory.pretty())
    print(f"Evaluation: {json.dumps(result.evaluation)}")


def _run_batch(pipe: Pipeline, *, max_retries: int) -> None:
    collector = MetricsCollector()
    for query, expected in _BATCH:
        result = pipe.run(query, max_retries=max_retries,
                          expected_route=expected, collector=collector)
        ok = "✓" if result.status == "success" else "✗"
        print(f"  {ok} {query!r:<62} -> {result.response}")
    print()
    print(collector.pretty())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="main.py", description="Run a query through the agentic pipeline.")
    parser.add_argument("query", nargs="?", help="the query to process")
    parser.add_argument("--json", action="store_true", help="emit the full result as JSON")
    parser.add_argument("--trajectory", action="store_true", help="print the step trajectory")
    parser.add_argument("--quiet", action="store_true", help="print only the response text")
    parser.add_argument("--max-retries", type=int, default=2, help="retries per tool (default 2)")
    parser.add_argument("--batch", action="store_true", help="run the built-in batch + metrics")
    args = parser.parse_args(argv)

    pipe = Pipeline()

    if args.batch:
        _run_batch(pipe, max_retries=args.max_retries)
        return 0

    if not args.query:
        parser.print_help()
        return 1

    _run_one(pipe, args.query, max_retries=args.max_retries, as_json=args.json,
             show_trajectory=args.trajectory, quiet=args.quiet)
    return 0


if __name__ == "__main__":
    sys.exit(main())
