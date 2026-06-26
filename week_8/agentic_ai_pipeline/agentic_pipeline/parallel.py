"""Sequential vs parallel tool execution (quiz Q7).

When tool calls are *independent* (no call needs another's output), running them
concurrently collapses their wall-clock time to roughly the slowest single call
instead of their sum. When calls are *dependent*, you must run them in order.

These helpers run a list of ``(tool, payload)`` tasks both ways so the demo can
show the speedup. A small ``latency`` makes the difference visible; with the
GIL, ``time.sleep`` releases the lock, so threaded I/O-style waits really do
overlap.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from .tools import Tool

Task = tuple[Tool, dict[str, Any]]


def run_sequential(tasks: list[Task], *, latency: float = 0.0) -> tuple[list[Any], float]:
    """Run tasks one after another; return (results, elapsed_ms)."""
    start = time.perf_counter()
    results = [tool(payload, latency=latency) for tool, payload in tasks]
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return results, elapsed_ms


def run_parallel(
    tasks: list[Task], *, latency: float = 0.0, max_workers: int | None = None
) -> tuple[list[Any], float]:
    """Run independent tasks concurrently; return (results, elapsed_ms).

    Results preserve input order regardless of completion order.
    """
    start = time.perf_counter()
    workers = max_workers or max(1, len(tasks))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(tool, payload, latency=latency) for tool, payload in tasks]
        results = [f.result() for f in futures]
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return results, elapsed_ms


def compare(tasks: list[Task], *, latency: float = 0.02) -> dict[str, Any]:
    """Run ``tasks`` both ways and report the timing and speedup."""
    _, seq_ms = run_sequential(tasks, latency=latency)
    _, par_ms = run_parallel(tasks, latency=latency)
    return {
        "num_tasks": len(tasks),
        "latency_per_call_ms": latency * 1000.0,
        "sequential_ms": round(seq_ms, 2),
        "parallel_ms": round(par_ms, 2),
        "speedup": round(seq_ms / par_ms, 2) if par_ms else None,
    }
