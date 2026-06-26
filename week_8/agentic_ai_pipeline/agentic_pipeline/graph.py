"""A minimal stateful directed-graph engine (quiz Q1, Q2, Q4).

This is a from-scratch take on the ``StateGraph`` idea popularised by LangGraph:

* **Nodes** are named callables ``fn(state) -> state`` (quiz Q2).
* **Edges** connect nodes. A *static* edge always goes to the same place; a
  *conditional* edge runs a router function over the state and picks the next
  node from a mapping (quiz Q3).
* Because an edge may point back to an earlier node, the graph supports
  **cycles / loops** (quiz Q4) — the retry loop is exactly such a back-edge.
* The shared ``AgentState`` gives the graph its **statefulness** (quiz Q1).

The engine records a :class:`Trajectory` as it goes and guards against runaway
loops with ``max_steps``.
"""

from __future__ import annotations

import time
from typing import Any, Callable

from .errors import GraphError
from .trajectory import Trajectory

# Sentinel meaning "stop here". Static edges to END (or a node with no outgoing
# edge) terminate the run.
END = "__end__"

NodeFn = Callable[[Any], Any]
RouterFn = Callable[[Any], str]


class StateGraph:
    """Builder + runner for a stateful directed graph.

    Build with ``add_node`` / ``add_edge`` / ``add_conditional_edges`` /
    ``set_entry_point``, call ``validate`` (optional — ``invoke`` does it too),
    then ``invoke(state)``.
    """

    def __init__(self, max_steps: int = 50) -> None:
        self.nodes: dict[str, NodeFn] = {}
        self.static_edges: dict[str, str] = {}
        self.conditional_edges: dict[str, tuple[RouterFn, dict[str, str]]] = {}
        self.entry_point: str | None = None
        # Backstop against an accidental infinite cycle; a healthy retry loop
        # finishes in a handful of steps, well under this ceiling.
        self.max_steps = max_steps

    # --- construction -----------------------------------------------------
    def add_node(self, name: str, fn: NodeFn) -> "StateGraph":
        if name == END:
            raise GraphError(f"{END!r} is a reserved node name")
        if name in self.nodes:
            raise GraphError(f"duplicate node {name!r}")
        self.nodes[name] = fn
        return self

    def add_edge(self, src: str, dst: str) -> "StateGraph":
        self.static_edges[src] = dst
        return self

    def add_conditional_edges(
        self, src: str, router: RouterFn, mapping: dict[str, str]
    ) -> "StateGraph":
        self.conditional_edges[src] = (router, mapping)
        return self

    def set_entry_point(self, name: str) -> "StateGraph":
        self.entry_point = name
        return self

    # --- validation -------------------------------------------------------
    def validate(self) -> None:
        """Check the graph is structurally sound before running it."""
        if self.entry_point is None:
            raise GraphError("no entry point set")
        if self.entry_point not in self.nodes:
            raise GraphError(f"entry point {self.entry_point!r} is not a node")

        def _check(target: str, context: str) -> None:
            if target != END and target not in self.nodes:
                raise GraphError(f"{context} points at unknown node {target!r}")

        for src, dst in self.static_edges.items():
            _check(src, "edge source")
            _check(dst, "edge target")
        for src, (_, mapping) in self.conditional_edges.items():
            _check(src, "conditional edge source")
            for key, dst in mapping.items():
                _check(dst, f"conditional edge {src!r}->{key!r}")

    # --- execution --------------------------------------------------------
    def _next_node(self, current: str, state: Any) -> str:
        """Decide where to go after ``current`` has run."""
        if current in self.conditional_edges:
            router, mapping = self.conditional_edges[current]
            key = router(state)
            if key not in mapping:
                raise GraphError(
                    f"router for {current!r} returned {key!r}, "
                    f"not in mapping {sorted(mapping)}"
                )
            return mapping[key]
        return self.static_edges.get(current, END)

    def invoke(self, state: Any, trajectory: Trajectory | None = None) -> tuple[Any, Trajectory]:
        """Run the graph from the entry point until it reaches END.

        Returns the final state and the trajectory of steps taken. Each node is
        executed inside a try/except backstop (quiz Q8): an unexpected error is
        recorded on the trajectory and re-raised as a :class:`GraphError`, so a
        crashing node never silently corrupts the run.
        """
        self.validate()
        traj = trajectory or Trajectory()
        current = self.entry_point
        assert current is not None  # guaranteed by validate()
        steps = 0

        while current != END:
            if steps >= self.max_steps:
                raise GraphError(
                    f"exceeded max_steps={self.max_steps} (likely an infinite loop); "
                    f"path so far: {' -> '.join(traj.path)}"
                )
            steps += 1
            fn = self.nodes[current]
            # Count errors before the node runs so we can tell whether *this*
            # node appended a new (non-fatal) error to the state's error log.
            errors_before = len(getattr(state, "errors", []))
            start = time.perf_counter()
            try:
                returned = fn(state)
                # Nodes may mutate-and-return, or just mutate; honour either.
                if returned is not None:
                    state = returned
            except Exception as exc:  # noqa: BLE001 — deliberate backstop
                duration_ms = (time.perf_counter() - start) * 1000.0
                snap = state.snapshot() if hasattr(state, "snapshot") else {}
                traj.add(current, "error", duration_ms, snap, error=str(exc))
                raise GraphError(f"node {current!r} crashed: {exc}") from exc

            duration_ms = (time.perf_counter() - start) * 1000.0
            snap = state.snapshot() if hasattr(state, "snapshot") else {}
            # This node "errored" (non-fatally) iff it recorded a new error,
            # e.g. a tool node whose call failed and will be retried (quiz Q8).
            errors_now = getattr(state, "errors", [])
            if len(errors_now) > errors_before:
                traj.add(current, "error", duration_ms, snap, error=errors_now[-1])
            else:
                traj.add(current, "ok", duration_ms, snap, error=None)

            current = self._next_node(current, state)

        return state, traj
