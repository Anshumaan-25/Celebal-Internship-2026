"""Tests for the stateful directed-graph engine (Q1, Q2, Q4)."""

import unittest
from dataclasses import dataclass, field

from agentic_pipeline.errors import GraphError
from agentic_pipeline.graph import END, StateGraph


@dataclass
class Counter:
    """A tiny state object good enough for the engine's contract."""

    n: int = 0
    errors: list = field(default_factory=list)

    def snapshot(self):
        return {"n": self.n}


class TestStateGraph(unittest.TestCase):
    def test_linear_run(self):
        g = StateGraph()
        g.add_node("a", lambda s: setattr(s, "n", s.n + 1) or s)
        g.add_node("b", lambda s: setattr(s, "n", s.n + 10) or s)
        g.set_entry_point("a")
        g.add_edge("a", "b")
        g.add_edge("b", END)
        final, traj = g.invoke(Counter())
        self.assertEqual(final.n, 11)
        self.assertEqual(traj.path, ["a", "b"])

    def test_conditional_routing(self):
        g = StateGraph()
        g.add_node("start", lambda s: s)
        g.add_node("even", lambda s: setattr(s, "n", -1) or s)
        g.add_node("odd", lambda s: setattr(s, "n", -2) or s)
        g.set_entry_point("start")
        g.add_conditional_edges(
            "start", lambda s: "even" if s.n % 2 == 0 else "odd",
            {"even": "even", "odd": "odd"})
        g.add_edge("even", END)
        g.add_edge("odd", END)
        self.assertEqual(g.invoke(Counter(n=4))[0].n, -1)
        self.assertEqual(g.invoke(Counter(n=3))[0].n, -2)

    def test_cycle_terminates(self):
        # A back-edge forms a loop; the router exits once n reaches 3.
        g = StateGraph(max_steps=20)
        g.add_node("inc", lambda s: setattr(s, "n", s.n + 1) or s)
        g.set_entry_point("inc")
        g.add_conditional_edges(
            "inc", lambda s: "stop" if s.n >= 3 else "go",
            {"go": "inc", "stop": END})
        final, traj = g.invoke(Counter())
        self.assertEqual(final.n, 3)
        self.assertEqual(traj.num_steps, 3)

    def test_infinite_loop_guarded(self):
        g = StateGraph(max_steps=5)
        g.add_node("loop", lambda s: s)
        g.set_entry_point("loop")
        g.add_edge("loop", "loop")
        with self.assertRaises(GraphError):
            g.invoke(Counter())

    def test_validate_rejects_dangling_edge(self):
        g = StateGraph()
        g.add_node("a", lambda s: s)
        g.set_entry_point("a")
        g.add_edge("a", "missing")
        with self.assertRaises(GraphError):
            g.validate()

    def test_router_returning_unknown_key_errors(self):
        g = StateGraph()
        g.add_node("a", lambda s: s)
        g.set_entry_point("a")
        g.add_conditional_edges("a", lambda s: "x", {"y": END})
        with self.assertRaises(GraphError):
            g.invoke(Counter())

    def test_missing_entry_point_errors(self):
        g = StateGraph()
        g.add_node("a", lambda s: s)
        with self.assertRaises(GraphError):
            g.validate()

    def test_crashing_node_becomes_graph_error(self):
        def boom(_):
            raise ValueError("kaboom")

        g = StateGraph()
        g.add_node("a", boom)
        g.set_entry_point("a")
        g.add_edge("a", END)
        with self.assertRaises(GraphError):
            g.invoke(Counter())


if __name__ == "__main__":
    unittest.main()
