# Agentic AI Pipeline — Single-Agent System as a Stateful Directed Graph

**Celebal Technologies · Week 8 Assignment — "Build an Agentic AI Pipeline"**

A single agent, implemented as a **stateful directed graph** built from scratch
in pure-Python (no agent framework, no LLM, no third-party packages). It
classifies each query, **routes** it to one of three tools, **retries**
recoverable failures in a loop, validates all tool I/O against **JSON schemas**,
records a full **trajectory**, and reports **completion-rate and cost** metrics.

Every concept from the Week 8 quiz is implemented and demonstrated — see the
mapping table below.

```text
$ python main.py "calculate 2 + 3 * 4" --trajectory
Query     : calculate 2 + 3 * 4
Intent    : calculate  ->  route: calculator
Response  : 2 + 3 * 4 = 14
Status    : success   cost: 1.0   latency: 0.06 ms
Trajectory:
   0 ✓ analyze      (  0.00 ms)
   1 ✓ route        (  0.00 ms)
   2 ✓ calculator   (  0.03 ms)
   3 ✓ validate     (  0.00 ms)
   4 ✓ respond      (  0.00 ms)
  path: analyze -> route -> calculator -> validate -> respond
Evaluation: {"completed": true, "chosen_route": "calculator", "num_steps": 5, "num_retries": 0, "had_error": false, "efficiency_score": 1.0, ...}
```

## Why this design

The quiz answers double as a spec. Read back as requirements, they describe
exactly one system: a single agent built as a stateful graph that conditionally
routes to tools, loops on failure, and is measured by its trajectory and cost.
This repo is that system — see [`docs/architecture.md`](docs/architecture.md)
for diagrams and the full design rationale.

## Quick start

No installation, no dependencies — just Python 3.9+ (developed on 3.13).

```bash
cd agentic_ai_pipeline

# Run a single query
python main.py "calculate (10 - 4) / 2"
python main.py "extract keywords from: nodes and edges define an agent graph"
python main.py "hello, what can you do?"

# Useful flags
python main.py "what is 8 ^ 2 + 1" --trajectory   # show the step path
python main.py --json "what is 10 / 4"            # machine-readable result
python main.py --batch                            # run a batch + aggregate metrics

# The full guided tour (every quiz concept, with live output)
python examples/demo.py

# The test suite (50 tests, standard-library unittest)
python -m unittest discover -s tests
```

Use it as a library:

```python
from agentic_pipeline import Pipeline, MetricsCollector

pipe = Pipeline()
result = pipe.run("calculate 2 + 3 * 4")

print(result.response)              # "2 + 3 * 4 = 14"
print(result.trajectory.pretty())   # the full step path
print(result.evaluation)            # completed? correct route? efficiency?
print(result.cost)                  # resource units consumed
```

## The three tools (conditional routing)

| Intent (rule) | Tool | Example query | Result |
|---|---|---|---|
| contains math / "calculate" | `calculator` | `calculate 2 + 3 * 4` | `2 + 3 * 4 = 14` |
| contains "keywords"/"extract" | `keyword_extractor` | `extract keywords from: ...` | `Top keywords: ...` |
| anything else | `general_response` | `hello there` | a templated reply |

The calculator uses a **safe AST evaluator** (no `eval`): it accepts
`+ - * / // % **` and parentheses and rejects names, attribute access, and
function calls.

## Retry loop & error handling

Tool failures carry a `recoverable` flag, which drives the loop:

- **Recoverable** (e.g. a transient fault, simulated by `FlakyTool`) → loop back
  and retry, up to the retry budget.
- **Non-recoverable** (malformed expression, division by zero) → give up
  immediately and answer gracefully, **without wasting retries**.

```text
$ python examples/demo.py   # excerpt — flaky calculator recovers on the 3rd try
   2 ✗ calculator   ! simulated transient failure (attempt 1)
   3 ✓ validate
   4 ✗ calculator   ! simulated transient failure (attempt 2)
   5 ✓ validate
   6 ✓ calculator
   7 ✓ validate
   8 ✓ respond
```

## Trajectory evaluation & metrics

- **Trajectory evaluation** (`evaluate_trajectory`) judges the *path*, not just
  the answer: did it complete, did it take the *correct* route, how many retries
  and steps, an efficiency score. It can flag a right-answer-wrong-route run that
  output-only checking would miss.
- **Metrics** (`MetricsCollector`) aggregate across runs: **completion rate**,
  **average/total cost**, steps, retries, and a per-intent breakdown.

## Sequential vs parallel tool calls

`agentic_pipeline/parallel.py` runs independent tool calls both ways and reports
the speedup (the demo shows ~4× on four independent extractions). Dependent
calls must stay sequential; independent ones overlap.

## Project structure

```
agentic_ai_pipeline/
├── main.py                  # CLI entry point
├── README.md
├── pyproject.toml           # metadata (no runtime deps)
├── requirements.txt         # intentionally empty — standard library only
├── agentic_pipeline/        # the package
│   ├── state.py             # AgentState (shared mutable state)
│   ├── graph.py             # StateGraph engine (nodes, edges, cycles)
│   ├── router.py            # rule-based intent classification
│   ├── tools.py             # the 3 tools + FlakyTool + base Tool
│   ├── schemas.py           # dependency-free JSON-schema validator
│   ├── nodes.py             # role nodes + edge routers
│   ├── pipeline.py          # build_graph + Pipeline/RunResult
│   ├── trajectory.py        # trajectory logging + evaluation
│   ├── metrics.py           # completion rate & cost
│   ├── parallel.py          # sequential vs parallel execution
│   └── errors.py            # exception hierarchy
├── examples/demo.py         # guided tour of every concept
├── tests/                   # 50 unittest tests
└── docs/architecture.md     # diagrams + design rationale + quiz mapping
```

## Quiz concept → implementation map

| # | Quiz concept | Implemented in |
|---|---|---|
| Q1 | Stateful directed graph vs linear pipeline | `graph.py` + `state.py` |
| Q2 | Nodes & edges | `StateGraph` API + `nodes.py` |
| Q3 | Conditional routing (3 query types) | `router.py` + `route` conditional edge |
| Q4 | Cycles / retry loop | `validate → tool` back-edge (`validate_router`) |
| Q5 | One agent simulating multi-agent | role nodes in `nodes.py` |
| Q6 | JSON-schema tools | `schemas.py` + every tool's schemas |
| Q7 | Sequential vs parallel tool calls | `parallel.py` |
| Q8 | Error handling | tool-node try/except + recoverable retries + engine backstop |
| Q9 | Trajectory evaluation | `trajectory.py` |
| Q10 | Task completion rate & cost | `metrics.py` |

## Notes

- **Deterministic by design**: no model, no randomness → reproducible routes,
  answers, and trajectories (which is what makes it testable).
- **Where an LLM would plug in**: `router.classify` (intent) and
  `GeneralResponseTool.run` (open-ended answers). Everything else is
  model-agnostic, so upgrading to a real model is a localized change.
