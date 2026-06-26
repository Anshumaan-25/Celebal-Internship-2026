# Week 8 — Build an Agentic AI Pipeline

Celebal Technologies Internship (2026) · Week 8 submission.

## Contents

- **`agentic_ai_pipeline/`** — the project: a single agent implemented as a
  **stateful directed graph** (pure-Python, no framework, no LLM, zero
  dependencies). It does conditional routing to three tools, a retry loop with
  error handling, JSON-schema-validated tool I/O, trajectory evaluation, and
  completion-rate / cost metrics. See
  [`agentic_ai_pipeline/README.md`](agentic_ai_pipeline/README.md) for full
  docs, architecture diagrams, and the quiz-concept → code mapping.
- **`week 8 quiz.docx`** — the completed conceptual quiz (10 questions on
  single-agent systems and agent pipelines).

## Quick start

```bash
cd agentic_ai_pipeline
python main.py "calculate 2 + 3 * 4"      # single query
python examples/demo.py                   # guided tour of every concept
python -m unittest discover -s tests      # 61 tests, standard-library only
```
