"""Custom exception hierarchy for the agentic pipeline.

Keeping a small, explicit hierarchy lets each layer raise something meaningful
and lets callers (and the retry loop) catch exactly what they care about.
"""

from __future__ import annotations


class PipelineError(Exception):
    """Base class for every error raised inside the pipeline."""


class GraphError(PipelineError):
    """Raised for structural problems in the graph itself.

    Examples: an edge points at a node that does not exist, no entry point was
    set, or the runner exceeded ``max_steps`` (a likely infinite loop).
    """


class SchemaValidationError(PipelineError):
    """Raised when data does not conform to a JSON schema.

    Carries the list of human-readable validation messages so the caller can
    log or surface exactly what was wrong.
    """

    def __init__(self, message: str, errors: list[str] | None = None) -> None:
        super().__init__(message)
        self.errors: list[str] = errors or []


class ToolError(PipelineError):
    """Raised when a tool cannot complete its job.

    This is the error the retry loop is built to react to: bad input, a failed
    computation, or a simulated transient fault.
    """

    def __init__(self, message: str, *, recoverable: bool = True) -> None:
        super().__init__(message)
        # ``recoverable`` lets a tool tell the retry loop whether trying again
        # could plausibly help (a transient fault) or not (malformed input).
        self.recoverable = recoverable


class MaxRetriesExceeded(PipelineError):
    """Raised/recorded when a step still fails after exhausting its retries."""
