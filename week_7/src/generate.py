"""Answer generation with Google Gemini, grounded in retrieved context.

Uses the current ``google-genai`` SDK (the old ``google-generativeai`` package
is end-of-life). The system instruction forces the model to answer *only* from
the retrieved passages, to cite them with bracketed numbers, and to abstain
when the answer is not present — the three behaviours that keep a RAG system
faithful and that we measure in the evaluation harness.

All calls funnel through ``complete()``, which applies a shared free-tier rate
limiter and retry/backoff, so both answer generation and the LLM judge stay
within quota.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from . import config
from .ratelimit import RateLimiter

# The exact phrase the model must use when the context is insufficient. The
# evaluator detects abstentions by matching this.
ABSTENTION = "I don't know based on the provided documents."

_INSTRUCTIONS = (
    "You are a careful research assistant. Answer the QUESTION using ONLY the "
    "numbered CONTEXT passages provided.\n"
    f'- If the answer is not contained in the context, reply exactly: "{ABSTENTION}"\n'
    "- Cite the passages you use with bracketed numbers like [1], [2].\n"
    "- Be concise and factual. Do not use any outside knowledge."
)

# Module-level singletons: configure the client + limiter once.
_CLIENT = None
_LIMITER = RateLimiter(config.GEMINI_RPM)


def _get_client():
    global _CLIENT
    if _CLIENT is None:
        from google import genai

        _CLIENT = genai.Client(api_key=config.require_api_key())
    return _CLIENT


def _extract_text(resp) -> str:
    try:
        if resp.text:
            return resp.text.strip()
    except Exception:
        pass
    try:  # fall back to walking the first candidate's parts
        return resp.candidates[0].content.parts[0].text.strip()
    except Exception:
        return ""


def complete(
    prompt: str,
    system_instruction: str | None = None,
    temperature: float = 0.2,
    max_output_tokens: int | None = None,
    max_retries: int = 4,
) -> str:
    """Low-level text completion: rate-limited, with exponential backoff."""
    from google.genai import types

    client = _get_client()
    cfg = types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=temperature,
        max_output_tokens=max_output_tokens or config.GENERATION_MAX_TOKENS,
    )
    for attempt in range(max_retries):
        try:
            _LIMITER.wait()
            resp = client.models.generate_content(
                model=config.GEMINI_MODEL, contents=prompt, config=cfg
            )
            return _extract_text(resp)
        except Exception as exc:
            # A per-DAY quota cap won't recover by retrying — fail fast so the
            # caller can stop gracefully and resume later.
            if "PerDay" in str(exc) or attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)  # 1s, 2s, 4s ... backoff for transient/per-minute errors
    return ""


def build_prompt(question: str, contexts: list[str]) -> str:
    """Build the user message (numbered context + question)."""
    if contexts:
        ctx = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(contexts))
    else:
        ctx = "(no context retrieved)"
    return f"CONTEXT:\n{ctx}\n\nQUESTION: {question}\n\nANSWER:"


@dataclass
class Generation:
    answer: str
    prompt: str
    model: str


def generate_answer(
    question: str,
    contexts: list[str],
    temperature: float | None = None,
) -> Generation:
    """Generate a grounded, cited answer from the retrieved context passages."""
    prompt = build_prompt(question, contexts)
    answer = complete(
        prompt,
        system_instruction=_INSTRUCTIONS,
        temperature=config.GENERATION_TEMPERATURE if temperature is None else temperature,
    )
    return Generation(answer=answer, prompt=prompt, model=config.GEMINI_MODEL)
