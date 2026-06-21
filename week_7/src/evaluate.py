"""Evaluation harness — the quantitative heart of the project.

Because the dataset ships ground truth (gold ``doc_id`` per query + reference
answers), we can measure the system rather than just eyeball it:

* **Retrieval** (objective, no LLM): Recall@k, Precision@k, Hit@k, nDCG@k and
  MRR — Recall@k vs k is our "loss-curve equivalent".
* **Answer quality** (LLM-as-judge): each answer is graded CORRECT / PARTIAL /
  INCORRECT against the reference; answers that abstain are detected directly.
* **Confusion matrices**: cross-tabulating the *objective* retrieval outcome
  against the *judged* answer outcome shows exactly how retrieval success drives
  answer correctness, and how often the model hallucinates vs. faithfully
  abstains when the answer was not retrieved.

Every Gemini call (answers + judge) is rate-limited (see ``ratelimit.py``), and
per-query results are cached so re-runs don't re-spend quota.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from . import config
from .generate import ABSTENTION, complete, generate_answer

# --------------------------------------------------------------------------
# Per-query record
# --------------------------------------------------------------------------
@dataclass
class QueryEval:
    query_id: str
    question: str
    gold_doc_id: str
    chunk_docs: list[str]    # doc_id of each retrieved chunk, in rank order (dups ok)
    ranked_docs: list[str]   # unique doc_ids, first-seen rank order
    reference_answer: str
    answer: str
    abstained: bool
    judge_label: str         # CORRECT | PARTIAL | INCORRECT | ABSTAIN | SKIPPED
    correct: bool


def _mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def _is_quota_error(exc: Exception) -> bool:
    s = str(exc)
    return "RESOURCE_EXHAUSTED" in s or "429" in s or "quota" in s.lower()


# --------------------------------------------------------------------------
# Retrieval metrics (objective — no LLM involved)
# --------------------------------------------------------------------------
def _ranked_unique_docs(chunks) -> list[str]:
    out: list[str] = []
    for c in chunks:
        if c.doc_id not in out:
            out.append(c.doc_id)
    return out


def _dcg(relevances: list[int]) -> float:
    return sum(r / math.log2(i + 2) for i, r in enumerate(relevances))


def aggregate_retrieval(rows: list[QueryEval], k_values: list[int]) -> tuple[dict, float]:
    """Mean Recall/Precision/Hit/nDCG at each k, plus overall MRR.

    There is one gold document per query, so Recall@k == Hit@k. Precision@k is
    measured at the *chunk* level (fraction of the top-k retrieved chunks that
    come from the gold document) — i.e. retrieved-context purity.
    """
    reciprocal_ranks = []
    for r in rows:
        rank = r.ranked_docs.index(r.gold_doc_id) + 1 if r.gold_doc_id in r.ranked_docs else 0
        reciprocal_ranks.append(1.0 / rank if rank > 0 else 0.0)

    per_k: dict[int, dict] = {}
    for k in k_values:
        recall, precision, hit, ndcg = [], [], [], []
        for r in rows:
            top_chunks = r.chunk_docs[:k]
            precision.append(sum(d == r.gold_doc_id for d in top_chunks) / k)
            top_docs = r.ranked_docs[:k]
            h = 1.0 if r.gold_doc_id in top_docs else 0.0
            hit.append(h)
            recall.append(h)  # single gold doc -> recall == hit
            rels = [1 if d == r.gold_doc_id else 0 for d in top_docs]
            ndcg.append(_dcg(rels))  # ideal DCG = 1 (one relevant doc at rank 1)
        per_k[k] = {
            "recall": _mean(recall),
            "precision": _mean(precision),
            "hit": _mean(hit),
            "ndcg": _mean(ndcg),
        }
    return per_k, _mean(reciprocal_ranks)


# --------------------------------------------------------------------------
# Answer judging (LLM-as-judge)
# --------------------------------------------------------------------------
_JUDGE_SYS = (
    "You are a strict grader. Compare the CANDIDATE ANSWER to the REFERENCE "
    "ANSWER for the same question and respond with exactly ONE word:\n"
    "- CORRECT: candidate conveys the same key facts as the reference (paraphrase ok)\n"
    "- PARTIAL: candidate is partially correct or incomplete\n"
    "- INCORRECT: candidate is wrong, contradicts the reference, or is irrelevant\n"
    "Output only the label word."
)


def is_abstention(answer: str) -> bool:
    a = (answer or "").strip().lower()
    if not a:
        return True
    return a.startswith("i don't know") or ABSTENTION.lower()[:30] in a


def judge_answer(question: str, reference: str, candidate: str) -> str:
    prompt = (
        f"QUESTION: {question}\n\nREFERENCE ANSWER: {reference}\n\n"
        f"CANDIDATE ANSWER: {candidate}\n\nLABEL:"
    )
    out = complete(prompt, system_instruction=_JUDGE_SYS, temperature=0.0, max_output_tokens=5)
    u = out.strip().upper()
    if "INCORRECT" in u:      # check first: "INCORRECT" contains "CORRECT"
        return "INCORRECT"
    if "PARTIAL" in u:
        return "PARTIAL"
    if "CORRECT" in u:
        return "CORRECT"
    return "INCORRECT"        # default: unparseable -> treat as incorrect


# --------------------------------------------------------------------------
# Report container
# --------------------------------------------------------------------------
@dataclass
class EvalReport:
    rows: list[QueryEval]
    retrieval_per_k: dict
    mrr: float
    k_values: list[int]
    label_counts: dict
    accuracy: float
    abstention_rate: float
    faithful_abstention_rate: float | None
    hallucination_rate: float | None

    def records_df(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(r) for r in self.rows])

    def retrieval_df(self) -> pd.DataFrame:
        df = pd.DataFrame(self.retrieval_per_k).T
        df.index.name = "k"
        return df[["recall", "precision", "hit", "ndcg"]]


def _build_report(rows: list[QueryEval], k_values: list[int]) -> EvalReport:
    per_k, mrr = aggregate_retrieval(rows, k_values)
    not_retrieved = [r for r in rows if r.gold_doc_id not in r.ranked_docs]
    faithful = _mean(r.abstained for r in not_retrieved) if not_retrieved else None
    halluc = (
        _mean((not r.abstained) and (not r.correct) for r in not_retrieved)
        if not_retrieved
        else None
    )
    return EvalReport(
        rows=rows,
        retrieval_per_k=per_k,
        mrr=mrr,
        k_values=k_values,
        label_counts=dict(Counter(r.judge_label for r in rows)),
        accuracy=_mean(r.correct for r in rows),
        abstention_rate=_mean(r.abstained for r in rows),
        faithful_abstention_rate=faithful,
        hallucination_rate=halluc,
    )


# --------------------------------------------------------------------------
# Caching of the (expensive) per-query rows
# --------------------------------------------------------------------------
def _save_rows(path: Path, rows: list[QueryEval]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump([asdict(r) for r in rows], fh)


def _load_rows(path: Path) -> list[QueryEval]:
    with open(path, encoding="utf-8") as fh:
        return [QueryEval(**d) for d in json.load(fh)]


def report_from_cache(cache_path, k_values=None) -> "EvalReport | None":
    """Rebuild an EvalReport from a cached per-query rows file (or None)."""
    p = Path(cache_path)
    if not p.exists():
        return None
    rows = _load_rows(p)
    if not rows:
        return None
    return _build_report(rows, list(k_values or config.TOP_K_VALUES_TO_COMPARE))


# --------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------
def evaluate_pipeline(
    pipeline,
    examples,
    k: int | None = None,
    k_values: list[int] | None = None,
    generate: bool = True,
    judge: bool = True,
    max_k: int | None = None,
    cache_path: str | Path | None = None,
    progress: bool = True,
) -> EvalReport:
    """Run retrieval + generation + judging over ``examples`` and aggregate.

    ``k`` is how many chunks feed the generator; ``k_values`` is the set of cut-offs
    for the retrieval curves. Results are cached to ``cache_path`` if given.
    """
    k = k or config.TOP_K
    k_values = list(k_values or config.TOP_K_VALUES_TO_COMPARE)
    max_k = max_k or max(k_values + [k])

    # Resume-aware: load any cached rows and only evaluate what's missing. Rows
    # are saved after every question so a quota interruption never loses work
    # and the run can be resumed later (important under tight free-tier caps).
    rows: list[QueryEval] = _load_rows(Path(cache_path)) if cache_path and Path(cache_path).exists() else []
    done_ids = {r.query_id for r in rows}
    remaining = [ex for ex in examples if ex.query_id not in done_ids]
    if rows and remaining:
        print(f"Resuming from cache: {len(done_ids)} done, {len(remaining)} remaining.")

    iterator = tqdm(remaining, desc="Evaluating") if progress else remaining
    for ex in iterator:
        chunks = pipeline.retriever.retrieve(ex.question, k=max_k)
        chunk_docs = [c.doc_id for c in chunks]
        ranked = _ranked_unique_docs(chunks)

        if not generate:
            # Retrieval-only: free (no Gemini call). Used by the ablations.
            answer, label, correct, abstained = "", "SKIPPED", False, False
        else:
            try:
                answer = generate_answer(ex.question, [c.text for c in chunks[:k]]).answer
                if is_abstention(answer):
                    label, correct, abstained = "ABSTAIN", False, True
                elif judge:
                    label = judge_answer(ex.question, ex.reference_answer, answer)
                    correct, abstained = (label == "CORRECT"), False
                else:
                    label, correct, abstained = "SKIPPED", False, False
            except Exception as exc:
                if _is_quota_error(exc):
                    print(f"\n⚠️  Gemini quota exhausted after {len(rows)} answers this run. "
                          f"Progress saved — re-run later to resume from here.")
                    break
                raise

        rows.append(
            QueryEval(
                query_id=ex.query_id,
                question=ex.question,
                gold_doc_id=ex.gold_doc_id,
                chunk_docs=chunk_docs,
                ranked_docs=ranked,
                reference_answer=ex.reference_answer,
                answer=answer,
                abstained=abstained,
                judge_label=label,
                correct=correct,
            )
        )
        if cache_path:
            _save_rows(Path(cache_path), rows)

    wanted = {ex.query_id for ex in examples}
    return _build_report([r for r in rows if r.query_id in wanted], k_values)


# --------------------------------------------------------------------------
# Confusion matrices (objective retrieval outcome x judged answer outcome)
# --------------------------------------------------------------------------
def confusion_retrieval_vs_correctness(rows: list[QueryEval]) -> pd.DataFrame:
    retrieved = ["Retrieved" if r.gold_doc_id in r.ranked_docs else "Not retrieved" for r in rows]
    correct = ["Correct" if r.correct else "Not correct" for r in rows]
    cm = pd.crosstab(
        pd.Series(retrieved, name="Gold document"),
        pd.Series(correct, name="Answer (LLM-judge)"),
    )
    return cm.reindex(index=["Retrieved", "Not retrieved"], columns=["Correct", "Not correct"], fill_value=0)


def confusion_retrieval_vs_attempt(rows: list[QueryEval]) -> pd.DataFrame:
    retrieved = ["Retrieved" if r.gold_doc_id in r.ranked_docs else "Not retrieved" for r in rows]
    attempt = ["Answered" if not r.abstained else "Abstained" for r in rows]
    cm = pd.crosstab(
        pd.Series(retrieved, name="Gold document"),
        pd.Series(attempt, name="System behaviour"),
    )
    return cm.reindex(index=["Retrieved", "Not retrieved"], columns=["Answered", "Abstained"], fill_value=0)


# --------------------------------------------------------------------------
# Plotting (saved to results/figures)
# --------------------------------------------------------------------------
def plot_recall_curve(report: EvalReport, save: str | Path | None = None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ks = report.k_values
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ks, [report.retrieval_per_k[k]["recall"] for k in ks], marker="o", label="Recall@k (=Hit@k)")
    ax.plot(ks, [report.retrieval_per_k[k]["ndcg"] for k in ks], marker="s", label="nDCG@k")
    ax.plot(ks, [report.retrieval_per_k[k]["precision"] for k in ks], marker="^", label="Precision@k (context purity)")
    ax.set_xlabel("k (top-k retrieved)")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(ks)
    ax.set_title("Retrieval performance vs k")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150)
    plt.close(fig)
    return save


def plot_confusions(report: EvalReport, prefix: str = "baseline", save_dir: str | Path | None = None) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    save_dir = Path(save_dir or config.FIGURES_DIR)
    matrices = {
        "retrieval_vs_correctness": (
            confusion_retrieval_vs_correctness(report.rows),
            "Retrieval outcome vs. answer correctness",
        ),
        "retrieval_vs_attempt": (
            confusion_retrieval_vs_attempt(report.rows),
            "Retrieval outcome vs. answer / abstain",
        ),
    }
    paths = {}
    for name, (cm, title) in matrices.items():
        fig, ax = plt.subplots(figsize=(4.8, 4.0))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
        ax.set_title(title)
        fig.tight_layout()
        path = save_dir / f"{prefix}_cm_{name}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths[name] = str(path)
    return paths


# --------------------------------------------------------------------------
# Pretty-printer
# --------------------------------------------------------------------------
def print_report(report: EvalReport) -> None:
    n = len(report.rows)
    print(f"\n=== Retrieval metrics (n={n}) ===")
    print(f"{'k':>4} {'Recall@k':>9} {'Prec@k':>8} {'Hit@k':>7} {'nDCG@k':>8}")
    for k in report.k_values:
        m = report.retrieval_per_k[k]
        print(f"{k:>4} {m['recall']:>9.3f} {m['precision']:>8.3f} {m['hit']:>7.3f} {m['ndcg']:>8.3f}")
    print(f"MRR (doc-level): {report.mrr:.3f}")

    print("\n=== Answer quality (LLM-as-judge) ===")
    print(f"Label distribution: {report.label_counts}")
    print(f"Accuracy (CORRECT):      {report.accuracy:.3f}")
    print(f"Abstention rate:         {report.abstention_rate:.3f}")
    if report.faithful_abstention_rate is not None:
        print("  When the gold doc was NOT retrieved:")
        print(f"    faithful-abstention rate: {report.faithful_abstention_rate:.3f}  (good)")
        print(f"    hallucination rate:       {report.hallucination_rate:.3f}  (bad)")
