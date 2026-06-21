"""Phase 4 & 5 experiments: ablations + advanced-technique comparison.

Retrieval ablations run with ``generate=False`` (no Gemini, no quota) so they
are fast and free; only the final answer-quality comparison spends LLM calls.
Each function returns a tidy pandas DataFrame; plotting helpers save figures.

Cache files and Chroma collections embed a dataset signature (``#docs x #queries``)
so changing the evaluation scale never silently reuses a smaller run's results.
"""
from __future__ import annotations

import pandas as pd

from . import config, evaluate
from .pipeline import RAGPipeline

K = config.TOP_K


def _sig(records, examples) -> str:
    return f"{len(records)}x{len(examples)}"


def _key_metrics(report: evaluate.EvalReport, k: int = K) -> dict:
    m = report.retrieval_per_k.get(k, report.retrieval_per_k[max(report.retrieval_per_k)])
    return {
        f"recall@{k}": round(m["recall"], 4),
        f"precision@{k}": round(m["precision"], 4),
        f"ndcg@{k}": round(m["ndcg"], 4),
        "mrr": round(report.mrr, 4),
        "accuracy": round(report.accuracy, 4),
        "abstention": round(report.abstention_rate, 4),
    }


def _index(records, **kw) -> RAGPipeline:
    pipe = RAGPipeline(reset=True, **kw)
    pipe.index_documents(records, show_progress=False)
    return pipe


# ---- Phase 5: retriever-method comparison ---------------------------------
def compare_retrievers(records, examples, methods=("dense", "bm25", "hybrid", "rerank"),
                       k=K, generate=False, judge=False, tag="cmp") -> pd.DataFrame:
    sig = _sig(records, examples)
    pipe = _index(records, collection_name=f"{tag}_{sig}_idx")
    rows = []
    for method in methods:
        pipe.set_retriever(method)
        cache = config.RESULTS_DIR / f"{tag}_{sig}_{method}_gen{int(generate)}.json"
        report = evaluate.evaluate_pipeline(
            pipe, examples, k=k, generate=generate, judge=judge, cache_path=cache
        )
        rows.append({"method": method, **_key_metrics(report, k)})
    return pd.DataFrame(rows)


# ---- Phase 4: chunk-size ablation -----------------------------------------
def ablate_chunk_size(records, examples, sizes=None, method="dense", k=K, tag="chunk") -> pd.DataFrame:
    sizes = sizes or config.CHUNK_SIZES_TO_COMPARE
    sig = _sig(records, examples)
    rows = []
    for size in sizes:
        pipe = _index(records, collection_name=f"{tag}_{sig}_{size}", chunk_size=size)
        pipe.set_retriever(method)
        report = evaluate.evaluate_pipeline(
            pipe, examples, k=k, generate=False, cache_path=config.RESULTS_DIR / f"{tag}_{sig}_{size}.json"
        )
        rows.append({"chunk_size": size, **_key_metrics(report, k)})
    return pd.DataFrame(rows)


# ---- Phase 4: embedding-model ablation ------------------------------------
def ablate_embedding_model(records, examples, models=None, method="dense", k=K, tag="emb") -> pd.DataFrame:
    models = models or config.EMBEDDING_MODELS_TO_COMPARE
    sig = _sig(records, examples)
    rows = []
    for i, model in enumerate(models):
        pipe = _index(records, collection_name=f"{tag}_{sig}_{i}", embedding_model=model)
        pipe.set_retriever(method)
        report = evaluate.evaluate_pipeline(
            pipe, examples, k=k, generate=False, cache_path=config.RESULTS_DIR / f"{tag}_{sig}_{i}.json"
        )
        rows.append({"embedding_model": model.split("/")[-1], **_key_metrics(report, k)})
    return pd.DataFrame(rows)


# ---- Phase 4/5: hybrid alpha sweep ----------------------------------------
def ablate_hybrid_alpha(records, examples, alphas=None, k=K, tag="alpha") -> pd.DataFrame:
    alphas = alphas or config.HYBRID_ALPHAS_TO_COMPARE
    sig = _sig(records, examples)
    pipe = _index(records, collection_name=f"{tag}_{sig}_idx")
    rows = []
    for a in alphas:
        pipe.set_retriever("hybrid", alpha=a)
        report = evaluate.evaluate_pipeline(
            pipe, examples, k=k, generate=False, cache_path=config.RESULTS_DIR / f"{tag}_{sig}_{a}.json"
        )
        rows.append({"alpha": a, **_key_metrics(report, k)})
    return pd.DataFrame(rows)


# ---- End-to-end answer-quality comparison (uses Gemini) -------------------
def compare_answers(records, examples, methods=("dense", "rerank"), k=K, tag="e2e"):
    sig = _sig(records, examples)
    pipe = _index(records, collection_name=f"{tag}_{sig}_idx")
    rows, reports = [], {}
    for method in methods:
        pipe.set_retriever(method)
        report = evaluate.evaluate_pipeline(
            pipe, examples, k=k, generate=True, judge=True,
            cache_path=config.RESULTS_DIR / f"{tag}_{sig}_{method}.json",
        )
        reports[method] = report
        rows.append({"method": method, **_key_metrics(report, k)})
    return pd.DataFrame(rows), reports


# ---- Plotting --------------------------------------------------------------
def plot_bars(df, label_col, metrics, title, save):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    labels = df[label_col].astype(str).tolist()
    x = np.arange(len(labels))
    w = 0.8 / len(metrics)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for i, m in enumerate(metrics):
        ax.bar(x + i * w, df[m], width=w, label=m)
    ax.set_xticks(x + w * (len(metrics) - 1) / 2)
    ax.set_xticklabels(labels, rotation=12)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score")
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save, dpi=150)
    plt.close(fig)
    return save


def plot_line(df, xcol, ycols, title, save, xlabel=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for y in ycols:
        ax.plot(df[xcol], df[y], marker="o", label=y)
    ax.set_xlabel(xlabel or xcol)
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save, dpi=150)
    plt.close(fig)
    return save
