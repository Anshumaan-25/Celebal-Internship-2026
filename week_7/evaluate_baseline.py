"""Phase 3 — run the full evaluation harness on the baseline RAG pipeline.

Examples:
    # quick verification (few questions, ~1-2 min)
    python evaluate_baseline.py --sample-size 8 --n-negatives 10

    # full run (120 questions; ~15-30 min under the free-tier rate limit)
    python evaluate_baseline.py

Outputs: a metrics table in the console, per-query CSV + cached JSON under
results/, and figures (Recall@k curve + confusion matrices) under
results/figures/. Re-running reuses the cache, so it won't re-spend quota.
"""
import argparse

from src import config, dataset, evaluate
from src.pipeline import RAGPipeline


def main(sample_size: int, n_negatives: int, judge: bool) -> None:
    print(f"Loading subset (sample_size={sample_size}, n_negatives={n_negatives})...")
    docs, examples = dataset.load_subset(sample_size=sample_size, n_negatives=n_negatives)
    print("Dataset summary:", dataset.summary(docs, examples))

    print(f"\nIndexing {len(docs)} documents...")
    pipe = RAGPipeline(collection_name=f"baseline_n{sample_size}", reset=True)
    n_chunks = pipe.index_documents(dataset.records(docs))
    print(f"  indexed {n_chunks} chunks")

    cache = config.RESULTS_DIR / f"baseline_eval_n{sample_size}.json"
    print(f"\nEvaluating (judge={judge}). Gemini calls are rate-limited to "
          f"{config.GEMINI_RPM} req/min — this is the slow part.")
    report = evaluate.evaluate_pipeline(pipe, examples, judge=judge, cache_path=cache)

    evaluate.print_report(report)

    # Persist artefacts.
    records_csv = config.RESULTS_DIR / f"baseline_records_n{sample_size}.csv"
    report.records_df().to_csv(records_csv, index=False)
    curve = evaluate.plot_recall_curve(report, save=config.FIGURES_DIR / f"baseline_recall_curve_n{sample_size}.png")
    cms = evaluate.plot_confusions(report, prefix=f"baseline_n{sample_size}")

    print("\nSaved:")
    print(f"  per-query records -> {records_csv}")
    print(f"  cached eval rows  -> {cache}")
    print(f"  recall@k curve    -> {curve}")
    for name, path in cms.items():
        print(f"  confusion matrix  -> {path}")
    print("\nConfusion matrix (retrieval vs. correctness):")
    print(evaluate.confusion_retrieval_vs_correctness(report.rows))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-size", type=int, default=120)
    ap.add_argument("--n-negatives", type=int, default=60)
    ap.add_argument("--no-judge", action="store_true", help="skip the LLM judge (retrieval metrics only)")
    args = ap.parse_args()
    main(args.sample_size, args.n_negatives, judge=not args.no_judge)
