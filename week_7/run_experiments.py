"""Phase 4 + 5 — run all ablations and the advanced-technique comparison, then
save tables (results/*.csv) and figures (results/figures/*.png) for the notebook.

    # quick verification (retrieval ablations free; small answer comparison)
    python run_experiments.py --sample-size 8 --n-negatives 20 --answer-sample 4

    # full run (retrieval ablations are free; answer comparison uses Gemini)
    python run_experiments.py --sample-size 120 --n-negatives 200 --answer-sample 40
"""
import argparse

from src import config, dataset, evaluate
from src.experiments import (
    ablate_chunk_size, ablate_embedding_model, ablate_hybrid_alpha,
    compare_answers, compare_retrievers, plot_bars, plot_line,
)

FIG = config.FIGURES_DIR
RES = config.RESULTS_DIR


def _show(title, df, csv):
    print(f"\n===== {title} =====")
    print(df.to_string(index=False))
    df.to_csv(csv, index=False)


def main(args):
    docs, examples = dataset.load_subset(sample_size=args.sample_size, n_negatives=args.n_negatives)
    records = dataset.records(docs)
    print("Dataset:", dataset.summary(docs, examples))
    k = config.TOP_K

    # 1) Retriever-method comparison (Phase 5) — retrieval-only / free
    methods = ["dense", "bm25", "hybrid", "rerank"]
    if args.with_query_rewrite:
        methods.append("query_rewrite")
    df_methods = compare_retrievers(records, examples, methods=methods, k=k)
    _show("Retriever comparison (retrieval metrics)", df_methods, RES / "cmp_retrievers.csv")
    plot_bars(df_methods, "method", [f"recall@{k}", f"ndcg@{k}", "mrr"],
              "Retriever comparison", FIG / "cmp_retrievers.png")

    # 2) Chunk-size ablation (Phase 4)
    df_chunk = ablate_chunk_size(records, examples, k=k)
    _show("Chunk-size ablation", df_chunk, RES / "ablation_chunk_size.csv")
    plot_line(df_chunk, "chunk_size", [f"recall@{k}", "mrr"],
              "Retrieval vs chunk size", FIG / "ablation_chunk_size.png", xlabel="chunk size (words)")

    # 3) Embedding-model ablation (Phase 4)
    df_emb = ablate_embedding_model(records, examples, k=k)
    _show("Embedding-model ablation", df_emb, RES / "ablation_embedding.csv")
    plot_bars(df_emb, "embedding_model", [f"recall@{k}", f"ndcg@{k}", "mrr"],
              "Embedding-model comparison", FIG / "ablation_embedding.png")

    # 4) Hybrid alpha sweep (Phase 4/5)
    df_alpha = ablate_hybrid_alpha(records, examples, k=k)
    _show("Hybrid alpha sweep (0=BM25 .. 1=dense)", df_alpha, RES / "ablation_alpha.csv")
    plot_line(df_alpha, "alpha", [f"recall@{k}", f"ndcg@{k}"],
              "Hybrid fusion: score vs alpha", FIG / "ablation_alpha.png", xlabel="alpha (0=BM25, 1=dense)")

    # 5) End-to-end answer-quality comparison (uses Gemini) — the Phase 5 payoff
    if not args.skip_answers:
        ans_examples = examples[: args.answer_sample] if args.answer_sample else examples
        print(f"\nAnswer comparison on {len(ans_examples)} questions "
              f"(rate-limited to {config.GEMINI_RPM} req/min)...")
        df_ans, reports = compare_answers(records, ans_examples, methods=("dense", "rerank"), k=k)
        completed = sum(len(r.rows) for r in reports.values())
        if completed == 0:
            print("\n⚠️  No answer-comparison results yet — Gemini daily quota is exhausted. "
                  "The retrieval ablations above are complete. Re-run this script later "
                  "(it resumes from cache) to fill in the answer comparison.")
        else:
            _show("Answer-quality comparison (dense vs hybrid+rerank)", df_ans, RES / "cmp_answers.csv")
            plot_bars(df_ans, "method", ["accuracy", "abstention"],
                      "Answer quality: baseline vs advanced", FIG / "cmp_answers.png")
            for method, report in reports.items():
                if report.rows:
                    evaluate.plot_confusions(report, prefix=f"e2e_{method}")

    print("\nAll experiments complete. Tables in results/, figures in results/figures/.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-size", type=int, default=120)
    ap.add_argument("--n-negatives", type=int, default=200)
    ap.add_argument("--answer-sample", type=int, default=40,
                    help="limit the costly answer comparison to first N questions (0=all)")
    ap.add_argument("--skip-answers", action="store_true", help="retrieval ablations only (no Gemini)")
    ap.add_argument("--with-query-rewrite", action="store_true",
                    help="include LLM query-rewrite in retriever comparison (slower, uses Gemini)")
    main(ap.parse_args())
