# Document Question Answering System (RAG)

> **Week 7 Project** — A Retrieval-Augmented Generation system that answers
> questions from custom documents by retrieving relevant passages and grounding
> a language model's answer in them.

This project goes beyond a "it answers!" demo: it is built around a **rigorous,
ground-truth evaluation harness** so that every claim is backed by a metric, and
it compares a baseline pipeline against several **advanced retrieval techniques**.

---

## What it does

```
            ┌─────────────┐   ┌──────────┐   ┌───────────┐   ┌──────────────┐
 Documents →│  Ingest /   │ → │  Chunk   │ → │  Embed    │ → │ Vector Store │
 (PDFs,     │  Load       │   │          │   │ (local)   │   │  (ChromaDB)  │
  arXiv)    └─────────────┘   └──────────┘   └───────────┘   └──────┬───────┘
                                                                    │
 Question ─────────────────────────────────────────► Retrieve top-k chunks
                                                                    │
                                                       ┌────────────▼───────────┐
                                                       │  Generate answer with  │
                                                       │  Gemini (grounded +    │
                                                       │  cited, abstains if    │
                                                       │  context insufficient) │
                                                       └────────────────────────┘
```

**Stack:** Sentence-Transformers (embeddings, local & free) · ChromaDB (vector
store) · Google **Gemini 2.5 Flash-Lite** via the `google-genai` SDK (generation,
free tier) · BM25 + cross-encoder (advanced retrieval) · custom + RAGAS evaluation.

---

## Dataset

[`vectara/open_ragbench`](https://huggingface.co/datasets/vectara/open_ragbench)
— an "Open RAG Benchmark" built from arXiv papers. Crucially it ships **ground
truth**, which is what makes quantitative evaluation possible:

| File | Contents |
|------|----------|
| `queries.json` | `query_id → {query, type, source}` |
| `answers.json` | `query_id → gold reference answer` |
| `qrels.json`   | `query_id → relevant {doc_id, section_id}` (relevance labels) |
| `corpus/{id}.json` | `{title, abstract, sections:[{text, tables, images}]}` |

We download the index files, draw a **fixed seeded subsample** of questions
(default 120 — enough for meaningful metrics while staying inside Gemini's free
tier), pull the gold documents plus a sample of **hard-negative** documents as
distractors, and cache the result under `data/`.

---

## Project structure

```
week_7/
├── README.md
├── requirements.txt
├── .env.example            # copy to .env, add your Gemini key
├── quickstart.py           # end-to-end smoke test
├── src/
│   ├── config.py           # all tunable settings (config-driven)
│   ├── dataset.py          # open_ragbench loader + ground-truth eval set
│   ├── ingest.py           # generic PDF / text ingestion (your own docs)
│   ├── chunk.py            # chunking strategies (fixed, recursive)
│   ├── embed.py            # pluggable embedding models
│   ├── store.py            # ChromaDB wrapper
│   ├── retrieve.py         # dense retriever (hybrid + rerank: Phase 5)
│   ├── generate.py         # Gemini grounded generation w/ citations
│   └── pipeline.py         # ties it all together (config-driven)
├── notebooks/
│   └── analysis.ipynb      # ALL experiments, metrics, curves (Phase 3+)
├── results/figures/        # generated plots
└── app.py                  # Streamlit demo (Phase 6)
```

---

## Setup

```bash
# 1. (recommended) create a virtual environment
python -m venv .venv && source .venv/bin/activate

# 2. install dependencies
pip install -r requirements.txt

# 3. add your FREE Gemini key
cp .env.example .env
#   then edit .env and set GOOGLE_API_KEY=...
#   (get one at https://aistudio.google.com/app/apikey)

# 4. verify everything works end-to-end
python quickstart.py
```

---

## Running the evaluation & demo

```bash
# Retrieval ablations + advanced-technique comparison.
# Retrieval metrics are FREE (no LLM); the answer comparison uses the Gemini
# free tier (~20 calls/day) and resumes automatically if interrupted.
python run_experiments.py --sample-size 120 --n-negatives 200 --answer-sample 40
python run_experiments.py --skip-answers          # retrieval-only, zero quota

# Single-config baseline evaluation (retrieval + answers + confusion matrix)
python evaluate_baseline.py --sample-size 120 --n-negatives 200

# Analysis notebook — all tables, Recall@k curves, confusion matrices
jupyter lab notebooks/analysis.ipynb

# Streamlit demo — upload a PDF, pick a retriever, get grounded + cited answers
streamlit run app.py
```

> **Free-tier note:** this key allows ~20 Gemini requests/day. Retrieval evaluation needs none;
> answer evaluation caches + resumes across days, or enable billing to run at scale.

## Roadmap

- [x] **Phase 1 — Scaffold:** config, deps, dataset loader, caching
- [x] **Phase 2 — Baseline RAG:** ingest → chunk → embed → store → retrieve → generate
- [x] **Phase 3 — Evaluation harness:** Recall@k / Precision@k / MRR / nDCG, LLM-judge confusion matrix, rate limiter
- [x] **Phase 4 — Ablations:** chunk size, embedding model, top-k curve, hybrid α-sweep
- [x] **Phase 5 — Advanced techniques:** BM25, hybrid fusion, cross-encoder re-ranking, LLM query rewriting
- [x] **Phase 6 — Streamlit demo + analysis notebook**

> **A note on "loss curves":** RAG has no training loop, so literal loss curves
> don't apply. We use the rigorous equivalents instead — **Recall@k curves**,
> **metric-vs-chunk-size curves**, and an answer-correctness **confusion matrix**.
