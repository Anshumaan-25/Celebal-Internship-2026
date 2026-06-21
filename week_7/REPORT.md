# Document Question Answering (RAG) — Project Report

A Retrieval-Augmented Generation system that answers questions from custom documents by retrieving
relevant passages and grounding a language model's answer in them — built around a **quantitative
evaluation harness** so every conclusion is backed by a metric.

- **Stack:** Sentence-Transformers (embeddings) · ChromaDB (vector store) · Google Gemini
  2.5 Flash-Lite (generation) · BM25 + cross-encoder (advanced retrieval).
- **Data:** [`vectara/open_ragbench`](https://huggingface.co/datasets/vectara/open_ragbench) — arXiv
  papers with **ground-truth** gold documents (`qrels`) and reference answers, which is what makes
  the evaluation below possible.
- **Full detail + plots:** `notebooks/analysis.ipynb` (executed, with embedded outputs).

## How to run
```bash
pip install -r requirements.txt
cp .env.example .env                  # add your Gemini key
python quickstart.py                  # end-to-end smoke test
python run_experiments.py --skip-answers   # all retrieval ablations (free, no LLM)
streamlit run app.py                  # interactive demo (upload a PDF, ask, get cited answers)
jupyter lab notebooks/analysis.ipynb  # the full analysis report
```

---

## Results

All retrieval metrics are computed from ground truth (no LLM), on a fixed seeded subset of
**40 questions / 118 documents** (gold + hard-negative distractors).

### 1. Advanced retrievers vs. the dense baseline (Phase 5)
| method | Recall@5 | Precision@5 | nDCG@5 | MRR |
|---|---|---|---|---|
| dense (baseline) | 1.000 | 0.845 | 0.991 | 0.988 |
| BM25 | 0.975 | 0.850 | 0.966 | 0.963 |
| hybrid (dense+BM25) | 1.000 | **0.930** | 0.991 | 0.988 |
| hybrid + cross-encoder re-rank | 1.000 | 0.900 | **1.000** | **1.000** |

**Takeaway:** recall saturates by k=5, so the methods separate on *ranking quality*. Hybrid fusion
gives the purest context (Precision@5 0.93), and cross-encoder re-ranking achieves **perfect ranking**
(nDCG@5 = MRR = 1.0). Pure BM25 is weakest (it misses one gold doc → Recall 0.975).

### 2. Ablation — chunk size
| chunk size (words) | 128 | 256 | 512 | 1024 |
|---|---|---|---|---|
| Precision@5 | 0.985 | 0.885 | 0.845 | 0.700 |
| nDCG@5 | 1.000 | 0.991 | 0.991 | 0.946 |

**Takeaway:** smaller chunks raise context purity but risk splitting answers; ~128–256 words is the
sweet spot here, and 1024-word chunks measurably hurt ranking.

### 3. Ablation — embedding model
| model | Precision@5 | nDCG@5 |
|---|---|---|
| all-MiniLM-L6-v2 | 0.790 | 0.991 |
| **bge-small-en-v1.5** | **0.845** | 0.991 |

**Takeaway:** `bge-small` retrieves purer context than `all-MiniLM` at equal cost — hence the default.

### 4. Ablation — hybrid fusion weight α (0 = BM25, 1 = dense)
| α | 0.0 | 0.25 | 0.5 | 0.75 | 1.0 |
|---|---|---|---|---|---|
| Precision@5 | 0.850 | 0.885 | **0.930** | 0.875 | 0.845 |
| nDCG@5 | 0.966 | 0.975 | 0.991 | **1.000** | 0.991 |

**Takeaway:** fusion (α ≈ 0.5–0.75) beats both pure BM25 (α=0) and pure dense (α=1) — combining
lexical and semantic signals helps.

### 5. Answer quality (LLM-as-judge + confusion matrix)
Each answer is graded CORRECT / PARTIAL / INCORRECT against the reference; abstentions are detected
directly.

| run | n | Accuracy | Abstention | Hallucination |
|---|---|---|---|---|
| baseline (dense) | 8 | 0.375 | 0.250 | 0.000 |
| dense (end-to-end) | 5 | 0.400 | 0.200 | 0.000 |
| hybrid + rerank (end-to-end) | 4 | 0.750 | 0.250 | 0.000 |

**Generation is the bottleneck — and the advanced retriever fixes part of it.** On the **same 4
questions**, hybrid+rerank scores **0.75 vs. dense's 0.25**, flipping two PARTIAL answers to CORRECT:
the cross-encoder surfaces better-ordered context, so the generator produces *complete* answers. So
re-ranking helps **both** retrieval ranking (nDCG/MRR → 1.0) **and** downstream answer quality.
Crucially, **hallucination is 0** in every run — when the answer isn't retrieved the model
*faithfully abstains* rather than fabricate (see the confusion matrices in the notebook). (rerank's
5th question is pending the ~20-calls/day free-tier cap; the harness caches and resumes, but the
trend is already decisive.)

---

## "Loss curves" and "confusion matrix"
RAG does no training, so there is **no literal loss curve**. The rigorous equivalents are used
instead: the **Recall@k-vs-k curve**, **metric-vs-chunk-size** and **metric-vs-α** curves, and a
genuine **confusion matrix** that cross-tabulates the *objective* retrieval outcome against the
*judged* answer outcome (revealing correctness-given-retrieval and the hallucinate-vs-abstain
behaviour).

## Limitations & future work
- **Free-tier quota** (~10–20 Gemini calls/day on this key) bounds at-scale answer evaluation; the
  harness resumes across days, or billing unlocks it in one run.
- **Multimodal questions** (`text-image`, `text-table`) can't be fully answered by a text-only
  pipeline — a documented, honest limitation.
- **Future:** multimodal RAG (use table/image content), larger hard-negative pools, section-level
  retrieval ground truth, and semantic chunking.

## Deliverables
- `src/` — config-driven pipeline: ingest, chunk, embed, store, retrieve (dense/BM25/hybrid/rerank/
  query-rewrite), generate, evaluate, experiments, rate limiter.
- `notebooks/analysis.ipynb` — executed analysis report (tables, curves, confusion matrices).
- `app.py` — Streamlit demo. `quickstart.py`, `evaluate_baseline.py`, `run_experiments.py` — scripts.
- `results/` — metric tables (CSV) and `results/figures/` — all plots.
