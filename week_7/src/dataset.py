"""Loader for the Open RAG Benchmark (``vectara/open_ragbench``).

The dataset is **not** a standard parquet/``datasets`` dataset — it is a file
tree on the HuggingFace Hub::

    pdf/arxiv/
    ├── queries.json          # query_id -> {"query", "type", "source"}
    ├── answers.json          # query_id -> gold reference answer
    ├── qrels.json            # query_id -> relevance label(s): {"doc_id","section_id"}
    ├── pdf_urls.json         # paper_id -> original PDF url
    └── corpus/
        └── {PAPER_ID}.json   # {"title","abstract","sections":[{text,tables,images}], ...}

So we download the small index files with ``huggingface_hub``, draw a fixed
*seeded* subsample of queries (to respect the Gemini free-tier rate limits),
pull only the corpus documents we actually need — the **gold** documents for
those queries plus a sample of **hard negatives** as realistic distractors —
flatten each document's sections into text, and cache the assembled subset
under ``data/`` so re-runs (and the Streamlit app) are instant.

The ``qrels``/``answers`` files give us the *ground truth* that powers the
quantitative evaluation in ``src/evaluate.py``.
"""
from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_files
from tqdm.auto import tqdm

from . import config

# Paths *inside* the HuggingFace dataset repo.
_ARXIV_DIR = "pdf/arxiv"
_QUERIES = f"{_ARXIV_DIR}/queries.json"
_ANSWERS = f"{_ARXIV_DIR}/answers.json"
_QRELS = f"{_ARXIV_DIR}/qrels.json"
_CORPUS_DIR = f"{_ARXIV_DIR}/corpus"


# --------------------------------------------------------------------------
# Data containers
# --------------------------------------------------------------------------
@dataclass
class Document:
    """A corpus document flattened to text (images dropped, tables kept)."""

    doc_id: str
    title: str
    text: str
    sections: list[str]          # per-section text, parallel to section indices
    is_gold: bool                # True if it is the answer doc for some query
    metadata: dict = field(default_factory=dict)

    def to_record(self) -> dict:
        """The minimal dict the pipeline needs for indexing."""
        return {"doc_id": self.doc_id, "title": self.title, "text": self.text}


@dataclass
class EvalExample:
    """A single ground-truth question for evaluation."""

    query_id: str
    question: str
    reference_answer: str
    gold_doc_id: str
    gold_section_ids: list[str]
    query_type: str              # "abstractive" | "extractive"
    query_source: str            # "text" | "text-image" | "text-table" | ...


# --------------------------------------------------------------------------
# Low-level download helpers
# --------------------------------------------------------------------------
def _download_json(filename: str) -> dict:
    path = hf_hub_download(
        repo_id=config.HF_DATASET_REPO, filename=filename, repo_type="dataset"
    )
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _download_corpus_doc(paper_id: str) -> dict | None:
    try:
        path = hf_hub_download(
            repo_id=config.HF_DATASET_REPO,
            filename=f"{_CORPUS_DIR}/{paper_id}.json",
            repo_type="dataset",
        )
    except Exception:
        return None  # some doc_ids may not have a corpus file; skip gracefully
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------
# Normalisation helpers (defensive against schema variations)
# --------------------------------------------------------------------------
def _normalize_qrel(entry) -> list[tuple[str, str]]:
    """Return a list of (doc_id, section_id) from a qrels entry."""
    out: list[tuple[str, str]] = []
    if isinstance(entry, dict):
        if "doc_id" in entry:
            out.append((str(entry["doc_id"]), str(entry.get("section_id", ""))))
        else:  # {doc_id: section_info, ...}
            for key, val in entry.items():
                if isinstance(val, dict) and "section_id" in val:
                    out.append((str(key), str(val.get("section_id", ""))))
                elif isinstance(val, (list, tuple)):
                    out.extend((str(key), str(s)) for s in val)
                else:
                    out.append((str(key), str(val)))
    elif isinstance(entry, list):
        for item in entry:
            out.extend(_normalize_qrel(item))
    return out


def _query_text(q) -> str:
    return q.get("query", "") if isinstance(q, dict) else str(q)


def _answer_text(a) -> str:
    if isinstance(a, dict):
        return a.get("answer") or a.get("text") or json.dumps(a)
    return str(a)


def _stringify(value) -> str:
    """Flatten a str / list / dict to text.

    Corpus sections store ``tables`` (and ``images``) as dicts keyed by id,
    each value being markdown, so we recursively join values into one string.
    """
    if not value:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(_stringify(v) for v in value if v)
    if isinstance(value, dict):
        return "\n".join(_stringify(v) for v in value.values() if v)
    return str(value)


def _flatten_doc(paper_id: str, raw: dict, is_gold: bool) -> Document:
    """Concatenate title, abstract and every section (text + markdown tables)."""
    title = _stringify(raw.get("title")) if isinstance(raw, dict) else ""
    abstract = _stringify(raw.get("abstract")) if isinstance(raw, dict) else ""
    sections_raw = raw.get("sections", []) if isinstance(raw, dict) else []

    section_texts: list[str] = []
    for sec in sections_raw:
        if isinstance(sec, dict):
            text = _stringify(sec.get("text")).strip()
            tables = _stringify(sec.get("tables")).strip()
            piece = "\n\n".join(p for p in (text, tables) if p)
            if piece:
                section_texts.append(piece)
        elif isinstance(sec, str) and sec.strip():
            section_texts.append(sec.strip())

    header = [p for p in (f"Title: {title}", f"Abstract: {abstract}") if p.split(": ", 1)[-1].strip()]
    full_text = "\n\n".join(header + section_texts)
    return Document(
        doc_id=paper_id,
        title=title,
        text=full_text,
        sections=section_texts,
        is_gold=is_gold,
        metadata={"abstract": abstract, "n_sections": len(section_texts)},
    )


# --------------------------------------------------------------------------
# Caching
# --------------------------------------------------------------------------
def _cache_path(sample_size: int, n_negatives: int, seed: int) -> Path:
    return config.DATA_DIR / f"open_ragbench_n{sample_size}_neg{n_negatives}_seed{seed}.json"


def _save_cache(path: Path, docs: list[Document], examples: list[EvalExample]) -> None:
    payload = {
        "documents": [asdict(d) for d in docs],
        "examples": [asdict(e) for e in examples],
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)


def _load_cache(path: Path) -> tuple[list[Document], list[EvalExample]]:
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    docs = [Document(**d) for d in payload["documents"]]
    examples = [EvalExample(**e) for e in payload["examples"]]
    return docs, examples


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def load_subset(
    sample_size: int | None = None,
    n_negatives: int = 60,
    seed: int | None = None,
    force_rebuild: bool = False,
) -> tuple[list[Document], list[EvalExample]]:
    """Build (or load from cache) a seeded evaluation subset.

    Returns ``(documents, examples)`` where ``documents`` is the retrieval
    corpus (gold docs + hard-negative distractors) and ``examples`` is the
    list of ground-truth questions.
    """
    sample_size = sample_size or config.EVAL_SAMPLE_SIZE
    seed = config.RANDOM_SEED if seed is None else seed
    cache = _cache_path(sample_size, n_negatives, seed)
    if cache.exists() and not force_rebuild:
        return _load_cache(cache)

    print("Downloading index files (queries / answers / qrels)...")
    queries = _download_json(_QUERIES)
    answers = _download_json(_ANSWERS)
    qrels = _download_json(_QRELS)

    # Keep only fully-labelled queries (have an answer AND a relevance judgement).
    valid_ids = [
        qid
        for qid in queries
        if qid in answers and qid in qrels and _normalize_qrel(qrels[qid])
    ]
    rng = random.Random(seed)
    rng.shuffle(valid_ids)
    chosen = valid_ids[:sample_size]

    examples: list[EvalExample] = []
    gold_ids: set[str] = set()
    for qid in chosen:
        rels = _normalize_qrel(qrels[qid])
        gold_doc = rels[0][0]
        gold_ids.add(gold_doc)
        qobj = queries[qid]
        examples.append(
            EvalExample(
                query_id=qid,
                question=_query_text(qobj),
                reference_answer=_answer_text(answers[qid]),
                gold_doc_id=gold_doc,
                gold_section_ids=[s for _, s in rels],
                query_type=qobj.get("type", "") if isinstance(qobj, dict) else "",
                query_source=qobj.get("source", "") if isinstance(qobj, dict) else "",
            )
        )

    # Add hard-negative distractor documents so retrieval is non-trivial.
    print("Listing corpus files to sample hard negatives...")
    corpus_files = [
        f
        for f in list_repo_files(config.HF_DATASET_REPO, repo_type="dataset")
        if f.startswith(_CORPUS_DIR + "/") and f.endswith(".json")
    ]
    corpus_ids = [Path(f).stem for f in corpus_files]
    negatives = [c for c in corpus_ids if c not in gold_ids]
    rng.shuffle(negatives)
    needed = list(gold_ids) + negatives[:n_negatives]

    docs: list[Document] = []
    for pid in tqdm(needed, desc="Downloading corpus docs"):
        raw = _download_corpus_doc(pid)
        if raw is not None:
            docs.append(_flatten_doc(pid, raw, is_gold=pid in gold_ids))

    # Drop any example whose gold doc failed to download, so metrics stay honest.
    available = {d.doc_id for d in docs}
    examples = [e for e in examples if e.gold_doc_id in available]

    _save_cache(cache, docs, examples)
    print(f"Built subset: {len(docs)} docs ({len(gold_ids)} gold), {len(examples)} questions.")
    return docs, examples


def records(docs: list[Document]) -> list[dict]:
    """Convert Documents to the minimal dicts the pipeline indexes."""
    return [d.to_record() for d in docs]


def summary(docs: list[Document], examples: list[EvalExample]) -> dict:
    """Quick descriptive stats — handy for the notebook's dataset section."""
    from collections import Counter

    return {
        "n_documents": len(docs),
        "n_gold_documents": sum(d.is_gold for d in docs),
        "n_questions": len(examples),
        "query_types": dict(Counter(e.query_type for e in examples)),
        "query_sources": dict(Counter(e.query_source for e in examples)),
        "avg_doc_words": round(
            sum(len(d.text.split()) for d in docs) / max(len(docs), 1), 1
        ),
    }
