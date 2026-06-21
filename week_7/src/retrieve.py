"""Retrievers — baseline + advanced techniques, all behind one interface.

Every retriever exposes ``retrieve(query, k) -> list[RetrievedChunk]`` so the
evaluation harness can compare them by swapping the object:

* ``DenseRetriever``       — vector similarity (the baseline)
* ``BM25Retriever``        — classic lexical / keyword search
* ``HybridRetriever``      — weighted fusion of dense + BM25 (alpha-tunable)
* ``RerankRetriever``      — cross-encoder re-ranks a base retriever's top-N
* ``QueryRewriteRetriever``— LLM multi-query expansion + reciprocal-rank fusion
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

from . import config
from .embed import Embedder
from .store import VectorStore


@dataclass
class RetrievedChunk:
    chunk_id: str
    doc_id: str
    text: str
    score: float   # higher = more relevant
    rank: int      # 0-based position in the result list


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


# --------------------------------------------------------------------------
# Baseline: dense vector retrieval
# --------------------------------------------------------------------------
class DenseRetriever:
    def __init__(self, store: VectorStore, embedder: Embedder):
        self.store = store
        self.embedder = embedder

    def retrieve(self, query: str, k: int | None = None) -> list[RetrievedChunk]:
        k = k or config.TOP_K
        q_vec = self.embedder.encode(query, is_query=True)[0]
        res = self.store.query(q_vec, k=k)
        out = []
        for rank, (cid, doc, meta, dist) in enumerate(
            zip(res["ids"], res["documents"], res["metadatas"], res["distances"])
        ):
            out.append(
                RetrievedChunk(cid, (meta or {}).get("doc_id", ""), doc, 1.0 - float(dist), rank)
            )
        return out


# --------------------------------------------------------------------------
# Lexical: BM25 over the same chunks
# --------------------------------------------------------------------------
class BM25Retriever:
    def __init__(self, chunk_records: list[dict]):
        from rank_bm25 import BM25Okapi

        self.records = chunk_records
        self._bm25 = BM25Okapi([_tokenize(r["text"]) for r in chunk_records])

    def retrieve(self, query: str, k: int | None = None) -> list[RetrievedChunk]:
        k = k or config.TOP_K
        scores = self._bm25.get_scores(_tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [
            RetrievedChunk(
                self.records[i]["chunk_id"], self.records[i]["doc_id"],
                self.records[i]["text"], float(scores[i]), rank,
            )
            for rank, i in enumerate(order)
        ]


# --------------------------------------------------------------------------
# Hybrid: weighted fusion of dense + BM25
# --------------------------------------------------------------------------
def _minmax(chunks: list[RetrievedChunk]) -> dict[str, tuple[RetrievedChunk, float]]:
    if not chunks:
        return {}
    scores = [c.score for c in chunks]
    lo, hi = min(scores), max(scores)
    span = (hi - lo) or 1.0
    return {c.chunk_id: (c, (c.score - lo) / span) for c in chunks}


class HybridRetriever:
    """``score = alpha * dense_norm + (1 - alpha) * bm25_norm`` over the union of
    each retriever's candidates (missing side scored 0 after min-max norm)."""

    def __init__(self, dense: DenseRetriever, bm25: BM25Retriever,
                 alpha: float | None = None, candidates: int | None = None):
        self.dense = dense
        self.bm25 = bm25
        self.alpha = config.HYBRID_ALPHA if alpha is None else alpha
        self.candidates = candidates or max(config.RERANK_CANDIDATES, config.TOP_K * 4)

    def retrieve(self, query: str, k: int | None = None) -> list[RetrievedChunk]:
        k = k or config.TOP_K
        dn = _minmax(self.dense.retrieve(query, k=self.candidates))
        bn = _minmax(self.bm25.retrieve(query, k=self.candidates))
        fused = []
        for cid in set(dn) | set(bn):
            d, b = dn.get(cid), bn.get(cid)
            chunk = (d or b)[0]
            score = self.alpha * (d[1] if d else 0.0) + (1 - self.alpha) * (b[1] if b else 0.0)
            fused.append((score, chunk))
        fused.sort(key=lambda x: x[0], reverse=True)
        return [
            RetrievedChunk(c.chunk_id, c.doc_id, c.text, float(s), rank)
            for rank, (s, c) in enumerate(fused[:k])
        ]


# --------------------------------------------------------------------------
# Cross-encoder re-ranking of any base retriever's candidates
# --------------------------------------------------------------------------
@lru_cache(maxsize=2)
def _load_cross_encoder(model_name: str):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name)


class RerankRetriever:
    def __init__(self, base, model_name: str | None = None,
                 candidates: int | None = None, top_n: int | None = None):
        self.base = base
        self.candidates = candidates or config.RERANK_CANDIDATES
        self.top_n = top_n or config.RERANK_TOP_N
        self._ce = _load_cross_encoder(model_name or config.RERANKER_MODEL)

    def retrieve(self, query: str, k: int | None = None) -> list[RetrievedChunk]:
        k = k or self.top_n
        cand = self.base.retrieve(query, k=self.candidates)
        if not cand:
            return []
        scores = self._ce.predict([[query, c.text] for c in cand])
        order = sorted(range(len(cand)), key=lambda i: scores[i], reverse=True)[:k]
        return [
            RetrievedChunk(cand[i].chunk_id, cand[i].doc_id, cand[i].text, float(scores[i]), rank)
            for rank, i in enumerate(order)
        ]


# --------------------------------------------------------------------------
# LLM multi-query expansion + reciprocal-rank fusion
# --------------------------------------------------------------------------
_REWRITE_SYS = (
    "You expand a search query into diverse alternative phrasings that could "
    "retrieve relevant passages. Output ONLY the alternatives, one per line, no "
    "numbering or commentary."
)


class QueryRewriteRetriever:
    def __init__(self, base, n_variants: int | None = None, rrf_k: int = 60):
        self.base = base
        self.n = n_variants or config.QUERY_REWRITE_N
        self.rrf_k = rrf_k

    def _variants(self, query: str) -> list[str]:
        from .generate import complete

        out = complete(
            f"Produce {self.n} alternative phrasings.\nQUERY: {query}",
            system_instruction=_REWRITE_SYS, temperature=0.7, max_output_tokens=128,
        )
        variants = [ln.strip("-•* ").strip() for ln in out.splitlines() if ln.strip()]
        return [query] + variants[: self.n]

    def retrieve(self, query: str, k: int | None = None) -> list[RetrievedChunk]:
        k = k or config.TOP_K
        scores: dict[str, float] = {}
        chunks: dict[str, RetrievedChunk] = {}
        for q in self._variants(query):
            for c in self.base.retrieve(q, k=max(k, config.TOP_K)):
                scores[c.chunk_id] = scores.get(c.chunk_id, 0.0) + 1.0 / (self.rrf_k + c.rank + 1)
                chunks[c.chunk_id] = c
        order = sorted(scores, key=lambda cid: scores[cid], reverse=True)[:k]
        return [
            RetrievedChunk(chunks[cid].chunk_id, chunks[cid].doc_id, chunks[cid].text, float(scores[cid]), rank)
            for rank, cid in enumerate(order)
        ]
