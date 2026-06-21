"""End-to-end RAG pipeline: ingest -> chunk -> embed -> store -> retrieve -> generate.

Everything is config-driven, so an experiment is just a different set of
constructor arguments. The evaluation harness builds many pipelines (one per
configuration) and compares them on the same ground-truth questions.
"""
from __future__ import annotations

from dataclasses import dataclass

from . import config
from .chunk import chunk_text
from .embed import get_embedder
from .generate import Generation, generate_answer
from .retrieve import DenseRetriever, RetrievedChunk
from .store import VectorStore


@dataclass
class RAGResult:
    question: str
    answer: str
    contexts: list[RetrievedChunk]
    generation: Generation


class RAGPipeline:
    def __init__(
        self,
        collection_name: str = "rag_baseline",
        embedding_model: str | None = None,
        chunk_strategy: str | None = None,
        chunk_size: int | None = None,
        overlap: int | None = None,
        reset: bool = False,
    ):
        self.chunk_strategy = chunk_strategy or config.CHUNK_STRATEGY
        self.chunk_size = chunk_size or config.CHUNK_SIZE
        self.overlap = config.CHUNK_OVERLAP if overlap is None else overlap

        self.embedder = get_embedder(embedding_model)
        self.store = VectorStore(name=collection_name, reset=reset)
        # {chunk_id, doc_id, text} kept in memory so BM25 / hybrid retrievers can
        # be built over the exact same chunks that went into the vector store.
        self.chunk_records: list[dict] = []
        self.retriever = DenseRetriever(self.store, self.embedder)

    # -- indexing ----------------------------------------------------------
    def index_documents(self, documents: list[dict], show_progress: bool = True) -> int:
        """Chunk, embed, and store a list of {doc_id, title, text} records.

        Returns the number of chunks indexed.
        """
        ids, texts, metas = [], [], []
        self.chunk_records = []
        for doc in documents:
            chunks = chunk_text(
                doc["text"], self.chunk_strategy, self.chunk_size, self.overlap
            )
            for ci, chunk in enumerate(chunks):
                cid = f"{doc['doc_id']}::chunk{ci}"
                ids.append(cid)
                texts.append(chunk)
                metas.append(
                    {
                        "doc_id": doc["doc_id"],
                        "title": doc.get("title", ""),
                        "chunk_index": ci,
                    }
                )
                self.chunk_records.append(
                    {"chunk_id": cid, "doc_id": doc["doc_id"], "text": chunk}
                )
        if not ids:
            return 0
        embeddings = self.embedder.encode(texts, show_progress=show_progress)
        self.store.add(ids, embeddings, texts, metas)
        return len(ids)

    # -- retrieval strategy ------------------------------------------------
    def make_retriever(self, kind: str = "dense", **kw):
        """Build a retriever over the already-indexed chunks.

        kind: "dense" | "bm25" | "hybrid" | "rerank" | "query_rewrite".
        """
        from .retrieve import (
            BM25Retriever, DenseRetriever, HybridRetriever,
            QueryRewriteRetriever, RerankRetriever,
        )

        kind = kind.lower()
        dense = DenseRetriever(self.store, self.embedder)
        if kind == "dense":
            return dense
        if kind == "bm25":
            return BM25Retriever(self.chunk_records)
        if kind == "hybrid":
            return HybridRetriever(dense, BM25Retriever(self.chunk_records), alpha=kw.get("alpha"))
        if kind == "rerank":
            base = self.make_retriever(kw.get("base", "hybrid"), **kw)
            return RerankRetriever(base, candidates=kw.get("candidates"), top_n=kw.get("top_n"))
        if kind == "query_rewrite":
            base = self.make_retriever(kw.get("base", "dense"))
            return QueryRewriteRetriever(base, n_variants=kw.get("n_variants"))
        raise ValueError(f"Unknown retriever kind: {kind!r}")

    def set_retriever(self, kind: str = "dense", **kw) -> None:
        self.retriever = self.make_retriever(kind, **kw)

    # -- querying ----------------------------------------------------------
    def answer(self, question: str, k: int | None = None) -> RAGResult:
        contexts = self.retriever.retrieve(question, k=k)
        gen = generate_answer(question, [c.text for c in contexts])
        return RAGResult(question=question, answer=gen.answer, contexts=contexts, generation=gen)
