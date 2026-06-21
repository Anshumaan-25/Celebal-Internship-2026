"""ChromaDB persistent vector store wrapper.

We compute embeddings ourselves (in ``embed.py``) and hand the vectors to
Chroma, rather than letting Chroma own an embedding function. That keeps the
embedding model fully pluggable and lets the same vectors feed the hybrid
retriever later.
"""
from __future__ import annotations

import chromadb

from . import config


class VectorStore:
    """A single named, cosine-space Chroma collection."""

    def __init__(self, name: str = "rag", persist: bool = True, reset: bool = False):
        self.name = name
        self.client = (
            chromadb.PersistentClient(path=str(config.CHROMA_DIR))
            if persist
            else chromadb.EphemeralClient()
        )
        if reset:
            try:
                self.client.delete_collection(name)
            except Exception:
                pass  # collection didn't exist yet
        self.collection = self.client.get_or_create_collection(
            name=name, metadata={"hnsw:space": "cosine"}
        )

    def add(self, ids, embeddings, documents, metadatas, batch_size: int = 512) -> None:
        emb = embeddings.tolist() if hasattr(embeddings, "tolist") else list(embeddings)
        for i in range(0, len(ids), batch_size):
            self.collection.add(
                ids=ids[i : i + batch_size],
                embeddings=emb[i : i + batch_size],
                documents=documents[i : i + batch_size],
                metadatas=metadatas[i : i + batch_size],
            )

    def query(self, query_embedding, k: int = 5) -> dict:
        qe = (
            query_embedding.tolist()
            if hasattr(query_embedding, "tolist")
            else list(query_embedding)
        )
        res = self.collection.query(
            query_embeddings=[qe],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        # Unwrap the single-query batch dimension.
        return {
            "ids": res["ids"][0],
            "documents": res["documents"][0],
            "metadatas": res["metadatas"][0],
            "distances": res["distances"][0],
        }

    def count(self) -> int:
        return self.collection.count()
