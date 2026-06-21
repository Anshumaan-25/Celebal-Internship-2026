"""Embedding models.

Local Sentence-Transformers by default (free, no API key). The model is
*pluggable* so the ablation study can compare several encoders. Embeddings are
L2-normalised, so a dot product equals cosine similarity — exactly what both
ChromaDB (cosine space) and our hybrid fusion expect.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from . import config

# BGE retrieval models work best when the *query* (not the passages) is
# prefixed with this instruction. Applied automatically for bge-* models.
_BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


@lru_cache(maxsize=4)
def _load_st_model(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


class Embedder:
    """Thin wrapper around a Sentence-Transformers model."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = _load_st_model(model_name)
        # Method was renamed across sentence-transformers versions.
        if hasattr(self._model, "get_embedding_dimension"):
            self.dim = self._model.get_embedding_dimension()
        else:
            self.dim = self._model.get_sentence_embedding_dimension()

    def encode(
        self,
        texts,
        is_query: bool = False,
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Encode text(s) to an L2-normalised float32 matrix (n, dim)."""
        if isinstance(texts, str):
            texts = [texts]
        if is_query and "bge" in self.model_name.lower():
            texts = [_BGE_QUERY_INSTRUCTION + t for t in texts]
        vecs = self._model.encode(
            list(texts),
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        return np.asarray(vecs, dtype=np.float32)


def get_embedder(model_name: str | None = None) -> Embedder:
    return Embedder(model_name or config.EMBEDDING_MODEL)
