"""Dense Vector Index for financial semantic search with cosine similarity."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


class DenseVectorIndex:
    """Dense vector index supporting cosine similarity and pluggable embedding functions."""

    def __init__(
        self,
        dimension: int = 128,
        embed_fn: Callable[[list[str]], np.ndarray] | None = None,
    ):
        self.dimension = dimension
        self.embed_fn = embed_fn or self._default_embedder
        self.doc_ids: list[str] = []
        self.vectors: np.ndarray | None = None  # Shape (N, D)

    def _default_embedder(self, texts: list[str]) -> np.ndarray:
        """Lightweight deterministic semantic projection embedding fallback."""
        # Generates deterministic float32 vectors based on n-gram hashing and character distribution
        embeddings = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for i, text in enumerate(texts):
            words = text.lower().split()
            if not words:
                continue
            vec = np.zeros(self.dimension, dtype=np.float32)
            for word in words:
                # 3-gram hashing into dimension buckets
                for j in range(max(1, len(word) - 2)):
                    gram = word[j : j + 3]
                    h = hash(gram) % self.dimension
                    sign = 1.0 if (hash(gram) // self.dimension) % 2 == 0 else -1.0
                    vec[h] += sign
            norm = np.linalg.norm(vec)
            if norm > 1e-6:
                embeddings[i] = vec / norm
            else:
                embeddings[i] = vec
        return embeddings

    def fit(self, doc_ids: list[str], texts: list[str], precomputed_vectors: np.ndarray | None = None) -> DenseVectorIndex:
        """Index texts or precomputed vectors."""
        self.doc_ids = list(doc_ids)
        if precomputed_vectors is not None:
            vectors = np.asarray(precomputed_vectors, dtype=np.float32)
        else:
            vectors = self.embed_fn(texts)

        # Normalize rows for cosine distance via inner product
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms < 1e-7] = 1.0
        self.vectors = vectors / norms
        self.dimension = self.vectors.shape[1] if self.vectors.ndim > 1 else self.dimension
        return self

    def search(self, query: str, top_k: int = 10, query_vector: np.ndarray | None = None) -> list[tuple[str, float]]:
        """Search query using cosine similarity against stored vectors."""
        if self.vectors is None or len(self.doc_ids) == 0:
            return []

        if query_vector is None:
            q_emb = self.embed_fn([query])[0]
        else:
            q_emb = np.asarray(query_vector, dtype=np.float32).flatten()

        q_norm = np.linalg.norm(q_emb)
        if q_norm > 1e-7:
            q_emb = q_emb / q_norm

        # Cosine similarity matrix multiplication: (N, D) @ (D,) -> (N,)
        scores = np.dot(self.vectors, q_emb)

        # Rank
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = [
            (self.doc_ids[idx], float(scores[idx]))
            for idx in top_indices
            if scores[idx] > -1.0
        ]
        return results
