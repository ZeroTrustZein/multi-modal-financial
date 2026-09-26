"""Dense Vector Index for financial semantic search with cosine similarity."""

from __future__ import annotations

import binascii
from collections.abc import Callable

import numpy as np


class DenseVectorIndex:
    """Dense vector index supporting cosine similarity and pluggable embedding functions."""

    FINANCIAL_CONCEPT_WEIGHTS = {
        "revenue": 1.5,
        "sales": 1.4,
        "income": 1.5,
        "profit": 1.4,
        "operating": 1.3,
        "margin": 1.4,
        "cash": 1.3,
        "flow": 1.3,
        "debt": 1.2,
        "equity": 1.2,
        "eps": 1.5,
        "ebitda": 1.5,
        "capex": 1.3,
        "asset": 1.2,
        "liability": 1.2,
    }

    def __init__(
        self,
        dimension: int = 128,
        embed_fn: Callable[[list[str]], np.ndarray] | None = None,
    ):
        self.dimension = dimension
        self.embed_fn = embed_fn or self._default_embedder
        self.doc_ids: list[str] = []
        self.doc_id_to_idx: dict[str, int] = {}
        self.vectors: np.ndarray | None = None  # Shape (N, D)

    def _hash_token(self, token: str) -> tuple[int, float]:
        """Produce deterministic bucket index and sign for token."""
        crc = binascii.crc32(token.encode("utf-8"))
        bucket = crc % self.dimension
        sign = 1.0 if (crc >> 16) % 2 == 0 else -1.0
        return bucket, sign

    def _default_embedder(self, texts: list[str]) -> np.ndarray:
        """Deterministic semantic projection embedding using CRC32 n-grams and financial weighting."""
        embeddings = np.zeros((len(texts), self.dimension), dtype=np.float32)

        for i, text in enumerate(texts):
            words = text.lower().split()
            if not words:
                continue
            vec = np.zeros(self.dimension, dtype=np.float32)

            for word in words:
                cleaned_word = "".join(c for c in word if c.isalnum() or c in "%$")
                if not cleaned_word:
                    continue

                w_mult = self.FINANCIAL_CONCEPT_WEIGHTS.get(cleaned_word, 1.0)

                # Word unigram projection
                bucket, sign = self._hash_token(cleaned_word)
                vec[bucket] += sign * 2.0 * w_mult

                # Character 3-gram projection for sub-word morphology
                if len(cleaned_word) >= 3:
                    for j in range(len(cleaned_word) - 2):
                        gram = cleaned_word[j : j + 3]
                        g_bucket, g_sign = self._hash_token(gram)
                        vec[g_bucket] += g_sign * 1.0 * w_mult

            norm = np.linalg.norm(vec)
            if norm > 1e-6:
                embeddings[i] = vec / norm
            else:
                embeddings[i] = vec

        return embeddings

    def fit(
        self,
        doc_ids: list[str],
        texts: list[str],
        precomputed_vectors: np.ndarray | None = None,
    ) -> DenseVectorIndex:
        """Index texts or precomputed vectors."""
        self.doc_ids = list(doc_ids)
        self.doc_id_to_idx = {doc_id: i for i, doc_id in enumerate(self.doc_ids)}

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

    def add(self, chunk_id: str, text: str, vector: list[float] | None = None) -> None:
        """Insert or index chunk embedding conforming to DenseIndexProtocol."""
        precomputed = np.asarray([vector], dtype=np.float32) if vector is not None else None
        self.add_documents([chunk_id], [text], precomputed_vectors=precomputed)

    def add_documents(
        self,
        doc_ids: list[str],
        texts: list[str],
        precomputed_vectors: np.ndarray | None = None,
    ) -> DenseVectorIndex:
        """Incrementally append documents to dense vector index."""
        if self.vectors is None or len(self.doc_ids) == 0:
            return self.fit(doc_ids, texts, precomputed_vectors)

        if precomputed_vectors is not None:
            new_vectors = np.asarray(precomputed_vectors, dtype=np.float32)
        else:
            new_vectors = self.embed_fn(texts)

        norms = np.linalg.norm(new_vectors, axis=1, keepdims=True)
        norms[norms < 1e-7] = 1.0
        norm_new = new_vectors / norms

        start_idx = len(self.doc_ids)
        for i, d_id in enumerate(doc_ids):
            self.doc_ids.append(d_id)
            self.doc_id_to_idx[d_id] = start_idx + i

        self.vectors = np.vstack([self.vectors, norm_new])
        return self

    def search(
        self,
        query: str | list[float],
        top_k: int = 10,
        query_vector: np.ndarray | None = None,
    ) -> list[tuple[str, float]]:
        """Search query using cosine similarity against stored vectors."""
        if self.vectors is None or len(self.doc_ids) == 0:
            return []

        if isinstance(query, (list, np.ndarray)):
            q_emb = np.asarray(query, dtype=np.float32).flatten()
        elif query_vector is None:
            q_emb = self.embed_fn([query])[0]
        else:
            q_emb = np.asarray(query_vector, dtype=np.float32).flatten()

        q_norm = np.linalg.norm(q_emb)
        if q_norm > 1e-7:
            q_emb = q_emb / q_norm

        scores = np.dot(self.vectors, q_emb)

        top_indices = np.argsort(scores)[::-1][:top_k]
        results = [
            (self.doc_ids[idx], float(scores[idx])) for idx in top_indices if scores[idx] > -1.0
        ]
        return results

    def batch_search(
        self,
        queries: list[str],
        top_k: int = 10,
    ) -> list[list[tuple[str, float]]]:
        """Execute vector search over a batch of query strings."""
        if self.vectors is None or len(self.doc_ids) == 0 or not queries:
            return [[] for _ in queries]

        q_embeddings = self.embed_fn(queries)
        q_norms = np.linalg.norm(q_embeddings, axis=1, keepdims=True)
        q_norms[q_norms < 1e-7] = 1.0
        q_normed = q_embeddings / q_norms

        # Matrix multiplication (Q, D) @ (D, N) -> (Q, N)
        score_matrix = np.dot(q_normed, self.vectors.T)

        batch_results: list[list[tuple[str, float]]] = []
        for row in score_matrix:
            ranked_idx = np.argsort(row)[::-1][:top_k]
            res = [(self.doc_ids[idx], float(row[idx])) for idx in ranked_idx]
            batch_results.append(res)

        return batch_results

    def get_vector(self, doc_id: str) -> np.ndarray | None:
        """Retrieve stored normalized embedding vector by doc ID."""
        idx = self.doc_id_to_idx.get(doc_id)
        if idx is not None and self.vectors is not None:
            return self.vectors[idx].copy()
        return None

    def similarity(self, doc_id_1: str, doc_id_2: str) -> float | None:
        """Compute pairwise cosine similarity between two indexed documents."""
        v1 = self.get_vector(doc_id_1)
        v2 = self.get_vector(doc_id_2)
        if v1 is None or v2 is None:
            return None
        return float(np.dot(v1, v2))
