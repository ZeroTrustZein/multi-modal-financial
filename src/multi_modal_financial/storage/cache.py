"""Embedding and query result caching subsystem."""

from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from collections.abc import Callable
from typing import Any

import numpy as np

from multi_modal_financial.types import AgentResponse


class EmbeddingCache:
    """LRU cache for dense embeddings to avoid redundant projection calculations."""

    def __init__(self, max_size: int = 5000):
        self.max_size = max_size
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _hash_text(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def get(self, text: str) -> np.ndarray | None:
        """Retrieve cached embedding vector for text."""
        h = self._hash_text(text)
        if h in self._cache:
            self._cache.move_to_end(h)
            self.hits += 1
            return self._cache[h].copy()
        self.misses += 1
        return None

    def put(self, text: str, vector: np.ndarray) -> None:
        """Store embedding vector in cache with LRU eviction."""
        h = self._hash_text(text)
        if h in self._cache:
            self._cache.move_to_end(h)
        self._cache[h] = np.asarray(vector, dtype=np.float32).copy()
        if len(self._cache) > self.max_size:
            self._cache.popitem(last=False)

    def get_or_compute(
        self, texts: list[str], compute_fn: Callable[[list[str]], np.ndarray]
    ) -> np.ndarray:
        """Batch lookup with fallback computation for uncached texts."""
        result_embeddings: list[np.ndarray | None] = [None] * len(texts)
        missing_indices: list[int] = []
        missing_texts: list[str] = []

        for idx, t in enumerate(texts):
            cached = self.get(t)
            if cached is not None:
                result_embeddings[idx] = cached
            else:
                missing_indices.append(idx)
                missing_texts.append(t)

        if missing_texts:
            computed = compute_fn(missing_texts)
            for m_idx, text, emb in zip(missing_indices, missing_texts, computed, strict=False):
                self.put(text, emb)
                result_embeddings[m_idx] = emb

        return np.vstack([res for res in result_embeddings if res is not None])

    def clear(self) -> None:
        """Clear cache state."""
        self._cache.clear()
        self.hits = 0
        self.misses = 0

    def stats(self) -> dict[str, Any]:
        """Cache performance statistics."""
        total = self.hits + self.misses
        rate = (self.hits / total) if total > 0 else 0.0
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(rate, 4),
        }


class QueryCache:
    """LRU cache for AgentResponse objects with optional TTL."""

    def __init__(self, max_size: int = 256, ttl_seconds: float = 3600.0):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict[str, tuple[float, AgentResponse]] = OrderedDict()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def make_key(
        query_str: str,
        ticker: str | None = None,
        period: str | None = None,
        year: int | None = None,
        top_k: int = 5,
        alpha: float = 0.5,
    ) -> str:
        """Construct normalized cache key from query parameters."""
        raw = f"{query_str.strip().lower()}|{ticker or ''}|{period or ''}|{year or ''}|{top_k}|{alpha:.2f}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, key: str) -> AgentResponse | None:
        """Retrieve cached response if within TTL."""
        if key in self._cache:
            timestamp, response = self._cache[key]
            if time.time() - timestamp <= self.ttl_seconds:
                self._cache.move_to_end(key)
                self.hits += 1
                return response
            else:
                del self._cache[key]

        self.misses += 1
        return None

    def put(self, key: str, response: AgentResponse) -> None:
        """Store query response in cache."""
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = (time.time(), response)
        if len(self._cache) > self.max_size:
            self._cache.popitem(last=False)

    def clear(self) -> None:
        """Clear query cache."""
        self._cache.clear()
        self.hits = 0
        self.misses = 0

    def stats(self) -> dict[str, Any]:
        """Query cache statistics."""
        total = self.hits + self.misses
        rate = (self.hits / total) if total > 0 else 0.0
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(rate, 4),
        }
