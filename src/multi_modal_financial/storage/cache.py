"""Embedding, query, and semantic similarity caching subsystem."""

from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from collections.abc import Callable
from typing import Any

import numpy as np

from multi_modal_financial.indexing.vector import DenseVectorIndex
from multi_modal_financial.types import (
    AgentResponse,
    CacheEvictionPolicy,
    CacheHitType,
    SemanticCacheConfig,
    SemanticCacheEntry,
    SemanticCacheLookupResult,
    SemanticCacheStats,
)


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


class SemanticCache:
    """Semantic vector similarity and exact match cache conforming to SemanticCacheProtocol."""

    def __init__(
        self,
        config: SemanticCacheConfig | None = None,
        embed_fn: Callable[[list[str]], np.ndarray] | None = None,
    ):
        self.config = config or SemanticCacheConfig()
        self._dense_index = DenseVectorIndex(dimension=128)
        self._embed_fn = embed_fn or self._dense_index.embed_fn
        self._entries: dict[str, SemanticCacheEntry] = {}
        self._exact_index: dict[str, str] = {}  # normalized_query -> key

        # Telemetry counters
        self._total_queries = 0
        self._exact_hits = 0
        self._semantic_hits = 0
        self._misses = 0
        self._evictions = 0
        self._total_latency_ms = 0.0

    @staticmethod
    def _normalize_query(query: str) -> str:
        """Normalize query string for deterministic exact lookups."""
        return " ".join(query.strip().lower().split())

    def _embed_query(self, query: str) -> list[float]:
        """Embed single query string using configured embedding projection."""
        vectors = self._embed_fn([query])
        vec = vectors[0]
        norm = np.linalg.norm(vec)
        if norm > 1e-7:
            vec = vec / norm
        return vec.tolist()

    def _compute_similarity(
        self, v1: list[float] | np.ndarray, v2: list[float] | np.ndarray
    ) -> float:
        """Compute similarity score according to configured distance metric."""
        arr1 = np.asarray(v1, dtype=np.float32)
        arr2 = np.asarray(v2, dtype=np.float32)

        if self.config.distance_metric == "cosine":
            norm1 = np.linalg.norm(arr1)
            norm2 = np.linalg.norm(arr2)
            if norm1 < 1e-7 or norm2 < 1e-7:
                return 0.0
            cos_sim = float(np.dot(arr1, arr2) / (norm1 * norm2))
            return max(-1.0, min(1.0, cos_sim))
        elif self.config.distance_metric == "euclidean":
            dist = float(np.linalg.norm(arr1 - arr2))
            return 1.0 / (1.0 + dist)
        else:  # Dot product
            return float(np.dot(arr1, arr2))

    def _delete_entry(self, key: str) -> None:
        """Remove entry from memory and secondary lookup index."""
        entry = self._entries.pop(key, None)
        if entry:
            norm_q = self._normalize_query(entry.query)
            if self._exact_index.get(norm_q) == key:
                self._exact_index.pop(norm_q, None)

    def get(
        self, query: str, query_vector: list[float] | None = None
    ) -> SemanticCacheLookupResult | None:
        """Lookup cached response using exact match fast path then vector similarity matching."""
        if not self.config.enabled:
            return None

        t0 = time.perf_counter()
        self._total_queries += 1
        now = time.time()

        norm_q = self._normalize_query(query)

        # 1. Exact match fast path
        exact_key = self._exact_index.get(norm_q)
        if exact_key and exact_key in self._entries:
            entry = self._entries[exact_key]
            if entry.is_expired(self.config.ttl_seconds, current_time=now):
                self._delete_entry(exact_key)
            else:
                entry.touch(current_time=now)
                self._exact_hits += 1
                latency = (time.perf_counter() - t0) * 1000.0
                self._total_latency_ms += latency
                return SemanticCacheLookupResult(
                    hit=True,
                    similarity=1.0,
                    matched_query=entry.query,
                    response=entry.response,
                    lookup_latency_ms=round(latency, 2),
                    hit_type=CacheHitType.EXACT,
                )

        # 2. Semantic vector match
        q_vec = query_vector if query_vector is not None else self._embed_query(query)

        best_sim = -1.0
        best_entry: SemanticCacheEntry | None = None
        expired_keys: list[str] = []

        for key, entry in self._entries.items():
            if entry.is_expired(self.config.ttl_seconds, current_time=now):
                expired_keys.append(key)
                continue

            if not entry.query_vector:
                continue

            sim = self._compute_similarity(q_vec, entry.query_vector)
            if sim > best_sim:
                best_sim = sim
                best_entry = entry

        # Cleanup expired entries discovered during scan
        for exp_key in expired_keys:
            self._delete_entry(exp_key)

        latency = (time.perf_counter() - t0) * 1000.0
        self._total_latency_ms += latency

        if best_entry is not None and best_sim >= self.config.similarity_threshold:
            best_entry.touch(current_time=now)
            self._semantic_hits += 1
            return SemanticCacheLookupResult(
                hit=True,
                similarity=round(best_sim, 4),
                matched_query=best_entry.query,
                response=best_entry.response,
                lookup_latency_ms=round(latency, 2),
                hit_type=CacheHitType.SEMANTIC,
            )

        self._misses += 1
        return SemanticCacheLookupResult(
            hit=False,
            similarity=round(best_sim, 4) if best_sim > 0.0 else 0.0,
            lookup_latency_ms=round(latency, 2),
            hit_type=CacheHitType.NONE,
        )

    def put(
        self,
        query: str,
        response: AgentResponse,
        query_vector: list[float] | None = None,
    ) -> None:
        """Store query, vector embedding, and response in semantic cache with policy-based eviction."""
        if not self.config.enabled:
            return

        now = time.time()
        norm_q = self._normalize_query(query)
        key = hashlib.sha256(norm_q.encode("utf-8")).hexdigest()

        q_vec = query_vector if query_vector is not None else self._embed_query(query)

        if key in self._entries:
            entry = self._entries[key]
            entry.response = response
            entry.query_vector = q_vec
            entry.touch(current_time=now)
            return

        # Check and enforce capacity limits
        if len(self._entries) >= self.config.max_entries:
            # First purge expired entries
            expired_keys = [
                k for k, e in self._entries.items() if e.is_expired(self.config.ttl_seconds, current_time=now)
            ]
            for exp_k in expired_keys:
                self._delete_entry(exp_k)

            # If still full, evict according to policy
            if len(self._entries) >= self.config.max_entries:
                evict_key: str | None = None
                if self.config.eviction_policy == CacheEvictionPolicy.LRU:
                    evict_key = min(self._entries.keys(), key=lambda k: self._entries[k].last_accessed_at)
                elif self.config.eviction_policy == CacheEvictionPolicy.LFU:
                    evict_key = min(
                        self._entries.keys(),
                        key=lambda k: (self._entries[k].access_count, self._entries[k].last_accessed_at),
                    )
                elif self.config.eviction_policy == CacheEvictionPolicy.FIFO:
                    evict_key = min(self._entries.keys(), key=lambda k: self._entries[k].created_at)

                if evict_key is not None:
                    self._delete_entry(evict_key)
                    self._evictions += 1

        new_entry = SemanticCacheEntry(
            key=key,
            query=query,
            query_vector=q_vec,
            response=response,
            similarity_score=1.0,
            created_at=now,
            last_accessed_at=now,
            access_count=1,
        )
        self._entries[key] = new_entry
        self._exact_index[norm_q] = key

    def clear(self) -> None:
        """Flush all cache entries and reset telemetry."""
        self._entries.clear()
        self._exact_index.clear()
        self._total_queries = 0
        self._exact_hits = 0
        self._semantic_hits = 0
        self._misses = 0
        self._evictions = 0
        self._total_latency_ms = 0.0

    def stats(self) -> SemanticCacheStats:
        """Return operational cache telemetry metrics."""
        total_hits = self._exact_hits + self._semantic_hits
        hit_rate = (total_hits / self._total_queries) if self._total_queries > 0 else 0.0
        avg_lat = (self._total_latency_ms / self._total_queries) if self._total_queries > 0 else 0.0

        return SemanticCacheStats(
            total_queries=self._total_queries,
            exact_hits=self._exact_hits,
            semantic_hits=self._semantic_hits,
            misses=self._misses,
            evictions=self._evictions,
            entry_count=len(self._entries),
            max_entries=self.config.max_entries,
            hit_rate=round(hit_rate, 4),
            avg_lookup_latency_ms=round(avg_lat, 2),
        )
