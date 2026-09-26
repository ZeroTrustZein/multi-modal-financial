"""Domain protocols and runtime-checkable abstract interfaces."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from multi_modal_financial.types import (
    AgentQuery,
    AgentResponse,
    ScoredChunk,
    SemanticCacheLookupResult,
    SemanticCacheStats,
)


@runtime_checkable
class CrossEncoderProtocol(Protocol):
    """Protocol for neural cross-encoder scoring models."""

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Compute cross-attention relevance scores for (query, document) pairs."""
        ...


@runtime_checkable
class RerankerProtocol(Protocol):
    """Protocol for second-stage candidate rerankers."""

    def rerank(
        self,
        query: str | AgentQuery,
        candidates: list[ScoredChunk],
        top_k: int | None = None,
    ) -> list[ScoredChunk]:
        """Rerank candidates based on neural and/or domain financial salience."""
        ...


@runtime_checkable
class SemanticCacheProtocol(Protocol):
    """Protocol for semantic query and response caching implementations."""

    def get(
        self,
        query: str,
        query_vector: list[float] | None = None,
        similarity_threshold: float | None = None,
    ) -> SemanticCacheLookupResult | None:
        """Lookup cached response using vector similarity matching."""
        ...

    def put(
        self,
        query: str,
        response: AgentResponse,
        query_vector: list[float] | None = None,
    ) -> None:
        """Store query, vector embedding, and response in semantic cache."""
        ...

    def clear(self) -> None:
        """Flush all cache entries."""
        ...

    def stats(self) -> SemanticCacheStats:
        """Return operational cache telemetry metrics."""
        ...


@runtime_checkable
class DenseIndexProtocol(Protocol):
    """Protocol for dense vector similarity indexing."""

    def search(self, query: str | list[float], top_k: int = 10) -> list[tuple[str, float]]:
        """Search nearest chunk IDs with similarity scores."""
        ...

    def add(self, chunk_id: str, text: str, vector: list[float] | None = None) -> None:
        """Insert or index chunk embedding."""
        ...


@runtime_checkable
class SparseIndexProtocol(Protocol):
    """Protocol for sparse lexical (BM25) indexing."""

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Search matching chunk IDs with lexical relevance scores."""
        ...

    def add(self, chunk_id: str, text: str) -> None:
        """Index document chunk tokens."""
        ...


@runtime_checkable
class RetrieverProtocol(Protocol):
    """Protocol for multi-modal hybrid retrieval engines."""

    def retrieve(
        self, query: str | AgentQuery, top_k: int = 10, alpha: float = 0.5
    ) -> list[ScoredChunk]:
        """Retrieve top ranked candidate chunks using hybrid scoring."""
        ...
