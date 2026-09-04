"""End-to-end Pipeline Orchestrator integrating all financial RAG subsystems."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Any

from multi_modal_financial.agent.pipeline import FinancialRAGPipeline
from multi_modal_financial.agent.router import QueryRouter
from multi_modal_financial.analytics.comparator import PeriodComparator, VarianceResult
from multi_modal_financial.analytics.ratios import FinancialRatioCalculator, RatioSummary
from multi_modal_financial.data.cleaner import FinancialDataCleaner
from multi_modal_financial.data.loader import BatchDocumentLoader
from multi_modal_financial.data.validator import FinancialTableValidator
from multi_modal_financial.grounding.audit import GroundingAuditReport, GroundingAuditor
from multi_modal_financial.grounding.verifier import GroundingVerifier
from multi_modal_financial.indexing.hybrid import HybridIndex
from multi_modal_financial.retrieval.fusion import HybridRetriever
from multi_modal_financial.retrieval.reranker import FinancialReranker
from multi_modal_financial.storage.cache import EmbeddingCache, QueryCache
from multi_modal_financial.storage.persistence import IndexPersistence
from multi_modal_financial.types import AgentQuery, AgentResponse, Document


@dataclass
class OrchestratorConfig:
    """Runtime configuration for FinancialPipelineOrchestrator."""

    enable_query_cache: bool = True
    enable_embedding_cache: bool = True
    clean_text: bool = True
    validate_tables: bool = True
    default_top_k: int = 5
    default_alpha: float = 0.5
    query_cache_size: int = 256
    query_cache_ttl: float = 3600.0
    embedding_cache_size: int = 5000
    confidence_threshold: float = 0.60
    max_hallucination_rate: float = 0.05
    persistence_dir: Path | None = None
    extra_options: dict[str, Any] = field(default_factory=dict)


class FinancialPipelineOrchestrator:
    """Master orchestrator integrating ingestion, indexing, retrieval, analytics, grounding, and storage."""

    def __init__(
        self,
        config: OrchestratorConfig | None = None,
        index: HybridIndex | None = None,
    ):
        self.config = config or OrchestratorConfig()

        # Cache Subsystem
        self.embedding_cache = (
            EmbeddingCache(max_size=self.config.embedding_cache_size)
            if self.config.enable_embedding_cache
            else None
        )
        self.query_cache = (
            QueryCache(
                max_size=self.config.query_cache_size,
                ttl_seconds=self.config.query_cache_ttl,
            )
            if self.config.enable_query_cache
            else None
        )

        # Storage & Persistence
        self.persistence = IndexPersistence()

        # Indexing & Search
        self.index = index or HybridIndex()
        self.retriever = HybridRetriever(self.index)
        self.reranker = FinancialReranker()
        self.router = QueryRouter()
        self.verifier = GroundingVerifier(confidence_threshold=self.config.confidence_threshold)

        # Core Agent Pipeline
        self.rag_pipeline = FinancialRAGPipeline(
            hybrid_index=self.index,
            retriever=self.retriever,
            reranker=self.reranker,
            verifier=self.verifier,
            router=self.router,
        )

        # Data & Preprocessing Subsystems
        self.cleaner = FinancialDataCleaner()
        self.validator = FinancialTableValidator()
        self.loader = BatchDocumentLoader(
            parser=self.rag_pipeline.parser,
            clean_text=self.config.clean_text,
            validate_tables=self.config.validate_tables,
        )

        # Analytics Subsystems
        self.ratios = FinancialRatioCalculator()
        self.comparator = PeriodComparator()

        # Grounding & Audit Subsystem
        self.auditor = GroundingAuditor(
            max_allowed_hallucination_rate=self.config.max_hallucination_rate,
            min_confidence=self.config.confidence_threshold,
        )

    def ingest_files(
        self,
        paths: list[str | Path] | str | Path,
        default_ticker: str | None = None,
    ) -> list[Document]:
        """Ingest single file, list of files, or directory into the pipeline index."""
        ingested: list[Document] = []
        if isinstance(paths, (str, Path)):
            p = Path(paths)
            if p.is_dir():
                docs = self.loader.load_directory(p, default_ticker=default_ticker)
            else:
                docs = [self.loader.load_file(p, default_ticker=default_ticker)]
        else:
            docs = []
            for item in paths:
                p = Path(item)
                if p.is_dir():
                    docs.extend(self.loader.load_directory(p, default_ticker=default_ticker))
                elif p.is_file():
                    docs.append(self.loader.load_file(p, default_ticker=default_ticker))

        for doc in docs:
            self.index.index_document(doc)
            ingested.append(doc)

        return ingested

    def run_query(
        self,
        query: str | AgentQuery,
        top_k: int | None = None,
        alpha: float | None = None,
        use_cache: bool = True,
    ) -> AgentResponse:
        """Execute query with automatic routing, caching, retrieval, reranking, and verification."""
        top_k = top_k or self.config.default_top_k
        alpha = alpha or self.config.default_alpha

        if isinstance(query, str):
            agent_q = self.router.build_agent_query(query, top_k=top_k)
            if alpha is not None:
                agent_q.alpha = alpha
        else:
            agent_q = query

        cache_key = None
        if use_cache and self.query_cache is not None:
            cache_key = QueryCache.make_key(
                query_str=agent_q.query_str,
                ticker=agent_q.ticker_filter,
                period=agent_q.period_filter,
                year=agent_q.year_filter,
                top_k=agent_q.top_k,
                alpha=agent_q.alpha,
            )
            cached_resp = self.query_cache.get(cache_key)
            if cached_resp is not None:
                return cached_resp

        # Run RAG execution
        resp = self.rag_pipeline.query(agent_q, top_k=agent_q.top_k, use_reranker=True)

        if use_cache and self.query_cache is not None and cache_key:
            self.query_cache.put(cache_key, resp)

        return resp

    def run_batch_queries(
        self,
        queries: list[str | AgentQuery],
        top_k: int | None = None,
    ) -> list[AgentResponse]:
        """Execute a batch of financial queries sequentially or concurrently."""
        responses: list[AgentResponse] = []
        for q in queries:
            resp = self.run_query(q, top_k=top_k)
            responses.append(resp)
        return responses

    def audit_queries(self, queries: list[str]) -> GroundingAuditReport:
        """Run batch queries and compile an end-to-end factual compliance audit report."""
        responses = self.run_batch_queries(queries)
        return self.auditor.audit_batch(responses)

    def analyze_document_ratios(self, doc_id: str) -> RatioSummary:
        """Compute financial ratios for an indexed document."""
        doc = self.index.get_document(doc_id)
        if not doc:
            raise KeyError(f"Document ID '{doc_id}' not found in index.")
        return self.ratios.compute_from_document(doc)

    def compare_documents(self, doc_id_base: str, doc_id_compare: str) -> list[VarianceResult]:
        """Compare financial performance metrics across two indexed documents."""
        doc_base = self.index.get_document(doc_id_base)
        doc_compare = self.index.get_document(doc_id_compare)
        if not doc_base or not doc_compare:
            raise KeyError(f"Missing one or both documents: {doc_id_base}, {doc_id_compare}")
        return self.comparator.compare_documents(doc_base, doc_compare)

    def save_index(self, target_path: str | Path, compress: bool = False) -> Path:
        """Persist current index to disk."""
        return self.persistence.save(self.index, target_path, compress=compress)

    def load_index(self, source_path: str | Path) -> None:
        """Load index from disk and re-link pipeline retriever."""
        self.index = self.persistence.load(source_path)
        self.rag_pipeline.index = self.index
        self.rag_pipeline.retriever = HybridRetriever(self.index)
        self.retriever = self.rag_pipeline.retriever
        if self.query_cache:
            self.query_cache.clear()

    def status(self) -> dict[str, Any]:
        """Summary diagnostics of the pipeline and subsystems."""
        stats = {
            "total_documents": self.index.total_documents(),
            "total_chunks": self.index.total_chunks(),
            "bm25_vocab_size": len(self.index.bm25.doc_freqs),
            "bm25_avgdl": self.index.bm25.avgdl,
            "vector_dimension": self.index.vector.dimension,
            "query_cache": self.query_cache.stats() if self.query_cache else None,
            "embedding_cache": self.embedding_cache.stats() if self.embedding_cache else None,
        }
        return stats
