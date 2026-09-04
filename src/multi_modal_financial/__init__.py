"""Multi-Modal Financial Document RAG Pipeline with Hybrid BM25 & Citation Grounding."""

__version__ = "0.1.0"

from multi_modal_financial.agent.pipeline import FinancialRAGPipeline
from multi_modal_financial.analytics.comparator import PeriodComparator, VarianceResult
from multi_modal_financial.analytics.ratios import FinancialRatioCalculator, RatioSummary
from multi_modal_financial.data.cleaner import FinancialDataCleaner
from multi_modal_financial.data.loader import BatchDocumentLoader
from multi_modal_financial.data.synthetic import SyntheticFilingGenerator
from multi_modal_financial.data.validator import (
    FinancialTableValidator,
    ReconciliationResult,
    StatementReconciler,
    ValidationIssue,
    ValidationReport,
)
from multi_modal_financial.grounding.audit import GroundingAuditor, GroundingAuditReport
from multi_modal_financial.indexing.hybrid import HybridIndex
from multi_modal_financial.pipeline.orchestrator import (
    FinancialPipelineOrchestrator,
    OrchestratorConfig,
)
from multi_modal_financial.retrieval.fusion import HybridRetriever
from multi_modal_financial.storage.cache import EmbeddingCache, QueryCache
from multi_modal_financial.storage.persistence import IndexManifest, IndexPersistence
from multi_modal_financial.types import (
    AgentQuery,
    AgentResponse,
    Chunk,
    Citation,
    Currency,
    Document,
    DocumentMetadata,
    DocumentType,
    ExtractionFilter,
    FigureData,
    FinancialMetric,
    FinancialStatementType,
    GroundingStatus,
    GroundingVerdict,
    ModalType,
    ProvenanceRecord,
    QueryIntent,
    RetrievalBenchmarkResult,
    ScoredChunk,
    TableData,
    UnitScale,
    scale_multiplier,
)

__all__ = [
    "__version__",
    "ModalType",
    "DocumentType",
    "FinancialStatementType",
    "Currency",
    "UnitScale",
    "scale_multiplier",
    "QueryIntent",
    "GroundingStatus",
    "FinancialMetric",
    "DocumentMetadata",
    "TableData",
    "FigureData",
    "Chunk",
    "Document",
    "ScoredChunk",
    "Citation",
    "GroundingVerdict",
    "AgentQuery",
    "AgentResponse",
    "RetrievalBenchmarkResult",
    "ProvenanceRecord",
    "ExtractionFilter",
    "HybridIndex",
    "HybridRetriever",
    "FinancialRAGPipeline",
    "FinancialDataCleaner",
    "FinancialTableValidator",
    "StatementReconciler",
    "ValidationIssue",
    "ValidationReport",
    "ReconciliationResult",
    "SyntheticFilingGenerator",
    "BatchDocumentLoader",
    "IndexPersistence",
    "IndexManifest",
    "EmbeddingCache",
    "QueryCache",
    "FinancialRatioCalculator",
    "RatioSummary",
    "PeriodComparator",
    "VarianceResult",
    "GroundingAuditor",
    "GroundingAuditReport",
    "FinancialPipelineOrchestrator",
    "OrchestratorConfig",
]
