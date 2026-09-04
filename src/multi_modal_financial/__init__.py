"""Multi-Modal Financial Document RAG Pipeline with Hybrid BM25 & Citation Grounding."""

__version__ = "0.1.0"

from multi_modal_financial.agent.pipeline import FinancialRAGPipeline
from multi_modal_financial.indexing.hybrid import HybridIndex
from multi_modal_financial.retrieval.fusion import HybridRetriever
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
]
