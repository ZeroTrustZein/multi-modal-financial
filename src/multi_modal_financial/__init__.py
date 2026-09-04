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
    Document,
    DocumentMetadata,
    GroundingVerdict,
    ModalType,
    ScoredChunk,
    TableData,
)

__all__ = [
    "__version__",
    "ModalType",
    "DocumentMetadata",
    "TableData",
    "Chunk",
    "Document",
    "ScoredChunk",
    "Citation",
    "GroundingVerdict",
    "AgentQuery",
    "AgentResponse",
    "HybridIndex",
    "HybridRetriever",
    "FinancialRAGPipeline",
]
