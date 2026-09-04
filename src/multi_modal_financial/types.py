"""Core data types for multi-modal financial RAG pipeline."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ModalType(str, Enum):
    """Supported modalities within financial documents."""
    TEXT = "text"
    TABLE = "table"
    FIGURE = "figure"
    METRIC = "metric"


class DocumentMetadata(BaseModel):
    """Metadata describing a financial filing or report."""
    doc_id: str
    filename: str
    ticker: str | None = None
    period: str | None = None  # e.g., "Q3", "FY"
    year: int | None = None
    doc_type: str = "filing"  # e.g., "10-K", "10-Q", "earnings_release", "analyst_report"
    page_count: int = 1
    extra: dict[str, Any] = Field(default_factory=dict)


class TableData(BaseModel):
    """Structured representation of a financial table."""
    headers: list[str] = Field(default_factory=list)
    rows: list[list[str]] = Field(default_factory=list)
    title: str | None = None
    unit: str | None = "USD"
    scale: str | None = "thousands"  # e.g., "millions", "billions"
    footnotes: list[str] = Field(default_factory=list)

    def to_markdown(self) -> str:
        """Render table into Markdown format for embedding and prompt feeding."""
        if not self.headers and not self.rows:
            return ""
        lines: list[str] = []
        if self.title:
            lines.append(f"**Table: {self.title}**")
        headers = self.headers or [f"Col {i+1}" for i in range(len(self.rows[0]))] if self.rows else []
        if headers:
            lines.append("| " + " | ".join(headers) + " |")
            lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for row in self.rows:
            lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
        return "\n".join(lines)


class FigureData(BaseModel):
    """Metadata and extracted summary of a chart or diagram."""
    figure_id: str
    caption: str | None = None
    chart_type: str | None = None  # e.g., "bar", "line", "waterfall"
    summary_text: str | None = None
    data_points: dict[str, float] = Field(default_factory=dict)


class Chunk(BaseModel):
    """Atomic unit of retrieved context, retaining modality and provenance."""
    chunk_id: str
    doc_id: str
    modal_type: ModalType = ModalType.TEXT
    content: str
    page_number: int = 1
    table_data: TableData | None = None
    figure_data: FigureData | None = None
    embedding: list[float] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Document(BaseModel):
    """Financial document container holding chunks and metadata."""
    doc_id: str
    metadata: DocumentMetadata
    chunks: list[Chunk] = Field(default_factory=list)
    raw_text: str | None = None


class ScoredChunk(BaseModel):
    """Retrieval candidate scored across sparse and dense modalities."""
    chunk: Chunk
    score: float
    dense_score: float = 0.0
    sparse_score: float = 0.0
    rank: int = 0


class Citation(BaseModel):
    """Grounding citation linking a synthesized claim to source evidence."""
    citation_id: str
    chunk_id: str
    doc_id: str
    page_number: int
    quote: str
    modal_type: ModalType = ModalType.TEXT
    confidence: float = 1.0


class GroundingVerdict(BaseModel):
    """Verification output validating claim attribution against retrieved source chunks."""
    claim: str
    citations: list[Citation] = Field(default_factory=list)
    is_supported: bool
    support_score: float  # [0.0, 1.0]
    reasoning: str = ""


class AgentQuery(BaseModel):
    """Input query to financial RAG agent."""
    query_str: str
    ticker_filter: str | None = None
    period_filter: str | None = None
    year_filter: int | None = None
    top_k: int = 5
    alpha: float = 0.5  # Weight for dense vs BM25 (0.0 = pure BM25, 1.0 = pure dense)


class AgentResponse(BaseModel):
    """Structured response from the financial RAG agent with citations and verification."""
    query: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    grounding_verdicts: list[GroundingVerdict] = Field(default_factory=list)
    retrieved_chunks: list[ScoredChunk] = Field(default_factory=list)
    execution_time_ms: float = 0.0
    overall_confidence: float = 0.0
