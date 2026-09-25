"""Core data types and domain models for multi-modal financial RAG pipeline."""

from __future__ import annotations

import math
import re
import time
from enum import Enum
from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict, Field, field_validator

T = TypeVar("T", bound="BaseFinancialModel")


class BaseFinancialModel(BaseModel):
    """Base domain model providing unified serialization and configuration."""

    model_config = ConfigDict(extra="ignore")

    def to_dict(self) -> dict[str, Any]:
        """Serialize model to dictionary."""
        return self.model_dump()

    @classmethod
    def from_dict(cls: type[T], data: dict[str, Any]) -> T:
        """Instantiate model from dictionary."""
        return cls.model_validate(data)


class ModalType(str, Enum):
    """Supported modalities within financial documents."""

    TEXT = "text"
    TABLE = "table"
    FIGURE = "figure"
    METRIC = "metric"
    HEADER = "header"
    FOOTNOTE = "footnote"
    SUMMARY = "summary"


class DocumentType(str, Enum):
    """Standard financial disclosure and SEC filing types."""

    TEN_K = "10-K"
    TEN_Q = "10-Q"
    EIGHT_K = "8-K"
    EARNINGS_RELEASE = "earnings_release"
    ANALYST_REPORT = "analyst_report"
    PROSPECTUS = "prospectus"
    PROXY = "DEF_14A"
    PRESS_RELEASE = "press_release"
    TRANSCRIPT = "earnings_transcript"
    FILING = "filing"
    OTHER = "other"


class FinancialStatementType(str, Enum):
    """Primary financial statement categories."""

    INCOME_STATEMENT = "income_statement"
    BALANCE_SHEET = "balance_sheet"
    CASH_FLOW = "cash_flow"
    COMPREHENSIVE_INCOME = "comprehensive_income"
    STOCKHOLDERS_EQUITY = "stockholders_equity"
    SEGMENT_METRICS = "segment_metrics"
    NOTES = "notes"
    UNKNOWN = "unknown"


class Currency(str, Enum):
    """Standard global currencies encountered in financial filings."""

    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"
    JPY = "JPY"
    CAD = "CAD"
    CHF = "CHF"
    CNY = "CNY"
    AUD = "AUD"
    OTHER = "OTHER"


class UnitScale(str, Enum):
    """Scale multipliers for reported financial metrics."""

    ONES = "ones"
    THOUSANDS = "thousands"
    MILLIONS = "millions"
    BILLIONS = "billions"
    TRILLIONS = "trillions"
    PERCENT = "percent"
    RATIO = "ratio"
    BPS = "bps"


class QueryIntent(str, Enum):
    """Classified intent of a financial query."""

    METRIC_LOOKUP = "metric_lookup"
    COMPARATIVE_ANALYSIS = "comparative_analysis"
    QUALITATIVE_RISK = "qualitative_risk"
    TREND_CALCULATION = "trend_calculation"
    GENERAL = "general"


class GroundingStatus(str, Enum):
    """Verification verdict status of a factual claim."""

    FULLY_SUPPORTED = "fully_supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    UNSUPPORTED = "unsupported"
    CONTRADICTED = "contradicted"


class RerankerStrategy(str, Enum):
    """Execution strategy for reranking retrieval candidates."""

    HEURISTIC = "heuristic"
    CROSS_ENCODER = "cross_encoder"
    HYBRID = "hybrid"


class RetrievalStrategy(str, Enum):
    """Retrieval ranking and fusion mode."""

    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID_RRF = "hybrid_rrf"
    HYBRID_CONVEX = "hybrid_convex"


class CacheEvictionPolicy(str, Enum):
    """Eviction algorithm for bounded caches."""

    LRU = "lru"
    LFU = "lfu"
    FIFO = "fifo"


class CacheHitType(str, Enum):
    """Categorization of cache lookup hit quality."""

    EXACT = "exact"
    SEMANTIC = "semantic"
    NONE = "none"


def scale_multiplier(scale: str | UnitScale | None) -> float:
    """Return numeric multiplier corresponding to scale denomination."""
    if scale is None:
        return 1.0
    s = scale.value if isinstance(scale, UnitScale) else str(scale).strip().lower()
    mapping: dict[str, float] = {
        "ones": 1.0,
        "units": 1.0,
        "one": 1.0,
        "thousands": 1_000.0,
        "thousand": 1_000.0,
        "k": 1_000.0,
        "millions": 1_000_000.0,
        "million": 1_000_000.0,
        "m": 1_000_000.0,
        "billions": 1_000_000_000.0,
        "billion": 1_000_000_000.0,
        "b": 1_000_000_000.0,
        "trillions": 1_000_000_000_000.0,
        "trillion": 1_000_000_000_000.0,
        "t": 1_000_000_000_000.0,
        "percent": 0.01,
        "percentage": 0.01,
        "%": 0.01,
        "ratio": 1.0,
        "bps": 0.0001,
        "basis_points": 0.0001,
    }
    return mapping.get(s, 1.0)


class FinancialMetric(BaseFinancialModel):
    """Structured financial metric extracted from text, tables, or figures."""

    name: str
    raw_value: str
    value: float | None = None
    unit: str = "USD"
    scale: str = "ones"
    normalized_value: float | None = None
    period: str | None = None
    year: int | None = None
    ticker: str | None = None
    context: str | None = None
    confidence: float = 1.0

    @classmethod
    def from_raw(
        cls,
        name: str,
        raw_value: str,
        unit: str | None = None,
        scale: str | None = None,
        year: int | None = None,
        period: str | None = None,
        ticker: str | None = None,
        context: str | None = None,
        confidence: float = 1.0,
    ) -> FinancialMetric:
        """Parse raw financial string (e.g. '$1,250M', '(45.2)%', '$3.5B') into FinancialMetric."""
        clean_s = raw_value.strip()

        # Detect currency
        detected_unit = unit or "USD"
        if "$" in clean_s:
            detected_unit = "USD"
        elif "€" in clean_s:
            detected_unit = "EUR"
        elif "£" in clean_s:
            detected_unit = "GBP"
        elif "¥" in clean_s:
            detected_unit = "JPY"

        # Detect accounting negative: (123.4)
        is_negative = False
        paren_match = re.search(r"^\((.*?)\)$", clean_s)
        if paren_match:
            is_negative = True
            clean_s = paren_match.group(1).strip()
        elif clean_s.startswith("-"):
            is_negative = True
            clean_s = clean_s.lstrip("-").strip()

        # Detect scale suffix
        detected_scale = scale or "ones"
        lower_s = clean_s.lower()
        if "%" in lower_s:
            detected_scale = "percent"
        elif lower_s.endswith("t") or "trillion" in lower_s:
            detected_scale = "trillions"
        elif lower_s.endswith("b") or "billion" in lower_s:
            detected_scale = "billions"
        elif lower_s.endswith("m") or "million" in lower_s:
            detected_scale = "millions"
        elif lower_s.endswith("k") or "thousand" in lower_s:
            detected_scale = "thousands"

        # Extract numeric core
        num_str = re.sub(r"[^\d.]", "", clean_s)
        val: float | None = None
        norm_val: float | None = None
        if num_str:
            try:
                base_val = float(num_str)
                if is_negative:
                    base_val = -base_val
                val = base_val
                multiplier = scale_multiplier(detected_scale)
                norm_val = round(base_val * multiplier, 6)
            except ValueError:
                val = None
                norm_val = None

        return cls(
            name=name.strip(),
            raw_value=raw_value,
            value=val,
            unit=detected_unit,
            scale=detected_scale,
            normalized_value=norm_val,
            period=period.upper() if period else None,
            year=year,
            ticker=ticker.upper() if ticker else None,
            context=context,
            confidence=confidence,
        )

    def formatted(self, include_unit: bool = True) -> str:
        """Format metric into human-readable financial string."""
        if self.value is None:
            return self.raw_value
        sign = "-" if self.value < 0 else ""
        abs_v = abs(self.value)
        unit_prefix = "$" if (include_unit and self.unit == "USD") else ""
        unit_suffix = (
            f" {self.unit}"
            if (include_unit and self.unit != "USD" and self.scale != "percent")
            else ""
        )
        if self.scale == "percent":
            return f"{sign}{abs_v:.1f}%"
        elif self.scale == "billions":
            return f"{sign}{unit_prefix}{abs_v:.2f}B{unit_suffix}"
        elif self.scale == "millions":
            return f"{sign}{unit_prefix}{abs_v:.1f}M{unit_suffix}"
        elif self.scale == "thousands":
            return f"{sign}{unit_prefix}{abs_v:,.0f}K{unit_suffix}"
        return f"{sign}{unit_prefix}{abs_v:,.2f}{unit_suffix}".strip()


class DocumentMetadata(BaseFinancialModel):
    """Metadata describing a financial filing or report."""

    doc_id: str
    filename: str
    ticker: str | None = None
    period: str | None = None  # e.g., "Q3", "FY"
    year: int | None = None
    doc_type: str = "filing"  # e.g., "10-K", "10-Q", "earnings_release", "analyst_report"
    page_count: int = 1
    company_name: str | None = None
    filing_date: str | None = None
    fiscal_period_end: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)

    @field_validator("ticker", mode="before")
    @classmethod
    def clean_ticker(cls, v: Any) -> str | None:
        """Ensure ticker is stripped and capitalized."""
        if v is None:
            return None
        return str(v).strip().upper()

    @field_validator("period", mode="before")
    @classmethod
    def clean_period(cls, v: Any) -> str | None:
        """Ensure period is stripped and capitalized."""
        if v is None:
            return None
        return str(v).strip().upper()

    @field_validator("page_count")
    @classmethod
    def validate_page_count(cls, v: int) -> int:
        """Validate page count is strictly positive."""
        return max(1, v)

    @field_validator("year")
    @classmethod
    def validate_year(cls, v: int | None) -> int | None:
        """Validate reasonable calendar year range."""
        if v is not None and (v < 1900 or v > 2100):
            raise ValueError(f"Year {v} outside valid range [1900, 2100].")
        return v

    def header_summary(self) -> str:
        """Produce formatted filing summary header."""
        parts: list[str] = []
        if self.ticker:
            parts.append(self.ticker)
        if self.period and self.year:
            parts.append(f"{self.period} {self.year}")
        elif self.year:
            parts.append(str(self.year))
        elif self.period:
            parts.append(self.period)
        parts.append(self.doc_type)
        return f"[{' | '.join(parts)}]"


class TableData(BaseFinancialModel):
    """Structured representation of a financial table."""

    table_id: str | None = None
    headers: list[str] = Field(default_factory=list)
    rows: list[list[str]] = Field(default_factory=list)
    title: str | None = None
    unit: str | None = "USD"
    scale: str | None = "thousands"  # e.g., "millions", "billions"
    statement_type: FinancialStatementType = FinancialStatementType.UNKNOWN
    footnotes: list[str] = Field(default_factory=list)

    @property
    def row_count(self) -> int:
        """Number of data rows in table."""
        return len(self.rows)

    @property
    def col_count(self) -> int:
        """Number of columns in table."""
        return len(self.headers) if self.headers else (len(self.rows[0]) if self.rows else 0)

    def to_markdown(self) -> str:
        """Render table into Markdown format for embedding and prompt feeding."""
        if not self.headers and not self.rows:
            return ""
        lines: list[str] = []
        if self.title:
            lines.append(f"**Table: {self.title}**")
        headers = (
            self.headers or [f"Col {i + 1}" for i in range(len(self.rows[0]))] if self.rows else []
        )
        if headers:
            lines.append("| " + " | ".join(headers) + " |")
            lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for row in self.rows:
            lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
        if self.footnotes:
            lines.append("")
            for fn in self.footnotes:
                lines.append(f"* {fn}")
        return "\n".join(lines)

    def to_records(self) -> list[dict[str, str]]:
        """Convert rows into list of dictionaries mapping header -> cell value."""
        records: list[dict[str, str]] = []
        if not self.headers:
            return records
        for row in self.rows:
            record: dict[str, str] = {}
            for idx, header in enumerate(self.headers):
                record[header] = str(row[idx]) if idx < len(row) else ""
            records.append(record)
        return records

    def get_cell(self, row_idx: int, col_idx: int) -> str | None:
        """Safely retrieve cell contents by 0-indexed row and column coordinates."""
        if 0 <= row_idx < len(self.rows):
            row = self.rows[row_idx]
            if 0 <= col_idx < len(row):
                return str(row[col_idx])
        return None

    def get_column(self, col_name_or_idx: str | int) -> list[str]:
        """Extract all column values by header name or 0-based column index."""
        if isinstance(col_name_or_idx, int):
            idx = col_name_or_idx
        else:
            name_lower = col_name_or_idx.strip().lower()
            try:
                idx = next(i for i, h in enumerate(self.headers) if h.strip().lower() == name_lower)
            except StopIteration:
                return []
        return [str(row[idx]) if idx < len(row) else "" for row in self.rows]

    def find_metric_row(self, metric_name: str) -> list[str] | None:
        """Find the first row whose first column contains the target metric name."""
        target = metric_name.strip().lower()
        for row in self.rows:
            if row and target in str(row[0]).strip().lower():
                return row
        return None

    def extract_metrics(self) -> list[FinancialMetric]:
        """Extract individual FinancialMetric entities across table rows and columns."""
        metrics: list[FinancialMetric] = []
        if len(self.headers) < 2 or not self.rows:
            return metrics

        for row in self.rows:
            if not row or len(row) < 2:
                continue
            metric_name = str(row[0]).strip()
            if not metric_name or metric_name.startswith("---"):
                continue

            for col_idx in range(1, min(len(self.headers), len(row))):
                header = self.headers[col_idx].strip()
                cell_val = str(row[col_idx]).strip()
                if not cell_val or cell_val in {"-", "—", "N/A", "n/a", "nil"}:
                    continue

                year_val: int | None = None
                year_match = re.search(r"\b(20\d{2}|19\d{2})\b", header)
                if year_match:
                    year_val = int(year_match.group(1))

                period_val: str | None = None
                period_match = re.search(r"\b(Q[1-4]|FY|FY\d{2,4})\b", header, re.IGNORECASE)
                if period_match:
                    period_val = period_match.group(1).upper()

                metric = FinancialMetric.from_raw(
                    name=metric_name,
                    raw_value=cell_val,
                    unit=self.unit or "USD",
                    scale=self.scale or "thousands",
                    year=year_val,
                    period=period_val,
                    context=f"Table: {self.title or 'Financial Table'} | Column: {header}",
                )
                metrics.append(metric)

        return metrics

    def find_metric_value(self, col_idx: int, *keywords: str) -> float | None:
        """Find numerical value in a given column prioritizing exact label match over substring."""
        # Pass 1: exact matches
        for kw in keywords:
            kw_clean = kw.lower().strip()
            for row in self.rows:
                if not row:
                    continue
                if row[0].lower().strip() == kw_clean:
                    if col_idx < len(row):
                        metric = FinancialMetric.from_raw(name=row[0], raw_value=row[col_idx])
                        if metric.value is not None and not math.isnan(metric.value):
                            return metric.value

        # Pass 2: substring matches excluding 'current' for total assets/liabilities
        for kw in keywords:
            kw_clean = kw.lower().strip()
            for row in self.rows:
                if not row:
                    continue
                row_label = row[0].lower().strip()
                if kw_clean in ["total assets", "assets"] and "current" in row_label:
                    continue
                if kw_clean in ["total liabilities", "liabilities"] and "current" in row_label:
                    continue
                if kw_clean in row_label:
                    if col_idx < len(row):
                        metric = FinancialMetric.from_raw(name=row[0], raw_value=row[col_idx])
                        if metric.value is not None and not math.isnan(metric.value):
                            return metric.value
        return None


class FigureData(BaseFinancialModel):
    """Metadata and extracted summary of a chart or diagram."""

    figure_id: str
    caption: str | None = None
    chart_type: str | None = None  # e.g., "bar", "line", "waterfall", "pie"
    summary_text: str | None = None
    data_points: dict[str, float] = Field(default_factory=dict)
    x_label: str | None = None
    y_label: str | None = None
    unit: str | None = None
    scale: str | None = None

    def total_value(self) -> float:
        """Sum of all numerical data points in figure."""
        return sum(self.data_points.values())

    def max_point(self) -> tuple[str, float] | None:
        """Key-value tuple of the largest data point."""
        if not self.data_points:
            return None
        return max(self.data_points.items(), key=lambda item: item[1])

    def min_point(self) -> tuple[str, float] | None:
        """Key-value tuple of the smallest data point."""
        if not self.data_points:
            return None
        return min(self.data_points.items(), key=lambda item: item[1])

    def to_markdown(self) -> str:
        """Render figure as Markdown text block."""
        lines: list[str] = []
        if self.caption:
            lines.append(f"**Figure: {self.caption}**")
        if self.summary_text:
            lines.append(self.summary_text)
        if self.data_points:
            unit_str = f" {self.unit}" if self.unit else ""
            for k, v in self.data_points.items():
                lines.append(f"- {k}: {v:g}{unit_str}")
        return "\n".join(lines)


class Chunk(BaseFinancialModel):
    """Atomic unit of retrieved context, retaining modality and provenance."""

    chunk_id: str
    doc_id: str
    modal_type: ModalType = ModalType.TEXT
    content: str = ""
    page_number: int = 1
    section_title: str | None = None
    table_data: TableData | None = None
    figure_data: FigureData | None = None
    metrics: list[FinancialMetric] = Field(default_factory=list)
    embedding: list[float] | None = None
    token_count: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("page_number")
    @classmethod
    def validate_page_number(cls, v: int) -> int:
        """Validate page number is at least 1."""
        return max(1, v)

    def word_count(self) -> int:
        """Count whitespace-delimited words in chunk content."""
        return len(self.content.split())

    def character_count(self) -> int:
        """Count total characters in chunk content."""
        return len(self.content)

    def has_numbers(self) -> bool:
        """Check whether chunk contains numerical data."""
        return bool(re.search(r"\d", self.content))

    def extract_numbers(self) -> list[float]:
        """Extract list of all numeric floating-point values in chunk text."""
        raw_nums = re.findall(r"-?\b\d+(?:,\d{3})*(?:\.\d+)?\b", self.content)
        parsed: list[float] = []
        for n in raw_nums:
            cleaned = n.replace(",", "")
            try:
                parsed.append(float(cleaned))
            except ValueError:
                pass
        return parsed


class Document(BaseFinancialModel):
    """Financial document container holding chunks and metadata."""

    doc_id: str
    metadata: DocumentMetadata
    chunks: list[Chunk] = Field(default_factory=list)
    raw_text: str | None = None

    def chunk_count(self) -> int:
        """Total number of chunks in document."""
        return len(self.chunks)

    def chunks_by_modal(self, modal_type: ModalType) -> list[Chunk]:
        """Filter chunks matching a specific modal type."""
        return [c for c in self.chunks if c.modal_type == modal_type]

    def get_tables(self) -> list[TableData]:
        """Collect all structured tables extracted in document."""
        return [c.table_data for c in self.chunks if c.table_data is not None]

    def get_figures(self) -> list[FigureData]:
        """Collect all figure summaries extracted in document."""
        return [c.figure_data for c in self.chunks if c.figure_data is not None]

    def get_all_metrics(self) -> list[FinancialMetric]:
        """Aggregate all extracted financial metrics across chunks and tables."""
        all_metrics: list[FinancialMetric] = []
        for c in self.chunks:
            all_metrics.extend(c.metrics)
            if c.table_data:
                all_metrics.extend(c.table_data.extract_metrics())
        return all_metrics

    def to_json(self, indent: int | None = 2) -> str:
        """Serialize document to JSON string."""
        return self.model_dump_json(indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> Document:
        """Instantiate document from JSON string."""
        return cls.model_validate_json(json_str)


class ScoredChunk(BaseFinancialModel):
    """Retrieval candidate scored across sparse and dense modalities."""

    chunk: Chunk
    score: float
    dense_score: float = 0.0
    sparse_score: float = 0.0
    rank: int = 0
    rerank_score: float | None = None
    cross_encoder_score: float | None = None
    semantic_score: float | None = None
    modality_bonus: float = 0.0
    explanation: str | None = None


class Citation(BaseFinancialModel):
    """Grounding citation linking a synthesized claim to source evidence."""

    citation_id: str
    chunk_id: str
    doc_id: str
    page_number: int
    quote: str
    modal_type: ModalType = ModalType.TEXT
    confidence: float = 1.0
    matched_numbers: list[str] = Field(default_factory=list)

    def format_inline(self) -> str:
        """Format citation as an inline reference badge."""
        return f"[{self.doc_id}:p{self.page_number}#{self.chunk_id}]"


class GroundingVerdict(BaseFinancialModel):
    """Verification output validating claim attribution against retrieved source chunks."""

    claim: str
    citations: list[Citation] = Field(default_factory=list)
    is_supported: bool
    support_score: float  # [0.0, 1.0]
    status: GroundingStatus = GroundingStatus.FULLY_SUPPORTED
    unsupported_numbers: list[str] = Field(default_factory=list)
    missing_entities: list[str] = Field(default_factory=list)
    reasoning: str = ""


class AgentQuery(BaseFinancialModel):
    """Input query to financial RAG agent."""

    query_str: str
    ticker_filter: str | None = None
    period_filter: str | None = None
    year_filter: int | None = None
    doc_type_filter: str | None = None
    modal_filter: ModalType | None = None
    intent: QueryIntent = QueryIntent.GENERAL
    top_k: int = Field(default=5, ge=1)
    alpha: float = Field(default=0.5, ge=0.0, le=1.0)  # Weight for dense vs BM25
    use_reranker: bool = True
    reranker_strategy: RerankerStrategy | None = None
    reranker_top_k: int | None = None
    reranker_threshold: float | None = None
    retrieval_strategy: RetrievalStrategy = RetrievalStrategy.HYBRID_RRF
    use_semantic_cache: bool = True
    similarity_threshold: float | None = None

    @field_validator("ticker_filter", mode="before")
    @classmethod
    def clean_ticker_filter(cls, v: Any) -> str | None:
        """Ensure ticker filter is capitalized and trimmed."""
        if v is None:
            return None
        return str(v).strip().upper()

    @field_validator("period_filter", mode="before")
    @classmethod
    def clean_period_filter(cls, v: Any) -> str | None:
        """Ensure period filter is capitalized and trimmed."""
        if v is None:
            return None
        return str(v).strip().upper()


class AgentResponse(BaseFinancialModel):
    """Structured response from the financial RAG agent with citations and verification."""

    query: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    grounding_verdicts: list[GroundingVerdict] = Field(default_factory=list)
    retrieved_chunks: list[ScoredChunk] = Field(default_factory=list)
    execution_time_ms: float = 0.0
    overall_confidence: float = 0.0
    cache_hit: bool = False
    cache_type: CacheHitType = CacheHitType.NONE
    cache_similarity: float | None = None
    rerank_explanations: list[RerankExplanation] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def is_fully_grounded(self) -> bool:
        """Return True if all synthesized claims are verified and supported."""
        if not self.grounding_verdicts:
            return False
        return all(v.is_supported for v in self.grounding_verdicts)

    def unsupported_claims(self) -> list[GroundingVerdict]:
        """Return list of verdicts that failed verification."""
        return [v for v in self.grounding_verdicts if not v.is_supported]

    def summary(self) -> str:
        """Return concise execution summary."""
        grounded_count = sum(1 for v in self.grounding_verdicts if v.is_supported)
        total_claims = len(self.grounding_verdicts)
        cache_part = f" | Cache: {self.cache_type.value}" if self.cache_hit else ""
        return (
            f"Query: '{self.query}' | Chunks: {len(self.retrieved_chunks)} | "
            f"Citations: {len(self.citations)} | Grounded: {grounded_count}/{total_claims} | "
            f"Conf: {self.overall_confidence:.2f} | Latency: {self.execution_time_ms:.1f}ms{cache_part}"
        )

    def to_json(self, indent: int | None = 2) -> str:
        """Serialize response to JSON string."""
        return self.model_dump_json(indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> AgentResponse:
        """Instantiate response from JSON string."""
        return cls.model_validate_json(json_str)


class RerankExplanation(BaseFinancialModel):
    """Auditable attribution explaining why a chunk's rank changed during reranking."""

    chunk_id: str
    initial_rank: int
    final_rank: int
    initial_score: float
    final_score: float
    cross_encoder_score: float | None = None
    heuristic_score: float | None = None
    modality_bonus: float = 0.0
    reasons: list[str] = Field(default_factory=list)


class RerankerConfig(BaseFinancialModel):
    """Configuration for cross-encoder and hybrid neural reranking."""

    strategy: RerankerStrategy = RerankerStrategy.HYBRID
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_k: int = Field(default=10, ge=1)
    score_threshold: float = Field(default=0.0)
    batch_size: int = Field(default=32, ge=1)
    table_boost: float = Field(default=0.2, ge=0.0)
    metric_boost: float = Field(default=0.25, ge=0.0)
    figure_boost: float = Field(default=0.15, ge=0.0)
    entity_boost: float = Field(default=0.2, ge=0.0)
    cross_encoder_weight: float = Field(default=0.7, ge=0.0, le=1.0)
    heuristic_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    device: str = "cpu"
    normalize_scores: bool = True


class HybridSearchConfig(BaseFinancialModel):
    """Configuration for sparse BM25 and dense embedding hybrid retrieval."""

    strategy: RetrievalStrategy = RetrievalStrategy.HYBRID_RRF
    top_k: int = Field(default=10, ge=1)
    alpha: float = Field(default=0.5, ge=0.0, le=1.0)
    rrf_k: int = Field(default=60, ge=1)
    rerank_top_k: int = Field(default=5, ge=1)
    score_threshold: float = Field(default=0.0, ge=0.0)


class SemanticCacheConfig(BaseFinancialModel):
    """Configuration for semantic vector similarity caching."""

    enabled: bool = True
    similarity_threshold: float = Field(default=0.88, ge=0.0, le=1.0)
    max_entries: int = Field(default=1000, ge=1)
    ttl_seconds: float = Field(default=3600.0, ge=0.0)
    eviction_policy: CacheEvictionPolicy = CacheEvictionPolicy.LRU
    distance_metric: str = "cosine"


class SemanticCacheEntry(BaseFinancialModel):
    """Cached entry stored in semantic cache with query vector representation."""

    key: str
    query: str
    query_vector: list[float] = Field(default_factory=list)
    response: AgentResponse
    similarity_score: float = 1.0
    created_at: float = Field(default_factory=time.time)
    last_accessed_at: float = Field(default_factory=time.time)
    access_count: int = 1
    metadata: dict[str, Any] = Field(default_factory=dict)

    def is_expired(self, ttl_seconds: float, current_time: float | None = None) -> bool:
        """Check whether entry has exceeded ttl."""
        now = time.time() if current_time is None else current_time
        return (now - self.created_at) > ttl_seconds

    def touch(self, current_time: float | None = None) -> None:
        """Update last accessed timestamp and increment hit counter."""
        self.last_accessed_at = time.time() if current_time is None else current_time
        self.access_count += 1


class SemanticCacheLookupResult(BaseFinancialModel):
    """Outcome of a semantic cache lookup request."""

    hit: bool
    similarity: float = 0.0
    matched_query: str | None = None
    response: AgentResponse | None = None
    lookup_latency_ms: float = 0.0
    hit_type: CacheHitType = CacheHitType.NONE


class SemanticCacheStats(BaseFinancialModel):
    """Telemetry and operational metrics for semantic caching."""

    total_queries: int = 0
    exact_hits: int = 0
    semantic_hits: int = 0
    misses: int = 0
    evictions: int = 0
    entry_count: int = 0
    max_entries: int = 1000
    hit_rate: float = 0.0
    avg_lookup_latency_ms: float = 0.0


class RetrievalBenchmarkResult(BaseFinancialModel):
    """Performance evaluation metrics for financial retrieval pipelines."""

    method_name: str
    recall_at_1: float = Field(ge=0.0, le=1.0)
    recall_at_k: float = Field(ge=0.0, le=1.0)
    mrr: float = Field(ge=0.0, le=1.0)
    ndcg: float = Field(default=0.0, ge=0.0, le=1.0)
    query_count: int = Field(default=0, ge=0)
    avg_latency_ms: float = Field(default=0.0, ge=0.0)


class ProvenanceRecord(BaseFinancialModel):
    """Fine-grained audit trail tracing a synthesized claim to source evidence."""

    record_id: str
    doc_id: str
    chunk_id: str
    page_number: int = 1
    modal_type: ModalType = ModalType.TEXT
    table_cell: tuple[int, int] | None = None
    source_snippet: str = ""
    verified: bool = False


class ExtractionFilter(BaseFinancialModel):
    """Filters applied during multi-modal document parsing and ingestion."""

    ticker: str | None = None
    period: str | None = None
    year: int | None = None
    doc_type: str | None = None
    modal_types: list[ModalType] = Field(default_factory=list)
    min_confidence: float = 0.0

    @field_validator("ticker", mode="before")
    @classmethod
    def clean_filter_ticker(cls, v: Any) -> str | None:
        """Capitalize and strip ticker."""
        return str(v).strip().upper() if v is not None else None
