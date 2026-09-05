# API Reference: Multi-Modal Financial RAG Pipeline

Complete programmatic reference for all modules, classes, methods, models, and helper functions in **`multi_modal_financial`**.

---

## Table of Contents
1. [Pipeline Orchestrator (`multi_modal_financial.pipeline.orchestrator`)](#1-pipeline-orchestrator)
2. [Domain Types & Models (`multi_modal_financial.types`)](#2-domain-types--models)
3. [Agent & Routing (`multi_modal_financial.agent`)](#3-agent--routing)
4. [Indexing Engine (`multi_modal_financial.indexing`)](#4-indexing-engine)
5. [Retrieval & Fusion (`multi_modal_financial.retrieval`)](#5-retrieval--fusion)
6. [Citation Grounding & Verification (`multi_modal_financial.grounding`)](#6-citation-grounding--verification)
7. [Data Preprocessing & Validation (`multi_modal_financial.data`)](#7-data-preprocessing--validation)
8. [Multi-Modal Parsing (`multi_modal_financial.parsing`)](#8-multi-modal-parsing)
9. [Financial Analytics (`multi_modal_financial.analytics`)](#9-financial-analytics)
10. [Storage & Caching (`multi_modal_financial.storage`)](#10-storage--caching)
11. [CLI & Formatters (`multi_modal_financial.cli`)](#11-cli--formatters)

---

## 1. Pipeline Orchestrator

### `OrchestratorConfig`
```python
from multi_modal_financial.pipeline.orchestrator import OrchestratorConfig
```
Configuration dataclass controlling caching, validation, retrieval defaults, and audit thresholds.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_query_cache` | `bool` | `True` | Cache query responses using TTL and composite hashing. |
| `enable_embedding_cache` | `bool` | `True` | Cache dense embedding vectors in LRU cache. |
| `clean_text` | `bool` | `True` | Automatically run text cleaner on ingested documents. |
| `validate_tables` | `bool` | `True` | Validate table structure and accounting balance during ingestion. |
| `default_top_k` | `int` | `5` | Default number of chunks retrieved per query. |
| `default_alpha` | `float` | `0.5` | Default hybrid weighting (0.0 = BM25, 1.0 = Dense). |
| `query_cache_size` | `int` | `256` | Maximum entries retained in query cache. |
| `query_cache_ttl` | `float` | `3600.0` | Cache time-to-live in seconds. |
| `embedding_cache_size` | `int` | `5000` | Maximum entries retained in embedding cache. |
| `confidence_threshold` | `float` | `0.60` | Minimum support score required for claims. |
| `max_hallucination_rate` | `float` | `0.05` | Maximum permissible hallucination rate in compliance audits. |
| `persistence_dir` | `Path \| None` | `None` | Optional default directory for index serialization. |

---

### `FinancialPipelineOrchestrator`
```python
from multi_modal_financial.pipeline.orchestrator import FinancialPipelineOrchestrator
```
Master pipeline controller integrating document loading, indexing, retrieval, verification, ratio calculations, variance analysis, and index persistence.

#### `__init__(config: OrchestratorConfig | None = None, index: HybridIndex | None = None)`
Initializes the orchestrator and all dependent subsystems.

#### `ingest_files(paths: list[str | Path] | str | Path, default_ticker: str | None = None) -> list[Document]`
Loads, cleans, validates, and indexes single files, lists of files, or entire directories. Supported formats: `.txt`, `.md`, `.json`, `.pdf`.

#### `run_query(query: str | AgentQuery, top_k: int | None = None, alpha: float | None = None, use_cache: bool = True) -> AgentResponse`
Executes an end-to-end financial query. Handles query routing, cache lookup, hybrid retrieval, reranking, synthesis, and sentence-level citation verification.

#### `run_batch_queries(queries: Sequence[str | AgentQuery], top_k: int | None = None) -> list[AgentResponse]`
Executes a sequence of queries sequentially, leveraging the internal cache.

#### `audit_queries(queries: list[str]) -> GroundingAuditReport`
Executes a suite of queries and compiles an institutional factual compliance audit report.

#### `analyze_document_ratios(doc_id: str) -> RatioSummary`
Extracts and computes key financial ratios (margins, liquidity, solvency) for an indexed document.

#### `compare_documents(doc_id_base: str, doc_id_compare: str) -> list[VarianceResult]`
Computes horizontal financial variances (nominal and percentage changes) between two indexed documents.

#### `save_index(target_path: str | Path, compress: bool = False) -> Path`
Serializes the current hybrid index to a directory or `.zip` archive.

#### `load_index(source_path: str | Path) -> None`
Loads a serialized index from disk and re-links the retrieval engine and caches.

#### `status() -> dict[str, Any]`
Returns diagnostic status including total documents, chunks, vocabulary size, vector dimension, and cache statistics.

---

## 2. Domain Types & Models

Located in `multi_modal_financial.types`.

### Enumerations
- **`ModalType`**: `TEXT`, `TABLE`, `FIGURE`, `METRIC`, `HEADER`, `FOOTNOTE`, `SUMMARY`
- **`DocumentType`**: `TEN_K`, `TEN_Q`, `EIGHT_K`, `EARNINGS_RELEASE`, `ANALYST_REPORT`, `PROSPECTUS`, `PROXY`, `PRESS_RELEASE`, `TRANSCRIPT`, `FILING`, `OTHER`
- **`FinancialStatementType`**: `INCOME_STATEMENT`, `BALANCE_SHEET`, `CASH_FLOW`, `COMPREHENSIVE_INCOME`, `STOCKHOLDERS_EQUITY`, `SEGMENT_METRICS`, `NOTES`, `UNKNOWN`
- **`Currency`**: `USD`, `EUR`, `GBP`, `JPY`, `CAD`, `CHF`, `CNY`, `AUD`, `OTHER`
- **`UnitScale`**: `ONES`, `THOUSANDS`, `MILLIONS`, `BILLIONS`, `TRILLIONS`, `PERCENT`, `RATIO`, `BPS`
- **`QueryIntent`**: `METRIC_LOOKUP`, `COMPARATIVE_ANALYSIS`, `QUALITATIVE_RISK`, `TREND_CALCULATION`, `GENERAL`
- **`GroundingStatus`**: `FULLY_SUPPORTED`, `PARTIALLY_SUPPORTED`, `UNSUPPORTED`, `CONTRADICTED`

---

### `FinancialMetric`
Represents an atomic quantitative metric parsed from financial disclosures.

```python
class FinancialMetric(BaseFinancialModel):
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
```
- **`from_raw(name, raw_value, unit=None, scale=None, year=None, period=None, ticker=None, context=None, confidence=1.0) -> FinancialMetric`**:
  Class method that automatically parses raw strings like `"$1,250M"`, `"(45.2)%"`, `"€3.2B"`, extracting currency, negative accounting parentheses, and numerical scale multipliers.

---

### `DocumentMetadata`
```python
class DocumentMetadata(BaseFinancialModel):
    doc_id: str
    ticker: str | None = None
    company_name: str | None = None
    document_type: DocumentType = DocumentType.OTHER
    period: str | None = None
    year: int | None = None
    filing_date: str | None = None
    source_path: str | None = None
    page_count: int = 1
```

---

### `TableData`
Structured representation of financial tables.
- **Attributes**: `table_id`, `doc_id`, `headers` (`list[str]`), `rows` (`list[list[str]]`), `title`, `statement_type`, `unit_scale`, `currency`, `page_number`, `footnotes`.
- **`to_markdown() -> str`**: Renders table into standard GitHub-flavored Markdown.
- **`to_records() -> list[dict[str, str]]`**: Converts rows into dictionaries keyed by header names.
- **`get_cell(row_idx: int, col_idx: int) -> str | None`**: Safe coordinate cell accessor.
- **`get_column(header: str) -> list[str]`**: Extracts entire column values by header name.
- **`find_metric_row(metric_name: str) -> list[str] | None`**: Locates row matching metric keyword.
- **`extract_metrics() -> list[FinancialMetric]`**: Automatically parses all numeric cells into structured `FinancialMetric` instances.

---

### `FigureData`
Structured representation of extracted charts and graphs.
- **Attributes**: `figure_id`, `doc_id`, `caption`, `chart_type`, `x_label`, `y_label`, `data_points` (`dict[str, float]`), `unit`, `page_number`.
- **`data_summary() -> str`**: Generates a readable text summary of chart data points.

---

### `Chunk`
Fundamental retrieval unit in the hybrid index.
- **Attributes**: `chunk_id`, `doc_id`, `content`, `modality` (`ModalType`), `page_number`, `section`, `metadata`, `numeric_values` (`list[float]`).

---

### `Document`
Top-level multi-modal container.
- **Attributes**: `doc_id`, `metadata`, `raw_text`, `chunks`, `tables`, `figures`.
- **`get_all_metrics() -> list[FinancialMetric]`**: Aggregates all extracted metrics across tables and narrative chunks.
- **`to_json() -> str` / `from_json(json_str: str) -> Document`**: Full JSON serialization.

---

### `ScoredChunk`, `Citation`, `GroundingVerdict`, `AgentQuery`, `AgentResponse`
- **`ScoredChunk`**: Pairs a `Chunk` with its final retrieval score and ranking breakdown.
- **`Citation`**: Source attribution linking a claim to a specific chunk (`doc_id`, `page_number`, `modality`, `excerpt`, `score`).
- **`GroundingVerdict`**: Result of sentence-level fact check (`status`, `confidence`, `verified_numbers`, `hallucinated_numbers`, `explanation`).
- **`AgentQuery`**: Query parameters (`query_str`, `ticker_filter`, `period_filter`, `year_filter`, `top_k`, `alpha`).
- **`AgentResponse`**: Structured RAG output (`query`, `answer`, `citations`, `retrieved_chunks`, `verdict`, `groundedness_score`, `execution_time_ms`).

---

## 3. Agent & Routing

### `QueryRouter` (`multi_modal_financial.agent.router`)
Analyzes query strings to classify intent and extract target metadata.
- **`classify_intent(query: str) -> QueryIntent`**: Classifies query into `METRIC_LOOKUP`, `COMPARATIVE_ANALYSIS`, `TREND_CALCULATION`, `QUALITATIVE_RISK`, or `GENERAL`.
- **`extract_ticker(query: str) -> str | None`**: Detects ticker symbols using financial word filtering.
- **`extract_period(query: str) -> tuple[str | None, int | None]`**: Extracts quarter and year.
- **`build_agent_query(query: str, top_k: int = 5) -> AgentQuery`**: Builds fully parameterized query object.

---

### `FinancialRAGPipeline` (`multi_modal_financial.agent.pipeline`)
Standard RAG execution pipeline.
- **`ingest_text(text: str, doc_id: str, ticker: str | None = None, period: str | None = None, year: int | None = None) -> Document`**: Ingests raw text or Markdown string.
- **`ingest_file(file_path: str | Path, ticker: str | None = None) -> Document`**: Ingests document file.
- **`query(query_input: str | AgentQuery, top_k: int = 5, use_reranker: bool = True) -> AgentResponse`**: Executes retrieval, reranking, synthesis, and verification.

---

## 4. Indexing Engine

### `BM25Index` (`multi_modal_financial.indexing.bm25`)
Okapi BM25 sparse lexical search engine.
- **`add_document(doc_id: str, text: str, metadata: dict | None = None) -> None`**
- **`add_documents(docs: Sequence[tuple[str, str, dict]]) -> None`**
- **`search(query: str, top_k: int = 10) -> list[tuple[str, float]]`**: Returns list of `(doc_id, score)`.
- **`tokenize(text: str) -> list[str]`**: Financial tokenizer preserving symbols, currencies, percentages, and tickers.

---

### `DenseVectorIndex` (`multi_modal_financial.indexing.vector`)
Deterministic dense vector index using orthographic hash projections.
- **`add_vector(doc_id: str, text: str, metadata: dict | None = None) -> None`**
- **`add_precomputed(doc_id: str, vector: np.ndarray, metadata: dict | None = None) -> None`**
- **`search(query: str | np.ndarray, top_k: int = 10) -> list[tuple[str, float]]`**
- **`embed(text: str) -> np.ndarray`**: Generates normalized vector embedding.

---

### `HybridIndex` (`multi_modal_financial.indexing.hybrid`)
Unified container encapsulating both `BM25Index` and `DenseVectorIndex`.
- **`index_document(doc: Document) -> None`**: Indexes document and all its chunks into both indices.
- **`index_chunk(chunk: Chunk) -> None`**: Indexes individual chunk.
- **`get_chunk(chunk_id: str) -> Chunk | None`**
- **`get_document(doc_id: str) -> Document | None`**
- **`filter_chunks(ticker: str | None = None, period: str | None = None, year: int | None = None, modality: ModalType | None = None) -> set[str]`**: Returns matching chunk IDs.

---

## 5. Retrieval & Fusion

### `HybridRetriever` (`multi_modal_financial.retrieval.fusion`)
Executes dual-mode retrieval over `HybridIndex`.
- **`retrieve(agent_query: AgentQuery, use_rrf: bool = True) -> list[ScoredChunk]`**: Primary retrieval method using either RRF or convex score blend.
- **`retrieve_rrf(query_str: str, top_k: int = 5, k_rrf: int = 60, allowed_ids: set[str] | None = None) -> list[ScoredChunk]`**: Reciprocal rank fusion.
- **`retrieve_convex(query_str: str, top_k: int = 5, alpha: float = 0.5, allowed_ids: set[str] | None = None) -> list[ScoredChunk]`**: Linear score blending.

#### Standalone Fusion Functions
- **`reciprocal_rank_fusion(bm25_ranks: list[str], dense_ranks: list[str], k: int = 60, weights: tuple[float, float] = (1.0, 1.0)) -> list[tuple[str, float]]`**
- **`min_max_normalize(scores: dict[str, float]) -> dict[str, float]`**
- **`z_score_normalize(scores: dict[str, float]) -> dict[str, float]`**

---

### `FinancialReranker` (`multi_modal_financial.retrieval.reranker`)
- **`rerank(query: str, scored_chunks: list[ScoredChunk], query_ticker: str | None = None) -> list[ScoredChunk]`**: Re-weights candidate chunks by modality (tables/metrics boosted), ticker/period exact match, and token alignment.

---

## 6. Citation Grounding & Verification

### `CitationGrounder` (`multi_modal_financial.grounding.citation`)
- **`extract_citations(text: str) -> list[int]`**: Parses bracket citations (`[1]`, `[2]`).
- **`extract_claims(text: str) -> list[str]`**: Slices answer into individual factual assertions.
- **`create_provenance_record(chunk: Chunk, query_id: str | None = None) -> ProvenanceRecord`**: Generates verifiable audit provenance record.

---

### `GroundingVerifier` (`multi_modal_financial.grounding.verifier`)
- **`verify_claim(claim: str, context_chunk: Chunk) -> GroundingVerdict`**: Validates numbers and entities in a claim against context.
- **`verify_response(response_text: str, retrieved_chunks: list[Chunk]) -> GroundingVerdict`**: Verifies full multi-sentence answer.

---

### `GroundingAuditor` (`multi_modal_financial.grounding.audit`)
- **`audit_response(response: AgentResponse) -> GroundingVerdict`**: Verifies single response.
- **`audit_batch(responses: list[AgentResponse]) -> GroundingAuditReport`**: Compiles comprehensive audit report across a batch of responses.
- **`GroundingAuditReport`**: Dataclass with `total_queries`, `passed_queries`, `hallucination_rate`, `mean_confidence`, `audit_passed` (`bool`), `to_dict()`, `export_json()`.

---

## 7. Data Preprocessing & Validation

### `FinancialDataCleaner` (`multi_modal_financial.data.cleaner`)
- **`normalize_dashes(text: str) -> str`**
- **`strip_html_sec_tags(text: str) -> str`**
- **`normalize_accounting_negatives(text: str) -> str`**: Converts `(125.0)` to `-125.0`.
- **`clean_cell_footnotes(text: str) -> str`**: Removes markers like `[1]`, `[a]`.
- **`repair_markdown_table(table_text: str) -> str`**
- **`clean_document_text(text: str) -> str`**: Complete cleaning pipeline.

---

### `FinancialTableValidator` (`multi_modal_financial.data.validator`)
- **`validate_table(table: TableData) -> ValidationReport`**: Evaluates structural consistency and numeric sanity.

---

### `StatementReconciler` (`multi_modal_financial.data.validator`)
- **`reconcile_balance_sheet(table: TableData, tolerance: float = 1.0) -> ReconciliationResult`**: Tests $Assets = Liabilities + Equity$.
- **`reconcile_income_statement(table: TableData, tolerance: float = 1.0) -> ReconciliationResult`**: Tests $GrossProfit = Revenue - COGS$ and $OperatingIncome = GrossProfit - OpEx$.

---

### `BatchDocumentLoader` (`multi_modal_financial.data.loader`)
- **`load_file(path: str | Path, default_ticker: str | None = None) -> Document`**
- **`load_directory(dir_path: str | Path, default_ticker: str | None = None) -> list[Document]`**

---

### `SyntheticFilingGenerator` (`multi_modal_financial.data.synthetic`)
Generates realistic financial statements and filings for testing.
- **`generate_income_statement(ticker: str = "ACME", year: int = 2025, period: str = "Q3") -> TableData`**
- **`generate_balance_sheet(ticker: str = "ACME", year: int = 2025, period: str = "Q3") -> TableData`**
- **`generate_filing_document(ticker: str = "ACME", year: int = 2025, period: str = "Q3") -> Document`**

---

## 8. Multi-Modal Parsing

### `FinancialDocumentParser` (`multi_modal_financial.parsing.extractor`)
- **`parse_text(text: str, doc_id: str, ticker: str | None = None, period: str | None = None, year: int | None = None) -> Document`**
- **`parse_pdf(file_path: str | Path, ticker: str | None = None) -> Document`**

---

### `TableParser` (`multi_modal_financial.parsing.table_parser`)
- **`parse_markdown(markdown_text: str, doc_id: str = "doc") -> list[TableData]`**
- **`parse_csv(csv_text: str, doc_id: str = "doc", delimiter: str = ",") -> TableData`**
- **`extract_tables(text: str, doc_id: str = "doc") -> list[TableData]`**

---

### `FigureParser` (`multi_modal_financial.parsing.figure_parser`)
- **`parse_figure_block(block_text: str, doc_id: str = "doc") -> FigureData | None`**
- **`extract_figures(text: str, doc_id: str = "doc") -> list[FigureData]`**

---

## 9. Financial Analytics

### `FinancialRatioCalculator` (`multi_modal_financial.analytics.ratios`)
- **`compute_margins(revenue: float, gross_profit: float | None = None, operating_income: float | None = None, net_income: float | None = None) -> dict[str, float]`**
- **`compute_liquidity(current_assets: float, current_liabilities: float, cash: float | None = None) -> dict[str, float]`**
- **`compute_solvency(total_debt: float, total_equity: float) -> dict[str, float]`**
- **`compute_from_table(table: TableData) -> RatioSummary`**
- **`compute_from_document(doc: Document) -> RatioSummary`**

---

### `PeriodComparator` (`multi_modal_financial.analytics.comparator`)
- **`calculate_variance(base_val: float, compare_val: float, metric_name: str = "Metric") -> VarianceResult`**
- **`compare_metrics(metrics_base: list[FinancialMetric], metrics_compare: list[FinancialMetric]) -> list[VarianceResult]`**
- **`compare_table_columns(table: TableData, col_base_idx: int, col_compare_idx: int) -> list[VarianceResult]`**
- **`compare_documents(doc_base: Document, doc_compare: Document) -> list[VarianceResult]`**

---

## 10. Storage & Caching

### `IndexPersistence` (`multi_modal_financial.storage.persistence`)
- **`save(index: HybridIndex, target_path: str | Path, compress: bool = False) -> Path`**
- **`load(source_path: str | Path) -> HybridIndex`**
- **`export_corpus(documents: list[Document], target_json: str | Path) -> Path`**
- **`import_corpus(source_json: str | Path) -> list[Document]`**

---

### `EmbeddingCache` & `QueryCache` (`multi_modal_financial.storage.cache`)
- **`EmbeddingCache(max_size: int = 5000)`**: LRU vector cache.
  - `get(text: str) -> np.ndarray | None`
  - `put(text: str, vector: np.ndarray) -> None`
  - `get_or_compute(text: str, compute_fn: Callable[[str], np.ndarray]) -> np.ndarray`
  - `stats() -> dict[str, Any]`
- **`QueryCache(max_size: int = 256, ttl_seconds: float = 3600.0)`**: TTL + LRU query response cache.
  - `get(key: str) -> AgentResponse | None`
  - `put(key: str, response: AgentResponse) -> None`
  - `make_key(...) -> str`
  - `stats() -> dict[str, Any]`
  - `clear() -> None`

---

## 11. CLI & Formatters

Located in `multi_modal_financial.cli`.

- **Commands**:
  - `multi-modal-financial info`: Prints system architecture and environment status.
  - `multi-modal-financial ingest <file_or_dir> [--ticker <TICKER>]`: Ingests filings into the pipeline.
  - `multi-modal-financial query "<prompt>" [--ticker <TICKER>] [--top-k <K>] [--alpha <ALPHA>]`: Runs hybrid RAG query with formatted citations.
  - `multi-modal-financial benchmark`: Runs baseline BM25 vs Hybrid RRF retrieval benchmarks.
- **Formatters** (`multi_modal_financial.cli.formatters`):
  - `format_architecture_table() -> Table`
  - `format_response_panel(answer: str, latency_ms: float) -> Panel`
  - `format_citations_table(citations: list[Citation]) -> Table`
  - `format_benchmark_table(bm25_r1, bm25_mrr, hybrid_r1, hybrid_mrr) -> Table`
