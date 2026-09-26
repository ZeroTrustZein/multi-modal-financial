# API Reference: Multi-Modal Financial RAG Pipeline

Complete programmatic reference for all modules, classes, methods, models, protocols, and helper functions in **`multi_modal_financial`**.

---

## Table of Contents
1. [Pipeline Orchestrator (`multi_modal_financial.pipeline.orchestrator`)](#1-pipeline-orchestrator)
2. [Runtime Protocols & Interfaces (`multi_modal_financial.interfaces`)](#2-runtime-protocols--interfaces)
3. [Domain Types & Models (`multi_modal_financial.types`)](#3-domain-types--models)
4. [Retrieval & Fusion Core (`multi_modal_financial.retrieval.fusion`)](#4-retrieval--fusion-core)
5. [Neural Cross-Encoder & Reranker (`multi_modal_financial.retrieval.reranker`)](#5-neural-cross-encoder--reranker)
6. [Storage, Multi-Tier Caching & Persistence (`multi_modal_financial.storage`)](#6-storage-multi-tier-caching--persistence)
7. [Agent & Routing (`multi_modal_financial.agent`)](#7-agent--routing)
8. [Indexing Engine (`multi_modal_financial.indexing`)](#8-indexing-engine)
9. [Citation Grounding & Verification (`multi_modal_financial.grounding`)](#9-citation-grounding--verification)
10. [Data Preprocessing & Validation (`multi_modal_financial.data`)](#10-data-preprocessing--validation)
11. [Multi-Modal Parsing (`multi_modal_financial.parsing`)](#11-multi-modal-parsing)
12. [Financial Analytics (`multi_modal_financial.analytics`)](#12-financial-analytics)
13. [CLI & Formatters (`multi_modal_financial.cli`)](#13-cli--formatters)

---

## 1. Pipeline Orchestrator

### `OrchestratorConfig`
```python
from multi_modal_financial.pipeline.orchestrator import OrchestratorConfig
```
Configuration dataclass controlling multi-tier caching, validation, retrieval strategies, neural reranking, and audit thresholds.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_query_cache` | `bool` | `True` | Cache query responses using TTL and composite hashing. |
| `enable_embedding_cache` | `bool` | `True` | Cache dense embedding vectors in LRU cache. |
| `enable_semantic_cache` | `bool` | `True` | Cache queries and responses via dense vector similarity matching. |
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
| `semantic_cache_config` | `SemanticCacheConfig` | `SemanticCacheConfig()` | Configuration for semantic cache threshold, capacity, and eviction policy. |
| `reranker_config` | `RerankerConfig` | `RerankerConfig()` | Configuration for second-stage cross-encoder and heuristic reranking. |
| `hybrid_search_config` | `HybridSearchConfig` | `HybridSearchConfig()` | Configuration for multi-strategy retrieval and RRF smoothing parameters. |

---

### `FinancialPipelineOrchestrator`
```python
from multi_modal_financial.pipeline.orchestrator import FinancialPipelineOrchestrator
```
Master pipeline controller integrating document loading, indexing, multi-strategy retrieval, neural reranking, verification, ratio calculations, variance analysis, and index persistence.

#### `__init__(config: OrchestratorConfig | None = None, index: HybridIndex | None = None)`
Initializes the orchestrator and all dependent subsystems including `SemanticCache`, `HybridRetriever`, `FinancialReranker`, `QueryRouter`, and `GroundingVerifier`.

#### `ingest_files(paths: list[str | Path] | str | Path, default_ticker: str | None = None) -> list[Document]`
Loads, cleans, validates, and indexes single files, lists of files, or entire directories. Supported formats: `.txt`, `.md`, `.json`, `.pdf`.

#### `run_query(query: str | AgentQuery, top_k: int | None = None, alpha: float | None = None, use_cache: bool = True) -> AgentResponse`
Executes an end-to-end financial query. Evaluates exact-match and semantic cache, runs query routing, hybrid retrieval, cross-encoder reranking, context synthesis, and sentence-level citation verification.

#### `run_batch_queries(queries: Sequence[str | AgentQuery], top_k: int | None = None) -> list[AgentResponse]`
Executes a sequence of queries sequentially, leveraging internal caching tiers.

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

#### `checkpoint(compress: bool = False) -> Path`
Persists the current index to the directory configured in `OrchestratorConfig.persistence_dir`.

#### `save_semantic_cache(output_path: str | Path) -> Path | None`
Serializes the semantic cache entries and vectors to disk.

#### `load_semantic_cache(source_path: str | Path) -> None`
Restores semantic cache entries from disk.

#### `benchmark_retrieval(test_queries: Sequence[tuple[AgentQuery | str, list[str]]], k: int = 5, use_rrf: bool = True) -> RetrievalBenchmarkResult`
Runs retrieval benchmark suite across labeled test query sets.

#### `status() -> dict[str, Any]`
Returns diagnostic status including total documents, chunks, vocabulary size, vector dimension, query cache stats, embedding cache stats, semantic cache telemetry, reranker config, and retriever configuration.

---

## 2. Runtime Protocols & Interfaces

Located in `multi_modal_financial.interfaces`. All protocols are decorated with `@runtime_checkable` conforming to PEP 544.

### `CrossEncoderProtocol`
```python
@runtime_checkable
class CrossEncoderProtocol(Protocol):
    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Compute cross-attention relevance scores for (query, document) pairs."""
        ...
```

### `RerankerProtocol`
```python
@runtime_checkable
class RerankerProtocol(Protocol):
    def rerank(
        self,
        query: str | AgentQuery,
        candidates: list[ScoredChunk],
        top_k: int | None = None,
    ) -> list[ScoredChunk]:
        """Rerank candidates based on neural and/or domain financial salience."""
        ...
```

### `SemanticCacheProtocol`
```python
@runtime_checkable
class SemanticCacheProtocol(Protocol):
    def get(
        self,
        query: str,
        query_vector: list[float] | None = None,
        similarity_threshold: float | None = None,
    ) -> SemanticCacheLookupResult | None: ...
    def put(
        self,
        query: str,
        response: AgentResponse,
        query_vector: list[float] | None = None,
    ) -> None: ...
    def clear(self) -> None: ...
    def stats(self) -> SemanticCacheStats: ...
```

### `DenseIndexProtocol`
```python
@runtime_checkable
class DenseIndexProtocol(Protocol):
    def search(self, query: str | list[float], top_k: int = 10) -> list[tuple[str, float]]: ...
    def add(self, chunk_id: str, text: str, vector: list[float] | None = None) -> None: ...
```

### `SparseIndexProtocol`
```python
@runtime_checkable
class SparseIndexProtocol(Protocol):
    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]: ...
    def add(self, chunk_id: str, text: str) -> None: ...
```

### `RetrieverProtocol`
```python
@runtime_checkable
class RetrieverProtocol(Protocol):
    def retrieve(
        self, query: str | AgentQuery, top_k: int = 10, alpha: float = 0.5
    ) -> list[ScoredChunk]: ...
```

---

## 3. Domain Types & Models

Located in `multi_modal_financial.types`.

### Enumerations
- **`ModalType`**: `TEXT`, `TABLE`, `FIGURE`, `METRIC`, `HEADER`, `FOOTNOTE`, `SUMMARY`
- **`DocumentType`**: `TEN_K`, `TEN_Q`, `EIGHT_K`, `EARNINGS_RELEASE`, `ANALYST_REPORT`, `PROSPECTUS`, `PROXY`, `PRESS_RELEASE`, `TRANSCRIPT`, `FILING`, `OTHER`
- **`FinancialStatementType`**: `INCOME_STATEMENT`, `BALANCE_SHEET`, `CASH_FLOW`, `COMPREHENSIVE_INCOME`, `STOCKHOLDERS_EQUITY`, `SEGMENT_METRICS`, `NOTES`, `UNKNOWN`
- **`Currency`**: `USD`, `EUR`, `GBP`, `JPY`, `CAD`, `CHF`, `CNY`, `AUD`, `OTHER`
- **`UnitScale`**: `ONES`, `THOUSANDS`, `MILLIONS`, `BILLIONS`, `TRILLIONS`, `PERCENT`, `RATIO`, `BPS`
- **`QueryIntent`**: `METRIC_LOOKUP`, `COMPARATIVE_ANALYSIS`, `QUALITATIVE_RISK`, `TREND_CALCULATION`, `GENERAL`
- **`GroundingStatus`**: `FULLY_SUPPORTED`, `PARTIALLY_SUPPORTED`, `UNSUPPORTED`, `CONTRADICTED`
- **`RerankerStrategy`**: `CROSS_ENCODER`, `MODALITY_HEURISTIC`, `HYBRID`, `NONE`
- **`RetrievalStrategy`**: `HYBRID_RRF`, `HYBRID_CONVEX`, `DENSE`, `SPARSE`
- **`CacheEvictionPolicy`**: `LRU`, `LFU`, `FIFO`
- **`CacheHitType`**: `EXACT`, `SEMANTIC`, `NONE`

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

### `ScoredChunk`
Pairs a `Chunk` with retrieval and reranking scores:
- **Attributes**: `chunk: Chunk`, `score: float`, `dense_score: float = 0.0`, `sparse_score: float = 0.0`, `rank: int = 0`, `rerank_score: float | None = None`, `cross_encoder_score: float | None = None`, `semantic_score: float | None = None`, `modality_bonus: float = 0.0`, `explanation: str | None = None`.

---

### `RerankExplanation`
Detailed explanation record for candidate reranking decisions:
- **Attributes**: `chunk_id: str`, `initial_rank: int`, `final_rank: int`, `initial_score: float`, `final_score: float`, `cross_encoder_score: float | None = None`, `heuristic_score: float | None = None`, `modality_bonus: float = 0.0`, `reasons: list[str]`.

---

### `RerankerConfig`
Configuration for second-stage reranking models:
- **Attributes**: `strategy: RerankerStrategy = RerankerStrategy.HYBRID`, `model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"`, `device: str = "cpu"`, `batch_size: int = 32`, `top_k: int = 5`, `score_threshold: float = 0.0`, `cross_encoder_weight: float = 0.7`, `heuristic_weight: float = 0.3`, `normalize_scores: bool = True`, `table_boost: float = 0.2`, `metric_boost: float = 0.25`, `figure_boost: float = 0.15`, `entity_boost: float = 0.2`.

---

### `SemanticCacheConfig`, `SemanticCacheEntry`, `SemanticCacheLookupResult`, `SemanticCacheStats`
Configuration, entries, lookup results, and operational telemetry for neural semantic cache:
- **`SemanticCacheConfig`**: `enabled: bool = True`, `similarity_threshold: float = 0.85`, `max_entries: int = 1000`, `ttl_seconds: float = 86400.0`, `eviction_policy: CacheEvictionPolicy = CacheEvictionPolicy.LRU`, `distance_metric: str = "cosine"`.
- **`SemanticCacheEntry`**: `key: str`, `query: str`, `query_vector: list[float]`, `response: AgentResponse`, `similarity_score: float = 1.0`, `created_at: float`, `last_accessed_at: float`, `access_count: int = 1`. Methods: `is_expired(ttl_seconds, current_time) -> bool`, `touch(current_time) -> None`.
- **`SemanticCacheLookupResult`**: `hit: bool`, `similarity: float = 0.0`, `matched_query: str | None = None`, `response: AgentResponse | None = None`, `lookup_latency_ms: float = 0.0`, `hit_type: CacheHitType = CacheHitType.NONE`.
- **`SemanticCacheStats`**: `total_queries: int`, `exact_hits: int`, `semantic_hits: int`, `misses: int`, `evictions: int`, `entry_count: int`, `max_entries: int`, `hit_rate: float`, `avg_lookup_latency_ms: float`. Method: `to_dict() -> dict[str, Any]`.

---

### `HybridSearchConfig`
Configuration model for hybrid retrieval fusion strategies:
- **Attributes**: `strategy: RetrievalStrategy = RetrievalStrategy.HYBRID_RRF`, `default_alpha: float = 0.5`, `top_k: int = 10`, `rrf_k: int = 60`, `normalization: str = "min_max"`, `score_threshold: float = 0.0`, `allowed_chunk_ids: set[str] | None = None`.

---

### `Citation`, `GroundingVerdict`, `AgentQuery`, `AgentResponse`
- **`Citation`**: Source attribution linking a claim to a specific chunk (`citation_id`, `doc_id`, `page_number`, `modality`, `excerpt`, `score`, `claim_text`).
- **`GroundingVerdict`**: Result of sentence-level fact check (`status`, `confidence`, `verified_numbers`, `hallucinated_numbers`, `explanation`).
- **`AgentQuery`**: Query parameters (`query_str`, `ticker_filter`, `period_filter`, `year_filter`, `doc_type_filter`, `modal_filter`, `top_k`, `alpha`, `retrieval_strategy`, `reranker_strategy`, `use_reranker`, `reranker_top_k`, `reranker_threshold`, `use_semantic_cache`, `enable_rerank_explanation`).
- **`AgentResponse`**: Structured RAG output (`query`, `answer`, `citations`, `retrieved_chunks`, `verdict`, `groundedness_score`, `execution_time_ms`, `overall_confidence`, `cache_hit`, `cache_type`, `cache_similarity`, `rerank_explanations`).

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
