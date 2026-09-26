# Multi-Modal Financial Document RAG Pipeline

[![CI](https://github.com/zein/multi-modal-financial/actions/workflows/ci.yml/badge.svg)](https://github.com/zein/multi-modal-financial/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Type Checked: Mypy](https://img.shields.io/badge/type%20checked-mypy-blue.svg)](https://mypy-lang.org/)

**Multi-Modal Financial Document RAG Pipeline** is an enterprise-grade retrieval-augmented generation framework engineered specifically for financial disclosures, SEC filings (10-K, 10-Q, 8-K), earnings releases, and quantitative equity research.

It combines Okapi BM25 lexical sparse search with dense semantic vector embeddings, Reciprocal Rank Fusion (RRF), cross-heuristic modality reranking, sentence-level citation grounding, automated GAAP financial statement reconciliation, and horizontal variance analytics.

---

## Key Features

- **Multi-Modal Document Ingestion & Parsing**:
  - Ingests plain text, markdown, JSON, and PDF financial documents.
  - Extracts narrative text chunks, tabular disclosures (markdown/CSV), and visual figures/charts.
  - Automatic detection of ticker symbols, reporting periods (`Q1`-`Q4`, `FY`), filing years, and financial statement types.
  - Cleans accounting artifacts: converts accounting negative parentheses `(125.0)` to `-125.0`, normalizes Unicode dashes, strips SEC tags, and removes table footnote markers.

- **Hybrid Retrieval Core**:
  - **BM25 Lexical Sparse Search**: Okapi BM25 ($k_1=1.5, b=0.75$) with financial-specific tokenization preserving dollar signs, percentages, ticker symbols, and hyphenated accounting terms.
  - **Dense Vector Semantic Search**: Deterministic orthographic projection embeddings or pre-computed vector ingestion with cosine similarity.
  - **Multi-Strategy Retrieval Engine**: Supports Reciprocal Rank Fusion (RRF, $k=60$), convex alpha score blending ($\alpha \cdot \text{Dense} + (1-\alpha) \cdot \text{BM25}$) with Min-Max or Z-score normalization, pure dense vector search, and pure sparse lexical search.
  - **Second-Stage Neural Cross-Encoder & Hybrid Reranker**: Fine-grained cross-token attention simulation (`FinancialCrossEncoder`) combined with financial domain heuristics (`FinancialReranker`), modality weighting (table/metric/figure chunks), ticker/period exact match boosting, and full explainability tracking (`RerankExplanation`).

- **Strict Citation Grounding & Hallucination Auditing**:
  - Sentence-level claim extraction linking assertions to discrete source citations (`[1]`, `[2]`).
  - Strict numeric fact verification comparing asserted quantities against source table cells and text excerpts.
  - Categorizes claims into `FULLY_SUPPORTED`, `PARTIALLY_SUPPORTED`, `UNSUPPORTED`, or `CONTRADICTED`.
  - `GroundingAuditor` for batch compliance auditing, hallucination rate scoring, and exportable JSON governance reports.

- **GAAP Statement Reconciliation & Ratio Analytics**:
  - Balance sheet invariant verification: $\text{Total Assets} = \text{Total Liabilities} + \text{Stockholders' Equity}$.
  - Income statement invariant verification: $\text{Gross Profit} = \text{Revenue} - \text{COGS}$.
  - Profitability ratios (Gross, Operating, Net Margins), Liquidity ratios (Current, Quick), and Solvency ratios (Debt-to-Equity).
  - Horizontal period comparisons (`PeriodComparator`) computing YoY/QoQ nominal and percentage variances.

- **Multi-Tier Caching & Index Persistence**:
  - **Semantic Vector Cache (`SemanticCache`)**: High-performance semantic similarity cache supporting configurable eviction policies (`LRU`, `LFU`, `FIFO`), exact-match fast path ($O(1)$), cosine and Euclidean distance metrics, TTL auto-expiration, and operational telemetry (`SemanticCacheStats`).
  - **Embedding & Query Caches**: Thread-safe Least-Recently-Used (LRU) vector embedding cache and Time-To-Live (TTL) query response cache.
  - **Atomic Checkpoint Persistence**: Directory and compressed `.zip` index serialization with JSON manifests and SHA-256 integrity validation.

- **Developer CLI & Rich Terminal Dashboard**:
  - Command-line interface powered by Click and Rich for inspection, batch ingestion, interactive queries with reranking explanations, retrieval benchmarks, cache telemetry (`cache-stats`), statement ratios (`ratios`), horizontal variance comparison (`compare`), and compliance audits (`audit`).

---

## Architecture

```
Financial Filings (SEC 10-K, 10-Q, 8-K, Earnings Releases, PDFs)
                          │
                          ▼
        ┌────────────────────────────────────┐
        │ Multi-Modal Parser & Data Cleaner  │
        │ - Text, Tables, Figures, Metadata  │
        │ - Accounting Parentheses Normalizer│
        │ - Table Validator & Reconciler     │
        └─────────────────┬──────────────────┘
                          │ Validated Chunks
                          ▼
        ┌────────────────────────────────────┐
        │        Hybrid Index Engine         │
        │   ├── BM25 Index (Lexical terms)   │
        │   └── Dense Vector Index (Cosine)  │
        │   ├── Multi-Tier Semantic Cache    │
        │   └── Checkpoint Persistence       │
        └─────────────────┬──────────────────┘
                          │ Top Candidates
                          ▼
        ┌────────────────────────────────────┐
        │   Multi-Strategy Hybrid Retriever  │
        │   ├── Reciprocal Rank Fusion (RRF) │
        │   ├── Convex Alpha Blending        │
        │   └── Dense-Only / Sparse-Only     │
        └─────────────────┬──────────────────┘
                          │ Candidate Pool
                          ▼
        ┌────────────────────────────────────┐
        │ Neural Cross-Encoder & Reranker    │
        │   ├── Pairwise Cross-Attention Sim │
        │   ├── Modality & Ticker Boosting   │
        │   └── Explainability Diagnostics   │
        └─────────────────┬──────────────────┘
                          │ Top-K Scored Chunks
                          ▼
        ┌────────────────────────────────────┐
        │  Financial Agent & Grounding Core  │
        │   ├── Structured Context Synthesis │
        │   ├── Citation Extraction          │
        │   ├── Grounding Numeric Verifier   │
        │   └── Enterprise Auditor           │
        └─────────────────┬──────────────────┘
                          │
                          ▼
               Verified AgentResponse
     (Answer + Provenance + Citations + Verdict)
```

---

## Installation

### Standard Installation
```bash
git clone https://github.com/zein/multi-modal-financial.git
cd multi-modal-financial
pip install -e .
```

### Development Suite (Tests, Linting, Type Checking)
```bash
pip install -e ".[dev]"
```

---

## Quickstart

### Python API

```python
from multi_modal_financial.pipeline.orchestrator import (
    FinancialPipelineOrchestrator,
    OrchestratorConfig,
)
from multi_modal_financial.types import (
    AgentQuery,
    CacheEvictionPolicy,
    RerankerConfig,
    RerankerStrategy,
    RetrievalStrategy,
    SemanticCacheConfig,
)

# 1. Initialize master orchestrator with semantic caching, hybrid retrieval, and neural reranker
config = OrchestratorConfig(
    enable_query_cache=True,
    enable_embedding_cache=True,
    enable_semantic_cache=True,
    semantic_cache_config=SemanticCacheConfig(
        similarity_threshold=0.85,
        eviction_policy=CacheEvictionPolicy.LRU,
    ),
    reranker_config=RerankerConfig(
        strategy=RerankerStrategy.HYBRID,
        cross_encoder_weight=0.7,
        heuristic_weight=0.3,
        table_boost=0.20,
    ),
    confidence_threshold=0.70,
)
orchestrator = FinancialPipelineOrchestrator(config=config)

# 2. Ingest financial filings
filing_text = """
Apple Inc. (AAPL) reported Q3 2025 financial results:
Total net sales were $94.9 billion, compared to $89.5 billion in Q3 2024.
Products revenue was $69.9 billion.
Services revenue reached a record $25.0 billion, up 14% year-over-year.
Operating income was $29.6 billion, representing an operating margin of 31.2%.
"""
orchestrator.rag_pipeline.ingest_text(
    filing_text, doc_id="aapl_q3_2025", ticker="AAPL", year=2025, period="Q3"
)

# 3. Query with hybrid retrieval, cross-encoder reranking, and explanations
query = AgentQuery(
    query_str="What was Apple's Services revenue and operating margin in Q3 2025?",
    ticker_filter="AAPL",
    top_k=3,
    retrieval_strategy=RetrievalStrategy.HYBRID_RRF,
    reranker_strategy=RerankerStrategy.HYBRID,
    enable_rerank_explanation=True,
)
response = orchestrator.run_query(query)

print("Answer:", response.answer)
print(f"From Cache: {response.cache_hit} ({response.cache_type})")
print("Groundedness Score:", response.groundedness_score)

# 4. Inspect retrieved citations and reranker explanations
for c in response.citations:
    print(f"[{c.citation_id}] Doc: {c.doc_id}, Page: {c.page_number} -> {c.excerpt}")

if response.rerank_explanations:
    for exp in response.rerank_explanations:
        print(f"Chunk: {exp.chunk_id} | Final: {exp.final_score:.4f} | Factors: {', '.join(exp.reasons)}")

# 5. Check factual grounding verdict
if response.verdict:
    print("Verification Status:", response.verdict.status.value)
    print("Verified Numbers:", response.verdict.verified_numbers)
    print("Hallucinated Numbers:", response.verdict.hallucinated_numbers)
```
    print("Hallucinated Numbers:", response.verdict.hallucinated_numbers)
```

### Financial Ratios & Variance Analysis

```python
from multi_modal_financial.data.synthetic import SyntheticFilingGenerator
from multi_modal_financial.pipeline.orchestrator import FinancialPipelineOrchestrator

orchestrator = FinancialPipelineOrchestrator()
gen = SyntheticFilingGenerator()

# Generate and index consecutive quarterly filings
doc_q2 = gen.generate_filing_document(ticker="MSFT", year=2025, period="Q2")
doc_q3 = gen.generate_filing_document(ticker="MSFT", year=2025, period="Q3")
orchestrator.index.index_document(doc_q2)
orchestrator.index.index_document(doc_q3)

# 1. Compute financial ratios for Q3
ratios = orchestrator.analyze_document_ratios("MSFT_2025_Q3")
print(f"Gross Margin: {ratios.gross_margin:.1%}")
print(f"Operating Margin: {ratios.operating_margin:.1%}")
print(f"Net Margin: {ratios.net_margin:.1%}")
print(f"Current Ratio: {ratios.current_ratio:.2f}")

# 2. Compare horizontal period variances (Q2 vs Q3)
variances = orchestrator.compare_documents("MSFT_2025_Q2", "MSFT_2025_Q3")
for v in variances:
    print(
        f"{v.metric_name}: Base ${v.base_value:,.1f}M -> Compare ${v.compare_value:,.1f}M ({v.percent_change:+.1f}%)"
    )
```

---

## Command-Line Interface (CLI)

The package includes a built-in CLI for terminal inspection, document ingestion, querying, benchmarking, financial analytics, and auditing.

### Display Environment & Architecture Diagnostics
```bash
# Basic subsystem overview
multi-modal-financial info

# Live status with cache occupancy and telemetry
multi-modal-financial info --status
```

### Ingest Documents
```bash
# Ingest single file with ticker attribution
multi-modal-financial ingest ./filings/apple_q3_2025.txt --ticker AAPL

# Ingest directory containing multiple filings
multi-modal-financial ingest ./samples
```

### Query with Hybrid Retrieval, Reranking & Explanations
```bash
# Standard hybrid query
multi-modal-financial query "What was total revenue in Q3 2025?" --ticker ACM --top-k 3

# Advanced query with neural cross-encoder reranking and factor explanations
multi-modal-financial query "What was revenue and operating income?" --ticker ACM --strategy hybrid_rrf --reranker-strategy hybrid --explain
```

### Semantic Cache Telemetry
```bash
multi-modal-financial cache-stats
```

### Compute Financial Statement Ratios
```bash
multi-modal-financial ratios ./samples/sample_filing.txt
```

### Horizontal Period Variance Analysis
```bash
multi-modal-financial compare ./samples/sample_q2.txt ./samples/sample_q3.txt
```

### Factual Grounding Compliance Audit
```bash
multi-modal-financial audit "What was revenue for ACM in Q3?" "What was gross profit?" --max-hallucination-rate 0.05
```

### Benchmark Retrieval Engine
```bash
multi-modal-financial benchmark
```
Example Output:
```
Running Retrieval Benchmark...

┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━┓
┃ Mode                ┃ Recall@1  ┃ MRR    ┃
┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━┩
│ BM25 (Lexical Only) │ 66.7%     │ 0.833  │
│ Dense (Vector Only) │ 100.0%    │ 1.000  │
│ Hybrid (Dense+BM25) │ 100.0%    │ 1.000  │
└─────────────────────┴───────────┴────────┘
```

---

## Project Structure

```
multi-modal-financial/
├── docs/
│   ├── ARCHITECTURE.md          # Technical deep-dive, math & subsystem design
│   ├── API_REFERENCE.md         # Comprehensive programmatic API documentation
│   └── USAGE_GUIDE.md           # End-to-end task recipes and cookbooks
├── src/
│   └── multi_modal_financial/
│       ├── __init__.py          # Package metadata and version export
│       ├── types.py             # Pydantic domain models, enums, dataclasses
│       ├── interfaces.py        # PEP 544 @runtime_checkable abstract protocols
│       ├── agent/               # Query routing and RAG synthesis pipeline
│       │   ├── pipeline.py      # Core FinancialRAGPipeline
│       │   └── router.py        # QueryRouter with intent classification
│       ├── analytics/           # Financial ratio calculators and period comparator
│       │   ├── ratios.py        # Margins, liquidity, and solvency ratios
│       │   └── comparator.py    # YoY/QoQ variance calculation engine
│       ├── cli/                 # Click CLI commands and Rich formatters
│       │   ├── main.py          # CLI entry points (info, ingest, query, cache-stats, ratios, compare, audit, benchmark)
│       │   └── formatters.py    # Terminal formatters, panels, and explanation tables
│       ├── data/                # Document loaders, cleaners, and validators
│       │   ├── cleaner.py       # FinancialDataCleaner (parentheses, dashes, tags)
│       │   ├── loader.py        # BatchDocumentLoader (TXT, MD, JSON, PDF)
│       │   ├── synthetic.py     # SyntheticFilingGenerator for testing
│       │   └── validator.py     # FinancialTableValidator and StatementReconciler
│       ├── grounding/           # Attribution and fact verification engine
│       │   ├── citation.py      # Citation extraction and claim parsing
│       │   ├── verifier.py      # GroundingVerifier (numeric consistency checks)
│       │   └── audit.py         # GroundingAuditor and compliance reporting
│       ├── indexing/            # Hybrid sparse-dense indexing layer
│       │   ├── bm25.py          # BM25Index with financial tokenization
│       │   ├── vector.py        # DenseVectorIndex with cosine similarity
│       │   └── hybrid.py        # HybridIndex unified container
│       ├── parsing/             # Multi-modal parsers
│       │   ├── extractor.py     # FinancialDocumentParser (text and PDF)
│       │   ├── figure_parser.py # FigureParser (charts, axes, data series)
│       │   └── table_parser.py  # TableParser (markdown and CSV tables)
│       ├── pipeline/            # Master workflow orchestration
│       │   └── orchestrator.py  # FinancialPipelineOrchestrator
│       ├── retrieval/           # Fusion & reranking core
│       │   ├── fusion.py        # HybridRetriever (RRF, convex blend, dense, sparse)
│       │   └── reranker.py      # FinancialCrossEncoder & FinancialReranker
│       ├── storage/             # Multi-tier caching and persistence
│       │   ├── cache.py         # EmbeddingCache (LRU), QueryCache (TTL), SemanticCache (Multi-policy)
│       │   └── persistence.py   # IndexPersistence (directory and zip archives)
│       └── py.typed             # PEP 561 typing marker
├── tests/                       # Comprehensive test suite (294 tests, 100% pass rate)
├── pyproject.toml               # Build system and dependency definitions
├── requirements.txt             # Runtime requirements
├── requirements-dev.txt         # Development & CI requirements
└── README.md                    # Project documentation
```

---

## In-Depth Documentation

For advanced technical details, consult the dedicated documentation files:
- [System Architecture & Design Deep Dive](docs/ARCHITECTURE.md): Mathematical formulations for Okapi BM25, Reciprocal Rank Fusion, convex alpha blending, cross-encoder neural scoring, semantic cache policies, and numeric verification algorithms.
- [Complete API Reference](docs/API_REFERENCE.md): Comprehensive signature, protocol, and parameter definitions for all classes and functions.
- [Practical Usage Guide & Recipes](docs/USAGE_GUIDE.md): Task-oriented workflows for multi-strategy retrieval, cross-encoder reranking, semantic caching, statement reconciliation, batch auditing, and index persistence.

---

## Verification & Testing

The project maintains 100% test pass rate across 294 test cases, comprehensive Mypy strict typing, and Ruff linting:

```bash
# Run test suite with coverage
python -m pytest --cov=multi_modal_financial --cov-report=term-missing

# Run Ruff linter and formatter check
python -m ruff check .
python -m ruff format --check .

# Run static type checker
python -m mypy src
```

```bash
# Run test suite with coverage
python -m pytest --cov=multi_modal_financial --cov-report=term-missing

# Run Ruff linter and formatter check
python -m ruff check .
python -m ruff format --check .

# Run static type checker
python -m mypy src
```

---

## License

This project is licensed under the [MIT License](LICENSE).
