# Multi-Modal Financial RAG: System Architecture & Technical Deep-Dive

This document provides a comprehensive technical breakdown of the architecture, algorithmic formulations, mathematical models, and subsystem designs behind **`multi-modal-financial`**.

---

## 1. System Overview & Design Philosophy

Financial document analysis imposes rigorous demands that generic Retrieval-Augmented Generation (RAG) pipelines fail to meet:
1. **Multi-Modal Data Heterogeneity**: Financial filings (SEC 10-K, 10-Q, 8-K, earnings releases) interleave prose narratives, dense quantitative tables (GAAP balance sheets, income statements), and visual figures/charts.
2. **Numeric Precision vs. Semantic Drift**: Semantic vector search frequently confuses numeric values or loses tabular column context (e.g., mistaking 2024 revenue for 2025 revenue). Lexical precision must be fused with semantic recall.
3. **Zero-Tolerance Hallucination & Auditability**: Financial assertions require verifiable sentence-level citation provenance linking directly back to audited source documents and exact table cells.
4. **Statement Reconciliation**: Numeric tables must be validated against accounting invariants ($Assets = Liabilities + Equity$) before indexing to filter corrupt OCR or malformed data.

To solve these challenges, `multi-modal-financial` is designed as a modular, decoupled pipeline of 8 core subsystems:

```
                                  [ Financial Filings ]
                           (SEC 10-K, 10-Q, 8-K, Earnings, PDF)
                                             │
                                             ▼
                      ┌─────────────────────────────────────────────┐
                      │     Ingestion, Parsing & Validation Layer   │
                      │  - FinancialDocumentParser (Text / PDF)     │
                      │  - TableParser (Markdown, CSV, Footnotes)   │
                      │  - FigureParser (Charts, Axes, Data series) │
                      │  - FinancialDataCleaner (Dashes, Parens)    │
                      │  - FinancialTableValidator & Reconciler     │
                      └──────────────────────┬──────────────────────┘
                                             │ Validated Chunks & Metadata
                                             ▼
                      ┌─────────────────────────────────────────────┐
                      │            Hybrid Indexing Engine           │
                      │  ┌────────────────────┬───────────────────┐ │
                      │  │   BM25 Lexical     │   Dense Vector    │ │
                      │  │   Sparse Index     │   Index (Cosine)  │ │
                      │  └─────────┬──────────┴─────────┬─────────┘ │
                      │            └──────────┬─────────┘           │
                      │                   HybridIndex               │
                      └───────────────────────┼─────────────────────┘
                                              │
                      ┌───────────────────────┼─────────────────────┐
                      │ Storage & Caching     ▼                     │
                      │ - EmbeddingCache (LRU)  IndexPersistence    │
                      │ - QueryCache (TTL)      (Directory / ZIP)   │
                      └───────────────────────┬─────────────────────┘
                                              │
       [ User Query ]                         ▼
             │        ┌─────────────────────────────────────────────┐
             ▼        │      Dual-Mode Retrieval & Reranking        │
     ┌──────────────┐ │  - Reciprocal Rank Fusion (RRF)             │
     │ QueryRouter  │ │  - Convex Alpha Blend (BM25 + Dense)        │
     │ Intent/Ticker│─┼─►- FinancialReranker (Modality & Ticker)    │
     └──────────────┘ └──────────────────────┬──────────────────────┘
                                             │ Top-K Scored Chunks
                                             ▼
                      ┌─────────────────────────────────────────────┐
                      │    Financial Agent & Grounding Engine       │
                      │  - FinancialRAGPipeline (Synthesis)         │
                      │  - CitationGrounder (Provenance Tracking)   │
                      │  - GroundingVerifier (Claim Verification)   │
                      │  - GroundingAuditor (Compliance Reporting)  │
                      └──────────────────────┬──────────────────────┘
                                             │
                      ┌──────────────────────┴──────────────────────┐
                      │         Financial Analytics Layer           │
                      │  - FinancialRatioCalculator (Margins, D/E)  │
                      │  - PeriodComparator (YoY/QoQ Variances)     │
                      └─────────────────────────────────────────────┘
                                             │
                                             ▼
                                     [ AgentResponse ]
                          (Answer + Citations + Verdict + Provenance)
```

---

## 2. Parsing, Ingestion & Data Validation Layer

### 2.1 Multi-Modal Document Parsing (`FinancialDocumentParser`)
The parser ingests unstructured text, structured financial disclosures, and raw PDF files:
- **Header & Metadata Extraction**: Automatically parses ticker symbols (`Ticker: AAPL`), fiscal periods (`Q1`, `Q2`, `Q3`, `Q4`, `FY`), reporting years (1900–2099), and filing types (`10-K`, `10-Q`, `8-K`).
- **Section Slicing**: Identifies financial statement boundaries (Item 1A Risk Factors, Item 7 MD&A, Item 8 Financial Statements).
- **Chunk Generation**: Partitions text into semantic chunks with overlapping windowing, attributing each chunk with its source document ID, page number, and modality (`text`, `table`, `figure`, `footnote`).

### 2.2 Table Extraction & Parsing (`TableParser`)
Financial tables encapsulate critical tabular statements:
- **Format Support**: Delimited CSV text and GitHub-flavored Markdown tables.
- **Scale & Currency Detection**: Automatically identifies currency signs (`$`, `€`, `£`, `¥`) and scale denominations (`in millions`, `thousands`, `billions`, `in thousands except per share data`).
- **Statement Type Inference**: Heuristically classifies tables into `income_statement`, `balance_sheet`, `cash_flow`, `stockholders_equity`, or `segment_metrics` based on line-item keywords.
- **Footnote Cleaning**: Strips footnote markers (e.g. `[1]`, `(a)`) from cell data while preserving underlying numeric values.

### 2.3 Figure & Chart Parsing (`FigureParser`)
Earnings presentations and analyst reports contain charts depicting financial trends:
- **Chart Type Detection**: Identifies `bar`, `line`, `pie`, `waterfall`, or `scatter` charts.
- **Data Series Extraction**: Extracts series labels, key-value coordinate points, and axis labels (`X-axis: Quarter`, `Y-axis: Revenue in Millions`).

### 2.4 Financial Data Cleaner (`FinancialDataCleaner`)
Raw financial text from EDGAR filings often contains noisy artifacts:
- **Dashes Normalization**: Converts em-dashes (`—`), en-dashes (`–`), and horizontal bars into standard hyphens.
- **Accounting Parentheses**: Automatically translates accounting negative conventions (e.g., `(142.5)` or `$(25.0)M`) into standard machine-readable floats (`-142.5`, `-25000000.0`).
- **SEC HTML/XML Tag Stripping**: Strips `<ix:nonNumeric>`, `<font>`, and `<div>` tags while retaining content.
- **Markdown Table Repair**: Adjusts ragged delimiter pipes and column mismatches.

### 2.5 Table Validation & Statement Reconciliation (`FinancialTableValidator`, `StatementReconciler`)
Before indexing, tables pass through structural and financial invariant validation:
- **Structural Integrity**: Validates consistent column counts, header presence, and non-empty rows.
- **Numeric Bounds Checking**: Ensures ratios and percentages fall within physically realistic bounds (e.g., gross margin $\le 100\%$, operating loss within valid ranges).
- **Balance Sheet Invariant**:
  $$\text{Total Assets} = \text{Total Liabilities} + \text{Stockholders' Equity}$$
  Allows a configurable margin of error $\epsilon$ (default $1.0$) to account for rounding differences.
- **Income Statement Invariant**:
  $$\text{Gross Profit} = \text{Revenue} - \text{Cost of Goods Sold (COGS)}$$
  $$\text{Operating Income} = \text{Gross Profit} - \text{Operating Expenses}$$

---

## 3. Hybrid Indexing Engine

Financial RAG requires both exact keyword matching (for specific dollar figures, CIK codes, and ticker identifiers) and semantic abstraction (for concept-based discovery like "headwinds in enterprise cloud").

### 3.1 Lexical Index: BM25 Okapi (`BM25Index`)
The lexical subsystem implements the Okapi BM25 ranking function with financial-specific tokenization:

$$\text{Score}_{\text{BM25}}(D, Q) = \sum_{t \in Q} \text{IDF}(t) \cdot \frac{f(t, D) \cdot (k_1 + 1)}{f(t, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}$$

Where:
- $f(t, D)$ is term frequency in chunk $D$.
- $|D|$ is the length of chunk $D$, and $\text{avgdl}$ is the average chunk length across the corpus.
- Free parameters: $k_1 = 1.5$ (term frequency saturation) and $b = 0.75$ (length normalization penalty).
- $\text{IDF}(t)$ is calculated using the standard smoothed formulation:

$$\text{IDF}(t) = \ln\left(1 + \frac{N - n(t) + 0.5}{n(t) + 0.5}\right)$$

**Financial Tokenization**: Unlike standard NLP tokenizers that discard symbols, the `BM25Index` preserves:
- Currency symbols and scale suffixes (`$120m`, `€45.2b`)
- Ticker symbols (`AAPL`, `MSFT`, `NVDA`)
- Percentage expressions (`15.4%`, `-3.2%`)
- Hyphenated accounting terms (`cost-of-goods`, `year-over-year`)

### 3.2 Semantic Index: Dense Vector Space (`DenseVectorIndex`)
The dense vector subsystem maps financial chunks into a $d$-dimensional continuous space ($d=64$ default projection):
- **Orthographic Projection**: Applies a deterministic, hash-seeded orthographic projection matrix $W \in \mathbb{R}^{V \times d}$ to term-frequency representations, providing consistent embeddings across restarts without external API dependencies.
- **L2 Unit Normalization**: All vectors are normalized to unit sphere $\|v\|_2 = 1.0$.
- **Cosine Similarity**: Cosine similarity is computed via inner product:
  $$\text{Sim}(u, v) = u \cdot v = \sum_{i=1}^d u_i v_i$$
- **Precomputed Embedding Ingestion**: Accepts external state-of-the-art embedding models (e.g. OpenAI `text-embedding-3-small`, BGE-Large-en) directly via `add_precomputed()`.

### 3.3 Unified Hybrid Index (`HybridIndex`)
`HybridIndex` acts as the single source of truth:
- Concurrently maintains the `BM25Index` and `DenseVectorIndex`.
- Maintains chunk repositories and parent document graphs.
- Supports strict metadata filtering by:
  - `ticker`: Restrict to specific equity symbol (e.g., `AAPL`).
  - `period`: Filter to specific quarter or year (`Q3`, `2025`).
  - `modality`: Filter by content type (`table`, `text`, `figure`).

---

## 4. Dual-Mode Retrieval & Reranking Core

`HybridRetriever` provides two distinct fusion strategies to combine sparse lexical scores with dense semantic vectors.

```
       Query: "operating margin Q3 2025"
                      │
           ┌──────────┴──────────┐
           ▼                     ▼
     [ BM25 Search ]       [ Dense Vector Search ]
     Ranked List L_bm25    Ranked List L_dense
           │                     │
           └──────────┬──────────┘
                      ▼
             [ Fusion Engine ]
         ├── Reciprocal Rank Fusion (RRF)
         └── Convex Alpha Combination (Min-Max / Z-Score)
                      │
                      ▼
             [ Financial Reranker ]
         ├── Modality Boosting (Table +15%, Metric +20%)
         ├── Ticker / Period Match Boosting (+25%)
         └── Factual Query Term Exact Boosting (+15%)
                      │
                      ▼
             Top-K Scored Chunks
```

### 4.1 Reciprocal Rank Fusion (RRF)
RRF merges ranked lists without requiring score calibration across different distributions:

$$\text{RRF\_Score}(d) = \sum_{m \in M} \frac{w_m}{k + r_m(d)}$$

Where:
- $M = \{\text{BM25}, \text{Dense}\}$
- $r_m(d) \in \{1, 2, \dots, K\}$ is the 1-based ordinal rank of document $d$ in system $m$.
- $k$ is the smoothing constant (default $k = 60$) that mitigates outlier impact.
- $w_m$ is the modality weight (default $w_{\text{dense}} = 1.0, w_{\text{bm25}} = 1.0$).

### 4.2 Convex Alpha Combination
When calibrated scores are preferred, scores are normalized via Min-Max or Z-Score:

$$\tilde{S}(d) = \frac{S(d) - S_{\min}}{S_{\max} - S_{\min} + \epsilon}$$

The final retrieval score is computed as:

$$S_{\text{hybrid}}(d) = \alpha \cdot \tilde{S}_{\text{dense}}(d) + (1 - \alpha) \cdot \tilde{S}_{\text{bm25}}(d)$$

- $\alpha = 0.0$: Pure lexical BM25 (ideal for precise financial line-item search).
- $\alpha = 1.0$: Pure dense vector semantic search (ideal for qualitative questions).
- $\alpha = 0.5$: Balanced hybrid retrieval (default).

### 4.3 Financial Cross-Heuristic Reranker (`FinancialReranker`)
The reranker applies multi-modal domain heuristics to the candidate set:
1. **Modality Weighting**: Prioritizes structured tables and metric chunks over narrative prose when the query targets quantitative metrics:
   $$\text{Score} \leftarrow \text{Score} \times 1.20 \quad (\text{if Table / Metric chunk})$$
2. **Ticker & Period Alignment**: Documents matching both the query's identified ticker and fiscal year receive a $+25\%$ score multiplier.
3. **Exact Token Coverage**: Chunks containing the exact query numeric tokens or accounting line-item headers receive a $+15\%$ relevance boost.

---

## 5. Query Routing & Agent Synthesis

### 5.1 Query Router (`QueryRouter`)
Queries are analyzed before retrieval to optimize pipeline execution:
- **Intent Classification**:
  - `METRIC_LOOKUP`: "What was Q3 revenue?"
  - `COMPARATIVE_ANALYSIS`: "Compare operating margin between 2024 and 2025"
  - `TREND_CALCULATION`: "What is the 3-year revenue CAGR?"
  - `QUALITATIVE_RISK`: "What are the regulatory risks in the 10-K?"
  - `GENERAL`: General queries
- **Metadata Extraction**: Regex extraction of ticker candidates (2–5 uppercase letters) and period markers (`Q1`-`Q4`, `FY`, `2020`-`2030`).
- **Dynamic Parameter Tuning**: Sets retrieval top-$k$ and $\alpha$ balance based on query intent (e.g., lower $\alpha$ for exact metrics, higher $\alpha$ for qualitative risk).

### 5.2 RAG Pipeline & Synthesis (`FinancialRAGPipeline`)
The pipeline coordinates retrieval, context formatting, and structured answer synthesis:
1. Retrieves top-$k$ chunks via `HybridRetriever`.
2. Re-scores chunks via `FinancialReranker`.
3. Synthesizes a structured response containing:
   - Concise narrative answer
   - Bracketed source citations (e.g., `[1]`, `[2]`)
   - Source provenance badges (e.g., `[src:acme_q3_2025:p.2]`)
   - Execution latency diagnostics

---

## 6. Citation Grounding & Hallucination Auditing

A cornerstone of `multi-modal-financial` is strict factual provenance.

```
       Synthesized Agent Answer:
       "Operating margin for Q3 2025 was 22.4% [1], while net income reached $215M [2]."
                                    │
                                    ▼
       ┌────────────────────────────────────────────────────────┐
       │     Citation Grounder: Extract Claims & Citations      │
       │  - Claim 1: "Operating margin for Q3 2025 was 22.4%"   │
       │  - Claim 2: "net income reached $215M"                 │
       └────────────────────────────┬───────────────────────────┘
                                    │
                                    ▼
       ┌────────────────────────────────────────────────────────┐
       │     Grounding Verifier: Numeric Verification           │
       │  - Claim 1 numbers: {22.4} vs Chunk 1 numbers {22.4}  │
       │    Verdict: Fully Supported (Confidence: 1.0)          │
       │  - Claim 2 numbers: {215.0} vs Chunk 2 numbers {215.0} │
       │    Verdict: Fully Supported (Confidence: 1.0)          │
       └────────────────────────────┬───────────────────────────┘
                                    │
                                    ▼
       ┌────────────────────────────────────────────────────────┐
       │     Grounding Auditor: Batch Auditing & Compliance     │
       │  - Hallucination Rate: 0.0% (Passed <= 5.0% threshold) │
       │  - Factual Support Score: 1.0                          │
       │  - Provenance Trace: 100% verified against corpus      │
       └────────────────────────────────────────────────────────┘
```

### 6.1 Citation Grounding (`CitationGrounder`)
- **Citation Parsing**: Detects both inline bracket citations (`[1]`, `[2]`) and explicit provenance tags (`[src:doc_id:page]`).
- **Claim Slicing**: Splits synthesized answers into discrete factual claims at sentence and clause boundaries.
- **Provenance Association**: Maps each claim to its corresponding retrieved chunk, capturing document ID, page number, and source modality.

### 6.2 Grounding Verification (`GroundingVerifier`)
The verifier performs numeric and entity consistency verification:
1. **Numeric Extraction**: Extracts all numeric entities, currency quantities, percentages, and multipliers from both the synthesized claim and the source chunk context.
2. **Numeric Fact Matching**:
   - For every numeric token $n_{\text{claim}}$, checks if $|n_{\text{claim}} - n_{\text{context}}| \le \delta$ (allowing tolerance for decimal rounding, $\delta = 10^{-4}$).
   - Verifies scale alignment (e.g. confirming that $1.25\text{ billion}$ matches $1,250\text{ million}$).
3. **Verdict Assignment**:
   - `FULLY_SUPPORTED`: $100\%$ of numerical claims and key entities found in context chunk.
   - `PARTIALLY_SUPPORTED`: $>50\%$ of numerical claims verified.
   - `UNSUPPORTED`: Claim asserts metrics not present in retrieved context.
   - `CONTRADICTED`: Claim asserts values that conflict with context metrics for the same line item.

### 6.3 Grounding Auditor (`GroundingAuditor`)
For institutional governance, `GroundingAuditor` evaluates batches of queries:
- Computes aggregate metrics: **Hallucination Rate**, **Average Factual Support Score**, **Citation Precision**, and **Unsupported Claim Count**.
- Evaluates against strict enterprise thresholds (e.g., maximum allowed hallucination rate $\le 5\%$).
- Generates a structured audit report exportable to JSON.

---

## 7. Financial Analytics & Comparative Subsystems

### 7.1 Financial Ratio Calculator (`FinancialRatioCalculator`)
Extracts and computes standard financial ratios from parsed tables or indexed documents:
- **Profitability Margins**:
  $$\text{Gross Margin} = \frac{\text{Gross Profit}}{\text{Revenue}}$$
  $$\text{Operating Margin} = \frac{\text{Operating Income}}{\text{Revenue}}$$
  $$\text{Net Profit Margin} = \frac{\text{Net Income}}{\text{Revenue}}$$
- **Liquidity Ratios**:
  $$\text{Current Ratio} = \frac{\text{Current Assets}}{\text{Current Liabilities}}$$
  $$\text{Quick Ratio} = \frac{\text{Cash} + \text{Marketable Securities} + \text{Accounts Receivable}}{\text{Current Liabilities}}$$
- **Solvency Ratios**:
  $$\text{Debt-to-Equity} = \frac{\text{Total Debt}}{\text{Total Stockholders' Equity}}$$

### 7.2 Period Comparator (`PeriodComparator`)
Performs horizontal financial analysis across reporting periods (YoY or QoQ):
- **Variance Calculation**:
  $$\Delta_{\text{nominal}} = V_{\text{compare}} - V_{\text{base}}$$
  $$\Delta_{\text{pct}} = \frac{V_{\text{compare}} - V_{\text{base}}}{|V_{\text{base}}|} \times 100\%$$
- **Variance Categorization**: Classifies performance changes as `growth`, `decline`, or `flat` (within $\pm 0.01\%$).
- **Multi-Period Table Comparison**: Compares full tabular statement columns across fiscal periods.

---

## 8. Storage, Caching & Persistence

### 8.1 Dual LRU & TTL Caching (`EmbeddingCache`, `QueryCache`)
- **`EmbeddingCache`**: Thread-safe Least-Recently-Used (LRU) cache for dense vector embeddings, avoiding re-computation for repeated queries or document chunks.
- **`QueryCache`**: Query response cache with configurable Time-To-Live (TTL, default 3,600s) and composite hashing over `(query_str, ticker, period, year, top_k, alpha)`.

### 8.2 Index Persistence (`IndexPersistence`)
- **Directory Persistence**: Saves `HybridIndex` state into modular files:
  - `manifest.json`: Versioning, document/chunk counts, vocabulary size.
  - `documents.json`: Serialized document graphs and metadata.
  - `chunks.json`: Multi-modal chunk data and pre-computed numeric values.
  - `bm25_index.json`: Inverted index frequencies, document lengths, IDF tables.
  - `vector_index.json`: Vector coordinate matrices and document mappings.
- **Compressed ZIP Export**: Bundles the entire index into a single self-contained `.zip` archive for distribution or snapshot backup.

---

## 9. Master Orchestration (`FinancialPipelineOrchestrator`)

The `FinancialPipelineOrchestrator` integrates all eight subsystems into a unified, high-level programmatic interface:
- Manages caching, persistence, ingestion, indexing, query routing, retrieval fusion, citation verification, audit reporting, and financial analytics.
- Exposes diagnostic status inspections (`status()`) returning live subsystem statistics.
