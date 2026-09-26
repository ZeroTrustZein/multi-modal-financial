# Multi-Modal Financial RAG: System Architecture & Technical Deep-Dive

This document provides a comprehensive technical breakdown of the architecture, algorithmic formulations, mathematical models, and subsystem designs behind **`multi-modal-financial`**.

---

## 1. System Overview & Design Philosophy

Financial document analysis imposes rigorous demands that generic Retrieval-Augmented Generation (RAG) pipelines fail to meet:
1. **Multi-Modal Data Heterogeneity**: Financial filings (SEC 10-K, 10-Q, 8-K, earnings releases) interleave prose narratives, dense quantitative tables (GAAP balance sheets, income statements), and visual figures/charts.
2. **Numeric Precision vs. Semantic Drift**: Semantic vector search frequently confuses numeric values or loses tabular column context (e.g., mistaking 2024 revenue for 2025 revenue). Lexical precision must be fused with semantic recall.
3. **Zero-Tolerance Hallucination & Auditability**: Financial assertions require verifiable sentence-level citation provenance linking directly back to audited source documents and exact table cells.
4. **Statement Reconciliation**: Numeric tables must be validated against accounting invariants ($Assets = Liabilities + Equity$) before indexing to filter corrupt OCR or malformed data.

To solve these challenges, `multi-modal-financial` is designed as a modular, decoupled pipeline of 9 core subsystems governed by runtime protocols:

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
                      │ Multi-Tier Storage    ▼                     │
                      │ - EmbeddingCache (LRU)  IndexPersistence    │
                      │ - QueryCache (TTL)      (Directory / ZIP)   │
                      │ - SemanticCache (LRU/LFU/FIFO + Vectors)    │
                      └───────────────────────┬─────────────────────┘
                                              │
       [ User Query ]                         ▼
             │        ┌─────────────────────────────────────────────┐
             ▼        │      Multi-Strategy Hybrid Retrieval        │
     ┌──────────────┐ │  - Reciprocal Rank Fusion (RRF, k=60)       │
     │ QueryRouter  │ │  - Convex Alpha Blend (BM25 + Dense)        │
     │ Intent/Ticker│─┼─►- Pure Dense Semantic / Pure Sparse Lexical│
     └──────────────┘ └──────────────────────┬──────────────────────┘
                                             │ Initial Candidate Pool
                                             ▼
                      ┌─────────────────────────────────────────────┐
                      │ Neural Cross-Encoder & Hybrid Reranking     │
                      │  - FinancialCrossEncoder (Cross-Attention)  │
                      │  - FinancialReranker (Modality & Ticker)    │
                      │  - RerankExplanation (Factor Attribution)   │
                      └──────────────────────┬──────────────────────┘
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

## 4. Multi-Strategy Hybrid Retrieval & Neural Cross-Encoder Reranking Core

`multi-modal-financial` implements a two-stage retrieval and reranking pipeline combining high-recall multi-strategy candidate generation with precision neural cross-encoder rescoring and domain heuristic boosting.

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
             [ Retrieval Engine ]
         ├── Reciprocal Rank Fusion (RRF, k=60)
         ├── Convex Alpha Blending (Min-Max / Z-Score)
         ├── Pure Dense Semantic Retrieval
         └── Pure Sparse Lexical Retrieval
                      │
                      ▼
         Initial Candidate Set (fetch_limit = max(4k, 30))
                      │
                      ▼
        [ Second-Stage Reranking Engine ]
         ├── FinancialCrossEncoder: Pairwise Cross-Attention Simulation
         │   ├── Financial Concept Overlap (Weights: revenue=1.5, eps=1.5, ...)
         │   ├── N-gram Sequence Alignment
         │   └── Numeric / Monetary Fact Congruence
         ├── Modality Boosting (Table +0.20, Metric +0.25, Figure +0.15)
         ├── Temporal Alignment (Fiscal Year +0.20, Quarter Period +0.20)
         └── Score Threshold Filtering & Top-K Pruning
                      │
                      ▼
             [ RerankExplanation ]
         (Initial Rank -> Final Rank, Score Delta, Component Factors)
                      │
                      ▼
             Top-K Scored Chunks
```

### 4.1 Multi-Strategy Retrieval Core (`HybridRetriever`, `HybridSearchConfig`)
The first retrieval stage generates candidate document chunks from `HybridIndex` using four configurable strategies defined by `RetrievalStrategy`:

1. **Reciprocal Rank Fusion (`HYBRID_RRF`)**:
   Merges ordinal rank positions across disparate lexical and semantic score distributions without calibration artifacts:

   $$\text{RRF\_Score}(d) = \sum_{m \in M} \frac{w_m}{k + r_m(d)}$$

   Where:
   - $M = \{\text{BM25}, \text{Dense}\}$
   - $r_m(d) \in \{1, 2, \dots, K\}$ is the 1-based ordinal rank of chunk $d$ in subsystem $m$.
   - $k$ is the smoothing constant (default $k = 60$, configurable via `rrf_k`).
   - $w_m$ represents modality retrieval weights (default $w_{\text{dense}} = 1.0, w_{\text{bm25}} = 1.0$).

2. **Convex Alpha Combination (`HYBRID_CONVEX`)**:
   Blends calibrated scores after Min-Max or Z-Score standardization:

   $$\tilde{S}(d) = \frac{S(d) - S_{\min}}{S_{\max} - S_{\min} + \epsilon}$$

   $$S_{\text{hybrid}}(d) = \alpha \cdot \tilde{S}_{\text{dense}}(d) + (1 - \alpha) \cdot \tilde{S}_{\text{bm25}}(d)$$

   - $\alpha = 0.0$: Pure lexical BM25 (ideal for precise financial line-item search).
   - $\alpha = 1.0$: Pure dense vector semantic search (ideal for conceptual thematic inquiries).
   - $\alpha = 0.5$: Balanced hybrid retrieval (default).

3. **Dense Semantic Only (`DENSE`)**:
   Bypasses lexical search, ranking chunks purely by unit-normalized cosine similarity:
   $$\text{Sim}(q, d) = \sum_{i=1}^{\text{dim}} q_i \cdot d_i$$

4. **Sparse Lexical Only (`SPARSE`)**:
   Bypasses dense vectors, returning candidates strictly ordered by normalized Okapi BM25 relevance.

**Metadata Filtering & Candidate Constraints**:
`HybridRetriever` filters candidate chunk IDs prior to scoring via `index.filter_chunk_ids()`, matching:
- `ticker`: Filters strictly to chunks attributed to the specified equity ticker.
- `period`: Filters to designated quarter or fiscal year (`Q1`-`Q4`, `FY`).
- `year`: Filters to specific filing reporting years.
- `modal_type`: Filters to desired modalities (`table`, `text`, `figure`, `metric`).
- `allowed_chunk_ids`: Whitelist constraint defined via `HybridSearchConfig`.

---

### 4.2 Neural Cross-Encoder Scoring Architecture (`FinancialCrossEncoder`)
Conforming to `CrossEncoderProtocol`, `FinancialCrossEncoder` simulates transformer-style cross-attention by jointly scoring candidate $(Q, D)$ pairs over financial semantic interactions:

1. **Financial Concept Weighted Term Overlap**:
   Terms are matched against a curated financial dictionary where critical disclosures carry elevated weights:

   $$S_{\text{term}} = \frac{\sum_{t \in Q \cap D} w_t}{\sum_{t \in Q} w_t}$$

   | Financial Concept | Weight $w_t$ | Financial Concept | Weight $w_t$ |
   |-------------------|--------------|-------------------|--------------|
   | `revenue`, `income`, `eps`, `ebitda` | $1.5$ | `sales`, `profit`, `margin` | $1.4$ |
   | `operating`, `cash`, `flow`, `capex`, `guidance` | $1.3$ | `debt`, `equity`, `asset`, `liability`, `growth` | $1.2$ |
   | Default unlisted terms | $1.0$ | | |

2. **N-Gram Sequence Alignment**:
   Measures consecutive bi-gram overlap between query tokens $B_Q$ and document tokens $B_D$:
   $$S_{\text{ngram}} = \frac{|B_Q \cap B_D|}{|B_Q|} \quad (|B_Q| \ge 2)$$

3. **Numeric & Monetary Fact Congruence**:
   Quantifies exact matching between numerical figures $N_Q$ in the query (e.g., `$1,250M`, `41.6%`, `2025`) and numbers $N_D$ present in the candidate chunk:
   $$S_{\text{num}} = \frac{|N_Q \cap N_D|}{|N_Q|}$$

4. **Joint Logit & Sigmoid Probability Mapping**:
   The individual factors are combined into a joint relevance logit and mapped to $[0, 1]$ via a clamped logistic sigmoid:

   $$\text{logit}(Q, D) = 2.5 \cdot S_{\text{term}} + 1.5 \cdot S_{\text{ngram}} + 1.2 \cdot S_{\text{num}} - 1.0$$

   $$P(\text{rel} \mid Q, D) = \sigma(\text{logit}) = \frac{1}{1 + \exp\left(-\max(\min(\text{logit}, 10.0), -10.0)\right)}$$

---

### 4.3 Second-Stage Domain Reranker (`FinancialReranker`)
Conforming to `RerankerProtocol`, `FinancialReranker` refines initial retrieval candidates using neural cross-attention, domain heuristics, and modality bonuses configured via `RerankerConfig`:

1. **Domain Heuristic Scoring**:
   Combines base retrieval score with lexical, numerical, and temporal bonuses:
   $$S_{\text{heur}} = S_{\text{init}} + 0.30 \cdot \text{Overlap}_{\text{lex}} + 0.35 \cdot \text{Overlap}_{\text{num}} + \text{Bonus}_{\text{temporal}}$$
   Where $\text{Bonus}_{\text{temporal}} = 0.20$ if fiscal years match $+ 0.20$ if reporting periods (`Q1`-`Q4`, `FY`) match.

2. **Modality Boosting**:
   Applies structural incentives depending on chunk content type and query intent:
   - **Table Chunk**: $+0.20$ (additional $+0.10$ if query contains keywords: `table`, `compare`, `margin`, `statement`, `breakdown`).
   - **Metric Chunk**: $+0.25$ bonus for pre-extracted atomic financial metrics.
   - **Figure / Chart Chunk**: $+0.15$ (additional $+0.10$ if query contains keywords: `chart`, `figure`, `graph`, `trend`, `trajectory`).

3. **Strategy Blending (`RerankerStrategy`)**:
   - `CROSS_ENCODER`: Uses pure normalized cross-encoder predictions plus modality bonuses:
     $$S_{\text{final}} = S_{\text{ce}} + \text{Bonus}_{\text{modality}}$$
   - `HYBRID`: Blends neural cross-encoder and domain heuristics via configurable weights:
     $$S_{\text{final}} = (w_{\text{ce}} \cdot S_{\text{ce}}) + (w_{\text{heur}} \cdot S_{\text{heur}}) + \text{Bonus}_{\text{modality}}$$
     *(Defaults: $w_{\text{ce}} = 0.7$, $w_{\text{heur}} = 0.3$)*
   - `MODALITY_HEURISTIC`: Applies domain heuristics and modality bonuses to initial retrieval scores.
   - `NONE`: Preserves initial retrieval candidate ordering.

4. **Candidate Pruning & Threshold Filtering**:
   Drops any candidate with $S_{\text{final}} < \text{score\_threshold}$ (default $0.0$) and caps results to `top_k` (or `reranker_top_k`).

---

### 4.4 Reranking Explainability Engine (`RerankExplanation`)
For auditing and regulatory transparency, every reranked candidate emits a structured `RerankExplanation`:
- `chunk_id`: Unique chunk identifier.
- `initial_rank` $\to$ `final_rank`: Direct rank shift observation (e.g. `#4 -> #1`).
- `initial_score` and `final_score`: Magnitude of score delta ($\Delta$).
- `cross_encoder_score`: Raw or normalized neural cross-attention score.
- `heuristic_score`: Composite domain heuristic score.
- `modality_bonus`: Additive structural bonus.
- `reasons`: Human-interpretable factors (e.g., `["Hybrid: CE=1.0000 (w=0.7) + Heur=0.4500 (w=0.3)", "Lexical overlap: 0.86", "Period match: Q3", "Table modality boost: +0.20"]`).

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

## 8. Storage, Multi-Tier Caching & Persistence

`multi-modal-financial` provides an enterprise-grade multi-tier caching hierarchy and atomic checkpoint persistence to minimize latency and ensure zero cold-start penalties in production environments.

### 8.1 Multi-Tier Caching Subsystem

```
                         Incoming User Query
                                  │
                                  ▼
               ┌─────────────────────────────────────┐
               │ Tier 1: QueryCache (Exact TTL Hash) │
               │ Hash: (query, ticker, period, top_k)│
               └──────────────────┬──────────────────┘
                                  │ Miss
                                  ▼
               ┌─────────────────────────────────────┐
               │ Tier 2: SemanticCache               │
               │  ├── Fast-Path Exact Lookup (O(1))  │
               │  └── Vector Similarity Search       │
               │      (Cosine / Euclidean / Dot)     │
               └──────────────────┬──────────────────┘
                                  │ Miss
                                  ▼
               ┌─────────────────────────────────────┐
               │ Full RAG Pipeline & Reranking Core  │
               │  ├── Tier 3: EmbeddingCache (LRU)   │
               │  │   (Reuses projection vectors)    │
               │  └── Hybrid Search & Verification   │
               └─────────────────────────────────────┘
```

1. **Embedding Cache (`EmbeddingCache`)**:
   - Thread-safe Least-Recently-Used (LRU) in-memory cache for dense vector embeddings.
   - Keys: SHA-256 digest of normalized chunk or query string.
   - Values: Pre-computed unit-normalized NumPy float32 embedding vectors.
   - Batch computation with fallback via `get_or_compute(texts, compute_fn)`.

2. **Exact Query Cache (`QueryCache`)**:
   - In-memory LRU cache storing full `AgentResponse` objects.
   - Composite Key: `(query_str, ticker, period, year, top_k, alpha)`.
   - Expiration: Configurable Time-To-Live (TTL, default 3,600s).

3. **Neural Semantic Cache (`SemanticCache`)**:
   Conforming to `SemanticCacheProtocol`, `SemanticCache` caches query-response pairs and evaluates semantic vector proximity:
   - **Exact Match Fast Path ($O(1)$)**: Checks normalized query string hash index before executing vector similarity scans, ensuring sub-millisecond retrieval on identical queries.
   - **Vector Proximity Search**: Computes similarity between the incoming query vector and active cache entry embeddings:
     - **Cosine Similarity** (default):
       $$\text{Sim}_{\cos}(u, v) = \frac{u \cdot v}{\|u\|_2 \|v\|_2}$$
     - **Euclidean Similarity**:
       $$\text{Sim}_{\text{euc}}(u, v) = \frac{1}{1 + \|u - v\|_2}$$
     - **Dot Product**:
       $$\text{Sim}_{\text{dot}}(u, v) = u \cdot v$$
   - **Similarity Gating**: Cache hits require $\text{Sim} \ge \tau$ (default $\tau = 0.85$, configurable via `SemanticCacheConfig.similarity_threshold`). Hits return a deep copy of `AgentResponse` marked with `cache_hit=True`, `cache_type=CacheHitType.SEMANTIC`, and the exact similarity score.
   - **Configurable Eviction Policies (`CacheEvictionPolicy`)**:
     - `LRU` (Least Recently Used): Evicts the entry with the oldest `last_accessed_at` timestamp.
     - `LFU` (Least Frequently Used): Evicts the entry with the lowest `access_count`, breaking ties by `last_accessed_at`.
     - `FIFO` (First In, First Out): Evicts the earliest inserted entry by `created_at`.
   - **TTL Lifecycle & Lazy/Eager Purge**: Entries beyond `ttl_seconds` are lazily removed upon search scans and proactively purged during capacity eviction in `put()`.
   - **Telemetry Monitoring (`SemanticCacheStats`)**: Captures operational telemetry: `total_queries`, `exact_hits`, `semantic_hits`, `misses`, `evictions`, `entry_count`, `max_entries`, `hit_rate`, and `avg_lookup_latency_ms`.

---

### 8.2 Checkpoint Index & Cache Persistence (`IndexPersistence`)

`IndexPersistence` serializes the complete state of `HybridIndex` and `SemanticCache` to disk or compressed ZIP archives:

```
persisted_index/ (or index_archive.zip)
├── manifest.json         # Checksum, version, document/chunk counts, vocabulary size, cache metadata
├── documents.json        # Serialized Document models, metadata, and section boundaries
├── chunks.json           # Chunk text, modalities, page numbers, and pre-extracted metric values
├── bm25.json             # Inverted index frequencies, document lengths, IDF tables, params (k1, b)
├── vectors.npz           # Compressed NumPy array of dense vectors and document ID mappings
└── semantic_cache.json   # Serialized SemanticCache entries, query vectors, responses, and access stats
```

- **Checksum Validation**: Computes a SHA-256 digest across all constituent files during serialization, preventing corruption during cold starts.
- **Archive Portability**: Compressed `.zip` archives can be transferred across cloud environments or pre-baked into container images for zero-latency startup.

---

## 9. Master Orchestration (`FinancialPipelineOrchestrator`)

The `FinancialPipelineOrchestrator` integrates all nine subsystems into a unified, enterprise-ready controller configured via `OrchestratorConfig`:
- Manages multi-tier caching (`EmbeddingCache`, `QueryCache`, `SemanticCache`), persistence checkpoints, and document ingestion.
- Automates text cleaning, table structural validation, and GAAP statement reconciliation prior to indexing.
- Coordinates query routing, multi-strategy hybrid retrieval (`HybridRetriever`), neural cross-encoder reranking (`FinancialReranker`), and citation grounding verification (`GroundingVerifier`).
- Provides financial ratio analytics (`FinancialRatioCalculator`) and horizontal period variance comparisons (`PeriodComparator`).
- Generates batch compliance reports (`GroundingAuditor`).
- Exposes diagnostic status inspections (`status()`) returning live metrics across documents, chunks, vocabulary, caches, reranker, and retrieval strategies.

---

## 10. Runtime Protocols & Abstract Interface Architecture

To ensure strict architectural decoupling, maintainability, and clean dependency inversion, all primary subsystems in `multi-modal-financial` conform to PEP 544 `@runtime_checkable` protocols defined in `multi_modal_financial.interfaces`:

```python
from multi_modal_financial.interfaces import (
    CrossEncoderProtocol,
    DenseIndexProtocol,
    RerankerProtocol,
    RetrieverProtocol,
    SemanticCacheProtocol,
    SparseIndexProtocol,
)
```

### 10.1 Protocol Specifications

1. **`CrossEncoderProtocol`**:
   Abstract interface for neural cross-attention models:
   ```python
   def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
       """Compute cross-attention relevance scores for (query, document) pairs."""
       ...
   ```

2. **`RerankerProtocol`**:
   Abstract interface for candidate rerankers:
   ```python
   def rerank(
       self,
       query: str | AgentQuery,
       candidates: list[ScoredChunk],
       top_k: int | None = None,
   ) -> list[ScoredChunk]:
       """Rerank candidates based on neural and/or domain financial salience."""
       ...
   ```

3. **`SemanticCacheProtocol`**:
   Abstract interface for semantic query and response caching:
   ```python
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

4. **`DenseIndexProtocol`**:
   Abstract interface for dense vector indexing:
   ```python
   def search(self, query: str | list[float], top_k: int = 10) -> list[tuple[str, float]]: ...
   def add(self, chunk_id: str, text: str, vector: list[float] | None = None) -> None: ...
   ```

5. **`SparseIndexProtocol`**:
   Abstract interface for lexical BM25 indexing:
   ```python
   def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]: ...
   def add(self, chunk_id: str, text: str) -> None: ...
   ```

6. **`RetrieverProtocol`**:
   Abstract interface for hybrid multi-modal retrieval engines:
   ```python
   def retrieve(
       self, query: str | AgentQuery, top_k: int = 10, alpha: float = 0.5
   ) -> list[ScoredChunk]: ...
   ```

### 10.2 Architectural Benefits
- **Zero-Cost Runtime Validation**: Applications can verify implementations dynamically using `isinstance(custom_model, CrossEncoderProtocol)` or `isinstance(custom_cache, SemanticCacheProtocol)`.
- **Pluggable Neural Backends**: Enterprise deployments can substitute local mock cross-encoders with GPU-accelerated Hugging Face transformers, ONNX Runtime engines, or remote inference endpoints without modifying pipeline code.
- **Modular Storage Tiers**: The semantic cache and indexing engines can be backed by Redis, Faiss, Qdrant, or Pinecone by implementing the respective protocol methods.
- **Isolated Testing & Mocking**: Unit tests verify interfaces without requiring neural model checkpoints or external databases.
