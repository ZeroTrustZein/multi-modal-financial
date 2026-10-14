# Practical Usage Guide: Multi-Modal Financial RAG

This guide provides real-world recipes and walkthroughs for integrating, configuring, and operating **`multi-modal-financial`** in quantitative research, hedge fund, and enterprise financial analysis environments.

---

## Table of Contents
1. [Recipe 1: Ingestion and Pipeline Setup](#recipe-1-ingestion-and-pipeline-setup)
2. [Recipe 2: Precision Financial Querying with Citation Provenance](#recipe-2-precision-financial-querying-with-citation-provenance)
3. [Recipe 3: Multi-Strategy Hybrid Retrieval](#recipe-3-multi-strategy-hybrid-retrieval)
4. [Recipe 4: Neural Cross-Encoder & Hybrid Reranking with Explanations](#recipe-4-neural-cross-encoder--hybrid-reranking-with-explanations)
5. [Recipe 5: Enterprise Semantic Caching & Eviction Policies](#recipe-5-enterprise-semantic-caching--eviction-policies)
6. [Recipe 6: Auditing Hallucinations and Factual Verification](#recipe-6-auditing-hallucinations-and-factual-verification)
7. [Recipe 7: Automated Financial Statement Reconciliation](#recipe-7-automated-financial-statement-reconciliation)
8. [Recipe 8: Financial Ratios and Horizontal Period Comparison](#recipe-8-financial-ratios-and-horizontal-period-comparison)
9. [Recipe 9: Index & Semantic Cache Checkpoint Persistence](#recipe-9-index--semantic-cache-checkpoint-persistence)
10. [Recipe 10: Advanced CLI Automation & Telemetry Dashboards](#recipe-10-advanced-cli-automation--telemetry-dashboards)

---

## Recipe 1: Ingestion and Pipeline Setup

### Using the Master Orchestrator

The `FinancialPipelineOrchestrator` provides an all-in-one interface with automatic text cleaning, table structural validation, multi-tier caching, and index persistence.

```python
from pathlib import Path
from multi_modal_financial.pipeline.orchestrator import (
    FinancialPipelineOrchestrator,
    OrchestratorConfig,
)
from multi_modal_financial.types import (
    CacheEvictionPolicy,
    RerankerConfig,
    RerankerStrategy,
    SemanticCacheConfig,
)

# 1. Configure the orchestrator with multi-tier caching and hybrid reranking
config = OrchestratorConfig(
    enable_query_cache=True,
    enable_embedding_cache=True,
    enable_semantic_cache=True,
    semantic_cache_config=SemanticCacheConfig(
        similarity_threshold=0.85,
        max_entries=1000,
        eviction_policy=CacheEvictionPolicy.LRU,
    ),
    reranker_config=RerankerConfig(
        strategy=RerankerStrategy.HYBRID,
        cross_encoder_weight=0.7,
        heuristic_weight=0.3,
        table_boost=0.20,
    ),
    clean_text=True,
    validate_tables=True,
    default_top_k=5,
    default_alpha=0.5,
    confidence_threshold=0.70,
)
orchestrator = FinancialPipelineOrchestrator(config=config)

# 2. Ingest documents (supports text, markdown, json, or PDF)
docs = orchestrator.ingest_files("./data/filings", default_ticker="AAPL")
print(f"Ingested {len(docs)} documents.")

# 3. Check live pipeline status
status = orchestrator.status()
print(f"Total documents: {status['total_documents']}")
print(f"Total chunks: {status['total_chunks']}")
print(f"BM25 vocabulary: {status['bm25_vocab_size']} terms")
```

---

## Recipe 2: Precision Financial Querying with Citation Provenance

Every query returns a verified answer with explicit source attribution, grounding confidence, and fact-checking verdicts.

```python
from multi_modal_financial.pipeline.orchestrator import FinancialPipelineOrchestrator
from multi_modal_financial.types import AgentQuery, RetrievalStrategy

orchestrator = FinancialPipelineOrchestrator()

# Ingest sample earnings disclosure
filing_text = """
Apple Inc. (AAPL) reported financial results for Q3 2025:
- Quarterly revenue was $94.9 billion, an increase of 6% year-over-year.
- Products revenue was $69.9 billion.
- Services revenue reached an all-time record of $25.0 billion, up 14% YoY.
- Operating income rose to $29.6 billion, yielding an operating margin of 31.2%.
- Diluted earnings per share was $1.40, up 11% compared to Q3 2024.
"""
orchestrator.rag_pipeline.ingest_text(
    filing_text, doc_id="aapl_q3_2025", ticker="AAPL", year=2025, period="Q3"
)

# Execute query with hybrid retrieval (RRF) and ticker filter
query = AgentQuery(
    query_str="What was Apple's Services revenue and growth rate in Q3 2025?",
    ticker_filter="AAPL",
    period_filter="Q3",
    year_filter=2025,
    top_k=3,
    retrieval_strategy=RetrievalStrategy.HYBRID_RRF,
)
response = orchestrator.run_query(query)

print("Answer:", response.answer)
print(f"Groundedness Score: {response.groundedness_score:.2f}")

# Inspect retrieved citations
for i, citation in enumerate(response.citations, 1):
    print(
        f"[{i}] Doc: {citation.doc_id}, Page: {citation.page_number}, Modality: {citation.modality.value}"
    )
    print(f"    Excerpt: {citation.excerpt}")

# Inspect factual grounding verdict
if response.verdict:
    print("Verdict Status:", response.verdict.status.value)
    print("Verified Numbers:", response.verdict.verified_numbers)
    print("Hallucinated Numbers:", response.verdict.hallucinated_numbers)
```

---

## Recipe 3: Multi-Strategy Hybrid Retrieval

`multi-modal-financial` supports four distinct retrieval strategies via `RetrievalStrategy` and `HybridSearchConfig`.

```python
from multi_modal_financial.pipeline.orchestrator import FinancialPipelineOrchestrator
from multi_modal_financial.types import AgentQuery, RetrievalStrategy

orchestrator = FinancialPipelineOrchestrator()

# Ingest test filings
orchestrator.ingest_files("./samples", default_ticker="ACM")

query_str = "Operating margin and gross profit"

# Strategy 1: Reciprocal Rank Fusion (RRF, k=60) - Default for balanced precision & recall
q_rrf = AgentQuery(query_str=query_str, retrieval_strategy=RetrievalStrategy.HYBRID_RRF, top_k=3)
resp_rrf = orchestrator.run_query(q_rrf)
print(f"RRF retrieved {len(resp_rrf.retrieved_chunks)} candidates.")

# Strategy 2: Convex Alpha Combination - Ideal for fine-grained semantic-lexical weighting
# alpha=0.2 (80% BM25 lexical, 20% dense vector)
q_convex = AgentQuery(
    query_str=query_str,
    retrieval_strategy=RetrievalStrategy.HYBRID_CONVEX,
    alpha=0.2,
    top_k=3,
)
resp_convex = orchestrator.run_query(q_convex)

# Strategy 3: Dense Vector Semantic Only - Ideal for qualitative risk and broad queries
q_dense = AgentQuery(query_str=query_str, retrieval_strategy=RetrievalStrategy.DENSE, top_k=3)
resp_dense = orchestrator.run_query(q_dense)

# Strategy 4: Sparse Lexical Only (Okapi BM25) - Ideal for exact financial line items
q_sparse = AgentQuery(query_str=query_str, retrieval_strategy=RetrievalStrategy.SPARSE, top_k=3)
resp_sparse = orchestrator.run_query(q_sparse)
```

---

## Recipe 4: Neural Cross-Encoder & Hybrid Reranking with Explanations

The second-stage reranker (`FinancialReranker`) refines initial candidates using cross-token attention simulation (`FinancialCrossEncoder`), modality boosting, and temporal alignment.

```python
from multi_modal_financial.pipeline.orchestrator import (
    FinancialPipelineOrchestrator,
    OrchestratorConfig,
)
from multi_modal_financial.types import (
    AgentQuery,
    RerankerConfig,
    RerankerStrategy,
)

# 1. Configure hybrid reranking weights and score thresholds
reranker_cfg = RerankerConfig(
    strategy=RerankerStrategy.HYBRID,
    cross_encoder_weight=0.7,
    heuristic_weight=0.3,
    score_threshold=0.10,  # Prune low-scoring noise chunks
    table_boost=0.20,
    metric_boost=0.25,
    figure_boost=0.15,
)

orchestrator = FinancialPipelineOrchestrator(
    config=OrchestratorConfig(reranker_config=reranker_cfg)
)

# 2. Ingest multi-modal filing containing text, table, and figure
filing_text = """
Q3 2025 Financial Summary for ACME Corp (Ticker: ACM):
Total revenue for the third quarter was $1,250 million, up 15% year-over-year.
Gross profit reached $520 million, yielding a gross margin of 41.6%.
Operating income was $280 million compared to $240 million in Q3 2024.

| Metric | Q3 2024 | Q3 2025 | YoY Change |
| Revenue | $1,087M | $1,250M | +15.0% |
| Net Income | $185M | $215M | +16.2% |
| Free Cash Flow | $140M | $175M | +25.0% |

Figure 1: Quarterly Revenue Breakdown by Segment
Cloud: 650.0, Enterprise: 400.0, Consumer: 200.0
"""
orchestrator.rag_pipeline.ingest_text(
    filing_text, doc_id="acme_q3_2025", ticker="ACM", period="Q3", year=2025
)

# 3. Query with reranking explanations enabled
query = AgentQuery(
    query_str="What was revenue and operating income for ACM in Q3?",
    ticker_filter="ACM",
    top_k=3,
    reranker_strategy=RerankerStrategy.HYBRID,
    enable_rerank_explanation=True,
)
response = orchestrator.run_query(query)

# 4. Inspect explainability breakdown
print("\n--- Reranker Factor Attribution ---")
for exp in response.rerank_explanations:
    print(f"Chunk ID: {exp.chunk_id}")
    print(f"  Rank Shift: #{exp.initial_rank} -> #{exp.final_rank}")
    print(
        f"  Score: {exp.initial_score:.4f} -> {exp.final_score:.4f} (Delta: {exp.final_score - exp.initial_score:+.4f})"
    )
    print(f"  Cross-Encoder Score: {exp.cross_encoder_score}")
    print(f"  Heuristic Score: {exp.heuristic_score}")
    print(f"  Modality Bonus: +{exp.modality_bonus:.2f}")
    print(f"  Reasons: {'; '.join(exp.reasons)}")
```

---

## Recipe 5: Enterprise Semantic Caching & Eviction Policies

The `SemanticCache` accelerates repetitive or semantically similar queries by returning deep-copied responses with vector proximity gating.

```python
from multi_modal_financial.pipeline.orchestrator import (
    FinancialPipelineOrchestrator,
    OrchestratorConfig,
)
from multi_modal_financial.types import (
    AgentQuery,
    CacheEvictionPolicy,
    SemanticCacheConfig,
)

# 1. Configure semantic cache with LFU eviction and 0.85 similarity threshold
cache_config = SemanticCacheConfig(
    enabled=True,
    similarity_threshold=0.85,
    max_entries=500,
    ttl_seconds=86400.0,  # 24-hour TTL
    eviction_policy=CacheEvictionPolicy.LFU,
    distance_metric="cosine",
)

orchestrator = FinancialPipelineOrchestrator(
    config=OrchestratorConfig(
        enable_semantic_cache=True,
        semantic_cache_config=cache_config,
    )
)
orchestrator.ingest_files("./samples", default_ticker="ACM")

# 2. Initial query: Cache miss -> full pipeline execution
q1 = AgentQuery(query_str="What was total revenue in Q3 2025 for ACM?")
resp1 = orchestrator.run_query(q1)
print(f"Q1 Latency: {resp1.execution_time_ms:.2f}ms | Cache Hit: {resp1.cache_hit}")

# 3. Exact query: Cache hit via O(1) fast-path
resp2 = orchestrator.run_query(q1)
print(
    f"Q2 Latency: {resp2.execution_time_ms:.2f}ms | Cache Hit: {resp2.cache_hit} ({resp2.cache_type.value})"
)

# 4. Semantically equivalent query: Cache hit via vector cosine proximity
q3 = AgentQuery(query_str="How much revenue did ACME generate in third quarter 2025?")
resp3 = orchestrator.run_query(q3)
print(
    f"Q3 Latency: {resp3.execution_time_ms:.2f}ms | Cache Hit: {resp3.cache_hit} ({resp3.cache_type.value})"
)
print(f"Semantic Cosine Similarity: {resp3.cache_similarity:.4f}")

# 5. Inspect operational cache telemetry
stats = orchestrator.semantic_cache.stats()
print(f"Total Lookups: {stats.total_queries}")
print(f"Exact Hits: {stats.exact_hits}")
print(f"Semantic Hits: {stats.semantic_hits}")
print(f"Hit Rate: {stats.hit_rate:.1%}")
print(f"Average Lookup Latency: {stats.avg_lookup_latency_ms:.2f}ms")
```

---

## Recipe 6: Auditing Hallucinations and Factual Verification

Use the `GroundingAuditor` to run governance audits over model responses, ensuring compliance with institutional risk standards.

```python
from multi_modal_financial.pipeline.orchestrator import FinancialPipelineOrchestrator

orchestrator = FinancialPipelineOrchestrator()

# Test queries to verify
audit_queries = [
    "What was Apple's total quarterly revenue in Q3 2025?",
    "How much did Services revenue increase year-over-year?",
    "What was diluted earnings per share?",
]

# Run automated batch audit
audit_report = orchestrator.audit_queries(audit_queries)

print("Audit Passed:", audit_report.audit_passed)
print("Hallucination Rate:", f"{audit_report.hallucination_rate * 100:.1f}%")
print("Mean Confidence:", f"{audit_report.mean_confidence:.2f}")

# Export audit findings to JSON for risk compliance
audit_report.export_json("audit_compliance_report.json")
```

---

## Recipe 7: Automated Financial Statement Reconciliation

Validate tabular disclosures against GAAP balance sheet and income statement accounting identities before allowing them into production indexes.

```python
from multi_modal_financial.data.synthetic import SyntheticFilingGenerator
from multi_modal_financial.data.validator import StatementReconciler

reconciler = StatementReconciler()
generator = SyntheticFilingGenerator()

# 1. Generate or parse a balance sheet
balance_sheet = generator.generate_balance_sheet(ticker="MSFT", year=2025, period="FY")

# 2. Test: Total Assets = Total Liabilities + Stockholders' Equity
bs_result = reconciler.reconcile_balance_sheet(balance_sheet, tolerance=1.0)
print(f"Balance Sheet Reconciled: {bs_result.is_balanced}")
print(f"Discrepancy: ${bs_result.discrepancy:,.2f}")
if not bs_result.is_balanced:
    print("Violations:", bs_result.messages)

# 3. Test: Gross Profit and Operating Income reconciliations
income_statement = generator.generate_income_statement(ticker="MSFT", year=2025, period="FY")
is_result = reconciler.reconcile_income_statement(income_statement, tolerance=1.0)
print(f"Income Statement Reconciled: {is_result.is_balanced}")
```

---

## Recipe 8: Financial Ratios and Horizontal Period Comparison

Compute profitability, liquidity, and solvency ratios directly from parsed documents and compare performance across fiscal periods.

```python
from multi_modal_financial.analytics.ratios import FinancialRatioCalculator
from multi_modal_financial.analytics.comparator import PeriodComparator
from multi_modal_financial.data.synthetic import SyntheticFilingGenerator

calculator = FinancialRatioCalculator()
comparator = PeriodComparator()
generator = SyntheticFilingGenerator()

# Ingest filings from two consecutive quarters
doc_q2 = generator.generate_filing_document(ticker="NVDA", year=2025, period="Q2")
doc_q3 = generator.generate_filing_document(ticker="NVDA", year=2025, period="Q3")

# 1. Compute financial ratios for Q3
ratios_q3 = calculator.compute_from_document(doc_q3)
print(f"Gross Margin: {ratios_q3.gross_margin:.1%}")
print(f"Operating Margin: {ratios_q3.operating_margin:.1%}")
print(f"Net Margin: {ratios_q3.net_margin:.1%}")
print(f"Current Ratio: {ratios_q3.current_ratio:.2f}")
print(f"Debt-to-Equity: {ratios_q3.debt_to_equity:.2f}")

# 2. Compare horizontal period variances (Q2 vs Q3)
variances = comparator.compare_documents(doc_q2, doc_q3)
for var in variances:
    print(
        f"{var.metric_name:<20} | Base: ${var.base_value:,.1f}M -> Compare: ${var.compare_value:,.1f}M | "
        f"Change: {var.percent_change:+.1f}% ({var.direction})"
    )
```

---

## Recipe 9: Index & Semantic Cache Checkpoint Persistence

Persist indexed documents, vector projection tables, and semantic cache state to disk or compressed ZIP archives to eliminate re-indexing latency on server warm-up.

```python
from multi_modal_financial.pipeline.orchestrator import FinancialPipelineOrchestrator

orchestrator = FinancialPipelineOrchestrator()
orchestrator.ingest_files("./samples", default_ticker="CORP")

# Execute queries to populate semantic cache
orchestrator.run_query("What was annual revenue in 2025?")

# 1. Save full index state to compressed .zip archive
zip_path = orchestrator.save_index("./checkpoints/financial_index.zip", compress=True)
print(f"Saved index archive: {zip_path}")

# 2. Persist semantic cache state separately
cache_path = orchestrator.save_semantic_cache("./checkpoints/semantic_cache.json")
print(f"Saved semantic cache: {cache_path}")

# 3. Cold restart: In a separate worker process
cold_orchestrator = FinancialPipelineOrchestrator()
cold_orchestrator.load_index("./checkpoints/financial_index.zip")
cold_orchestrator.load_semantic_cache("./checkpoints/semantic_cache.json")

print(f"Restored index with {cold_orchestrator.index.total_chunks()} chunks.")
print(f"Restored cache with {cold_orchestrator.semantic_cache.stats().entry_count} entries.")

# Query served instantaneously from warm semantic cache
res = cold_orchestrator.run_query("What was annual revenue in 2025?")
print(f"Response served in {res.execution_time_ms:.2f}ms (Cache Hit: {res.cache_hit})")
```

---

## Recipe 10: Advanced CLI Automation & Telemetry Dashboards

The CLI provides terminal inspection, ingestion, querying, ratio extraction, period comparison, cache telemetry, and compliance auditing.

### Check Pipeline Architecture and Live Cache Status
```bash
# Architecture overview
multi-modal-financial info

# Live cache occupancy and telemetry
multi-modal-financial info --status
```

### Ingest Filings
```bash
# Ingest single filing
multi-modal-financial ingest ./filings/apple_10k_2025.txt --ticker AAPL

# Ingest directory with automatic table validation
multi-modal-financial ingest ./filings --validate
```

### Query with Hybrid Reranking & Factor Explanations
```bash
multi-modal-financial query "What was revenue and operating income for ACM in Q3?" \
    --ticker ACM \
    --strategy hybrid_rrf \
    --reranker-strategy hybrid \
    --explain
```

### Semantic Cache Telemetry Table
```bash
multi-modal-financial cache-stats
```

### Compute Statement Ratios from CLI
```bash
multi-modal-financial ratios ./filings/apple_q3_2025.txt
```

### Calculate Horizontal Variances Across Filings
```bash
multi-modal-financial compare ./filings/aapl_q2_2025.txt ./filings/aapl_q3_2025.txt
```

### Factual Grounding Compliance Audit
```bash
multi-modal-financial audit \
    "What was Apple's Services revenue?" \
    "What was diluted EPS in Q3 2025?" \
    --max-hallucination-rate 0.05
```

### Run Multi-Modal Retrieval Benchmark Suite
```bash
multi-modal-financial benchmark --top-k 3
```
Output:
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
