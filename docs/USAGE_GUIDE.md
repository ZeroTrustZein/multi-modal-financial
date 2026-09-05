# Practical Usage Guide: Multi-Modal Financial RAG

This guide provides real-world recipes and walkthroughs for integrating, configuring, and operating **`multi-modal-financial`** in production and quantitative research environments.

---

## Table of Contents
1. [Recipe 1: Ingestion and Pipeline Setup](#recipe-1-ingestion-and-pipeline-setup)
2. [Recipe 2: Precision Financial Querying with Citation Provenance](#recipe-2-precision-financial-querying-with-citation-provenance)
3. [Recipe 3: Auditing Hallucinations and Factual Verification](#recipe-3-auditing-hallucinations-and-factual-verification)
4. [Recipe 4: Automated Financial Statement Reconciliation](#recipe-4-automated-financial-statement-reconciliation)
5. [Recipe 5: Financial Ratios and Horizontal Period Comparison](#recipe-5-financial-ratios-and-horizontal-period-comparison)
6. [Recipe 6: Index Persistence, Cold Starts & Caching](#recipe-6-index-persistence-cold-starts--caching)
7. [Recipe 7: CLI in Automation Pipelines](#recipe-7-cli-in-automation-pipelines)

---

## Recipe 1: Ingestion and Pipeline Setup

### Using the Master Orchestrator

The `FinancialPipelineOrchestrator` provides an all-in-one interface with automatic text cleaning, table structural validation, and index persistence.

```python
from pathlib import Path
from multi_modal_financial.pipeline.orchestrator import (
    FinancialPipelineOrchestrator,
    OrchestratorConfig,
)

# 1. Configure the orchestrator
config = OrchestratorConfig(
    enable_query_cache=True,
    enable_embedding_cache=True,
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

Every query returns a verified answer with explicit source attribution and grounding confidence.

```python
from multi_modal_financial.pipeline.orchestrator import FinancialPipelineOrchestrator
from multi_modal_financial.types import AgentQuery

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
orchestrator.rag_pipeline.ingest_text(filing_text, doc_id="aapl_q3_2025", ticker="AAPL", year=2025)

# Execute query with hybrid retrieval (alpha=0.5)
query = AgentQuery(
    query_str="What was Apple's Services revenue and growth rate in Q3 2025?",
    ticker_filter="AAPL",
    top_k=3,
    alpha=0.5,
)
response = orchestrator.run_query(query)

print("Answer:", response.answer)
print("Groundedness Score:", response.groundedness_score)

# Inspect retrieved citations
for i, citation in enumerate(response.citations, 1):
    print(
        f"[{i}] Doc: {citation.doc_id}, Page: {citation.page_number}, Modality: {citation.modality.value}"
    )
    print(f"    Excerpt: {citation.excerpt}")
```

---

## Recipe 3: Auditing Hallucinations and Factual Verification

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

## Recipe 4: Automated Financial Statement Reconciliation

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

## Recipe 5: Financial Ratios and Horizontal Period Comparison

Compute profitability, liquidity, and solvency ratios directly from parsed documents and compare performance across fiscal periods.

```python
from multi_modal_financial.analytics.ratios import FinancialRatioCalculator
from multi_modal_financial.analytics.comparator import PeriodComparator
from multi_modal_financial.data.synthetic import SyntheticFilingGenerator

calculator = FinancialRatioCalculator()
comparator = PeriodComparator()
generator = SyntheticFilingGenerator()

# Ingest filings from two different quarters
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

## Recipe 6: Index Persistence, Cold Starts & Caching

Persist indexed documents and vector projection tables to disk to eliminate re-indexing latency on server warm-up.

```python
from multi_modal_financial.pipeline.orchestrator import FinancialPipelineOrchestrator

# Initialize and populate index
orchestrator = FinancialPipelineOrchestrator()
orchestrator.ingest_files("./samples", default_ticker="CORP")

# 1. Save index to a directory or compressed .zip file
save_path = orchestrator.save_index("./checkpoints/financial_index", compress=True)
print(f"Saved compressed index to {save_path}")

# 2. In a separate cold-start worker:
cold_orchestrator = FinancialPipelineOrchestrator()
cold_orchestrator.load_index("./checkpoints/financial_index.zip")
print(
    f"Loaded index with {cold_orchestrator.index.total_chunks()} chunks ready for immediate search."
)

# 3. Fast cached queries
resp1 = cold_orchestrator.run_query("What was the annual revenue?")
print(f"Query 1 latency: {resp1.execution_time_ms:.2f} ms")

# Repeated query served from in-memory QueryCache (<1ms)
resp2 = cold_orchestrator.run_query("What was the annual revenue?")
print(f"Query 2 (cached) latency: {resp2.execution_time_ms:.2f} ms")
```

---

## Recipe 7: CLI in Automation Pipelines

The CLI offers terminal inspection, ingestion, querying, and benchmarking for automated shell scripts and CI/CD pipelines.

### Check Pipeline Environment
```bash
multi-modal-financial info
```

### Batch Ingest Filings
```bash
# Ingest single file
multi-modal-financial ingest ./filings/apple_10k_2025.txt --ticker AAPL

# Ingest entire directory
multi-modal-financial ingest ./filings
```

### Query with Terminal Citations
```bash
# Query with hybrid alpha balance (0.5)
multi-modal-financial query "What was total revenue in Q3 2025?" --ticker ACM --top-k 3 --alpha 0.5
```

### Run Retrieval Benchmarks
```bash
multi-modal-financial benchmark
```
Outputs:
```
┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━┓
┃ Mode                ┃ Recall@1  ┃ MRR    ┃
┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━┩
│ BM25 (Lexical Only) │ 66.7%     │ 0.833  │
│ Hybrid (Dense+BM25) │ 100.0%    │ 1.000  │
└─────────────────────┴───────────┴────────┘
```
