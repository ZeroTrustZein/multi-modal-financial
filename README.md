# Multi-Modal Financial Document RAG Pipeline

Multi-modal financial document RAG agent pipeline with hybrid BM25 dense vector reranking and citation grounding.

## Features

- **Multi-Modal Document Parsing**: Extract text, financial tables, metrics, and figures from earnings releases, SEC filings (10-K, 10-Q), and analyst reports.
- **Hybrid Retrieval**: Combines BM25 lexical sparse search with dense semantic vector embeddings.
- **RRF & Dense-Sparse Reranking**: Reciprocal Rank Fusion (RRF) and score normalization for precision retrieval over financial disclosures.
- **Citation Grounding & Attribution**: Hallucination suppression with strict provenance tracking, sentence-level citation verification, and factual support scores.
- **Agent Reasoning & Routing**: Specialized financial routing for comparative balance sheets, cash flows, guidance analysis, and risk metrics.
- **CLI & Evaluation Suite**: Complete command-line interface for indexing, querying, grounding audits, and retrieval benchmarks.

## Architecture

```
Financial Document (PDF / HTML / Text)
        │
        ▼
[ Multi-Modal Parser ] ───► Text Chunks + Financial Tables + Figures
        │
        ▼
[ Hybrid Index ]
   ├── Sparse BM25 Index (Lexical terms, ticker symbols, table headers)
   └── Dense Vector Index (Semantic embeddings, cosine similarity)
        │
        ▼
[ Hybrid Retriever & Reranker ]
   ├── Reciprocal Rank Fusion (RRF) / Alpha Blend
   └── Cross-Encoder / Relevance Scoring
        │
        ▼
[ Financial Agent & Grounding Engine ]
   ├── Structured Context Synthesis
   ├── Citation Extraction & Verification
   └── Provenance & Factual Grounding Verdict
```

## Installation

```bash
pip install -e .
# Dev dependencies
pip install -e ".[dev]"
```

## Quick Start

```bash
# Ingest and index financial documents
multi-modal-financial ingest --input-dir ./samples

# Query with hybrid retrieval and citation grounding
multi-modal-financial query "What was the operating margin in Q3 2025?"

# Run retrieval benchmark
multi-modal-financial benchmark
```
