"""End-to-end integration test suite for the Multi-Modal Financial RAG Pipeline.

Validates the full lifecycle:
Synthetic Generation -> Batch Loading & Cleaning -> Validation & Reconciliation ->
Multi-Modal Indexing -> Hybrid Retrieval & Reranking -> Agent Synthesis & Grounding ->
Audit Verification -> Checkpoint Persistence & Recovery.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pypdf
import pytest

from multi_modal_financial.agent.router import QueryIntent
from multi_modal_financial.analytics.comparator import PeriodComparator
from multi_modal_financial.analytics.ratios import FinancialRatioCalculator
from multi_modal_financial.data.loader import BatchDocumentLoader
from multi_modal_financial.data.synthetic import SyntheticFilingGenerator
from multi_modal_financial.data.validator import (
    StatementReconciler,
)
from multi_modal_financial.pipeline.orchestrator import (
    FinancialPipelineOrchestrator,
    OrchestratorConfig,
)
from multi_modal_financial.storage.cache import EmbeddingCache, QueryCache
from multi_modal_financial.storage.persistence import IndexPersistence
from multi_modal_financial.types import (
    AgentQuery,
    AgentResponse,
    DocumentType,
)


class TestFullLifecycleIntegration:
    """End-to-end integration testing across all pipeline subsystems."""

    @pytest.fixture
    def corpus_dir(self, tmp_path: Path) -> Path:
        """Create a directory with multiple synthetic filing documents and formats."""
        corpus = tmp_path / "financial_corpus"
        corpus.mkdir()

        gen = SyntheticFilingGenerator(seed=42)

        # 1. Synthetic 10-K for AAPL
        doc_aapl = gen.generate_filing_document(
            ticker="AAPL", year=2025, period="FY", doc_type=DocumentType.TEN_K
        )
        (corpus / "AAPL_10K_2025.json").write_text(json.dumps(doc_aapl.to_dict()), encoding="utf-8")

        # 2. Synthetic 10-Q for MSFT
        doc_msft = gen.generate_filing_document(
            ticker="MSFT", year=2025, period="Q3", doc_type=DocumentType.TEN_Q
        )
        (corpus / "MSFT_10Q_2025_Q3.json").write_text(
            json.dumps(doc_msft.to_dict()), encoding="utf-8"
        )

        # 3. Text filing with HTML tags and tables for NVDA
        nvda_content = """
        <DOCUMENT>
        <TYPE>10-Q
        <COMPANY>NVIDIA CORP
        NVIDIA Corporation announced strong growth in Data Center computing and AI infrastructure.
        Revenue for the quarter was $35,100 million, up 94% year-over-year.
        Gross margin expanded to 75.1% compared to 74.0% in the prior year period.
        Operating income totaled $21,800 million.

        | Metric | Q3 2024 | Q3 2025 | YoY Change |
        | Total Revenue | $18,120M | $35,100M | +93.7% |
        | Gross Profit | $13,400M | $26,360M | +96.7% |
        | Operating Income | $10,417M | $21,800M | +109.3% |
        | Net Income | $9,243M | $19,309M | +108.9% |

        Figure 1: Revenue by Market Segment
        Data Center: 30800.0, Gaming: 3200.0, Professional Visualization: 480.0, Automotive: 440.0
        </DOCUMENT>
        """
        (corpus / "NVDA_10Q_2025_Q3.txt").write_text(nvda_content, encoding="utf-8")

        # 4. JSON file holding raw text/content field
        raw_json_doc = {
            "title": "Amazon Q3 Highlights",
            "content": "Amazon reported operating cash flow of $112.7 billion. Net sales were $158.9 billion.",
        }
        (corpus / "AMZN_10Q_2025.json").write_text(json.dumps(raw_json_doc), encoding="utf-8")

        # 5. Simple 1-page PDF filing
        writer = pypdf.PdfWriter()
        writer.add_blank_page(width=72, height=72)
        pdf_path = corpus / "GOOGL_10K_2025.pdf"
        with open(pdf_path, "wb") as f:
            writer.write(f)

        # 6. Corrupt file that should be skipped safely
        (corpus / "corrupt_file.txt").write_text("just random binary or invalid", encoding="utf-8")

        return corpus

    def test_batch_loader_and_cleaning(self, corpus_dir: Path):
        """Test batch loading, text cleaning, table validation, and format parsing."""
        loader = BatchDocumentLoader(clean_text=True, validate_tables=True)
        docs = loader.load_directory(corpus_dir)
        assert len(docs) >= 4

        tickers = {d.metadata.ticker for d in docs if d.metadata.ticker}
        assert "AAPL" in tickers
        assert "MSFT" in tickers
        assert "NVDA" in tickers

        # Verify streaming iter_documents
        streamed = list(loader.iter_documents(corpus_dir))
        assert len(streamed) == len(docs)

    def test_reconciliation_and_ratios(self, corpus_dir: Path):
        """Verify accounting reconciliations and ratio calculations across ingested documents."""
        loader = BatchDocumentLoader()
        aapl_doc = loader.load_file(corpus_dir / "AAPL_10K_2025.json")

        tables = aapl_doc.get_tables()
        assert len(tables) >= 2

        # Reconcile statements
        income_tbl = next(
            (t for t in tables if t.title and "operations" in t.title.lower()), tables[0]
        )
        inc_reconcile = StatementReconciler.reconcile_income_statement(income_tbl)
        assert inc_reconcile.is_balanced

        balance_tbl = next(
            (t for t in tables if t.title and "balance" in t.title.lower()), tables[1]
        )
        bal_reconcile = StatementReconciler.reconcile_balance_sheet(balance_tbl)
        assert bal_reconcile.is_balanced

        # Calculate ratios
        ratios = FinancialRatioCalculator.compute_from_document(aapl_doc)
        assert ratios.gross_margin_pct is not None
        assert ratios.current_ratio is not None

    def test_orchestrator_multi_query_and_grounding(self, corpus_dir: Path):
        """Test orchestrator querying, routing, verification, and audit generation."""
        cfg = OrchestratorConfig(
            default_top_k=3,
            clean_text=True,
            validate_tables=True,
        )
        orch = FinancialPipelineOrchestrator(config=cfg)
        orch.ingest_files(corpus_dir)

        # 1. Quantitative Query for NVIDIA
        q1 = AgentQuery(query_str="What was total revenue and gross margin for NVDA in Q3 2025?")
        resp1 = orch.run_query(q1)
        assert resp1.execution_time_ms > 0
        assert len(resp1.retrieved_chunks) > 0
        assert any(c.chunk.doc_id.upper().startswith("NVDA") for c in resp1.retrieved_chunks)
        assert len(resp1.citations) > 0

        # 2. Comparative Analysis Query (invoking automatic routing)
        resp2 = orch.run_query("Compare operating income and revenue between 2024 and 2025")
        assert resp2.metadata.get("intent") == QueryIntent.COMPARATIVE_ANALYSIS

        # 3. Qualitative Risk Query (invoking automatic routing)
        resp3 = orch.run_query("What are the key macroeconomic and supply chain headwinds?")
        assert resp3.metadata.get("intent") == QueryIntent.QUALITATIVE_RISK

        # Run Grounding Audit across batch
        audit_report = orch.auditor.audit_batch([resp1, resp2, resp3])
        assert audit_report.total_queries == 3
        assert hasattr(audit_report, "hallucination_rate")
        assert audit_report.compliance_status in {"PASS", "WARN", "FAIL"}

    def test_period_comparator_across_filings(self, corpus_dir: Path):
        """Test period comparator using multi-column table and multi-period metrics."""
        loader = BatchDocumentLoader()
        nvda_doc = loader.load_file(corpus_dir / "NVDA_10Q_2025_Q3.txt", default_ticker="NVDA")

        tables = nvda_doc.get_tables()
        assert len(tables) > 0
        tbl = tables[0]

        # Compare column 1 (Q3 2024) vs column 2 (Q3 2025)
        variances = PeriodComparator.compare_table_columns(tbl, base_col=1, compare_col=2)
        assert len(variances) >= 3

        rev_var = next((v for v in variances if "revenue" in v.metric_name.lower()), None)
        assert rev_var is not None
        assert rev_var.trend == "UP"
        assert rev_var.percentage_change is not None and rev_var.percentage_change > 50.0

    def test_checkpoint_persistence_and_recovery(self, corpus_dir: Path, tmp_path: Path):
        """Test exporting index checkpoint to zip archive, reloading, and querying."""
        orch = FinancialPipelineOrchestrator()
        orch.ingest_files(corpus_dir)

        archive_zip = tmp_path / "index_checkpoint.zip"
        orch.save_index(archive_zip, compress=True)
        assert archive_zip.exists()
        assert archive_zip.stat().st_size > 0

        # Reload into a completely new orchestrator
        orch_restored = FinancialPipelineOrchestrator()
        orch_restored.load_index(archive_zip)

        assert orch_restored.index.bm25.corpus_size == orch.index.bm25.corpus_size
        assert len(orch_restored.index.chunks) == len(orch.index.chunks)

        # Query restored index
        resp = orch_restored.run_query("NVIDIA Data Center performance")
        assert len(resp.retrieved_chunks) > 0

    def test_caches_and_invalidation(self):
        """Test embedding and query caches with eviction and TTL expiration."""
        # Embedding Cache
        emb_cache = EmbeddingCache(max_size=2)
        v1 = np.array([1.0, 0.0], dtype=np.float32)
        v2 = np.array([0.0, 1.0], dtype=np.float32)
        v3 = np.array([0.5, 0.5], dtype=np.float32)

        emb_cache.put("text1", v1)
        emb_cache.put("text2", v2)
        # Re-insert text1 to exercise move_to_end
        emb_cache.put("text1", v1)
        # Exceed max_size
        emb_cache.put("text3", v3)
        assert len(emb_cache._cache) == 2
        assert emb_cache.get("text2") is None  # evicted
        assert emb_cache.get("text1") is not None

        emb_cache.clear()
        assert len(emb_cache._cache) == 0
        assert emb_cache.hits == 0

        # Query Cache with immediate expiration
        q_cache = QueryCache(max_size=2, ttl_seconds=0.01)
        resp = AgentResponse(
            query="test", answer="answer", citations=[], grounding_verdicts=[], retrieved_chunks=[]
        )
        key = q_cache.make_key("test")
        q_cache.put(key, resp)
        # Overwrite same key to exercise move_to_end
        q_cache.put(key, resp)

        time.sleep(0.02)  # expire TTL
        assert q_cache.get(key) is None
        assert q_cache.misses >= 1

        # Test capacity eviction
        q_cache_lru = QueryCache(max_size=1, ttl_seconds=100.0)
        q_cache_lru.put("k1", resp)
        q_cache_lru.put("k2", resp)
        assert len(q_cache_lru._cache) == 1
        q_cache_lru.clear()
        assert len(q_cache_lru._cache) == 0

    def test_persistence_corrupt_and_missing_errors(self, tmp_path: Path):
        """Test IndexPersistence raises proper exceptions for invalid inputs."""
        # Non-existent path
        with pytest.raises(FileNotFoundError):
            IndexPersistence.load(tmp_path / "ghost_archive.zip")

        # Corrupt archive missing manifest.json
        corrupt_dir = tmp_path / "corrupt_idx"
        corrupt_dir.mkdir()
        (corrupt_dir / "dummy.txt").write_text("no manifest here", encoding="utf-8")
        with pytest.raises(ValueError, match="missing manifest.json"):
            IndexPersistence.load(corrupt_dir)
