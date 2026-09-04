"""Tests for FinancialPipelineOrchestrator end-to-end integration."""

from pathlib import Path
import pytest

from multi_modal_financial.data.synthetic import SyntheticFilingGenerator
from multi_modal_financial.pipeline.orchestrator import (
    FinancialPipelineOrchestrator,
    OrchestratorConfig,
)


class TestFinancialPipelineOrchestrator:
    """Unit and integration tests for FinancialPipelineOrchestrator."""

    @pytest.fixture
    def setup_sample_corpus(self, tmp_path: Path) -> Path:
        """Create sample filings on disk."""
        gen = SyntheticFilingGenerator(seed=42)
        corpus_dir = tmp_path / "corpus"
        corpus_dir.mkdir()

        # Write two quarterly filings for Apple
        doc_q2 = gen.generate_filing_document(ticker="AAPL", year=2025, period="Q2")
        doc_q3 = gen.generate_filing_document(ticker="AAPL", year=2025, period="Q3")

        p2 = corpus_dir / "AAPL_10Q_2025_Q2.txt"
        p3 = corpus_dir / "AAPL_10Q_2025_Q3.txt"

        # Export chunks content as text
        p2.write_text("\n\n".join(c.content for c in doc_q2.chunks), encoding="utf-8")
        p3.write_text("\n\n".join(c.content for c in doc_q3.chunks), encoding="utf-8")

        return corpus_dir

    def test_orchestrator_ingest_and_query(self, setup_sample_corpus: Path):
        cfg = OrchestratorConfig(enable_query_cache=True, default_top_k=3)
        orch = FinancialPipelineOrchestrator(config=cfg)

        docs = orch.ingest_files(setup_sample_corpus)
        assert len(docs) == 2
        assert orch.index.total_documents() == 2
        assert orch.index.total_chunks() > 0

        # Query
        resp1 = orch.run_query("What was total revenue in Q3 2025?", use_cache=True)
        assert resp1.answer is not None
        assert len(resp1.retrieved_chunks) > 0
        assert resp1.overall_confidence > 0.0

        # Second query should hit cache
        resp2 = orch.run_query("What was total revenue in Q3 2025?", use_cache=True)
        assert resp2.answer == resp1.answer
        if orch.query_cache:
            assert orch.query_cache.hits == 1

    def test_orchestrator_batch_queries_and_audit(self, setup_sample_corpus: Path):
        orch = FinancialPipelineOrchestrator()
        orch.ingest_files(setup_sample_corpus)

        queries = [
            "Total Revenue for Q3",
            "Gross Profit for Apple",
        ]
        responses = orch.run_batch_queries(queries)
        assert len(responses) == 2

        audit_report = orch.audit_queries(queries)
        assert audit_report.total_queries == 2
        assert audit_report.compliance_status in {"PASS", "WARN"}

    def test_orchestrator_checkpoint_persistence(self, setup_sample_corpus: Path, tmp_path: Path):
        orch = FinancialPipelineOrchestrator()
        orch.ingest_files(setup_sample_corpus)

        save_loc = tmp_path / "checkpoint_dir"
        persisted_path = orch.save_index(save_loc)
        assert persisted_path.exists()

        # New orchestrator loads checkpoint
        orch2 = FinancialPipelineOrchestrator()
        orch2.load_index(save_loc)
        assert orch2.index.total_documents() == 2
        assert orch2.index.total_chunks() == orch.index.total_chunks()

        # Query works on restored orchestrator
        resp = orch2.run_query("Apple revenue", top_k=2)
        assert len(resp.retrieved_chunks) > 0

    def test_orchestrator_status_diagnostics(self, setup_sample_corpus: Path):
        orch = FinancialPipelineOrchestrator()
        orch.ingest_files(setup_sample_corpus)
        status = orch.status()

        assert status["total_documents"] == 2
        assert status["total_chunks"] > 0
        assert status["bm25_vocab_size"] > 0
        assert status["vector_dimension"] > 0
        assert status["query_cache"] is not None
