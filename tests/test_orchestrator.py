"""Tests for FinancialPipelineOrchestrator end-to-end integration."""

from pathlib import Path

import pytest

from multi_modal_financial.data.synthetic import SyntheticFilingGenerator
from multi_modal_financial.pipeline.orchestrator import (
    FinancialPipelineOrchestrator,
    OrchestratorConfig,
)
from multi_modal_financial.types import (
    AgentQuery,
    HybridSearchConfig,
    RerankerConfig,
    RerankerStrategy,
    RetrievalBenchmarkResult,
    RetrievalStrategy,
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

    def test_orchestrator_analytics_and_comparison(self, setup_sample_corpus: Path):
        orch = FinancialPipelineOrchestrator()
        docs = orch.ingest_files(setup_sample_corpus)
        doc_ids = [d.doc_id for d in docs]

        # Analyze ratios for first document
        ratios = orch.analyze_document_ratios(doc_ids[0])
        assert ratios is not None

        # Compare two documents
        variances = orch.compare_documents(doc_ids[0], doc_ids[1])
        assert isinstance(variances, list)

        # Non-existent doc error
        with pytest.raises(KeyError):
            orch.analyze_document_ratios("non_existent_id")

        with pytest.raises(KeyError):
            orch.compare_documents(doc_ids[0], "missing_id")

    def test_orchestrator_hybrid_search_config(self, setup_sample_corpus: Path):
        cfg = OrchestratorConfig(
            hybrid_search_config=HybridSearchConfig(
                strategy=RetrievalStrategy.HYBRID_RRF,
                rrf_k=40,
                rerank_top_k=3,
            ),
            reranker_config=RerankerConfig(
                strategy=RerankerStrategy.HYBRID,
                score_threshold=0.0,
            ),
        )
        orch = FinancialPipelineOrchestrator(config=cfg)
        orch.ingest_files(setup_sample_corpus)

        assert orch.retriever.config.rrf_k == 40
        assert orch.reranker.config.strategy == RerankerStrategy.HYBRID

        resp = orch.run_query("Apple revenue Q3")
        assert len(resp.retrieved_chunks) > 0
        st = orch.status()
        assert st["retriever"]["rrf_k"] == 40
        assert st["reranker"]["strategy"] == "hybrid"

    def test_orchestrator_checkpoint_dir_auto(self, setup_sample_corpus: Path, tmp_path: Path):
        persist_dir = tmp_path / "auto_ckpt"
        cfg = OrchestratorConfig(persistence_dir=persist_dir)
        orch = FinancialPipelineOrchestrator(config=cfg)
        orch.ingest_files(setup_sample_corpus)

        saved = orch.checkpoint()
        assert saved.exists()
        assert (persist_dir / "manifest.json").exists()

        # Without persistence_dir configured
        orch_no_dir = FinancialPipelineOrchestrator()
        with pytest.raises(ValueError, match="persistence_dir is not configured"):
            orch_no_dir.checkpoint()

    def test_orchestrator_save_and_load_semantic_cache(
        self, setup_sample_corpus: Path, tmp_path: Path
    ):
        orch = FinancialPipelineOrchestrator()
        orch.ingest_files(setup_sample_corpus)

        # Run query to populate semantic cache
        q1 = AgentQuery(query_str="What was Apple revenue in Q3 2025?", top_k=2)
        resp1 = orch.run_query(q1)
        assert resp1 is not None

        cache_path = tmp_path / "cache_dump.json"
        saved = orch.save_semantic_cache(cache_path)
        assert saved is not None and saved.exists()

        orch2 = FinancialPipelineOrchestrator()
        orch2.load_semantic_cache(cache_path)
        assert orch2.semantic_cache is not None
        assert len(orch2.semantic_cache._entries) >= 1

    def test_orchestrator_benchmark_retrieval(self, setup_sample_corpus: Path):
        orch = FinancialPipelineOrchestrator()
        docs = orch.ingest_files(setup_sample_corpus)
        target_chunk_id = docs[0].chunks[0].chunk_id

        labeled_queries = [
            ("Apple revenue Q3", [target_chunk_id]),
        ]
        result = orch.benchmark_retrieval(labeled_queries, k=5)
        assert isinstance(result, RetrievalBenchmarkResult)
        assert result.query_count == 1
        assert result.avg_latency_ms >= 0.0
