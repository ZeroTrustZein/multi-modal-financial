"""Comprehensive unit and functional test suite for the CLI interface."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

from multi_modal_financial import __version__
from multi_modal_financial.cli.main import cli


class TestCLI:
    """Test suite for Click CLI commands."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        return CliRunner()

    def test_cli_help(self, runner: CliRunner):
        """Test root --help command."""
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "Multi-Modal Financial Document RAG CLI." in result.output
        assert "info" in result.output
        assert "ingest" in result.output
        assert "query" in result.output
        assert "benchmark" in result.output

    def test_cli_version(self, runner: CliRunner):
        """Test root --version command."""
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert __version__ in result.output

    def test_info_command(self, runner: CliRunner):
        """Test info command displays architecture components."""
        result = runner.invoke(cli, ["info"])
        assert result.exit_code == 0
        assert "Multi-Modal Financial RAG - Architecture" in result.output
        assert "BM25 Okapi" in result.output
        assert "DenseVectorIndex" in result.output
        assert "FinancialReranker" in result.output
        assert "GroundingVerifier" in result.output

    def test_ingest_single_file(self, runner: CliRunner, tmp_path: Path):
        """Test ingest command with a single text document."""
        doc_file = tmp_path / "AAPL_10Q.txt"
        doc_file.write_text(
            "Apple reported net income of $21.4 billion for the fiscal quarter ended June 2025.",
            encoding="utf-8",
        )

        result = runner.invoke(cli, ["ingest", str(doc_file)])
        assert result.exit_code == 0
        assert "Ingested 'AAPL_10Q'" in result.output

    def test_ingest_single_file_with_ticker(self, runner: CliRunner, tmp_path: Path):
        """Test ingest command specifying ticker option."""
        doc_file = tmp_path / "summary.txt"
        doc_file.write_text(
            "Quarterly revenue grew by 18% driven by enterprise cloud adoption.",
            encoding="utf-8",
        )

        result = runner.invoke(cli, ["ingest", str(doc_file), "--ticker", "MSFT"])
        assert result.exit_code == 0
        assert "Ingested 'summary'" in result.output

    def test_ingest_directory(self, runner: CliRunner, tmp_path: Path):
        """Test ingest command targeting a folder of documents."""
        (tmp_path / "file1.txt").write_text("Revenue was $100M.", encoding="utf-8")
        (tmp_path / "file2.md").write_text(
            "| Revenue | $200M |\n| Net Income | $50M |", encoding="utf-8"
        )
        (tmp_path / "ignore.csv").write_text("col1,col2\n1,2", encoding="utf-8")

        result = runner.invoke(cli, ["ingest", str(tmp_path), "--ticker", "NVDA"])
        assert result.exit_code == 0
        assert "Successfully ingested 2 documents" in result.output

    def test_ingest_nonexistent_file(self, runner: CliRunner, tmp_path: Path):
        """Test ingest command error handling when path does not exist."""
        non_existent = tmp_path / "ghost_filing.txt"
        result = runner.invoke(cli, ["ingest", str(non_existent)])
        assert result.exit_code != 0
        assert "does not exist" in result.output.lower() or "error" in result.output.lower()

    def test_query_default(self, runner: CliRunner):
        """Test standard financial query execution."""
        result = runner.invoke(cli, ["query", "What was revenue for ACM in Q3?"])
        assert result.exit_code == 0
        assert "Synthesized Response" in result.output
        assert "Citations & Grounding" in result.output

    def test_query_with_options(self, runner: CliRunner):
        """Test query command passing custom ticker, top-k, and alpha."""
        result = runner.invoke(
            cli,
            [
                "query",
                "Operating income comparison",
                "--ticker",
                "ACM",
                "--top-k",
                "2",
                "--alpha",
                "0.7",
            ],
        )
        assert result.exit_code == 0
        assert "Synthesized Response" in result.output
        assert "Citations & Grounding" in result.output

    def test_benchmark_command(self, runner: CliRunner):
        """Test benchmark command executes synthetic retrieval evaluation."""
        result = runner.invoke(cli, ["benchmark"])
        assert result.exit_code == 0
        assert "Running Retrieval Benchmark..." in result.output
        assert "Benchmark Results (Recall@1 & MRR)" in result.output
        assert "BM25 (Sparse)" in result.output
        assert "Hybrid RRF + Reranker" in result.output

    def test_info_command_with_status(self, runner: CliRunner):
        """Test info command with --status flag showing cache diagnostics."""
        result = runner.invoke(cli, ["info", "--status"])
        assert result.exit_code == 0
        assert "Multi-Modal Financial RAG - Architecture" in result.output
        assert "Semantic Cache Telemetry" in result.output

    def test_query_with_reranker_and_retrieval_strategies(self, runner: CliRunner):
        """Test query command across various reranking and retrieval strategies."""
        for strat in ["hybrid_rrf", "hybrid_convex", "dense", "sparse"]:
            res = runner.invoke(
                cli,
                [
                    "query",
                    "ACME revenue growth",
                    "--strategy",
                    strat,
                    "--reranker-strategy",
                    "cross_encoder",
                ],
            )
            assert res.exit_code == 0
            assert "Synthesized Response" in res.output

        for r_strat in ["heuristic", "cross_encoder", "hybrid"]:
            res = runner.invoke(
                cli,
                [
                    "query",
                    "ACME revenue growth",
                    "--reranker-strategy",
                    r_strat,
                ],
            )
            assert res.exit_code == 0
            assert "Synthesized Response" in res.output

    def test_query_with_explain(self, runner: CliRunner):
        """Test query command with --explain flag."""
        result = runner.invoke(cli, ["query", "ACME Q3 operating income", "--explain"])
        assert result.exit_code == 0
        assert "Synthesized Response" in result.output
        assert "Reranker Scoring & Explanations" in result.output

    def test_query_threshold_and_top_k(self, runner: CliRunner):
        """Test query with reranker thresholds and custom top-k."""
        result = runner.invoke(
            cli,
            [
                "query",
                "ACME gross margin",
                "--reranker-top-k",
                "2",
                "--reranker-threshold",
                "0.05",
                "--no-rerank",
                "--no-semantic-cache",
            ],
        )
        assert result.exit_code == 0
        assert "Synthesized Response" in result.output

    def test_cache_stats_command(self, runner: CliRunner):
        """Test cache-stats CLI command."""
        result = runner.invoke(cli, ["cache-stats"])
        assert result.exit_code == 0
        assert "Semantic Cache Telemetry" in result.output
        assert "Total Lookups" in result.output
        assert "Exact Hits" in result.output

    def test_ratios_command(self, runner: CliRunner, tmp_path: Path):
        """Test ratios CLI command on a document."""
        doc_file = tmp_path / "financials.txt"
        doc_file.write_text(
            """
            Statement of Operations for Q3 2025:
            | Metric | Amount |
            | Total Revenue | $5,000M |
            | Gross Profit | $2,500M |
            | Operating Income | $1,250M |
            | Net Income | $1,000M |
            """,
            encoding="utf-8",
        )
        result = runner.invoke(cli, ["ratios", str(doc_file)])
        assert result.exit_code == 0
        assert "Financial Ratios Summary" in result.output
        assert "Gross Margin" in result.output

    def test_compare_command(self, runner: CliRunner, tmp_path: Path):
        """Test compare CLI command between two filing periods."""
        doc1 = tmp_path / "q1_filing.txt"
        doc1.write_text(
            """
            | Metric | 2024 |
            | Revenue | $1,000M |
            | Net Income | $200M |
            """,
            encoding="utf-8",
        )
        doc2 = tmp_path / "q2_filing.txt"
        doc2.write_text(
            """
            | Metric | 2025 |
            | Revenue | $1,200M |
            | Net Income | $250M |
            """,
            encoding="utf-8",
        )
        result = runner.invoke(cli, ["compare", str(doc1), str(doc2)])
        assert result.exit_code == 0
        assert "Financial Period Comparison" in result.output

    def test_audit_command(self, runner: CliRunner):
        """Test audit CLI command."""
        result = runner.invoke(cli, ["audit", "What was revenue for ACM in Q3?"])
        assert result.exit_code == 0
        assert "Grounding Compliance Audit Report" in result.output
        assert "Total Queries Audited" in result.output

    def test_formatters_unit(self):
        """Direct tests for formatters module components."""
        from multi_modal_financial.analytics.comparator import VarianceResult
        from multi_modal_financial.analytics.ratios import RatioSummary
        from multi_modal_financial.cli.formatters import (
            format_audit_table,
            format_cache_stats_table,
            format_comparison_table,
            format_ratios_table,
            format_rerank_explanations_table,
        )
        from multi_modal_financial.grounding.audit import GroundingAuditReport
        from multi_modal_financial.types import (
            CacheHitType,
            RerankExplanation,
            SemanticCacheStats,
        )

        # 1. Rerank explanations table
        exp = RerankExplanation(
            chunk_id="c1",
            initial_rank=1,
            final_rank=1,
            initial_score=0.8,
            final_score=0.95,
            reasons=["Lexical overlap: 0.50"],
        )
        tbl_exp = format_rerank_explanations_table([exp])
        assert tbl_exp.title == "Reranker Scoring & Explanations"

        tbl_empty_exp = format_rerank_explanations_table([])
        assert tbl_empty_exp.title == "Reranker Scoring & Explanations"

        # 2. Cache stats table
        stats = SemanticCacheStats(
            total_entries=5,
            max_entries=100,
            exact_hits=2,
            semantic_hits=3,
            misses=5,
            total_lookups=10,
            hit_rate_pct=50.0,
            evictions=0,
            expired_count=0,
        )
        tbl_cache = format_cache_stats_table(stats)
        assert tbl_cache.title == "Semantic Cache Telemetry"

        # 3. Ratios table
        ratios = RatioSummary(
            gross_margin_pct=50.0,
            operating_margin_pct=25.0,
            net_margin_pct=20.0,
            current_ratio=2.0,
        )
        tbl_ratios = format_ratios_table(ratios, doc_id="TEST_DOC")
        assert "TEST_DOC" in tbl_ratios.title

        # 4. Comparison table
        variance = VarianceResult(
            metric_name="Revenue",
            base_value=100.0,
            compare_value=120.0,
            absolute_change=20.0,
            percentage_change=20.0,
            trend="UP",
        )
        tbl_comp = format_comparison_table([variance], base_id="Q1", comp_id="Q2")
        assert "Q1 vs Q2" in tbl_comp.title

        # 5. Audit table
        rep = GroundingAuditReport(
            total_queries=2,
            total_claims=4,
            supported_claims=4,
            partially_supported_claims=0,
            unsupported_claims=0,
            hallucination_rate=0.0,
            mean_confidence=0.95,
            compliance_status="PASS",
        )
        tbl_audit = format_audit_table(rep)
        assert tbl_audit.title == "Grounding Compliance Audit Report"

    def test_main_module_execution(self, monkeypatch):
        """Test python -m multi_modal_financial entrypoint."""
        monkeypatch.setattr(sys, "argv", ["multi_modal_financial", "--help"])
        with pytest.raises(SystemExit) as exc_info:
            runpy.run_module("multi_modal_financial.__main__", run_name="__main__")
        assert exc_info.value.code == 0
