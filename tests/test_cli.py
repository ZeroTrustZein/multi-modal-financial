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

    def test_main_module_execution(self, monkeypatch):
        """Test python -m multi_modal_financial entrypoint."""
        monkeypatch.setattr(sys, "argv", ["multi_modal_financial", "--help"])
        with pytest.raises(SystemExit) as exc_info:
            runpy.run_module("multi_modal_financial.__main__", run_name="__main__")
        assert exc_info.value.code == 0
