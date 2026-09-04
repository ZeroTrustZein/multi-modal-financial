"""Tests for synthetic financial filings and batch document loader."""

from pathlib import Path

from multi_modal_financial.data.loader import BatchDocumentLoader
from multi_modal_financial.data.synthetic import SyntheticFilingGenerator
from multi_modal_financial.data.validator import StatementReconciler
from multi_modal_financial.types import DocumentType, FinancialStatementType


class TestSyntheticFilingGenerator:
    """Unit tests for SyntheticFilingGenerator."""

    def test_generate_income_statement(self):
        gen = SyntheticFilingGenerator(seed=123)
        tbl = gen.generate_income_statement(ticker="AAPL", year=2025, period="Q3")
        assert tbl.statement_type == FinancialStatementType.INCOME_STATEMENT
        assert len(tbl.rows) >= 8
        assert "Total Revenue" in tbl.rows[0][0]

        # Reconcile generated table
        reconciliation = StatementReconciler.reconcile_income_statement(tbl, col_idx=1)
        assert reconciliation.is_balanced

    def test_generate_balance_sheet(self):
        gen = SyntheticFilingGenerator(seed=123)
        tbl = gen.generate_balance_sheet(ticker="MSFT", year=2025, period="Q3")
        assert tbl.statement_type == FinancialStatementType.BALANCE_SHEET
        assert len(tbl.rows) >= 10

        # Reconcile generated table
        reconciliation = StatementReconciler.reconcile_balance_sheet(tbl, col_idx=1)
        assert reconciliation.is_balanced

    def test_generate_filing_document(self):
        gen = SyntheticFilingGenerator(seed=999)
        doc = gen.generate_filing_document(ticker="NVDA", year=2025, period="Q3", doc_type=DocumentType.TEN_Q)
        assert doc.metadata.ticker == "NVDA"
        assert doc.metadata.period == "Q3"
        assert doc.metadata.year == 2025
        assert len(doc.chunks) > 0

        # Check presence of multi-modal chunks
        has_table = any(c.table_data is not None for c in doc.chunks)
        has_figure = any(c.figure_data is not None for c in doc.chunks)
        assert has_table
        assert has_figure


class TestBatchDocumentLoader:
    """Unit tests for BatchDocumentLoader."""

    def test_extract_metadata_from_path(self):
        loader = BatchDocumentLoader()
        meta = loader.extract_metadata_from_path(Path("AAPL_10Q_2025_Q3.txt"))
        assert meta.ticker == "AAPL"
        assert meta.doc_type == "10Q"
        assert meta.year == 2025
        assert meta.period == "Q3"

    def test_load_text_file(self, tmp_path: Path):
        file_path = tmp_path / "MSFT_10K_2024.txt"
        file_path.write_text(
            """
            # Microsoft Corporation Annual Report
            Total revenue reached $245,000 million.
            Cloud revenue grew 25% year-over-year.

            | Segment | 2023 | 2024 |
            |---|---|---|
            | Productivity | $69,000 | $77,000 |
            | Cloud | $87,000 | $105,000 |
            """,
            encoding="utf-8",
        )
        loader = BatchDocumentLoader(clean_text=True, validate_tables=True)
        doc = loader.load_file(file_path)
        assert doc.metadata.ticker == "MSFT"
        assert doc.metadata.year == 2024
        assert len(doc.chunks) >= 2

    def test_load_directory_and_iter(self, tmp_path: Path):
        # Create multiple dummy files
        for ticker in ["AAPL", "GOOGL", "AMZN"]:
            p = tmp_path / f"{ticker}_10Q_2025_Q1.txt"
            p.write_text(f"Filing summary for {ticker} in Q1 2025.", encoding="utf-8")

        loader = BatchDocumentLoader()
        docs = loader.load_directory(tmp_path)
        assert len(docs) == 3
        tickers = {d.metadata.ticker for d in docs}
        assert tickers == {"AAPL", "GOOGL", "AMZN"}

        # Test streaming iter_documents
        streamed = list(loader.iter_documents(tmp_path))
        assert len(streamed) == 3
