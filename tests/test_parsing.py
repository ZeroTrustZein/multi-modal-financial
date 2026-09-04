"""Comprehensive unit tests for parsing core logic: tables, figures, and document extraction."""

from __future__ import annotations

from unittest.mock import MagicMock

from multi_modal_financial.parsing.extractor import FinancialDocumentParser
from multi_modal_financial.parsing.figure_parser import FigureParser
from multi_modal_financial.parsing.table_parser import TableParser
from multi_modal_financial.types import FinancialStatementType


class TestTableParser:
    """Unit tests for TableParser."""

    def test_scale_and_unit_detection(self) -> None:
        scale_mil, unit_usd = TableParser.detect_scale_and_unit(
            "Consolidated figures (in millions, except per share amounts) in USD"
        )
        assert scale_mil == "millions"
        assert unit_usd == "USD"

        scale_bil, unit_eur = TableParser.detect_scale_and_unit(
            "European operations (in billions) €"
        )
        assert scale_bil == "billions"
        assert unit_eur == "EUR"

        scale_thous, unit_gbp = TableParser.detect_scale_and_unit("UK branch £ in thousands")
        assert scale_thous == "thousands"
        assert unit_gbp == "GBP"

        scale_pct, unit_jpy = TableParser.detect_scale_and_unit("Percentage margins in ¥ (percent)")
        assert scale_pct == "percent"
        assert unit_jpy == "JPY"

    def test_detect_statement_type(self) -> None:
        assert (
            TableParser.detect_statement_type(
                ["Item", "2024"], [["Operating Income", "$100M"]], title="Consolidated Operations"
            )
            == FinancialStatementType.INCOME_STATEMENT
        )

        assert (
            TableParser.detect_statement_type(
                ["Asset Category", "Amount"],
                [["Total Assets", "$500B"]],
                title="Consolidated Balance Sheet",
            )
            == FinancialStatementType.BALANCE_SHEET
        )

        assert (
            TableParser.detect_statement_type(
                ["Activity", "Amount"],
                [["Operating Activities", "$50B"]],
                title="Cash Flow Statement",
            )
            == FinancialStatementType.CASH_FLOW
        )

        assert (
            TableParser.detect_statement_type(
                ["Item", "2025"],
                [["Foreign Currency Translation", "$10M"]],
                title="Comprehensive Income",
            )
            == FinancialStatementType.COMPREHENSIVE_INCOME
        )

        assert (
            TableParser.detect_statement_type(
                ["Category", "Shares"],
                [["Common Stock", "1000"]],
                title="Statement of Stockholders' Equity",
            )
            == FinancialStatementType.STOCKHOLDERS_EQUITY
        )

        assert (
            TableParser.detect_statement_type(
                ["Region", "Revenue"],
                [["Americas", "$200M"], ["Europe", "$150M"]],
                title="Geographic Breakdown",
            )
            == FinancialStatementType.SEGMENT_METRICS
        )

        assert (
            TableParser.detect_statement_type(
                ["Policy", "Description"],
                [["Revenue Recognition", "ASC 606"]],
                title="Note 1 Summary",
            )
            == FinancialStatementType.NOTES
        )

        assert (
            TableParser.detect_statement_type(
                ["Random", "Data"], [["Foo", "Bar"]], title="Miscellaneous"
            )
            == FinancialStatementType.UNKNOWN
        )

    def test_parse_markdown_table_with_footnotes(self) -> None:
        raw_table = """
        **Table: Balance Sheet Summary (in millions) in USD**
        | Item | 2024 | 2025 |
        | --- | --- | --- |
        | Cash and Cash Equivalents | $29,965 | $30,215 |
        | Total Assets | $352,583 | $364,980 |
        * Excludes restricted cash.
        (1) Subject to annual audit reconciliation.
        """
        table = TableParser.parse_markdown_table(raw_table)
        assert table.headers == ["Item", "2024", "2025"]
        assert table.row_count == 2
        assert table.scale == "millions"
        assert table.unit == "USD"
        assert table.statement_type == FinancialStatementType.BALANCE_SHEET
        assert len(table.footnotes) == 2
        assert "Excludes restricted cash." in table.footnotes[0]

    def test_parse_delimited_csv(self) -> None:
        csv_data = """Line Item,FY2024,FY2025
Operating Revenue,$100M,$120M
Cost of Goods Sold,$60M,$70M
Gross Profit,$40M,$50M
"""
        table = TableParser.parse_delimited_table(csv_data, delimiter=",", title="Margin Breakdown")
        assert table.headers == ["Line Item", "FY2024", "FY2025"]
        assert table.row_count == 3
        assert table.statement_type == FinancialStatementType.INCOME_STATEMENT
        assert table.rows[0][1] == "$100M"

    def test_extract_tables_from_text(self) -> None:
        doc_text = """
        Here is the first disclosure:

        | Metric | Q1 | Q2 |
        | --- | --- | --- |
        | EBITDA | $50M | $55M |

        Some intervening text.

        | Metric | Q3 | Q4 |
        | --- | --- | --- |
        | Free Cash Flow | $30M | $35M |
        """
        tables = TableParser.extract_tables_from_text(doc_text)
        assert len(tables) == 2
        assert tables[0].rows[0][0] == "EBITDA"
        assert tables[1].rows[0][0] == "Free Cash Flow"


class TestFigureParser:
    """Unit tests for FigureParser."""

    def test_chart_type_detection(self) -> None:
        assert FigureParser.detect_chart_type("Figure 1: Waterfall of EBITDA growth") == "waterfall"
        assert (
            FigureParser.detect_chart_type("Chart 2: Pie chart of revenue by business unit")
            == "pie"
        )
        assert FigureParser.detect_chart_type("Bar chart showing quarterly headcount") == "bar"
        assert FigureParser.detect_chart_type("Line trajectory of subscription revenue") == "line"
        assert FigureParser.detect_chart_type("Scatter plot of valuation multiples") == "scatter"
        assert FigureParser.detect_chart_type("Area chart of gross profit margins") == "area"
        assert FigureParser.detect_chart_type("Diagram of manufacturing plant") == "chart"

    def test_axis_label_extraction(self) -> None:
        text = "Figure 3: Revenue trajectory.\nX-Axis: Fiscal Quarter\nY-Axis: Millions USD"
        x, y = FigureParser.extract_axis_labels(text)
        assert x == "Fiscal Quarter"
        assert y == "Millions USD"

    def test_figure_parsing_with_bullets(self) -> None:
        raw_fig = """
        Figure 4: Global Sales Distribution (in billions)
        X-Axis: Region
        Y-Axis: USD Billions
        - Americas: $45.5B
        - Europe: $25.2B
        - Greater China: $15.0B
        - Rest of Asia: $8.5B
        """
        fig = FigureParser.parse_figure_block(raw_fig, figure_id="fig_geo")
        assert fig.caption == "Global Sales Distribution (in billions)"
        assert fig.chart_type == "bar" or fig.chart_type == "chart"
        assert fig.unit == "USD"
        assert fig.scale == "billions"
        assert fig.x_label == "Region"
        assert fig.y_label == "USD Billions"
        assert fig.data_points["Americas"] == 45.5
        assert fig.data_points["Europe"] == 25.2
        assert fig.total_value() == 94.2

    def test_extract_figures_from_text(self) -> None:
        text = """
        Overview of corporate performance.

        Figure 1: Revenue by segment
        Cloud: 500
        Enterprise: 300

        Further narrative commentary.

        [Figure 2: Operating margin trend]
        """
        figures = FigureParser.extract_figures_from_text(text)
        assert len(figures) >= 2
        assert figures[0].data_points.get("Cloud") == 500.0


class TestFinancialDocumentParser:
    """Unit tests for FinancialDocumentParser."""

    def test_detect_metadata_from_text(self) -> None:
        text = """
        UNITED STATES SECURITIES AND EXCHANGE COMMISSION
        Washington, D.C. 20549
        FORM 10-K
        ANNUAL REPORT PURSUANT TO SECTION 13 OR 15(d) OF THE SECURITIES EXCHANGE ACT OF 1934
        For the fiscal year ended September 27, 2025
        Commission File Number: 001-36743
        Ticker: NVDA
        NVIDIA CORPORATION
        """
        meta = FinancialDocumentParser.detect_metadata_from_text(text, filename="nvda_10k.txt")
        assert meta.ticker == "NVDA"
        assert meta.year == 2025
        assert meta.doc_type == "10-K"

    def test_section_tracking_and_inline_metrics(self) -> None:
        doc_raw = """
        ITEM 1. BUSINESS OVERVIEW

        The company operates in cloud infrastructure. Total revenue was $85,000 million in fiscal 2025.

        ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS

        Operating margin expanded to 34.5% driven by AI chip demand. Diluted EPS reached $4.25.

        | Segment | Revenue |
        | Data Center | $40,000M |
        | Gaming | $15,000M |
        """
        parser = FinancialDocumentParser(chunk_size=200)
        doc = parser.parse_text(doc_raw)

        assert doc.chunk_count() >= 3
        # Check section assignments
        sections = [c.section_title for c in doc.chunks if c.section_title is not None]
        assert any("ITEM 1" in s for s in sections)
        assert any("ITEM 7" in s for s in sections)

        # Check inline metrics
        all_metrics = doc.get_all_metrics()
        metric_names = [m.name for m in all_metrics]
        assert any("revenue" in n.lower() for n in metric_names)

    def test_mock_pdf_parsing(self, monkeypatch) -> None:
        """Test parse_pdf with mocked pypdf reader."""
        mock_page_1 = MagicMock()
        mock_page_1.extract_text.return_value = "FORM 10-Q\nTicker: AAPL\nFiscal year 2025 Q3"
        mock_page_2 = MagicMock()
        mock_page_2.extract_text.return_value = "Operating Cash Flow was $32,000 million."

        mock_reader = MagicMock()
        mock_reader.pages = [mock_page_1, mock_page_2]

        import pypdf

        monkeypatch.setattr(pypdf, "PdfReader", lambda source: mock_reader)

        parser = FinancialDocumentParser()
        doc = parser.parse_pdf("dummy.pdf")
        assert doc.metadata.ticker == "AAPL"
        assert doc.metadata.page_count == 2
        assert len(doc.chunks) >= 2
        assert doc.chunks[0].page_number == 1
        assert doc.chunks[1].page_number == 2
