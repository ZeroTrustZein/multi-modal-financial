"""Tests for financial document text cleaner and table repair subsystem."""

from multi_modal_financial.data.cleaner import FinancialDataCleaner


class TestFinancialDataCleaner:
    """Unit tests for FinancialDataCleaner."""

    def test_normalize_dashes(self):
        text = "Q3\u20132025 revenue was $1,500\u2014up 15% year\u2212over\u2010year."
        normalized = FinancialDataCleaner.normalize_dashes(text)
        assert "Q3-2025" in normalized
        assert "$1,500-up" in normalized
        assert "year-over-year" in normalized

    def test_strip_html_and_sec_tags(self):
        sec_sample = """
        <DOCUMENT>
        <TYPE>10-Q
        <SEQUENCE>1
        <FILENAME>aapl-20250628.htm
        <DESCRIPTION>FORM 10-Q
        <TEXT>
        <div>Revenue was &nbsp;$94,500 &amp; gross margin was 45.2% &mdash; an increase.</div>
        </TEXT>
        </DOCUMENT>
        """
        cleaned = FinancialDataCleaner.strip_html_and_sec_tags(sec_sample)
        assert "<DOCUMENT>" not in cleaned
        assert "<TEXT>" not in cleaned
        assert "<div>" not in cleaned
        assert "Revenue was  $94,500 & gross margin was 45.2% - an increase." in cleaned

    def test_normalize_accounting_negatives(self):
        text = "Operating loss was ($450.5) million, net margin was ( 12.5% ), and adjustment was (120)."
        converted = FinancialDataCleaner.normalize_accounting_negatives(text)
        assert "-$450.5" in converted
        assert "-12.5%" in converted
        assert "-120" in converted

    def test_clean_cell_footnotes(self):
        assert FinancialDataCleaner.clean_cell_footnotes("$1,250[1]") == "$1,250"
        assert (
            FinancialDataCleaner.clean_cell_footnotes("Operating Income(a)") == "Operating Income"
        )
        assert FinancialDataCleaner.clean_cell_footnotes("14.5%*") == "14.5%"
        assert FinancialDataCleaner.clean_cell_footnotes("Net Assets †") == "Net Assets"

    def test_repair_markdown_table(self):
        ragged_table = """
        | Metric | Q3 2024 | Q3 2025
        |---|---|---
        | Revenue[1] | $100 | $120
        | Operating Profit | ($15)
        """
        repaired = FinancialDataCleaner.repair_markdown_table(ragged_table)
        lines = repaired.strip().splitlines()
        assert len(lines) == 4
        # All lines start and end with pipe
        for line in lines:
            assert line.startswith("|") and line.endswith("|")
            # Should have 3 columns
            parts = [p.strip() for p in line.split("|")[1:-1]]
            assert len(parts) == 3

        # Check footnote cleaned
        assert "Revenue |" in repaired or "Revenue " in repaired

    def test_clean_document_text_end_to_end(self):
        raw_doc = """
        <DOCUMENT>
        <TYPE>10-K
        Table of Contents
        Page 14 of 120

        Operating income was ($250) million in Q3\u20132025.

        Gross margin expanded to 42.0%.
        </DOCUMENT>
        """
        cleaned = FinancialDataCleaner.clean_document_text(raw_doc)
        assert "Table of Contents" not in cleaned
        assert "Page 14 of 120" not in cleaned
        assert "-$250 million in Q3-2025." in cleaned
        assert "Gross margin expanded to 42.0%." in cleaned
        # Check blank line collapsing
        assert "\n\n\n\n" not in cleaned
