"""Tests for SEC section hierarchy parser and chunking."""

from __future__ import annotations

from multi_modal_financial.parsing.hierarchy import SectionHierarchyParser

SAMPLE_FILING_TEXT = """
Apple Inc. 10-K Annual Report

ITEM 1. BUSINESS
Apple designs, manufactures, and markets smartphones, personal computers, tablets, wearables, and accessories.
The Company's fiscal year is the 52- or 53-week period that ends on the last Saturday of September.

ITEM 1A. RISK FACTORS
The Company's business, results of operations, and financial condition can be adversely affected by global economic conditions.
Supply chain disruptions, geopolitical tensions, and trade restrictions represent key operational risks.

ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION
Total net sales were $383,285 million in 2023, compared to $394,328 million in 2022.
Gross margin was 44.1% compared to 43.3% in the prior year.

ITEM 8. FINANCIAL STATEMENTS
Condensed Consolidated Statements of Operations
Net sales: $383,285 million
Cost of sales: $214,137 million
Gross profit: $169,148 million
"""


class TestSectionHierarchyParser:
    def test_detect_sections(self) -> None:
        parser = SectionHierarchyParser()
        sections = parser.detect_sections(SAMPLE_FILING_TEXT)
        assert len(sections) == 4
        labels = [s[1] for s in sections]
        assert "Item 1: Business" in labels
        assert "Item 1A: Risk Factors" in labels
        assert "Item 7: MD&A" in labels
        assert "Item 8: Financial Statements and Supplementary Data" in labels

    def test_assign_sections_to_text(self) -> None:
        parser = SectionHierarchyParser()
        blocks = parser.assign_sections_to_text(SAMPLE_FILING_TEXT)
        assert len(blocks) >= 4
        assert blocks[0][0] == "Filing Header & Preamble"
        assert blocks[1][0] == "Item 1: Business"
        assert "smartphones" in blocks[1][1]

    def test_split_into_hierarchical_chunks(self) -> None:
        parser = SectionHierarchyParser(max_chunk_tokens=50)
        chunks = parser.split_into_hierarchical_chunks(
            doc_id="aapl_10k", text=SAMPLE_FILING_TEXT, base_page=1
        )
        assert len(chunks) >= 4
        for chunk in chunks:
            assert chunk.section_title is not None
            assert chunk.metadata.get("hierarchical") is True
            assert chunk.metadata.get("section") is not None
