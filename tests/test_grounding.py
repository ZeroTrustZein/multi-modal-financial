"""Comprehensive unit tests for grounding core logic: citations, verifier, and provenance."""

from __future__ import annotations

import pytest

from multi_modal_financial.grounding.citation import CitationGrounder
from multi_modal_financial.grounding.verifier import GroundingVerifier
from multi_modal_financial.types import Chunk, GroundingStatus, ModalType


class TestCitationGrounder:
    """Unit tests for CitationGrounder."""

    @pytest.fixture
    def sample_chunks(self) -> list[Chunk]:
        return [
            Chunk(
                chunk_id="chk_rev",
                doc_id="aapl_2025",
                modal_type=ModalType.TEXT,
                content="Apple reported net sales of $94,930 million for the quarter.",
                page_number=2,
            ),
            Chunk(
                chunk_id="chk_margin",
                doc_id="aapl_2025",
                modal_type=ModalType.TABLE,
                content="Gross margin expanded to 46.2% compared to 45.2% prior year.",
                page_number=5,
            ),
        ]

    def test_extract_citations_with_brackets(self, sample_chunks: list[Chunk]) -> None:
        text = "Net sales were strong [1] while gross margin expanded [2]."
        cleaned_text, citations = CitationGrounder.extract_citations_from_text(text, sample_chunks)
        assert len(citations) == 2
        assert citations[0].chunk_id == "chk_rev"
        assert citations[1].chunk_id == "chk_margin"

    def test_extract_citations_with_provenance_badges(self, sample_chunks: list[Chunk]) -> None:
        text = "Revenue was confirmed [aapl_2025:p2#chk_rev]."
        _, citations = CitationGrounder.extract_citations_from_text(text, sample_chunks)
        assert len(citations) == 1
        assert citations[0].doc_id == "aapl_2025"
        assert citations[0].page_number == 2
        assert citations[0].chunk_id == "chk_rev"

    def test_auto_ground_claim(self, sample_chunks: list[Chunk]) -> None:
        claim = "Net sales reached $94,930 million."
        cites = CitationGrounder.auto_ground_claim(claim, sample_chunks)
        assert len(cites) > 0
        assert cites[0].chunk_id == "chk_rev"
        assert "94930" in cites[0].matched_numbers

    def test_create_provenance_record(self, sample_chunks: list[Chunk]) -> None:
        claim = "Net sales were $94,930 million."
        cites = CitationGrounder.auto_ground_claim(claim, sample_chunks)
        rec = CitationGrounder.create_provenance_record(
            cites[0], sample_chunks[0], cell_coord=(1, 2)
        )
        assert rec.chunk_id == "chk_rev"
        assert rec.table_cell == (1, 2)
        assert rec.verified is True


class TestGroundingVerifier:
    """Unit tests for GroundingVerifier."""

    @pytest.fixture
    def evidence_chunks(self) -> list[Chunk]:
        return [
            Chunk(
                chunk_id="chk_1",
                doc_id="doc_1",
                content="Total revenue was $150,000 million and operating income was $45,000 million in 2025.",
            ),
            Chunk(
                chunk_id="chk_2",
                doc_id="doc_1",
                content="Diluted earnings per share was $3.85 with free cash flow of $38,000 million.",
            ),
        ]

    def test_verify_supported_claim(self, evidence_chunks: list[Chunk]) -> None:
        verifier = GroundingVerifier(confidence_threshold=0.6)
        verdict = verifier.verify_claim(
            "Operating income was $45,000 million in 2025.", evidence_chunks
        )
        assert verdict.is_supported is True
        assert verdict.status == GroundingStatus.FULLY_SUPPORTED
        assert len(verdict.unsupported_numbers) == 0

    def test_verify_hallucinated_numbers(self, evidence_chunks: list[Chunk]) -> None:
        verifier = GroundingVerifier(confidence_threshold=0.6)
        verdict = verifier.verify_claim(
            "Operating income reached $999,999 million in 2025.", evidence_chunks
        )
        assert "999,999" in verdict.unsupported_numbers or "999999" in verdict.unsupported_numbers
        assert "Warning: Numbers" in verdict.reasoning

    def test_verify_response_multiple_sentences(self, evidence_chunks: list[Chunk]) -> None:
        verifier = GroundingVerifier(confidence_threshold=0.6)
        response_text = (
            "Total revenue reached $150,000 million in 2025. "
            "EPS was reported as $3.85 per share. "
            "However, dividends paid were $12,345 million."
        )
        verdicts = verifier.verify_response(response_text, evidence_chunks)
        assert len(verdicts) == 3
        assert verdicts[0].is_supported is True
        assert verdicts[1].is_supported is True
        # Third sentence has hallucinated number 12,345
        assert verdicts[2].is_supported is False

        hallucination_rate = verifier.calculate_hallucination_rate(verdicts)
        assert round(hallucination_rate, 2) == 0.33

        summary = verifier.audit_summary(verdicts)
        assert summary["total_claims"] == 3
        assert summary["supported_claims"] == 2
        assert len(summary["unsupported_numbers"]) > 0
