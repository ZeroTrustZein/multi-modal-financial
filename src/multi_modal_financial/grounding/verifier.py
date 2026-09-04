"""Grounding verifier and hallucination detector for financial figures."""

from __future__ import annotations

import re
from typing import Any

from multi_modal_financial.grounding.citation import CitationGrounder
from multi_modal_financial.types import Chunk, GroundingStatus, GroundingVerdict


class GroundingVerifier:
    """Verifies that synthesized financial claims are strictly backed by source evidence."""

    def __init__(self, confidence_threshold: float = 0.6):
        self.confidence_threshold = confidence_threshold

    @staticmethod
    def extract_numbers(text: str) -> list[str]:
        """Extract canonical numeric strings for financial accuracy checking."""
        return re.findall(r"\b\d+(?:,\d{3})*(?:\.\d+)?\b", text)

    def verify_claim(self, claim: str, source_chunks: list[Chunk]) -> GroundingVerdict:
        """Verify whether an individual factual claim is grounded in source chunks."""
        citations = CitationGrounder.auto_ground_claim(claim, source_chunks)

        # Check numerical fidelity: any number in claim must be present in at least one source chunk
        claim_numbers = self.extract_numbers(claim)
        source_text = " ".join(c.content for c in source_chunks)
        clean_source = source_text.replace(",", "")

        missing_numbers: list[str] = []
        for num in claim_numbers:
            # Check raw or cleaned number without commas
            clean_num = num.replace(",", "")
            if num not in source_text and clean_num not in clean_source:
                missing_numbers.append(num)

        if not citations:
            return GroundingVerdict(
                claim=claim,
                citations=[],
                is_supported=False,
                support_score=0.0,
                status=GroundingStatus.UNSUPPORTED,
                unsupported_numbers=missing_numbers,
                reasoning="No source chunks met minimum lexical overlap threshold.",
            )

        top_score = citations[0].confidence

        # If numbers mentioned in claim are missing from sources, penalize heavily
        if missing_numbers:
            penalty = 0.4 * (len(missing_numbers) / max(len(claim_numbers), 1))
            final_score = max(0.0, top_score - penalty)
            is_supported = final_score >= self.confidence_threshold
            status = (
                GroundingStatus.PARTIALLY_SUPPORTED if is_supported else GroundingStatus.UNSUPPORTED
            )
            return GroundingVerdict(
                claim=claim,
                citations=citations,
                is_supported=is_supported,
                support_score=round(final_score, 4),
                status=status,
                unsupported_numbers=missing_numbers,
                reasoning=f"Warning: Numbers {missing_numbers} not found in retrieved chunks.",
            )

        is_supported = top_score >= self.confidence_threshold
        status = (
            GroundingStatus.FULLY_SUPPORTED if is_supported else GroundingStatus.PARTIALLY_SUPPORTED
        )
        return GroundingVerdict(
            claim=claim,
            citations=citations,
            is_supported=is_supported,
            support_score=round(top_score, 4),
            status=status,
            unsupported_numbers=[],
            reasoning="Claim fully supported with numerical and lexical alignment.",
        )

    def verify_response(
        self,
        response_text: str,
        source_chunks: list[Chunk],
    ) -> list[GroundingVerdict]:
        """Split synthesized response into sentences/claims and verify each."""
        sentences = [
            s.strip()
            for s in re.split(r"(?<=[.!?])\s+", response_text)
            if len(s.strip()) > 10 and not s.strip().startswith("#")
        ]

        verdicts: list[GroundingVerdict] = []
        for sentence in sentences:
            verdict = self.verify_claim(sentence, source_chunks)
            verdicts.append(verdict)

        return verdicts

    def calculate_hallucination_rate(self, verdicts: list[GroundingVerdict]) -> float:
        """Calculate the proportion of ungrounded or contradictory claims."""
        if not verdicts:
            return 0.0
        unsupported = sum(1 for v in verdicts if not v.is_supported)
        return round(unsupported / len(verdicts), 4)

    def audit_summary(self, verdicts: list[GroundingVerdict]) -> dict[str, Any]:
        """Produce structured grounding metrics audit."""
        total = len(verdicts)
        if total == 0:
            return {
                "total_claims": 0,
                "supported_claims": 0,
                "hallucination_rate": 0.0,
                "avg_support_score": 0.0,
                "unsupported_numbers": [],
            }

        supported = sum(1 for v in verdicts if v.is_supported)
        all_unsupported_nums = [num for v in verdicts for num in v.unsupported_numbers]
        avg_score = sum(v.support_score for v in verdicts) / total

        return {
            "total_claims": total,
            "supported_claims": supported,
            "hallucination_rate": self.calculate_hallucination_rate(verdicts),
            "avg_support_score": round(avg_score, 4),
            "unsupported_numbers": list(set(all_unsupported_nums)),
        }
