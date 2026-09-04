"""Grounding verifier and hallucination detector for financial figures."""

from __future__ import annotations

import re

from multi_modal_financial.grounding.citation import CitationGrounder
from multi_modal_financial.types import Chunk, GroundingVerdict


class GroundingVerifier:
    """Verifies that synthesized financial claims are strictly backed by source evidence."""

    def __init__(self, confidence_threshold: float = 0.6):
        self.confidence_threshold = confidence_threshold

    def verify_claim(self, claim: str, source_chunks: list[Chunk]) -> GroundingVerdict:
        """Verify whether an individual factual claim is grounded in source chunks."""
        citations = CitationGrounder.auto_ground_claim(claim, source_chunks)

        # Check numerical fidelity: any number in claim must be present in at least one source chunk
        claim_numbers = re.findall(r"\b\d+(?:\.\d+)?\b", claim)
        source_text = " ".join(c.content for c in source_chunks)
        missing_numbers = [num for num in claim_numbers if num not in source_text]

        if not citations:
            return GroundingVerdict(
                claim=claim,
                citations=[],
                is_supported=False,
                support_score=0.0,
                reasoning="No source chunks met minimum lexical overlap threshold.",
            )

        top_score = citations[0].confidence

        # If numbers mentioned in claim are missing from sources, penalize heavily
        if missing_numbers:
            penalty = 0.4 * (len(missing_numbers) / max(len(claim_numbers), 1))
            final_score = max(0.0, top_score - penalty)
            is_supported = final_score >= self.confidence_threshold
            return GroundingVerdict(
                claim=claim,
                citations=citations,
                is_supported=is_supported,
                support_score=round(final_score, 4),
                reasoning=f"Warning: Numbers {missing_numbers} not found in retrieved chunks.",
            )

        is_supported = top_score >= self.confidence_threshold
        return GroundingVerdict(
            claim=claim,
            citations=citations,
            is_supported=is_supported,
            support_score=round(top_score, 4),
            reasoning="Claim fully supported with numerical and lexical alignment.",
        )

    def verify_response(
        self,
        response_text: str,
        source_chunks: list[Chunk],
    ) -> list[GroundingVerdict]:
        """Split synthesized response into sentences/claims and verify each."""
        # Split text into distinct sentences / claims
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
