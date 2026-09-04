"""Citation extraction and provenance grounding."""

from __future__ import annotations

import re

from multi_modal_financial.types import Chunk, Citation


class CitationGrounder:
    """Extracts explicit citations and aligns synthesized text with source chunks."""

    @staticmethod
    def extract_citations_from_text(
        text: str,
        source_chunks: list[Chunk],
    ) -> tuple[str, list[Citation]]:
        """Find citation markers like [doc_id:p1] or [1] in text and produce Citation objects."""
        # Find numeric bracket references like [1], [2]
        bracket_pattern = re.compile(r"\[(\d+)\]")
        citations: list[Citation] = []

        chunk_map = {idx + 1: c for idx, c in enumerate(source_chunks)}

        for match in bracket_pattern.finditer(text):
            ref_num = int(match.group(1))
            if ref_num in chunk_map:
                target_chunk = chunk_map[ref_num]
                # Extract surrounding sentence as the quoted span
                start_pos = max(0, match.start() - 100)
                end_pos = min(len(text), match.end() + 100)
                quote_snippet = text[start_pos:end_pos].strip()

                citations.append(
                    Citation(
                        citation_id=f"cite_{ref_num}_{len(citations)+1}",
                        chunk_id=target_chunk.chunk_id,
                        doc_id=target_chunk.doc_id,
                        page_number=target_chunk.page_number,
                        quote=quote_snippet,
                        modal_type=target_chunk.modal_type,
                        confidence=0.95,
                    )
                )

        return text, citations

    @staticmethod
    def auto_ground_claim(
        claim: str,
        candidate_chunks: list[Chunk],
        min_overlap: float = 0.25,
    ) -> list[Citation]:
        """Ground a claim by finding highest lexical and numerical overlap in source chunks."""
        claim_words = set(re.findall(r"\b[A-Za-z0-9_\.]+\b", claim.lower()))
        if not claim_words:
            return []

        grounded: list[Citation] = []

        for idx, chunk in enumerate(candidate_chunks):
            chunk_words = set(re.findall(r"\b[A-Za-z0-9_\.]+\b", chunk.content.lower()))
            overlap = len(claim_words.intersection(chunk_words))
            score = overlap / len(claim_words)

            # Extra weight if exact numbers match
            claim_nums = set(re.findall(r"\b\d+(?:\.\d+)?\b", claim))
            if claim_nums:
                chunk_nums = set(re.findall(r"\b\d+(?:\.\d+)?\b", chunk.content))
                num_match = len(claim_nums.intersection(chunk_nums)) / len(claim_nums)
                score = (score * 0.5) + (num_match * 0.5)

            if score >= min_overlap:
                grounded.append(
                    Citation(
                        citation_id=f"auto_cite_{idx+1}",
                        chunk_id=chunk.chunk_id,
                        doc_id=chunk.doc_id,
                        page_number=chunk.page_number,
                        quote=chunk.content[:200],
                        modal_type=chunk.modal_type,
                        confidence=round(score, 4),
                    )
                )

        grounded.sort(key=lambda c: c.confidence, reverse=True)
        return grounded
