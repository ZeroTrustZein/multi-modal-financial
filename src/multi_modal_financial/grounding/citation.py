"""Citation extraction and provenance grounding."""

from __future__ import annotations

import re

from multi_modal_financial.types import Chunk, Citation, ModalType, ProvenanceRecord


class CitationGrounder:
    """Extracts explicit citations and aligns synthesized text with source chunks."""

    PROVENANCE_BADGE_PATTERN = re.compile(r"\[([a-zA-Z0-9_\-]+):p(\d+)#([a-zA-Z0-9_\-]+)\]")
    BRACKET_PATTERN = re.compile(r"\[(\d+)\]")

    @staticmethod
    def extract_numbers_from_string(text: str) -> list[str]:
        """Extract clean numeric tokens for provenance matching."""
        nums = re.findall(r"-?\b\d+(?:,\d{3})*(?:\.\d+)?\b", text)
        return [n.replace(",", "") for n in nums]

    @classmethod
    def extract_citations_from_text(
        cls,
        text: str,
        source_chunks: list[Chunk],
    ) -> tuple[str, list[Citation]]:
        """Find citation markers like [1] or [doc_id:p1#chunk_id] in text and produce Citation objects."""
        citations: list[Citation] = []
        chunk_map_by_idx = {idx + 1: c for idx, c in enumerate(source_chunks)}
        chunk_map_by_id = {c.chunk_id: c for c in source_chunks}

        # 1. Look for inline badges [doc_id:p1#chunk_id]
        for match in cls.PROVENANCE_BADGE_PATTERN.finditer(text):
            doc_id = match.group(1)
            page_num = int(match.group(2))
            chunk_id = match.group(3)
            target_chunk = chunk_map_by_id.get(chunk_id)

            start_pos = max(0, match.start() - 120)
            end_pos = min(len(text), match.end() + 120)
            span = text[start_pos:end_pos].strip()

            claim_nums = set(cls.extract_numbers_from_string(span))
            matched_nums: list[str] = []
            if target_chunk:
                c_nums = set(cls.extract_numbers_from_string(target_chunk.content))
                matched_nums = sorted(list(claim_nums.intersection(c_nums)))

            citations.append(
                Citation(
                    citation_id=f"cite_badge_{len(citations) + 1}",
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    page_number=page_num,
                    quote=span,
                    modal_type=target_chunk.modal_type if target_chunk else ModalType.TEXT,
                    confidence=0.98,
                    matched_numbers=matched_nums,
                )
            )

        # 2. Look for numeric bracket references like [1], [2]
        for match in cls.BRACKET_PATTERN.finditer(text):
            ref_num = int(match.group(1))
            if ref_num in chunk_map_by_idx:
                target_chunk = chunk_map_by_idx[ref_num]
                start_pos = max(0, match.start() - 100)
                end_pos = min(len(text), match.end() + 100)
                quote_snippet = text[start_pos:end_pos].strip()

                claim_nums = set(cls.extract_numbers_from_string(quote_snippet))
                c_nums = set(cls.extract_numbers_from_string(target_chunk.content))
                matched_nums = sorted(list(claim_nums.intersection(c_nums)))

                citations.append(
                    Citation(
                        citation_id=f"cite_{ref_num}_{len(citations) + 1}",
                        chunk_id=target_chunk.chunk_id,
                        doc_id=target_chunk.doc_id,
                        page_number=target_chunk.page_number,
                        quote=quote_snippet,
                        modal_type=target_chunk.modal_type,
                        confidence=0.95,
                        matched_numbers=matched_nums,
                    )
                )

        return text, citations

    @classmethod
    def auto_ground_claim(
        cls,
        claim: str,
        candidate_chunks: list[Chunk],
        min_overlap: float = 0.25,
    ) -> list[Citation]:
        """Ground an un-annotated claim by finding highest lexical and numerical overlap in source chunks."""
        claim_words = set(re.findall(r"\b[A-Za-z0-9_\.]+\b", claim.lower()))
        if not claim_words:
            return []

        claim_nums = set(cls.extract_numbers_from_string(claim))
        grounded: list[Citation] = []

        for idx, chunk in enumerate(candidate_chunks):
            chunk_words = set(re.findall(r"\b[A-Za-z0-9_\.]+\b", chunk.content.lower()))
            overlap = len(claim_words.intersection(chunk_words))
            lex_score = overlap / len(claim_words)

            num_score = 0.0
            matched_nums: list[str] = []
            if claim_nums:
                chunk_nums = set(cls.extract_numbers_from_string(chunk.content))
                intersection = claim_nums.intersection(chunk_nums)
                matched_nums = sorted(list(intersection))
                num_score = len(intersection) / len(claim_nums)
                composite_score = (lex_score * 0.4) + (num_score * 0.6)
            else:
                composite_score = lex_score

            if composite_score >= min_overlap:
                grounded.append(
                    Citation(
                        citation_id=f"auto_cite_{idx + 1}",
                        chunk_id=chunk.chunk_id,
                        doc_id=chunk.doc_id,
                        page_number=chunk.page_number,
                        quote=chunk.content[:250].strip(),
                        modal_type=chunk.modal_type,
                        confidence=round(composite_score, 4),
                        matched_numbers=matched_nums,
                    )
                )

        grounded.sort(key=lambda c: c.confidence, reverse=True)
        return grounded

    @classmethod
    def create_provenance_record(
        cls,
        citation: Citation,
        chunk: Chunk,
        cell_coord: tuple[int, int] | None = None,
    ) -> ProvenanceRecord:
        """Construct fine-grained provenance audit trail item."""
        return ProvenanceRecord(
            record_id=f"prov_{citation.citation_id}",
            doc_id=citation.doc_id,
            chunk_id=citation.chunk_id,
            page_number=citation.page_number,
            modal_type=citation.modal_type,
            table_cell=cell_coord,
            source_snippet=citation.quote,
            verified=citation.confidence >= 0.7,
        )
