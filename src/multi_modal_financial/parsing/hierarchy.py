"""Hierarchical section-aware document splitter for SEC filings and financial disclosures."""

from __future__ import annotations

import re
from typing import Any

from multi_modal_financial.types import Chunk, ModalType

# Standard SEC 10-K and 10-Q Item structure patterns
SEC_ITEM_PATTERNS = [
    (re.compile(r"^\s*item\s+1\.\s+business\b", re.IGNORECASE | re.MULTILINE), "Item 1: Business"),
    (
        re.compile(r"^\s*item\s+1a\.\s+risk\s+factors\b", re.IGNORECASE | re.MULTILINE),
        "Item 1A: Risk Factors",
    ),
    (
        re.compile(
            r"^\s*item\s+1b\.\s+unresolved\s+staff\s+comments\b", re.IGNORECASE | re.MULTILINE
        ),
        "Item 1B: Unresolved Staff Comments",
    ),
    (
        re.compile(r"^\s*item\s+2\.\s+properties\b", re.IGNORECASE | re.MULTILINE),
        "Item 2: Properties",
    ),
    (
        re.compile(r"^\s*item\s+3\.\s+legal\s+proceedings\b", re.IGNORECASE | re.MULTILINE),
        "Item 3: Legal Proceedings",
    ),
    (
        re.compile(
            r"^\s*item\s+7\.\s+management['']?s\s+discussion\s+and\s+analysis\b",
            re.IGNORECASE | re.MULTILINE,
        ),
        "Item 7: MD&A",
    ),
    (
        re.compile(
            r"^\s*item\s+7a\.\s+quantitative\s+and\s+qualitative\s+disclosures\b",
            re.IGNORECASE | re.MULTILINE,
        ),
        "Item 7A: Market Risk Disclosures",
    ),
    (
        re.compile(r"^\s*item\s+8\.\s+financial\s+statements\b", re.IGNORECASE | re.MULTILINE),
        "Item 8: Financial Statements and Supplementary Data",
    ),
    (
        re.compile(
            r"^\s*item\s+9\.\s+changes\s+in\s+and\s+disagreements\b", re.IGNORECASE | re.MULTILINE
        ),
        "Item 9: Accounting Disagreements",
    ),
]


class SectionHierarchyParser:
    """Detects and assigns hierarchical SEC sections to document text and chunks."""

    def __init__(self, max_chunk_tokens: int = 500, chunk_overlap_tokens: int = 50):
        self.max_chunk_tokens = max_chunk_tokens
        self.chunk_overlap_tokens = chunk_overlap_tokens

    def detect_sections(self, text: str) -> list[tuple[int, str]]:
        """Find character start indices and labels of SEC sections."""
        found_sections: list[tuple[int, str]] = []
        for pattern, label in SEC_ITEM_PATTERNS:
            for match in pattern.finditer(text):
                found_sections.append((match.start(), label))

        # Sort by character offset
        found_sections.sort(key=lambda x: x[0])
        return found_sections

    def assign_sections_to_text(self, text: str) -> list[tuple[str, str]]:
        """Split raw text into (section_label, section_content) pairs."""
        sections = self.detect_sections(text)
        if not sections:
            return [("General Filing", text)]

        result: list[tuple[str, str]] = []
        if sections[0][0] > 0:
            preamble = text[: sections[0][0]].strip()
            if preamble:
                result.append(("Filing Header & Preamble", preamble))

        for idx, (start_idx, label) in enumerate(sections):
            end_idx = sections[idx + 1][0] if idx + 1 < len(sections) else len(text)
            content = text[start_idx:end_idx].strip()
            if content:
                result.append((label, content))

        return result

    def split_into_hierarchical_chunks(
        self,
        doc_id: str,
        text: str,
        base_page: int = 1,
        metadata: dict[str, Any] | None = None,
    ) -> list[Chunk]:
        """Convert document text into section-attributed atomic chunks."""
        section_blocks = self.assign_sections_to_text(text)
        chunks: list[Chunk] = []
        chunk_idx = 1
        meta = metadata or {}

        for sec_label, sec_text in section_blocks:
            paragraphs = [p.strip() for p in sec_text.split("\n\n") if p.strip()]
            current_buffer: list[str] = []
            current_word_count = 0

            for p in paragraphs:
                words = p.split()
                p_len = len(words)

                if current_word_count + p_len > self.max_chunk_tokens and current_buffer:
                    content_str = "\n\n".join(current_buffer)
                    chunk_id = f"{doc_id}_chunk_{chunk_idx:04d}"
                    chunk_meta = dict(meta)
                    chunk_meta["section"] = sec_label
                    chunk_meta["hierarchical"] = True

                    chunks.append(
                        Chunk(
                            chunk_id=chunk_id,
                            doc_id=doc_id,
                            modal_type=ModalType.TEXT,
                            content=content_str,
                            page_number=base_page,
                            section_title=sec_label,
                            token_count=len(content_str.split()),
                            metadata=chunk_meta,
                        )
                    )
                    chunk_idx += 1
                    current_buffer = [p]
                    current_word_count = p_len
                else:
                    current_buffer.append(p)
                    current_word_count += p_len

            if current_buffer:
                content_str = "\n\n".join(current_buffer)
                chunk_id = f"{doc_id}_chunk_{chunk_idx:04d}"
                chunk_meta = dict(meta)
                chunk_meta["section"] = sec_label
                chunk_meta["hierarchical"] = True

                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        doc_id=doc_id,
                        modal_type=ModalType.TEXT,
                        content=content_str,
                        page_number=base_page,
                        section_title=sec_label,
                        token_count=len(content_str.split()),
                        metadata=chunk_meta,
                    )
                )
                chunk_idx += 1

        return chunks
