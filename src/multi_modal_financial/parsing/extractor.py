"""Financial document text and multi-modal chunk extractor."""

from __future__ import annotations

import io
import re
from pathlib import Path

from multi_modal_financial.parsing.figure_parser import FigureParser
from multi_modal_financial.parsing.table_parser import TableParser
from multi_modal_financial.types import Chunk, Document, DocumentMetadata, ModalType


class FinancialDocumentParser:
    """Parses financial filings (10-K, 10-Q, earnings reports) into multi-modal chunks."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def parse_text(
        self,
        text: str,
        metadata: DocumentMetadata | None = None,
    ) -> Document:
        """Parse raw text/markdown into structured multi-modal Document."""
        if metadata is None:
            metadata = DocumentMetadata(
                doc_id="doc_default",
                filename="document.txt",
                page_count=1,
            )

        chunks: list[Chunk] = []
        chunk_idx = 0

        # Split text into sections / paragraphs
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        for p in paragraphs:
            # Check if paragraph is markdown table
            if "|" in p and ("---" in p or p.count("|") >= 4):
                table_data = TableParser.parse_markdown_table(p)
                chunks.append(
                    Chunk(
                        chunk_id=f"{metadata.doc_id}_chunk_{chunk_idx}",
                        doc_id=metadata.doc_id,
                        modal_type=ModalType.TABLE,
                        content=p,
                        page_number=1,
                        table_data=table_data,
                        metadata={"type": "table"},
                    )
                )
                chunk_idx += 1
                continue

            # Check if paragraph describes a figure/chart
            if re.search(r"^(?:Figure|Chart)\s*\d*:", p, re.IGNORECASE) or "[Figure" in p:
                fig_data = FigureParser.parse_figure_block(p, figure_id=f"fig_{chunk_idx}")
                chunks.append(
                    Chunk(
                        chunk_id=f"{metadata.doc_id}_chunk_{chunk_idx}",
                        doc_id=metadata.doc_id,
                        modal_type=ModalType.FIGURE,
                        content=p,
                        page_number=1,
                        figure_data=fig_data,
                        metadata={"type": "figure"},
                    )
                )
                chunk_idx += 1
                continue

            # Check if paragraph is a single financial metric declaration
            if re.search(r"^[A-Za-z\s]+:\s*\$?[0-9]+(?:\.[0-9]+)?%?\s*(?:million|billion|M|B)?$", p.strip(), re.IGNORECASE):
                chunks.append(
                    Chunk(
                        chunk_id=f"{metadata.doc_id}_chunk_{chunk_idx}",
                        doc_id=metadata.doc_id,
                        modal_type=ModalType.METRIC,
                        content=p,
                        page_number=1,
                        metadata={"type": "metric"},
                    )
                )
                chunk_idx += 1
                continue

            # Standard narrative text: chunk if longer than threshold
            words = p.split()
            if len(words) > self.chunk_size:
                step = self.chunk_size - self.chunk_overlap
                for i in range(0, len(words), step):
                    sub_words = words[i : i + self.chunk_size]
                    sub_text = " ".join(sub_words)
                    chunks.append(
                        Chunk(
                            chunk_id=f"{metadata.doc_id}_chunk_{chunk_idx}",
                            doc_id=metadata.doc_id,
                            modal_type=ModalType.TEXT,
                            content=sub_text,
                            page_number=1,
                            metadata={"type": "narrative_slice", "start_idx": i},
                        )
                    )
                    chunk_idx += 1
            else:
                chunks.append(
                    Chunk(
                        chunk_id=f"{metadata.doc_id}_chunk_{chunk_idx}",
                        doc_id=metadata.doc_id,
                        modal_type=ModalType.TEXT,
                        content=p,
                        page_number=1,
                        metadata={"type": "narrative"},
                    )
                )
                chunk_idx += 1

        return Document(
            doc_id=metadata.doc_id,
            metadata=metadata,
            chunks=chunks,
            raw_text=text,
        )

    def parse_pdf(
        self,
        pdf_source: str | Path | bytes | io.BytesIO,
        metadata: DocumentMetadata | None = None,
    ) -> Document:
        """Extract pages and text from a PDF file."""
        import pypdf

        if isinstance(pdf_source, (str, Path)):
            reader = pypdf.PdfReader(str(pdf_source))
            filename = Path(pdf_source).name
        elif isinstance(pdf_source, bytes):
            reader = pypdf.PdfReader(io.BytesIO(pdf_source))
            filename = "document.pdf"
        else:
            reader = pypdf.PdfReader(pdf_source)
            filename = "document.pdf"

        page_count = len(reader.pages)
        if metadata is None:
            metadata = DocumentMetadata(
                doc_id=re.sub(r"[^a-zA-Z0-9_]", "_", filename),
                filename=filename,
                page_count=page_count,
            )
        else:
            metadata.page_count = page_count

        extracted_text_list: list[str] = []
        chunks: list[Chunk] = []
        chunk_idx = 0

        for page_idx, page in enumerate(reader.pages):
            page_num = page_idx + 1
            page_text = page.extract_text() or ""
            extracted_text_list.append(page_text)

            # Sub-parse page content
            sub_doc = self.parse_text(page_text, metadata=metadata)
            for c in sub_doc.chunks:
                c.chunk_id = f"{metadata.doc_id}_p{page_num}_{chunk_idx}"
                c.page_number = page_num
                chunks.append(c)
                chunk_idx += 1

        return Document(
            doc_id=metadata.doc_id,
            metadata=metadata,
            chunks=chunks,
            raw_text="\n\n--- PAGE BREAK ---\n\n".join(extracted_text_list),
        )
