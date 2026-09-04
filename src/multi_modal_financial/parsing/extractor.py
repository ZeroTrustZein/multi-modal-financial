"""Financial document text and multi-modal chunk extractor."""

from __future__ import annotations

import io
import re
from pathlib import Path

from multi_modal_financial.parsing.figure_parser import FigureParser
from multi_modal_financial.parsing.table_parser import TableParser
from multi_modal_financial.types import (
    Chunk,
    Document,
    DocumentMetadata,
    FinancialMetric,
    ModalType,
)


class FinancialDocumentParser:
    """Parses financial filings (10-K, 10-Q, earnings reports) into multi-modal chunks."""

    SECTION_HEADER_PATTERN = re.compile(
        r"^(?:ITEM\s+[0-9A-Z]+[\.\s\-:]+|NOTE\s+[0-9]+[\.\s\-:]+|PART\s+[IVX]+[\.\s\-:]+|"
        r"MANAGEMENT'S DISCUSSION|CONSOLIDATED STATEMENTS|FINANCIAL STATEMENTS|"
        r"BUSINESS OVERVIEW|RISK FACTORS|EXECUTIVE SUMMARY|OUTLOOK|HIGHLIGHTS)",
        re.IGNORECASE,
    )

    METRIC_EXTRACT_PATTERN = re.compile(
        r"\b([A-Za-z\s]{3,30}?)\s*(?:was|is|reached|of|amounted to|increased to|decreased to)?\s*"
        r"(\$?[0-9]+(?:,[0-9]{3})*(?:\.[0-9]+)?\s*(?:billion|million|thousand|B|M|K|%)?)\b",
        re.IGNORECASE,
    )

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @staticmethod
    def detect_metadata_from_text(
        text: str,
        filename: str = "document.txt",
        default_id: str | None = None,
    ) -> DocumentMetadata:
        """Infer document metadata (ticker, filing year, period, doc_type) from text preamble."""
        header_sample = text[:2500]
        doc_id = default_id or re.sub(r"[^a-zA-Z0-9_]", "_", Path(filename).stem)

        # Detect ticker
        ticker: str | None = None
        ticker_match = re.search(
            r"(?:ticker|symbol|nasdaq|nyse)\s*[:=\(\[]?\s*([A-Z]{1,5})\b",
            header_sample,
            re.IGNORECASE,
        )
        if ticker_match:
            ticker = ticker_match.group(1).upper()

        # Detect year: prioritize fiscal/period year over statutory act citations
        year: int | None = None
        fiscal_year_match = re.search(
            r"(?:fiscal\s+year\s+ended|year\s+ended|period\s+ended|for\s+the\s+year|fiscal\s+year|fy)\s*[A-Za-z0-9,\s]*?\b(20\d{2})\b",
            header_sample,
            re.IGNORECASE,
        )
        if fiscal_year_match:
            year = int(fiscal_year_match.group(1))
        else:
            # Fallback to general 2000s year before 1900s
            year_match_2000 = re.search(r"\b(20\d{2})\b", header_sample)
            if year_match_2000:
                year = int(year_match_2000.group(1))
            else:
                year_match_1900 = re.search(r"\b(19\d{2})\b", header_sample)
                if year_match_1900:
                    year = int(year_match_1900.group(1))

        # Detect fiscal period
        period: str | None = None
        period_match = re.search(
            r"\b(Q[1-4]|FY|FY\d{2,4}|Annual|Quarterly)\b", header_sample, re.IGNORECASE
        )
        if period_match:
            period = period_match.group(1).upper()

        # Detect doc type
        doc_type = "filing"
        if "10-k" in header_sample.lower():
            doc_type = "10-K"
        elif "10-q" in header_sample.lower():
            doc_type = "10-Q"
        elif "8-k" in header_sample.lower():
            doc_type = "8-K"
        elif (
            "earnings release" in header_sample.lower() or "press release" in header_sample.lower()
        ):
            doc_type = "earnings_release"

        return DocumentMetadata(
            doc_id=doc_id,
            filename=filename,
            ticker=ticker,
            period=period,
            year=year,
            doc_type=doc_type,
            page_count=1,
        )

    def _extract_metrics_from_text(self, text: str) -> list[FinancialMetric]:
        """Scan text for inline financial metrics and key values."""
        metrics: list[FinancialMetric] = []
        for match in self.METRIC_EXTRACT_PATTERN.finditer(text):
            label = match.group(1).strip()
            raw_val = match.group(2).strip()
            if any(
                term in label.lower()
                for term in [
                    "revenue",
                    "sales",
                    "income",
                    "profit",
                    "margin",
                    "eps",
                    "cash",
                    "flow",
                    "expense",
                    "ebitda",
                    "debt",
                    "equity",
                    "capex",
                ]
            ):
                metric = FinancialMetric.from_raw(label, raw_val, context=text[:120])
                if metric.value is not None:
                    metrics.append(metric)
        return metrics

    def parse_text(
        self,
        text: str,
        metadata: DocumentMetadata | None = None,
    ) -> Document:
        """Parse raw text/markdown into structured multi-modal Document."""
        if metadata is None:
            metadata = self.detect_metadata_from_text(text)

        chunks: list[Chunk] = []
        chunk_idx = 0
        current_section: str | None = None

        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        for p in paragraphs:
            # Check for section header
            first_line = p.splitlines()[0].strip()
            if self.SECTION_HEADER_PATTERN.match(first_line):
                current_section = first_line.rstrip(":")

            # Check if paragraph is markdown table
            if "|" in p and ("---" in p or p.count("|") >= 4):
                table_data = TableParser.parse_markdown_table(p)
                tbl_metrics = table_data.extract_metrics()
                chunks.append(
                    Chunk(
                        chunk_id=f"{metadata.doc_id}_chunk_{chunk_idx}",
                        doc_id=metadata.doc_id,
                        modal_type=ModalType.TABLE,
                        content=p,
                        page_number=1,
                        section_title=current_section,
                        table_data=table_data,
                        metrics=tbl_metrics,
                        token_count=int(len(p.split()) * 1.3),
                        metadata={"type": "table", "rows": table_data.row_count},
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
                        section_title=current_section,
                        figure_data=fig_data,
                        token_count=int(len(p.split()) * 1.3),
                        metadata={"type": "figure", "chart_type": fig_data.chart_type},
                    )
                )
                chunk_idx += 1
                continue

            # Check if paragraph is a single financial metric declaration
            if re.search(
                r"^[A-Za-z\s]+:\s*\$?[0-9]+(?:\.[0-9]+)?%?\s*(?:million|billion|M|B)?$",
                p.strip(),
                re.IGNORECASE,
            ):
                parts = p.strip().split(":", 1)
                metric_obj = FinancialMetric.from_raw(parts[0], parts[1].strip())
                chunks.append(
                    Chunk(
                        chunk_id=f"{metadata.doc_id}_chunk_{chunk_idx}",
                        doc_id=metadata.doc_id,
                        modal_type=ModalType.METRIC,
                        content=p,
                        page_number=1,
                        section_title=current_section,
                        metrics=[metric_obj] if metric_obj.value is not None else [],
                        token_count=int(len(p.split()) * 1.3),
                        metadata={"type": "metric"},
                    )
                )
                chunk_idx += 1
                continue

            # Standard narrative text: chunk if longer than threshold
            words = p.split()
            inline_metrics = self._extract_metrics_from_text(p)

            if len(words) > self.chunk_size:
                step = self.chunk_size - self.chunk_overlap
                for i in range(0, len(words), step):
                    sub_words = words[i : i + self.chunk_size]
                    sub_text = " ".join(sub_words)
                    sub_metrics = [m for m in inline_metrics if m.raw_value in sub_text]
                    chunks.append(
                        Chunk(
                            chunk_id=f"{metadata.doc_id}_chunk_{chunk_idx}",
                            doc_id=metadata.doc_id,
                            modal_type=ModalType.TEXT,
                            content=sub_text,
                            page_number=1,
                            section_title=current_section,
                            metrics=sub_metrics,
                            token_count=int(len(sub_words) * 1.3),
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
                        section_title=current_section,
                        metrics=inline_metrics,
                        token_count=int(len(words) * 1.3),
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
        first_page_text = reader.pages[0].extract_text() if page_count > 0 else ""

        if metadata is None:
            metadata = self.detect_metadata_from_text(first_page_text, filename=filename)
            metadata.page_count = page_count
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
