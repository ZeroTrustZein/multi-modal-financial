"""Batch document loader and dataset manager for financial disclosures."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from pathlib import Path

from multi_modal_financial.data.cleaner import FinancialDataCleaner
from multi_modal_financial.data.validator import FinancialTableValidator
from multi_modal_financial.parsing.extractor import FinancialDocumentParser
from multi_modal_financial.types import Document, DocumentMetadata


class BatchDocumentLoader:
    """Loads and preprocesses batches of financial documents from disk."""

    SUPPORTED_EXTENSIONS = {".txt", ".md", ".csv", ".json", ".pdf"}

    # Filename heuristic pattern: TICKER_DOCTYPE_YEAR_PERIOD (e.g. AAPL_10Q_2025_Q3.txt)
    FILENAME_META_PATTERN = re.compile(
        r"^(?P<ticker>[A-Za-z]{1,5})[_-](?P<doctype>[A-Za-z0-9]+)[_-](?P<year>20\d\d)[_-]?(?P<period>[Qq][1-4]|[Ff][Yy]\d\d?)?",
        re.IGNORECASE,
    )

    def __init__(
        self,
        parser: FinancialDocumentParser | None = None,
        clean_text: bool = True,
        validate_tables: bool = True,
    ):
        self.parser = parser or FinancialDocumentParser()
        self.clean_text = clean_text
        self.validate_tables = validate_tables

    def extract_metadata_from_path(self, file_path: Path) -> DocumentMetadata:
        """Derive metadata fields from filename and directory heuristics."""
        stem = file_path.stem
        match = self.FILENAME_META_PATTERN.match(stem)

        ticker = None
        doc_type = "FILING"
        year = None
        period = None

        if match:
            gd = match.groupdict()
            ticker = gd.get("ticker", "").upper() if gd.get("ticker") else None
            doc_type = gd.get("doctype", "FILING").upper() if gd.get("doctype") else "FILING"
            year = int(gd["year"]) if gd.get("year") else None
            period = gd.get("period", "").upper() if gd.get("period") else None

        return DocumentMetadata(
            doc_id=stem,
            filename=file_path.name,
            ticker=ticker,
            doc_type=doc_type,
            year=year,
            period=period,
        )

    def load_file(self, file_path: str | Path, default_ticker: str | None = None) -> Document:
        """Load, clean, parse, and validate a single financial document file."""
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")

        meta = self.extract_metadata_from_path(path)
        if default_ticker and not meta.ticker:
            meta.ticker = default_ticker.upper()

        ext = path.suffix.lower()

        if ext == ".pdf":
            doc = self.parser.parse_pdf(path)
            if default_ticker and not doc.metadata.ticker:
                doc.metadata.ticker = default_ticker.upper()
        elif ext == ".json":
            raw_json = path.read_text(encoding="utf-8", errors="replace")
            data = json.loads(raw_json)
            # If JSON is already a serialized Document
            if "doc_id" in data and "chunks" in data:
                doc = Document.from_dict(data)
            else:
                text_content = data.get("content", data.get("text", str(data)))
                if self.clean_text:
                    text_content = FinancialDataCleaner.clean_document_text(text_content)
                doc = self.parser.parse_text(text_content, metadata=meta)
        else:
            raw_text = path.read_text(encoding="utf-8", errors="replace")
            if self.clean_text:
                raw_text = FinancialDataCleaner.clean_document_text(raw_text)
            doc = self.parser.parse_text(raw_text, metadata=meta)

        # Validate parsed tables if requested
        if self.validate_tables:
            for chunk in doc.chunks:
                if chunk.table_data:
                    report = FinancialTableValidator.validate_table_structure(chunk.table_data)
                    chunk.metadata["validation_errors"] = len(report.errors())
                    chunk.metadata["validation_warnings"] = len(report.warnings())

        return doc

    def load_directory(
        self,
        directory: str | Path,
        pattern: str = "*.*",
        recursive: bool = True,
        default_ticker: str | None = None,
    ) -> list[Document]:
        """Load all matching documents in a directory."""
        dir_path = Path(directory)
        if not dir_path.is_dir():
            return []

        file_iter = dir_path.rglob(pattern) if recursive else dir_path.glob(pattern)
        documents: list[Document] = []

        for file_path in sorted(file_iter):
            if file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                try:
                    doc = self.load_file(file_path, default_ticker=default_ticker)
                    documents.append(doc)
                except Exception:
                    continue

        return documents

    def iter_documents(
        self,
        directory: str | Path,
        pattern: str = "*.*",
        default_ticker: str | None = None,
    ) -> Iterator[Document]:
        """Streaming generator yielding documents one-by-one to conserve memory."""
        dir_path = Path(directory)
        if not dir_path.is_dir():
            return

        for file_path in sorted(dir_path.rglob(pattern)):
            if file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                try:
                    yield self.load_file(file_path, default_ticker=default_ticker)
                except Exception:
                    continue
