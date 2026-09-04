"""Hybrid Index combining BM25 Okapi and Dense Vector search."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from multi_modal_financial.indexing.bm25 import BM25Index
from multi_modal_financial.indexing.vector import DenseVectorIndex
from multi_modal_financial.types import Chunk, Document, FinancialStatementType, ModalType


class HybridIndex:
    """Manages dual sparse-dense indexes and document storage."""

    def __init__(
        self,
        dimension: int = 128,
        embed_fn: Callable[[list[str]], np.ndarray] | None = None,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
    ):
        self.bm25 = BM25Index(k1=bm25_k1, b=bm25_b)
        self.vector = DenseVectorIndex(dimension=dimension, embed_fn=embed_fn)
        self.chunks: dict[str, Chunk] = {}
        self.documents: dict[str, Document] = {}

    def _build_enriched_text(self, chunk: Chunk) -> str:
        """Construct information-dense representation for indexing."""
        parts: list[str] = []

        if chunk.section_title:
            parts.append(f"[Section: {chunk.section_title}]")

        if chunk.table_data:
            tbl = chunk.table_data
            if tbl.title:
                parts.append(f"Table: {tbl.title}")
            if tbl.statement_type and tbl.statement_type != FinancialStatementType.UNKNOWN:
                parts.append(f"Statement: {tbl.statement_type}")
            if tbl.scale:
                parts.append(f"Scale: {tbl.scale}")
            parts.append(chunk.content)
            if tbl.footnotes:
                parts.append("Footnotes: " + " ".join(tbl.footnotes))
        elif chunk.figure_data:
            fig = chunk.figure_data
            if fig.caption:
                parts.append(f"Figure: {fig.caption}")
            if fig.chart_type:
                parts.append(f"Chart Type: {fig.chart_type}")
            if fig.data_points:
                pts_str = ", ".join(f"{k}={v:g}" for k, v in fig.data_points.items())
                parts.append(f"Data: {pts_str}")
            parts.append(chunk.content)
        else:
            parts.append(chunk.content)

        if chunk.metrics:
            metrics_summary = "; ".join(f"{m.name}: {m.formatted()}" for m in chunk.metrics)
            parts.append(f"[Metrics: {metrics_summary}]")

        return "\n".join(parts)

    def index_document(self, document: Document) -> None:
        """Add document and all its chunks to the hybrid index."""
        self.documents[document.doc_id] = document
        for chunk in document.chunks:
            self.chunks[chunk.chunk_id] = chunk

        self._rebuild_indices()

    def index_chunks(self, chunks: list[Chunk]) -> None:
        """Add individual chunks to the hybrid index."""
        for chunk in chunks:
            self.chunks[chunk.chunk_id] = chunk

        self._rebuild_indices()

    def add_document(self, document: Document) -> None:
        """Alias for index_document."""
        self.index_document(document)

    def _rebuild_indices(self) -> None:
        """Re-fit BM25 and dense vector indexes over all active chunks."""
        chunk_ids = list(self.chunks.keys())
        contents = [self._build_enriched_text(self.chunks[cid]) for cid in chunk_ids]

        self.bm25.fit(chunk_ids, contents)
        self.vector.fit(chunk_ids, contents)

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        """Retrieve chunk by ID."""
        return self.chunks.get(chunk_id)

    def get_document(self, doc_id: str) -> Document | None:
        """Retrieve indexed document container by ID."""
        return self.documents.get(doc_id)

    def get_all_chunks(self) -> list[Chunk]:
        """Return list of all indexed chunks."""
        return list(self.chunks.values())

    def total_chunks(self) -> int:
        """Return total number of chunks indexed."""
        return len(self.chunks)

    def total_documents(self) -> int:
        """Return total number of documents indexed."""
        return len(self.documents)

    def filter_chunk_ids(
        self,
        ticker: str | None = None,
        period: str | None = None,
        year: int | None = None,
        modal_type: ModalType | None = None,
        statement_type: FinancialStatementType | None = None,
        doc_type: str | None = None,
    ) -> list[str]:
        """Filter chunk IDs based on metadata and statement criteria."""
        valid_ids: list[str] = []
        for cid, chunk in self.chunks.items():
            if modal_type and chunk.modal_type != modal_type:
                continue

            if statement_type:
                if not chunk.table_data or chunk.table_data.statement_type != statement_type:
                    continue

            doc = self.documents.get(chunk.doc_id)
            if doc:
                if ticker and doc.metadata.ticker and doc.metadata.ticker.upper() != ticker.upper():
                    continue
                if period and doc.metadata.period and doc.metadata.period.upper() != period.upper():
                    continue
                if year and doc.metadata.year and doc.metadata.year != year:
                    continue
                if doc_type and doc.metadata.doc_type.lower() != doc_type.lower():
                    continue

            valid_ids.append(cid)
        return valid_ids
