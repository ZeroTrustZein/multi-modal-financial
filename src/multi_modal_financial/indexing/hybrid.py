"""Hybrid Index combining BM25 Okapi and Dense Vector search."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from multi_modal_financial.indexing.bm25 import BM25Index
from multi_modal_financial.indexing.vector import DenseVectorIndex
from multi_modal_financial.types import Chunk, Document, ModalType


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

    def _rebuild_indices(self) -> None:
        """Re-fit BM25 and dense vector indexes over all active chunks."""
        chunk_ids = list(self.chunks.keys())
        # Enhance text representation with table metadata if present
        contents: list[str] = []
        for cid in chunk_ids:
            chunk = self.chunks[cid]
            text = chunk.content
            if chunk.table_data and chunk.table_data.title:
                text = f"{chunk.table_data.title}\n{text}"
            if chunk.figure_data and chunk.figure_data.caption:
                text = f"{chunk.figure_data.caption}\n{text}"
            contents.append(text)

        self.bm25.fit(chunk_ids, contents)

        # Vector index
        self.vector.fit(chunk_ids, contents)

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        """Retrieve chunk by ID."""
        return self.chunks.get(chunk_id)

    def filter_chunk_ids(
        self,
        ticker: str | None = None,
        period: str | None = None,
        year: int | None = None,
        modal_type: ModalType | None = None,
    ) -> list[str]:
        """Filter chunk IDs based on metadata criteria."""
        valid_ids: list[str] = []
        for cid, chunk in self.chunks.items():
            if modal_type and chunk.modal_type != modal_type:
                continue
            doc = self.documents.get(chunk.doc_id)
            if doc:
                if ticker and doc.metadata.ticker and doc.metadata.ticker.upper() != ticker.upper():
                    continue
                if period and doc.metadata.period and doc.metadata.period.upper() != period.upper():
                    continue
                if year and doc.metadata.year and doc.metadata.year != year:
                    continue
            valid_ids.append(cid)
        return valid_ids
