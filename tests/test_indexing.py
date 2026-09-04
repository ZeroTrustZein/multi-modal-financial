"""Comprehensive unit tests for indexing core logic: BM25, Dense Vector, and Hybrid Index."""

from __future__ import annotations

import numpy as np
import pytest

from multi_modal_financial.indexing.bm25 import BM25Index
from multi_modal_financial.indexing.hybrid import HybridIndex
from multi_modal_financial.indexing.vector import DenseVectorIndex
from multi_modal_financial.types import (
    Chunk,
    Document,
    DocumentMetadata,
    FinancialMetric,
    FinancialStatementType,
    ModalType,
    TableData,
)


class TestBM25Index:
    """Unit tests for BM25 Okapi Index."""

    def test_financial_tokenization(self) -> None:
        text = "Operating loss was $(1,234.5M) representing -12.4% for AAPL in Q3 FY2025."
        tokens = BM25Index.tokenize(text)
        assert "-1234.5m" in tokens or "1234.5m" in tokens
        assert "aapl" in tokens
        assert "q3" in tokens
        assert "fy2025" in tokens
        assert "operating" in tokens

    def test_incremental_add_documents(self) -> None:
        index = BM25Index()
        index.fit(["d1"], ["Revenue was $100M in 2024."])
        assert index.corpus_size == 1

        index.add_documents(["d2"], ["Net income was $20M in 2025."])
        assert index.corpus_size == 2

        res = index.search("net income")
        assert len(res) > 0
        assert res[0][0] == "d2"

    def test_term_frequency_and_stats(self) -> None:
        index = BM25Index()
        index.fit(["d1"], ["Revenue revenue revenue profit"])
        assert index.get_doc_term_frequency("d1", "revenue") == 3
        assert index.get_doc_term_frequency("d1", "profit") == 1
        assert index.get_doc_term_frequency("d1", "nonexistent") == 0
        assert index.get_doc_term_frequency("unknown_doc", "revenue") == 0

        stats = index.corpus_statistics()
        assert stats["corpus_size"] == 1
        assert stats["vocabulary_size"] == 2


class TestDenseVectorIndex:
    """Unit tests for DenseVectorIndex."""

    def test_deterministic_projection_and_similarity(self) -> None:
        v_idx = DenseVectorIndex(dimension=32)
        v_idx.fit(
            ["c1", "c2"],
            ["High revenue cloud software sales", "Semiconductor chip hardware manufacturing"],
        )

        # Check normalization
        vec1 = v_idx.get_vector("c1")
        assert vec1 is not None
        assert np.isclose(np.linalg.norm(vec1), 1.0, atol=1e-4)

        # Pairwise similarity
        sim = v_idx.similarity("c1", "c2")
        assert sim is not None
        assert -1.0 <= sim <= 1.0

    def test_batch_search(self) -> None:
        v_idx = DenseVectorIndex(dimension=32)
        v_idx.fit(["c1", "c2"], ["Artificial intelligence GPUs", "Retail grocery stores"])

        batch_res = v_idx.batch_search(["GPUs hardware", "Supermarket food"], top_k=1)
        assert len(batch_res) == 2
        assert batch_res[0][0][0] == "c1"
        assert batch_res[1][0][0] == "c2"

    def test_incremental_add(self) -> None:
        v_idx = DenseVectorIndex(dimension=32)
        v_idx.fit(["c1"], ["Initial disclosure document"])
        assert len(v_idx.doc_ids) == 1

        v_idx.add_documents(["c2"], ["Subsequent earnings filing"])
        assert len(v_idx.doc_ids) == 2
        assert v_idx.get_vector("c2") is not None


class TestHybridIndex:
    """Unit tests for HybridIndex."""

    @pytest.fixture
    def populated_index(self) -> HybridIndex:
        index = HybridIndex(dimension=32)

        meta1 = DocumentMetadata(
            doc_id="doc_aapl", filename="aapl.txt", ticker="AAPL", year=2025, period="Q3"
        )
        tbl = TableData(
            title="Income Statement",
            headers=["Item", "2025"],
            rows=[["Revenue", "$90,000M"], ["Operating Income", "$28,000M"]],
            statement_type=FinancialStatementType.INCOME_STATEMENT,
        )
        c1 = Chunk(
            chunk_id="chk_aapl_1",
            doc_id="doc_aapl",
            modal_type=ModalType.TABLE,
            content="Table of Operations",
            table_data=tbl,
            metrics=[FinancialMetric.from_raw("Revenue", "$90,000M")],
            section_title="ITEM 8. FINANCIAL STATEMENTS",
        )
        doc1 = Document(doc_id="doc_aapl", metadata=meta1, chunks=[c1])

        meta2 = DocumentMetadata(
            doc_id="doc_msft", filename="msft.txt", ticker="MSFT", year=2024, period="FY"
        )
        c2 = Chunk(
            chunk_id="chk_msft_1",
            doc_id="doc_msft",
            modal_type=ModalType.TEXT,
            content="Cloud commercial revenue increased by 22% in constant currency.",
            section_title="MANAGEMENT DISCUSSION",
        )
        doc2 = Document(doc_id="doc_msft", metadata=meta2, chunks=[c2])

        index.index_document(doc1)
        index.index_document(doc2)
        return index

    def test_index_storage_and_counts(self, populated_index: HybridIndex) -> None:
        assert populated_index.total_documents() == 2
        assert populated_index.total_chunks() == 2
        assert populated_index.get_chunk("chk_aapl_1") is not None
        assert populated_index.get_document("doc_aapl") is not None

    def test_filter_chunk_ids(self, populated_index: HybridIndex) -> None:
        aapl_chunks = populated_index.filter_chunk_ids(ticker="AAPL")
        assert aapl_chunks == ["chk_aapl_1"]

        year_chunks = populated_index.filter_chunk_ids(year=2024)
        assert year_chunks == ["chk_msft_1"]

        table_chunks = populated_index.filter_chunk_ids(modal_type=ModalType.TABLE)
        assert table_chunks == ["chk_aapl_1"]

        statement_chunks = populated_index.filter_chunk_ids(
            statement_type=FinancialStatementType.INCOME_STATEMENT
        )
        assert statement_chunks == ["chk_aapl_1"]
