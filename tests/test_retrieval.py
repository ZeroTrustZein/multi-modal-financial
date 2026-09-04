"""Comprehensive unit tests for retrieval core logic: RRF, normalization, hybrid retrieval, and reranking."""

from __future__ import annotations

import pytest

from multi_modal_financial.indexing.hybrid import HybridIndex
from multi_modal_financial.retrieval.fusion import (
    HybridRetriever,
    min_max_normalize,
    reciprocal_rank_fusion,
    z_score_normalize,
)
from multi_modal_financial.retrieval.reranker import FinancialReranker
from multi_modal_financial.types import (
    AgentQuery,
    Chunk,
    Document,
    DocumentMetadata,
    ModalType,
    QueryIntent,
    TableData,
)


class TestFusionAlgorithms:
    """Unit tests for fusion and normalization algorithms."""

    def test_rrf_scoring(self) -> None:
        dense = [("doc1", 0.9), ("doc2", 0.8), ("doc3", 0.7)]
        sparse = [("doc2", 15.0), ("doc1", 12.0), ("doc4", 10.0)]

        fused = reciprocal_rank_fusion(dense, sparse, k=60)
        assert len(fused) == 4
        # doc1 and doc2 appear in both, so should have higher scores than doc3/doc4
        top_ids = [cid for cid, _ in fused[:2]]
        assert "doc1" in top_ids
        assert "doc2" in top_ids

    def test_min_max_normalize(self) -> None:
        scores = [10.0, 20.0, 30.0]
        norm = min_max_normalize(scores)
        assert norm[0] == 0.0
        assert norm[1] == 0.5
        assert norm[2] == 1.0

        empty_norm = min_max_normalize([])
        assert empty_norm == []

        identical = min_max_normalize([5.0, 5.0])
        assert identical == [1.0, 1.0]

    def test_z_score_normalize(self) -> None:
        scores = [10.0, 20.0, 30.0]
        z = z_score_normalize(scores)
        assert len(z) == 3
        assert round(sum(z), 4) == 0.0


class TestHybridRetriever:
    """Unit tests for HybridRetriever and benchmarking."""

    @pytest.fixture
    def setup_retriever(self) -> HybridRetriever:
        index = HybridIndex(dimension=32)
        meta = DocumentMetadata(doc_id="sec_doc", filename="filing.txt", ticker="TSLA", year=2025)

        c1 = Chunk(
            chunk_id="tsla_rev",
            doc_id="sec_doc",
            modal_type=ModalType.TEXT,
            content="Automotive revenues reached $25,182 million in Q4 2025.",
        )
        tbl = TableData(
            title="Production and Deliveries",
            headers=["Vehicle", "Delivered"],
            rows=[["Model 3/Y", "460,000"], ["Model S/X", "22,000"]],
        )
        c2 = Chunk(
            chunk_id="tsla_table",
            doc_id="sec_doc",
            modal_type=ModalType.TABLE,
            content="Vehicle production metrics",
            table_data=tbl,
        )
        doc = Document(doc_id="sec_doc", metadata=meta, chunks=[c1, c2])
        index.index_document(doc)
        return HybridRetriever(index)

    def test_retrieve_rrf(self, setup_retriever: HybridRetriever) -> None:
        query = AgentQuery(query_str="automotive revenues Q4 2025", top_k=2)
        results = setup_retriever.retrieve(query, use_rrf=True)
        assert len(results) > 0
        assert results[0].chunk.chunk_id == "tsla_rev"

    def test_retrieve_convex_alpha(self, setup_retriever: HybridRetriever) -> None:
        query = AgentQuery(query_str="vehicle deliveries model 3", alpha=0.7, top_k=2)
        results = setup_retriever.retrieve(query, use_rrf=False)
        assert len(results) > 0

    def test_benchmark_suite(self, setup_retriever: HybridRetriever) -> None:
        test_queries = [
            (AgentQuery(query_str="automotive revenues"), ["tsla_rev"]),
            (AgentQuery(query_str="vehicle deliveries"), ["tsla_table"]),
        ]
        bench = setup_retriever.benchmark(test_queries, k=2, use_rrf=True)
        assert bench.query_count == 2
        assert bench.recall_at_k > 0.0
        assert bench.mrr > 0.0
        assert bench.avg_latency_ms >= 0.0


class TestFinancialReranker:
    """Unit tests for FinancialReranker."""

    def test_entity_and_modality_boosting(self) -> None:
        c_text = Chunk(
            chunk_id="chk_txt",
            doc_id="d1",
            modal_type=ModalType.TEXT,
            content="General market overview for 2024.",
        )
        c_tbl = Chunk(
            chunk_id="chk_tbl",
            doc_id="d1",
            modal_type=ModalType.TABLE,
            content="Income Statement 2025: Operating Income was $45,000 million.",
        )

        from multi_modal_financial.types import ScoredChunk

        sc_text = ScoredChunk(chunk=c_text, score=0.5, rank=1)
        sc_tbl = ScoredChunk(chunk=c_tbl, score=0.5, rank=2)

        reranker = FinancialReranker()
        reranked = reranker.rerank(
            AgentQuery(
                query_str="operating income in 2025 table", intent=QueryIntent.METRIC_LOOKUP
            ),
            [sc_text, sc_tbl],
        )

        assert len(reranked) == 2
        # Table chunk should be boosted to rank 1 due to year match (2025), number match ($45,000), and table modality
        assert reranked[0].chunk.chunk_id == "chk_tbl"
        assert reranked[0].rank == 1
        assert reranked[0].modality_bonus > 0.0
