"""Comprehensive unit tests for retrieval core logic: RRF, normalization, hybrid retrieval, and reranking."""

from __future__ import annotations

import pytest

from multi_modal_financial.indexing.hybrid import HybridIndex
from multi_modal_financial.interfaces import (
    CrossEncoderProtocol,
    RerankerProtocol,
    RetrieverProtocol,
)
from multi_modal_financial.retrieval.fusion import (
    HybridRetriever,
    min_max_normalize,
    reciprocal_rank_fusion,
    z_score_normalize,
)
from multi_modal_financial.retrieval.reranker import FinancialCrossEncoder, FinancialReranker
from multi_modal_financial.types import (
    AgentQuery,
    Chunk,
    Document,
    DocumentMetadata,
    HybridSearchConfig,
    ModalType,
    QueryIntent,
    RerankerConfig,
    RerankerStrategy,
    RetrievalStrategy,
    ScoredChunk,
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


class TestFinancialCrossEncoder:
    """Unit tests for FinancialCrossEncoder implementation."""

    def test_protocol_conformance(self) -> None:
        encoder = FinancialCrossEncoder()
        assert isinstance(encoder, CrossEncoderProtocol)

    def test_predict_scoring(self) -> None:
        encoder = FinancialCrossEncoder()
        pairs = [
            ("operating revenue for 2025", "Operating revenue reached $150,000 million in fiscal year 2025."),
            ("operating revenue for 2025", "The weather was sunny in Seattle yesterday afternoon."),
        ]
        scores = encoder.predict(pairs)
        assert len(scores) == 2
        assert all(0.0 <= s <= 1.0 for s in scores)
        # Relevant financial pair should score higher than unrelated passage
        assert scores[0] > scores[1]

    def test_predict_empty_pairs(self) -> None:
        encoder = FinancialCrossEncoder()
        assert encoder.predict([]) == []


class TestFinancialRerankerExtended:
    """Unit tests for neural cross-encoder and hybrid reranker strategies."""

    @pytest.fixture
    def candidates(self) -> list[ScoredChunk]:
        c1 = Chunk(
            chunk_id="c_text",
            doc_id="doc1",
            modal_type=ModalType.TEXT,
            content="Overview of competitive dynamics in the cloud industry.",
        )
        c2 = Chunk(
            chunk_id="c_metric",
            doc_id="doc1",
            modal_type=ModalType.METRIC,
            content="Net sales totaled $391,035 million for fiscal 2025.",
        )
        c3 = Chunk(
            chunk_id="c_table",
            doc_id="doc1",
            modal_type=ModalType.TABLE,
            content="Consolidated Operations: Net Sales $391,035M, Cost of sales $210,352M.",
        )
        return [
            ScoredChunk(chunk=c1, score=0.4, rank=1),
            ScoredChunk(chunk=c2, score=0.5, rank=2),
            ScoredChunk(chunk=c3, score=0.6, rank=3),
        ]

    def test_protocol_conformance(self) -> None:
        reranker = FinancialReranker()
        assert isinstance(reranker, RerankerProtocol)

    def test_cross_encoder_strategy(self, candidates: list[ScoredChunk]) -> None:
        config = RerankerConfig(strategy=RerankerStrategy.CROSS_ENCODER, top_k=2)
        reranker = FinancialReranker(config=config)
        results = reranker.rerank("Net sales in fiscal 2025", candidates)

        assert len(results) == 2
        assert results[0].cross_encoder_score is not None
        assert results[0].rank == 1
        assert len(reranker.last_explanations) == 2
        exp = reranker.last_explanations[0]
        assert exp.final_rank == 1
        assert exp.cross_encoder_score is not None

    def test_hybrid_strategy_with_explanations(self, candidates: list[ScoredChunk]) -> None:
        config = RerankerConfig(
            strategy=RerankerStrategy.HYBRID,
            cross_encoder_weight=0.6,
            heuristic_weight=0.4,
            top_k=3,
        )
        reranker = FinancialReranker(config=config)
        query = AgentQuery(query_str="Net sales in 2025 table", top_k=3)
        results = reranker.rerank(query, candidates)

        assert len(results) == 3
        # Table or metric with exact number & year match should be top
        assert results[0].chunk.chunk_id in ("c_table", "c_metric")
        assert results[0].explanation is not None
        assert "Hybrid:" in results[0].explanation
        assert len(reranker.last_explanations) == 3
        assert reranker.last_explanations[0].heuristic_score is not None
        assert reranker.last_explanations[0].cross_encoder_score is not None

    def test_score_threshold_filtering(self, candidates: list[ScoredChunk]) -> None:
        config = RerankerConfig(
            strategy=RerankerStrategy.HEURISTIC,
            score_threshold=1.5,  # Very high threshold to filter out low-score items
        )
        reranker = FinancialReranker(config=config)
        results = reranker.rerank("totally unrelated query", candidates)
        # Should only return items meeting threshold or empty
        for item in results:
            assert item.score >= 1.5

    def test_empty_candidates_handling(self) -> None:
        reranker = FinancialReranker()
        assert reranker.rerank("Query", []) == []
        assert reranker.last_explanations == []


class TestHybridRetrieverStrategies:
    """Unit tests for HybridRetriever with different RetrievalStrategy modes."""

    @pytest.fixture
    def setup_retriever(self) -> HybridRetriever:
        index = HybridIndex(dimension=32)
        meta = DocumentMetadata(doc_id="doc_aapl", filename="aapl.txt", ticker="AAPL", year=2025)
        c1 = Chunk(
            chunk_id="chunk_1",
            doc_id="doc_aapl",
            modal_type=ModalType.TEXT,
            content="Total net sales reached $391,035 million in fiscal year 2025.",
        )
        c2 = Chunk(
            chunk_id="chunk_2",
            doc_id="doc_aapl",
            modal_type=ModalType.TEXT,
            content="Operating expenses were $55,000 million for research and development.",
        )
        doc = Document(doc_id="doc_aapl", metadata=meta, chunks=[c1, c2])
        index.index_document(doc)
        return HybridRetriever(index)

    def test_retriever_protocol_conformance(self, setup_retriever: HybridRetriever) -> None:
        assert isinstance(setup_retriever, RetrieverProtocol)

    def test_dense_only_strategy(self, setup_retriever: HybridRetriever) -> None:
        config = HybridSearchConfig(strategy=RetrievalStrategy.DENSE, top_k=2)
        setup_retriever.config = config
        hits = setup_retriever.retrieve("net sales 2025", strategy=RetrievalStrategy.DENSE)
        assert len(hits) > 0
        assert hits[0].dense_score != 0.0

    def test_sparse_only_strategy(self, setup_retriever: HybridRetriever) -> None:
        hits = setup_retriever.retrieve("research and development", strategy=RetrievalStrategy.SPARSE)
        assert len(hits) > 0
        assert hits[0].sparse_score > 0.0
        assert hits[0].chunk.chunk_id == "chunk_2"

    def test_convex_blend_strategy(self, setup_retriever: HybridRetriever) -> None:
        hits = setup_retriever.retrieve("net sales", strategy=RetrievalStrategy.HYBRID_CONVEX, alpha=0.8)
        assert len(hits) > 0
        assert hits[0].rank == 1

    def test_string_query_input(self, setup_retriever: HybridRetriever) -> None:
        hits = setup_retriever.retrieve("Total net sales", top_k=1)
        assert len(hits) == 1
        assert hits[0].chunk.chunk_id == "chunk_1"
