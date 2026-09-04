"""Comprehensive unit tests for agent core logic: router and end-to-end RAG pipeline."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from multi_modal_financial.agent.pipeline import FinancialRAGPipeline
from multi_modal_financial.agent.router import QueryRouter
from multi_modal_financial.types import (
    AgentQuery,
    AgentResponse,
    ModalType,
    QueryIntent,
)


class TestQueryRouter:
    """Unit tests for QueryRouter."""

    def test_routing_intents(self) -> None:
        router = QueryRouter()

        r_comp = router.route("Compare EBITDA between MSFT and GOOG in 2025")
        assert r_comp["intent"] == QueryIntent.COMPARATIVE_ANALYSIS
        assert r_comp["preferred_modal"] == ModalType.TABLE

        r_trend = router.route("What was the revenue growth trajectory over the last 3 years?")
        assert r_trend["intent"] == QueryIntent.TREND_CALCULATION

        r_risk = router.route("Discuss regulatory and litigation risks in European markets")
        assert r_risk["intent"] == QueryIntent.QUALITATIVE_RISK
        assert r_risk["preferred_modal"] == ModalType.TEXT
        assert r_risk["alpha"] == 0.6

        r_metric = router.route("What was the diluted EPS in Q3 2025?")
        assert r_metric["intent"] == QueryIntent.METRIC_LOOKUP
        assert r_metric["preferred_modal"] == ModalType.METRIC
        assert r_metric["year"] == 2025
        assert r_metric["period"] == "Q3"

    def test_ticker_extraction_and_stopwords(self) -> None:
        router = QueryRouter()

        r1 = router.route("Can we analyze AAPL performance?")
        assert r1["ticker"] == "AAPL"  # 'CAN' excluded from tickers

        r2 = router.route("What is NVDA guidance for next quarter?")
        assert r2["ticker"] == "NVDA"

    def test_build_agent_query(self) -> None:
        router = QueryRouter()
        q = router.build_agent_query("What was operating income for AMZN in 2024?", top_k=3)
        assert isinstance(q, AgentQuery)
        assert q.ticker_filter == "AMZN"
        assert q.year_filter == 2024
        assert q.top_k == 3


class TestFinancialRAGPipeline:
    """Unit tests for FinancialRAGPipeline."""

    @pytest.fixture
    def rag_pipeline(self) -> FinancialRAGPipeline:
        pipeline = FinancialRAGPipeline()
        text = """
        Apple Inc. Fiscal 2025 Form 10-K

        Total net sales for fiscal 2025 were $391,035 million.
        Operating income was $123,216 million with operating margin of 31.5%.

        | Segment | 2024 | 2025 |
        | Products | $298,085M | $302,500M |
        | Services | $85,200M | $88,535M |
        """
        pipeline.ingest_text(text, doc_id="aapl_10k_2025", ticker="AAPL", year=2025, period="FY")
        return pipeline

    def test_end_to_end_query(self, rag_pipeline: FinancialRAGPipeline) -> None:
        response = rag_pipeline.query("What was total net sales in 2025 for AAPL?")
        assert isinstance(response, AgentResponse)
        assert len(response.retrieved_chunks) > 0
        assert len(response.citations) > 0
        assert len(response.grounding_verdicts) > 0
        assert response.execution_time_ms > 0
        assert response.overall_confidence > 0.5
        assert "391,035" in response.answer or "391035" in response.answer

    def test_empty_query_response(self) -> None:
        pipeline = FinancialRAGPipeline()
        response = pipeline.query("Nonexistent company earnings")
        assert "No relevant financial disclosures found" in response.answer
        assert len(response.retrieved_chunks) == 0

    def test_ingest_directory(self) -> None:
        pipeline = FinancialRAGPipeline()
        with tempfile.TemporaryDirectory() as tmpdir:
            f1 = Path(tmpdir) / "doc1.txt"
            f1.write_text("Revenue was $100M in 2024.", encoding="utf-8")
            f2 = Path(tmpdir) / "doc2.md"
            f2.write_text("EBITDA was $25M in 2025.", encoding="utf-8")

            ingested = pipeline.ingest_directory(tmpdir, pattern="*.*", default_ticker="ACME")
            assert len(ingested) == 2
            assert pipeline.index.total_documents() == 2
            assert pipeline.index.total_chunks() == 2
