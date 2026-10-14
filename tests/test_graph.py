"""Tests for Financial Knowledge Graph and entity lineage engine."""

from __future__ import annotations

from multi_modal_financial.data.synthetic import SyntheticFilingGenerator
from multi_modal_financial.indexing.graph import (
    FinancialKnowledgeGraph,
)


class TestFinancialKnowledgeGraph:
    def test_add_nodes_and_edges(self) -> None:
        graph = FinancialKnowledgeGraph()
        n1 = graph.add_node("TICKER_AAPL", "TICKER", "AAPL", industry="Tech")
        n2 = graph.add_node("METRIC_REV", "METRIC", "Revenues", value=94900000000)
        edge = graph.add_edge("TICKER_AAPL", "METRIC_REV", "HAS_METRIC", weight=1.0)

        assert n1.id == "TICKER_AAPL"
        assert n2.properties["value"] == 94900000000
        assert edge.relation == "HAS_METRIC"
        assert len(graph.nodes) == 2
        assert len(graph.edges) == 1

    def test_get_neighbors(self) -> None:
        graph = FinancialKnowledgeGraph()
        graph.add_node("A", "ENTITY", "A")
        graph.add_node("B", "ENTITY", "B")
        graph.add_node("C", "ENTITY", "C")
        graph.add_edge("A", "B", "REL_1")
        graph.add_edge("A", "C", "REL_2")

        out_neighbors = graph.get_neighbors("A")
        assert len(out_neighbors) == 2
        rel1_neighbors = graph.get_neighbors("A", relation="REL_1")
        assert len(rel1_neighbors) == 1
        assert rel1_neighbors[0].id == "B"

    def test_query_multi_hop(self) -> None:
        graph = FinancialKnowledgeGraph()
        graph.add_node("AAPL", "TICKER", "AAPL")
        graph.add_node("P_2025_Q3", "PERIOD", "2025 Q3")
        graph.add_node("REV", "METRIC", "Total Revenue")

        graph.add_edge("AAPL", "P_2025_Q3", "REPORTED_IN")
        graph.add_edge("P_2025_Q3", "REV", "HAS_METRIC")

        results = graph.query_multi_hop("AAPL", ["REPORTED_IN", "HAS_METRIC"])
        assert len(results) == 1
        assert results[0].id == "REV"

    def test_build_from_document(self) -> None:
        gen = SyntheticFilingGenerator()
        doc = gen.generate_filing_document(ticker="MSFT", year=2025, period="Q3")

        graph = FinancialKnowledgeGraph()
        graph.build_from_document(doc)

        assert "TICKER_MSFT" in graph.nodes
        assert "PERIOD_MSFT_2025_Q3" in graph.nodes
        assert len(graph.edges) > 0

    def test_dict_serialization_roundtrip(self) -> None:
        graph = FinancialKnowledgeGraph()
        graph.add_node("NVDA", "TICKER", "NVDA", market_cap="3T")
        graph.add_node("DATA_CENTER", "SEGMENT", "Data Center", revenue=30800000000)
        graph.add_edge("NVDA", "DATA_CENTER", "SEGMENT_OF")

        serialized = graph.to_dict()
        reconstructed = FinancialKnowledgeGraph.from_dict(serialized)

        assert len(reconstructed.nodes) == 2
        assert len(reconstructed.edges) == 1
        assert reconstructed.nodes["NVDA"].properties["market_cap"] == "3T"

    def test_to_dot_export(self) -> None:
        graph = FinancialKnowledgeGraph()
        graph.add_node("A", "TICKER", "A")
        graph.add_node("B", "METRIC", "B")
        graph.add_edge("A", "B", "HAS_METRIC")

        dot = graph.to_dot()
        assert "digraph FinancialKnowledgeGraph" in dot
        assert '"A" -> "B"' in dot
