"""Tests for SEC EDGAR client and live data ingestion."""

from __future__ import annotations

from multi_modal_financial.data.edgar import SecEdgarClient


class TestSecEdgarClient:
    def test_cik_lookup_known_tickers(self) -> None:
        client = SecEdgarClient()
        assert client.get_cik("AAPL") == "0000320193"
        assert client.get_cik("msft") == "0000789019"
        assert client.get_cik("NVDA") == "0001045810"

    def test_cik_lookup_unlisted_fallback(self) -> None:
        client = SecEdgarClient()
        cik = client.get_cik("XYZCORP")
        assert len(cik) == 10
        assert cik.isdigit()

    def test_fetch_company_facts_structure(self) -> None:
        client = SecEdgarClient()
        facts = client.fetch_company_facts("AAPL")
        assert "facts" in facts
        assert "us-gaap" in facts["facts"]
        assert "Revenues" in facts["facts"]["us-gaap"]
        assert "OperatingIncomeLoss" in facts["facts"]["us-gaap"]

    def test_convert_facts_to_document(self) -> None:
        client = SecEdgarClient()
        doc = client.convert_facts_to_document("AAPL", year=2025, period="Q3")
        assert doc.metadata.ticker == "AAPL"
        assert doc.metadata.year == 2025
        assert doc.metadata.period == "Q3"
        assert doc.metadata.doc_type == "10-Q"
        assert len(doc.chunks) > 0
        assert len(doc.get_tables()) > 0

        # Verify revenue and operating income parsed
        metrics = doc.get_all_metrics()
        metric_names = [m.name.lower() for m in metrics]
        assert any("revenue" in name for name in metric_names)
