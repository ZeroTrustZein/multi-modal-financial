"""Unit test suite for multi-modal financial types and domain models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from multi_modal_financial.types import (
    AgentQuery,
    AgentResponse,
    Chunk,
    Citation,
    Currency,
    Document,
    DocumentMetadata,
    DocumentType,
    ExtractionFilter,
    FigureData,
    FinancialMetric,
    FinancialStatementType,
    GroundingStatus,
    GroundingVerdict,
    ModalType,
    ProvenanceRecord,
    QueryIntent,
    RetrievalBenchmarkResult,
    ScoredChunk,
    TableData,
    UnitScale,
    scale_multiplier,
)


class TestEnums:
    """Test enumeration values and string compatibility."""

    def test_modal_type(self) -> None:
        assert ModalType.TEXT == "text"
        assert ModalType.TABLE == "table"
        assert ModalType.FIGURE == "figure"
        assert ModalType.METRIC == "metric"
        assert ModalType.HEADER == "header"
        assert ModalType.FOOTNOTE == "footnote"
        assert ModalType.SUMMARY == "summary"
        assert issubclass(ModalType, str)

    def test_document_type(self) -> None:
        assert DocumentType.TEN_K == "10-K"
        assert DocumentType.TEN_Q == "10-Q"
        assert DocumentType.EIGHT_K == "8-K"
        assert DocumentType.EARNINGS_RELEASE == "earnings_release"
        assert DocumentType.ANALYST_REPORT == "analyst_report"
        assert DocumentType.PROSPECTUS == "prospectus"
        assert DocumentType.PROXY == "DEF_14A"
        assert DocumentType.PRESS_RELEASE == "press_release"
        assert DocumentType.TRANSCRIPT == "earnings_transcript"
        assert DocumentType.FILING == "filing"
        assert DocumentType.OTHER == "other"

    def test_statement_type(self) -> None:
        assert FinancialStatementType.INCOME_STATEMENT == "income_statement"
        assert FinancialStatementType.BALANCE_SHEET == "balance_sheet"
        assert FinancialStatementType.CASH_FLOW == "cash_flow"
        assert FinancialStatementType.COMPREHENSIVE_INCOME == "comprehensive_income"
        assert FinancialStatementType.STOCKHOLDERS_EQUITY == "stockholders_equity"
        assert FinancialStatementType.SEGMENT_METRICS == "segment_metrics"
        assert FinancialStatementType.NOTES == "notes"
        assert FinancialStatementType.UNKNOWN == "unknown"

    def test_currency(self) -> None:
        assert Currency.USD == "USD"
        assert Currency.EUR == "EUR"
        assert Currency.GBP == "GBP"
        assert Currency.JPY == "JPY"
        assert Currency.CAD == "CAD"
        assert Currency.CHF == "CHF"
        assert Currency.CNY == "CNY"
        assert Currency.AUD == "AUD"
        assert Currency.OTHER == "OTHER"

    def test_unit_scale(self) -> None:
        assert UnitScale.ONES == "ones"
        assert UnitScale.THOUSANDS == "thousands"
        assert UnitScale.MILLIONS == "millions"
        assert UnitScale.BILLIONS == "billions"
        assert UnitScale.TRILLIONS == "trillions"
        assert UnitScale.PERCENT == "percent"
        assert UnitScale.RATIO == "ratio"
        assert UnitScale.BPS == "bps"

    def test_query_intent(self) -> None:
        assert QueryIntent.METRIC_LOOKUP == "metric_lookup"
        assert QueryIntent.COMPARATIVE_ANALYSIS == "comparative_analysis"
        assert QueryIntent.QUALITATIVE_RISK == "qualitative_risk"
        assert QueryIntent.TREND_CALCULATION == "trend_calculation"
        assert QueryIntent.GENERAL == "general"

    def test_grounding_status(self) -> None:
        assert GroundingStatus.FULLY_SUPPORTED == "fully_supported"
        assert GroundingStatus.PARTIALLY_SUPPORTED == "partially_supported"
        assert GroundingStatus.UNSUPPORTED == "unsupported"
        assert GroundingStatus.CONTRADICTED == "contradicted"


class TestScaleMultiplier:
    """Test scale multiplier conversion function."""

    def test_standard_scales(self) -> None:
        assert scale_multiplier("ones") == 1.0
        assert scale_multiplier("thousands") == 1_000.0
        assert scale_multiplier("k") == 1_000.0
        assert scale_multiplier("millions") == 1_000_000.0
        assert scale_multiplier("m") == 1_000_000.0
        assert scale_multiplier("billions") == 1_000_000_000.0
        assert scale_multiplier("b") == 1_000_000_000.0
        assert scale_multiplier("trillions") == 1_000_000_000_000.0
        assert scale_multiplier("percent") == 0.01
        assert scale_multiplier("%") == 0.01
        assert scale_multiplier("bps") == 0.0001
        assert scale_multiplier(None) == 1.0
        assert scale_multiplier(UnitScale.MILLIONS) == 1_000_000.0
        assert scale_multiplier("unknown_unit") == 1.0


class TestFinancialMetric:
    """Test FinancialMetric creation, parsing, formatting, and serialization."""

    def test_from_raw_dollar_billions(self) -> None:
        metric = FinancialMetric.from_raw("Revenue", "$94.9B", year=2025, ticker="AAPL")
        assert metric.name == "Revenue"
        assert metric.raw_value == "$94.9B"
        assert metric.value == 94.9
        assert metric.unit == "USD"
        assert metric.scale == "billions"
        assert metric.normalized_value == 94_900_000_000.0
        assert metric.year == 2025
        assert metric.ticker == "AAPL"

    def test_from_raw_accounting_negative(self) -> None:
        metric = FinancialMetric.from_raw("Operating Loss", "(150.5M)", unit="USD")
        assert metric.value == -150.5
        assert metric.scale == "millions"
        assert metric.normalized_value == -150_500_000.0
        assert "-" in metric.formatted()

    def test_from_raw_standard_negative(self) -> None:
        metric = FinancialMetric.from_raw("Capex", "-$45.2B")
        assert metric.value == -45.2
        assert metric.scale == "billions"
        assert metric.normalized_value == -45_200_000_000.0

    def test_from_raw_percentage(self) -> None:
        metric = FinancialMetric.from_raw("Gross Margin", "42.5%")
        assert metric.value == 42.5
        assert metric.scale == "percent"
        assert metric.normalized_value == 0.425
        assert metric.formatted() == "42.5%"

    def test_from_raw_foreign_currencies(self) -> None:
        eur = FinancialMetric.from_raw("Revenue EU", "€500M")
        assert eur.unit == "EUR"
        assert eur.value == 500.0

        gbp = FinancialMetric.from_raw("Sales UK", "£25.5B")
        assert gbp.unit == "GBP"
        assert gbp.value == 25.5

        jpy = FinancialMetric.from_raw("Profit JP", "¥3,500")
        assert jpy.unit == "JPY"
        assert jpy.value == 3500.0

    def test_from_raw_non_numeric(self) -> None:
        metric = FinancialMetric.from_raw("Metric", "N/A")
        assert metric.value is None
        assert metric.normalized_value is None
        assert metric.formatted() == "N/A"

    def test_formatted_outputs(self) -> None:
        m_bil = FinancialMetric(
            name="Rev", raw_value="$1.25B", value=1.25, unit="USD", scale="billions"
        )
        assert m_bil.formatted() == "$1.25B"

        m_mil = FinancialMetric(
            name="Net", raw_value="$350M", value=350.0, unit="USD", scale="millions"
        )
        assert m_mil.formatted() == "$350.0M"

        m_thous = FinancialMetric(
            name="Op", raw_value="$4,500K", value=4500.0, unit="USD", scale="thousands"
        )
        assert m_thous.formatted() == "$4,500K"

        m_other = FinancialMetric(
            name="EPS", raw_value="$2.50", value=2.50, unit="USD", scale="ones"
        )
        assert m_other.formatted() == "$2.50"

    def test_dict_roundtrip(self) -> None:
        metric = FinancialMetric.from_raw(
            "EBITDA", "$1,200M", year=2024, period="Q3", ticker="NVDA"
        )
        d = metric.to_dict()
        assert d["name"] == "EBITDA"
        assert d["ticker"] == "NVDA"
        reconstructed = FinancialMetric.from_dict(d)
        assert reconstructed.value == metric.value
        assert reconstructed.ticker == metric.ticker


class TestDocumentMetadata:
    """Test DocumentMetadata validation, normalization, and formatting."""

    def test_metadata_normalization(self) -> None:
        meta = DocumentMetadata(
            doc_id="doc_1",
            filename="report.pdf",
            ticker="aapl ",
            period="q3 ",
            year=2025,
            page_count=0,
        )
        assert meta.ticker == "AAPL"
        assert meta.period == "Q3"
        assert meta.page_count == 1  # Clamped to minimum 1

    def test_metadata_year_validation(self) -> None:
        with pytest.raises(ValidationError):
            DocumentMetadata(doc_id="d", filename="f.pdf", year=1800)

        with pytest.raises(ValidationError):
            DocumentMetadata(doc_id="d", filename="f.pdf", year=2200)

        valid = DocumentMetadata(doc_id="d", filename="f.pdf", year=2024)
        assert valid.year == 2024

    def test_header_summary(self) -> None:
        m1 = DocumentMetadata(
            doc_id="d1", filename="f.txt", ticker="msft", period="fy", year=2025, doc_type="10-K"
        )
        assert m1.header_summary() == "[MSFT | FY 2025 | 10-K]"

        m2 = DocumentMetadata(doc_id="d2", filename="f.txt", ticker="goog", doc_type="8-K")
        assert m2.header_summary() == "[GOOG | 8-K]"

    def test_dict_roundtrip(self) -> None:
        m = DocumentMetadata(
            doc_id="d3", filename="f.txt", ticker="AMZN", year=2024, extra={"key": "val"}
        )
        d = m.to_dict()
        assert d["extra"]["key"] == "val"
        rebuilt = DocumentMetadata.from_dict(d)
        assert rebuilt.ticker == "AMZN"
        assert rebuilt.extra["key"] == "val"


class TestTableData:
    """Test TableData manipulation, markdown rendering, metric extraction, and cell lookups."""

    @pytest.fixture
    def populated_table(self) -> TableData:
        return TableData(
            title="Consolidated Income Statements",
            headers=["Line Item", "2024", "2025"],
            rows=[
                ["Total Net Sales", "$383,285", "$391,035"],
                ["Cost of Sales", "$214,137", "$210,352"],
                ["Operating Income", "$114,301", "$123,216"],
            ],
            unit="USD",
            scale="millions",
            statement_type=FinancialStatementType.INCOME_STATEMENT,
            footnotes=["Unaudited results."],
        )

    def test_table_properties(self, populated_table: TableData) -> None:
        assert populated_table.row_count == 3
        assert populated_table.col_count == 3

    def test_empty_table_markdown(self) -> None:
        empty = TableData()
        assert empty.to_markdown() == ""
        assert empty.row_count == 0
        assert empty.col_count == 0

    def test_markdown_rendering(self, populated_table: TableData) -> None:
        md = populated_table.to_markdown()
        assert "**Table: Consolidated Income Statements**" in md
        assert "| Line Item | 2024 | 2025 |" in md
        assert "| Total Net Sales | $383,285 | $391,035 |" in md
        assert "* Unaudited results." in md

    def test_to_records(self, populated_table: TableData) -> None:
        records = populated_table.to_records()
        assert len(records) == 3
        assert records[0]["Line Item"] == "Total Net Sales"
        assert records[0]["2024"] == "$383,285"
        assert records[0]["2025"] == "$391,035"

    def test_get_cell(self, populated_table: TableData) -> None:
        assert populated_table.get_cell(0, 0) == "Total Net Sales"
        assert populated_table.get_cell(0, 2) == "$391,035"
        assert populated_table.get_cell(2, 1) == "$114,301"
        assert populated_table.get_cell(10, 0) is None
        assert populated_table.get_cell(0, 10) is None

    def test_get_column(self, populated_table: TableData) -> None:
        col_2025 = populated_table.get_column("2025")
        assert col_2025 == ["$391,035", "$210,352", "$123,216"]

        col_idx_0 = populated_table.get_column(0)
        assert col_idx_0 == ["Total Net Sales", "Cost of Sales", "Operating Income"]

        assert populated_table.get_column("NonExistent") == []

    def test_find_metric_row(self, populated_table: TableData) -> None:
        row = populated_table.find_metric_row("operating income")
        assert row is not None
        assert row[0] == "Operating Income"
        assert row[2] == "$123,216"

        assert populated_table.find_metric_row("NonExistentMetric") is None

    def test_extract_metrics(self, populated_table: TableData) -> None:
        metrics = populated_table.extract_metrics()
        assert len(metrics) == 6  # 3 rows x 2 numerical columns

        sales_2025 = next(m for m in metrics if m.name == "Total Net Sales" and m.year == 2025)
        assert sales_2025.value == 391035.0
        assert sales_2025.scale == "millions"
        assert sales_2025.normalized_value == 391_035_000_000.0

    def test_dict_roundtrip(self, populated_table: TableData) -> None:
        d = populated_table.to_dict()
        assert d["statement_type"] == "income_statement"
        rebuilt = TableData.from_dict(d)
        assert rebuilt.row_count == populated_table.row_count
        assert rebuilt.title == populated_table.title


class TestFigureData:
    """Test FigureData methods, markdown formatting, and aggregation."""

    def test_figure_aggregation(self) -> None:
        fig = FigureData(
            figure_id="fig_geo",
            caption="Revenue by Region",
            chart_type="bar",
            summary_text="Americas led regional sales.",
            data_points={"Americas": 150.0, "Europe": 80.0, "APAC": 50.0},
            unit="USD",
            scale="millions",
        )
        assert fig.total_value() == 280.0
        assert fig.max_point() == ("Americas", 150.0)
        assert fig.min_point() == ("APAC", 50.0)

        md = fig.to_markdown()
        assert "**Figure: Revenue by Region**" in md
        assert "Americas led regional sales." in md
        assert "- Americas: 150 USD" in md

    def test_empty_figure(self) -> None:
        fig = FigureData(figure_id="fig_empty")
        assert fig.total_value() == 0.0
        assert fig.max_point() is None
        assert fig.min_point() is None

    def test_dict_roundtrip(self) -> None:
        fig = FigureData(figure_id="f1", caption="Test", data_points={"A": 10.0})
        d = fig.to_dict()
        rebuilt = FigureData.from_dict(d)
        assert rebuilt.figure_id == "f1"
        assert rebuilt.data_points["A"] == 10.0


class TestChunk:
    """Test Chunk methods, numerical extraction, word counting, and validation."""

    def test_chunk_computations(self) -> None:
        content = "Net income rose to $2,300 million with EPS of $1.85, an increase of 14.5%."
        chunk = Chunk(
            chunk_id="chk_1",
            doc_id="doc_1",
            modal_type=ModalType.TEXT,
            content=content,
            page_number=2,
        )
        assert chunk.word_count() == 14
        assert chunk.character_count() == len(content)
        assert chunk.has_numbers() is True

        numbers = chunk.extract_numbers()
        assert 2300.0 in numbers
        assert 1.85 in numbers
        assert 14.5 in numbers

    def test_chunk_page_number_clamping(self) -> None:
        chunk = Chunk(chunk_id="c0", doc_id="d0", content="test", page_number=0)
        assert chunk.page_number == 1

    def test_chunk_dict_roundtrip(self) -> None:
        chunk = Chunk(chunk_id="c1", doc_id="d1", content="Sample", metadata={"tag": "fin"})
        d = chunk.to_dict()
        rebuilt = Chunk.from_dict(d)
        assert rebuilt.chunk_id == "c1"
        assert rebuilt.metadata["tag"] == "fin"


class TestDocument:
    """Test Document container, modal filtering, table/figure collections, and JSON serialization."""

    def test_document_aggregations(self) -> None:
        meta = DocumentMetadata(doc_id="sec_10k", filename="10k.txt", ticker="AAPL", year=2025)
        tbl = TableData(title="T1", headers=["A", "B"], rows=[["1", "2"]])
        fig = FigureData(figure_id="fig_1", caption="F1", data_points={"X": 1.0})

        c1 = Chunk(chunk_id="c1", doc_id="sec_10k", modal_type=ModalType.TEXT, content="Summary")
        c2 = Chunk(
            chunk_id="c2",
            doc_id="sec_10k",
            modal_type=ModalType.TABLE,
            content="Table MD",
            table_data=tbl,
        )
        c3 = Chunk(
            chunk_id="c3",
            doc_id="sec_10k",
            modal_type=ModalType.FIGURE,
            content="Fig MD",
            figure_data=fig,
        )

        doc = Document(doc_id="sec_10k", metadata=meta, chunks=[c1, c2, c3])

        assert doc.chunk_count() == 3
        assert len(doc.chunks_by_modal(ModalType.TEXT)) == 1
        assert len(doc.chunks_by_modal(ModalType.TABLE)) == 1
        assert len(doc.chunks_by_modal(ModalType.FIGURE)) == 1
        assert len(doc.get_tables()) == 1
        assert len(doc.get_figures()) == 1

    def test_document_get_all_metrics(self) -> None:
        meta = DocumentMetadata(doc_id="sec_10k", filename="10k.txt")
        tbl = TableData(title="T1", headers=["Category", "2025"], rows=[["Net Sales", "$1,000M"]])
        c1 = Chunk(
            chunk_id="c1",
            doc_id="sec_10k",
            content="EPS disclosure",
            metrics=[FinancialMetric.from_raw("EPS", "$2.50")],
        )
        c2 = Chunk(chunk_id="c2", doc_id="sec_10k", content="Table", table_data=tbl)
        doc = Document(doc_id="sec_10k", metadata=meta, chunks=[c1, c2])
        all_metrics = doc.get_all_metrics()
        assert len(all_metrics) == 2
        assert any(m.name == "EPS" for m in all_metrics)
        assert any(m.name == "Net Sales" for m in all_metrics)

    def test_document_json_roundtrip(self) -> None:
        meta = DocumentMetadata(doc_id="doc_json", filename="f.txt", ticker="MSFT", year=2025)
        c1 = Chunk(chunk_id="c1", doc_id="doc_json", content="Test text")
        doc = Document(doc_id="doc_json", metadata=meta, chunks=[c1])

        json_str = doc.to_json()
        assert "MSFT" in json_str

        rebuilt = Document.from_json(json_str)
        assert rebuilt.doc_id == "doc_json"
        assert rebuilt.metadata.ticker == "MSFT"
        assert len(rebuilt.chunks) == 1


class TestScoredChunk:
    """Test ScoredChunk serialization and scoring attributes."""

    def test_scored_chunk_dict(self) -> None:
        chunk = Chunk(chunk_id="chk_s", doc_id="d1", content="Content")
        sc = ScoredChunk(
            chunk=chunk,
            score=0.885,
            dense_score=0.85,
            sparse_score=0.92,
            rank=1,
            rerank_score=0.95,
            modality_bonus=0.2,
        )
        d = sc.to_dict()
        assert d["score"] == 0.885
        assert d["rank"] == 1
        assert d["rerank_score"] == 0.95
        rebuilt = ScoredChunk.from_dict(d)
        assert rebuilt.score == 0.885
        assert rebuilt.chunk.chunk_id == "chk_s"


class TestCitation:
    """Test Citation formatting and serialization."""

    def test_citation_inline_format(self) -> None:
        cite = Citation(
            citation_id="cite_1",
            chunk_id="chk_22",
            doc_id="aapl_10k",
            page_number=4,
            quote="Total net sales were $391,035M.",
            confidence=0.98,
            matched_numbers=["391,035"],
        )
        assert cite.format_inline() == "[aapl_10k:p4#chk_22]"
        d = cite.to_dict()
        rebuilt = Citation.from_dict(d)
        assert rebuilt.citation_id == "cite_1"
        assert rebuilt.matched_numbers == ["391,035"]


class TestGroundingVerdict:
    """Test GroundingVerdict status, verdicts, and serialization."""

    def test_grounding_verdict(self) -> None:
        v = GroundingVerdict(
            claim="Operating income was $123,216M in 2025.",
            citations=[],
            is_supported=True,
            support_score=0.92,
            status=GroundingStatus.FULLY_SUPPORTED,
            reasoning="Exact numerical alignment.",
        )
        assert v.is_supported is True
        assert v.status == GroundingStatus.FULLY_SUPPORTED

        d = v.to_dict()
        rebuilt = GroundingVerdict.from_dict(d)
        assert rebuilt.support_score == 0.92


class TestAgentQuery:
    """Test AgentQuery validation, filters, and defaults."""

    def test_valid_query(self) -> None:
        q = AgentQuery(
            query_str="operating margin 2025",
            ticker_filter="aapl ",
            period_filter="q3 ",
            top_k=10,
            alpha=0.75,
        )
        assert q.ticker_filter == "AAPL"
        assert q.period_filter == "Q3"
        assert q.top_k == 10
        assert q.alpha == 0.75

    def test_query_bounds_validation(self) -> None:
        with pytest.raises(ValidationError):
            AgentQuery(query_str="test", top_k=0)

        with pytest.raises(ValidationError):
            AgentQuery(query_str="test", alpha=1.5)

        with pytest.raises(ValidationError):
            AgentQuery(query_str="test", alpha=-0.1)

    def test_query_dict_roundtrip(self) -> None:
        q = AgentQuery(query_str="test", ticker_filter="NVDA", intent=QueryIntent.METRIC_LOOKUP)
        d = q.to_dict()
        rebuilt = AgentQuery.from_dict(d)
        assert rebuilt.ticker_filter == "NVDA"
        assert rebuilt.intent == QueryIntent.METRIC_LOOKUP


class TestAgentResponse:
    """Test AgentResponse groundness checks, unsupported claims extraction, summary, and JSON roundtrip."""

    def test_response_groundedness_true(self) -> None:
        v1 = GroundingVerdict(claim="Claim 1", is_supported=True, support_score=0.9)
        v2 = GroundingVerdict(claim="Claim 2", is_supported=True, support_score=0.85)
        resp = AgentResponse(
            query="test",
            answer="Claim 1. Claim 2.",
            grounding_verdicts=[v1, v2],
            execution_time_ms=12.5,
            overall_confidence=0.875,
        )
        assert resp.is_fully_grounded() is True
        assert len(resp.unsupported_claims()) == 0
        summary = resp.summary()
        assert "Grounded: 2/2" in summary
        assert "Conf: 0.88" in summary

    def test_response_groundedness_false(self) -> None:
        v1 = GroundingVerdict(claim="Claim 1", is_supported=True, support_score=0.9)
        v2 = GroundingVerdict(claim="Claim 2", is_supported=False, support_score=0.2)
        resp = AgentResponse(
            query="test",
            answer="Claim 1. Claim 2.",
            grounding_verdicts=[v1, v2],
        )
        assert resp.is_fully_grounded() is False
        assert len(resp.unsupported_claims()) == 1
        assert resp.unsupported_claims()[0].claim == "Claim 2"

    def test_response_json_roundtrip(self) -> None:
        resp = AgentResponse(
            query="What was revenue?",
            answer="Revenue was $100M.",
            execution_time_ms=14.2,
            overall_confidence=0.95,
        )
        json_str = resp.to_json()
        assert "What was revenue?" in json_str
        rebuilt = AgentResponse.from_json(json_str)
        assert rebuilt.answer == "Revenue was $100M."
        assert rebuilt.execution_time_ms == 14.2


class TestSupportingTypes:
    """Test RetrievalBenchmarkResult, ProvenanceRecord, and ExtractionFilter."""

    def test_benchmark_result(self) -> None:
        bm = RetrievalBenchmarkResult(
            method_name="Hybrid RRF",
            recall_at_1=0.85,
            recall_at_k=0.95,
            mrr=0.89,
            ndcg=0.91,
            query_count=50,
            avg_latency_ms=15.4,
        )
        assert bm.recall_at_1 == 0.85
        d = bm.to_dict()
        rebuilt = RetrievalBenchmarkResult.from_dict(d)
        assert rebuilt.method_name == "Hybrid RRF"

    def test_provenance_record(self) -> None:
        rec = ProvenanceRecord(
            record_id="rec_01",
            doc_id="sec_10k",
            chunk_id="chk_05",
            page_number=3,
            modal_type=ModalType.TABLE,
            table_cell=(2, 1),
            source_snippet="$123,216",
            verified=True,
        )
        assert rec.verified is True
        assert rec.table_cell == (2, 1)
        d = rec.to_dict()
        rebuilt = ProvenanceRecord.from_dict(d)
        assert rebuilt.source_snippet == "$123,216"

    def test_extraction_filter(self) -> None:
        flt = ExtractionFilter(
            ticker="aapl ",
            period="FY",
            year=2025,
            modal_types=[ModalType.TABLE, ModalType.METRIC],
            min_confidence=0.7,
        )
        assert flt.ticker == "AAPL"
        assert len(flt.modal_types) == 2
        d = flt.to_dict()
        rebuilt = ExtractionFilter.from_dict(d)
        assert rebuilt.year == 2025
