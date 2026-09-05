"""Comprehensive tests for edge cases, error conditions, and boundary values."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np

from multi_modal_financial.agent.pipeline import FinancialRAGPipeline
from multi_modal_financial.analytics.comparator import PeriodComparator
from multi_modal_financial.analytics.ratios import FinancialRatioCalculator, RatioSummary
from multi_modal_financial.data.cleaner import FinancialDataCleaner
from multi_modal_financial.data.validator import StatementReconciler
from multi_modal_financial.indexing.bm25 import BM25Index
from multi_modal_financial.indexing.hybrid import HybridIndex
from multi_modal_financial.indexing.vector import DenseVectorIndex
from multi_modal_financial.parsing.extractor import FinancialDocumentParser
from multi_modal_financial.parsing.figure_parser import FigureParser
from multi_modal_financial.parsing.table_parser import TableParser
from multi_modal_financial.pipeline.orchestrator import (
    FinancialPipelineOrchestrator,
)
from multi_modal_financial.retrieval.fusion import (
    HybridRetriever,
    min_max_normalize,
    reciprocal_rank_fusion,
    z_score_normalize,
)
from multi_modal_financial.types import (
    AgentQuery,
    AgentResponse,
    Chunk,
    Document,
    DocumentMetadata,
    FinancialMetric,
    GroundingVerdict,
    ModalType,
    TableData,
)


class TestTypesEdgeCases:
    """Edge cases for domain data structures and parsing helpers."""

    def test_financial_metric_trillions_and_thousands(self):
        m_t = FinancialMetric.from_raw(name="US GDP", raw_value="$28.5T")
        assert m_t.scale == "trillions"
        assert m_t.normalized_value == 28.5 * 1e12

        m_k = FinancialMetric.from_raw(name="Small Expense", raw_value="$450K")
        assert m_k.scale == "thousands"
        assert m_k.normalized_value == 450000.0

    def test_financial_metric_invalid_numeric_fallback(self):
        m = FinancialMetric.from_raw(name="Status", raw_value="Not Disclosed")
        assert m.value is None
        assert m.normalized_value is None
        assert m.formatted() == "Not Disclosed"

    def test_document_metadata_period_only(self):
        meta = DocumentMetadata(doc_id="doc_q1", filename="q1.txt", period="Q1", ticker="XYZ")
        summary = meta.header_summary()
        assert "XYZ" in summary and "Q1" in summary

        meta_year = DocumentMetadata(
            doc_id="doc_2024", filename="2024.txt", year=2024, ticker="XYZ"
        )
        assert "XYZ" in meta_year.header_summary() and "2024" in meta_year.header_summary()

    def test_table_data_empty_and_ragged_edges(self):
        tbl = TableData(headers=[], rows=[])
        assert tbl.to_records() == []
        assert tbl.extract_metrics() == []
        assert tbl.get_column("nonexistent") == []
        assert tbl.get_cell(10, 10) is None

        tbl_ragged = TableData(
            headers=["Item", "2024"],
            rows=[
                ["--- separator ---", "---"],
                ["Short Row"],
                ["Valid Row", "-"],
                ["Valid Row 2", "N/A"],
                ["Valid Row 3", "$500"],
            ],
        )
        metrics = tbl_ragged.extract_metrics()
        assert len(metrics) == 1
        assert metrics[0].name == "Valid Row 3"
        assert metrics[0].value == 500.0

    def test_chunk_numeric_values_empty_and_invalid(self):
        c = Chunk(
            chunk_id="c_empty", doc_id="d1", modal_type=ModalType.TEXT, content="No numbers here."
        )
        assert c.extract_numbers() == []

    def test_agent_response_unsupported_and_summary(self):
        resp = AgentResponse(
            query="Revenue query",
            answer="Revenue was $100M.",
            citations=[],
            grounding_verdicts=[],
            retrieved_chunks=[],
        )
        # Empty verdicts should return False for fully grounded
        assert resp.is_fully_grounded() is False
        assert resp.unsupported_claims() == []
        assert "Query: 'Revenue query'" in resp.summary()

        # Add an unsupported claim
        fake_verdict = GroundingVerdict(
            claim="Fake claim",
            is_supported=False,
            support_score=0.1,
            missing_entities=["$999M"],
        )
        resp.grounding_verdicts = [fake_verdict]
        assert resp.is_fully_grounded() is False
        assert len(resp.unsupported_claims()) == 1

        d = resp.to_dict()
        assert d["query"] == "Revenue query"
        restored = AgentResponse.from_dict(d)
        assert restored.query == resp.query


class TestCleanerEdgeCases:
    """Edge cases for text cleaner and table repair utilities."""

    def test_clean_document_text_empty(self):
        assert FinancialDataCleaner.clean_document_text("") == ""

    def test_clean_cell_footnotes(self):
        assert FinancialDataCleaner.clean_cell_footnotes("$100 [1]") == "$100"
        assert FinancialDataCleaner.clean_cell_footnotes("$200 (a)") == "$200"
        assert FinancialDataCleaner.clean_cell_footnotes("$300 *") == "$300"
        assert FinancialDataCleaner.clean_cell_footnotes("$400 †") == "$400"

    def test_repair_markdown_table_empty_and_non_table(self):
        assert FinancialDataCleaner.repair_markdown_table("") == ""
        assert (
            FinancialDataCleaner.repair_markdown_table("Just regular text\nNo table here")
            == "Just regular text\nNo table here"
        )

    def test_repair_markdown_table_ragged_lines(self):
        ragged = """
        | Col A | Col B |
        | --- | --- |
        | Value 1 | Value 2 | Extra 3 |
        | Value 4 |
        """
        repaired = FinancialDataCleaner.repair_markdown_table(ragged)
        lines = repaired.strip().splitlines()
        assert len(lines) == 4
        assert lines[0].count("|") >= 4


class TestParsingEdgeCases:
    """Edge cases for table and figure parsers."""

    def test_figure_parser_currencies_and_scales(self):
        eur_text = "Figure 1: Capital Expenditure in Europe. Total capex reached €500 million."
        scale, unit = FigureParser.detect_scale_and_unit(eur_text)
        assert scale == "millions"
        assert unit == "EUR"

        gbp_text = "Figure 2: UK Revenue. £120 thousand."
        scale_gbp, unit_gbp = FigureParser.detect_scale_and_unit(gbp_text)
        assert scale_gbp == "thousands"
        assert unit_gbp == "GBP"

        jpy_text = "Figure 3: Japan segment generated ¥10 billion."
        scale_jpy, unit_jpy = FigureParser.detect_scale_and_unit(jpy_text)
        assert scale_jpy == "billions"
        assert unit_jpy == "JPY"

    def test_figure_parser_bracket_caption(self):
        bracket_text = """
        [Figure 1: Global Segment Revenue Distribution]
        Cloud: 450.0
        Hardware: 250.0
        Services: 150.0
        """
        fig = FigureParser.parse_figure_block(bracket_text, figure_id="fig_bracket")
        assert fig.caption is not None
        assert fig.caption.rstrip("]") == "Global Segment Revenue Distribution"
        assert len(fig.data_points) == 3
        assert fig.data_points["Cloud"] == 450.0

    def test_extractor_narrative_slicing(self):
        """Verify long paragraphs exceed chunk_size and trigger multi-chunk slicing."""
        parser = FinancialDocumentParser(chunk_size=20, chunk_overlap=5)
        long_paragraph = " ".join([f"word{i}" for i in range(75)])
        doc = parser.parse_text(long_paragraph)
        assert len(doc.chunks) > 1
        assert all(c.modal_type == ModalType.TEXT for c in doc.chunks)

    def test_extractor_pdf_bytes_and_stream(self, tmp_path: Path):
        """Verify parse_pdf accepts bytes and BytesIO directly."""
        import pypdf

        # Create a simple 1-page PDF using pypdf
        writer = pypdf.PdfWriter()
        writer.add_blank_page(width=72, height=72)
        pdf_bytes_io = io.BytesIO()
        writer.write(pdf_bytes_io)
        pdf_bytes = pdf_bytes_io.getvalue()

        parser = FinancialDocumentParser()
        # Parse from bytes
        doc_from_bytes = parser.parse_pdf(pdf_bytes)
        assert doc_from_bytes.doc_id == "document"
        assert doc_from_bytes.metadata.page_count == 1

        # Parse from BytesIO with explicit metadata
        custom_meta = DocumentMetadata(doc_id="custom_pdf", filename="report.pdf", ticker="GOOG")
        pdf_stream = io.BytesIO(pdf_bytes)
        doc_from_stream = parser.parse_pdf(pdf_stream, metadata=custom_meta)
        assert doc_from_stream.doc_id == "custom_pdf"
        assert doc_from_stream.metadata.ticker == "GOOG"


class TestIndexingEdgeCases:
    """Edge cases for BM25 and Dense vector indexes."""

    def test_bm25_empty_and_unknown_terms(self):
        bm25 = BM25Index()
        # Search on empty index
        assert bm25.search("revenue") == []

        # Fit index
        bm25.fit(["d1"], ["Operating revenue grew 10%"])
        # Query with terms not in vocab
        assert bm25.search("unseen nonfinancial keywords xyz") == []

    def test_bm25_add_documents_empty_and_update(self):
        bm25 = BM25Index()
        # Add documents when corpus is initially empty
        bm25.add_documents(["d1"], ["Initial disclosure text"])
        assert bm25.corpus_size == 1

        # Update existing document
        bm25.add_documents(["d1"], ["Updated disclosure with higher revenue"])
        assert bm25.corpus_size == 1
        results = bm25.search("higher revenue")
        assert len(results) > 0
        assert results[0][0] == "d1"

    def test_dense_vector_empty_and_zero_norm(self):
        v_index = DenseVectorIndex(dimension=32)
        # Empty search
        assert v_index.search("query") == []
        assert v_index.batch_search(["query1", "query2"]) == [[], []]

        # Text with no valid characters (only punctuation/spaces)
        v_index.fit(["d1"], ["   "])
        assert len(v_index.doc_ids) == 1

    def test_dense_vector_precomputed_vectors(self):
        dim = 16
        v_index = DenseVectorIndex(dimension=dim)
        precomputed = np.random.randn(2, dim).astype(np.float32)
        v_index.fit(["c1", "c2"], ["text1", "text2"], precomputed_vectors=precomputed)
        assert v_index.vectors is not None
        assert v_index.vectors.shape == (2, dim)

        # Add documents with precomputed
        new_vecs = np.random.randn(1, dim).astype(np.float32)
        v_index.add_documents(["c3"], ["text3"], precomputed_vectors=new_vecs)
        assert v_index.vectors is not None
        assert v_index.vectors.shape == (3, dim)


class TestRetrievalFusionEdgeCases:
    """Edge cases for RRF, score normalization, and hybrid retriever."""

    def test_reciprocal_rank_fusion_partial_weights(self):
        list1 = [("d1", 10.0), ("d2", 8.0)]
        list2 = [("d2", 15.0), ("d3", 5.0)]
        # Provide single weight for 2 lists -> should extend with 1.0
        fused = reciprocal_rank_fusion(list1, list2, weights=[0.8])
        assert len(fused) == 3
        # d2 is in both lists, should rank highest
        assert fused[0][0] == "d2"

    def test_normalization_constant_and_empty(self):
        assert min_max_normalize([]) == []
        # Constant non-zero values
        assert min_max_normalize([5.0, 5.0, 5.0]) == [1.0, 1.0, 1.0]
        # Constant zero values
        assert min_max_normalize([0.0, 0.0]) == [0.0, 0.0]

        assert z_score_normalize([]) == []
        # Constant values variance = 0
        assert z_score_normalize([4.0, 4.0, 4.0]) == [0.0, 0.0, 0.0]

    def test_retriever_convex_alpha_allowed_ids(self, sample_document: Document):
        h_index = HybridIndex(dimension=32)
        h_index.index_document(sample_document)
        retriever = HybridRetriever(h_index)

        # Retrieve with convex combination (use_rrf=False) and ticker filter (which sets allowed_ids)
        q = AgentQuery(query_str="operating income", top_k=2, alpha=0.6, ticker_filter="AAPL")
        results = retriever.retrieve(q, use_rrf=False)
        assert len(results) > 0
        assert all(r.chunk.doc_id == sample_document.doc_id for r in results)

    def test_retriever_benchmark_empty_queries(self, sample_document: Document):
        h_index = HybridIndex(dimension=32)
        h_index.index_document(sample_document)
        retriever = HybridRetriever(h_index)

        bench = retriever.benchmark(test_queries=[])
        assert bench.query_count == 0
        assert bench.recall_at_1 == 0.0


class TestValidatorAndReconcilerEdgeCases:
    """Edge cases for table validator and accounting reconciler."""

    def test_balance_sheet_missing_total_assets(self):
        tbl = TableData(
            headers=["Item", "Amount"],
            rows=[["Total Liabilities", "$500"], ["Stockholders' Equity", "$500"]],
        )
        res = StatementReconciler.reconcile_balance_sheet(tbl)
        assert res.is_balanced is False
        assert res.expected_value == 0.0

    def test_balance_sheet_with_explicit_liabilities_and_equity_row(self):
        tbl = TableData(
            headers=["Item", "Amount"],
            rows=[
                ["Total Assets", "$1,000"],
                ["Total Liabilities and Stockholders' Equity", "$1,000"],
            ],
        )
        res = StatementReconciler.reconcile_balance_sheet(tbl)
        assert res.is_balanced is True
        assert res.difference == 0.0

    def test_balance_sheet_missing_liabilities_or_equity(self):
        tbl = TableData(
            headers=["Item", "Amount"],
            rows=[["Total Assets", "$1,000"], ["Cash", "$500"]],
        )
        res = StatementReconciler.reconcile_balance_sheet(tbl)
        assert res.is_balanced is False

    def test_income_statement_incomplete_rows(self):
        tbl = TableData(
            headers=["Metric", "Amount"],
            rows=[["Total Revenue", "$1,000"]],  # Missing COGS and Gross Profit
        )
        res = StatementReconciler.reconcile_income_statement(tbl)
        assert res.is_balanced is False


class TestAnalyticsEdgeCases:
    """Edge cases for financial ratios and period comparator."""

    def test_ratios_zero_and_negative_revenue(self):
        # Revenue is zero
        res_zero = FinancialRatioCalculator.compute_margins(revenue=0.0, gross_profit=10.0)
        assert res_zero == {}

        # Revenue is negative
        res_neg = FinancialRatioCalculator.compute_margins(revenue=-100.0, gross_profit=10.0)
        assert res_neg == {}

    def test_liquidity_zero_and_negative_liabilities(self):
        res = FinancialRatioCalculator.compute_liquidity(
            current_assets=500.0, current_liabilities=0.0
        )
        assert res == {}

        res_neg = FinancialRatioCalculator.compute_liquidity(
            current_assets=500.0, current_liabilities=-50.0
        )
        assert res_neg == {}

    def test_solvency_zero_and_negative_equity(self):
        res = FinancialRatioCalculator.compute_solvency(
            total_debt=200.0, total_equity=0.0, total_assets=500.0
        )
        assert "debt_to_equity" not in res
        assert res.get("debt_to_assets") == round(200.0 / 500.0, 3)

    def test_ratio_summary_to_dict(self):
        summary = RatioSummary(gross_margin_pct=45.2, current_ratio=1.85)
        d = summary.to_dict()
        assert d["gross_margin_pct"] == 45.2
        assert d["current_ratio"] == 1.85

    def test_comparator_base_zero(self):
        # Base value is 0.0 -> pct_change is None
        var = PeriodComparator.calculate_variance(
            "New Segment Revenue", base_val=0.0, compare_val=150.0
        )
        assert var.percentage_change is None
        assert var.trend == "UP"
        assert var.absolute_change == 150.0

        d = var.to_dict()
        assert d["metric_name"] == "New Segment Revenue"

    def test_comparator_table_columns_out_of_range(self):
        tbl = TableData(headers=["Metric", "2025"], rows=[["Revenue", "$100"]])
        # Ask for base_col=5 which is out of range
        vars_res = PeriodComparator.compare_table_columns(tbl, base_col=5, compare_col=1)
        assert vars_res == []

    def test_comparator_compare_documents(self):
        # Compare two documents with matching metric names
        doc1 = Document(
            doc_id="doc_2024",
            metadata=DocumentMetadata(doc_id="doc_2024", filename="2024.txt", year=2024),
            chunks=[
                Chunk(
                    chunk_id="c1",
                    doc_id="doc_2024",
                    modal_type=ModalType.METRIC,
                    content="Revenue: $1,000",
                    metrics=[FinancialMetric.from_raw(name="Revenue", raw_value="$1,000")],
                )
            ],
        )
        doc2 = Document(
            doc_id="doc_2025",
            metadata=DocumentMetadata(doc_id="doc_2025", filename="2025.txt", year=2025),
            chunks=[
                Chunk(
                    chunk_id="c2",
                    doc_id="doc_2025",
                    modal_type=ModalType.METRIC,
                    content="Revenue: $1,250",
                    metrics=[FinancialMetric.from_raw(name="Revenue", raw_value="$1,250")],
                )
            ],
        )
        variances = PeriodComparator.compare_documents(doc1, doc2)
        assert len(variances) == 1
        assert variances[0].metric_name == "Revenue"
        assert variances[0].percentage_change == 25.0
        assert variances[0].trend == "UP"


class TestPipelineOrchestratorEdgeCases:
    """Edge cases for RAG pipeline and full orchestrator."""

    def test_pipeline_ingest_directory_nonexistent(self, tmp_path: Path):
        pipeline = FinancialRAGPipeline()
        docs = pipeline.ingest_directory(tmp_path / "non_existent_dir")
        assert docs == []

    def test_pipeline_query_without_reranker(self, populated_pipeline: FinancialRAGPipeline):
        resp = populated_pipeline.query("Total net sales for AAPL", top_k=2, use_reranker=False)
        assert resp.answer != ""
        assert len(resp.retrieved_chunks) <= 2

    def test_orchestrator_ingest_paths_list(self, tmp_path: Path):
        f1 = tmp_path / "doc1.txt"
        f1.write_text("Revenue reached $1,000M.", encoding="utf-8")
        sub_dir = tmp_path / "sub"
        sub_dir.mkdir()
        f2 = sub_dir / "doc2.txt"
        f2.write_text("Net income reached $200M.", encoding="utf-8")

        orch = FinancialPipelineOrchestrator()
        docs = orch.ingest_files(paths=[f1, sub_dir], default_ticker="TEST")
        assert len(docs) == 2

    def test_orchestrator_query_cache(self, populated_pipeline: FinancialRAGPipeline):
        orch = FinancialPipelineOrchestrator()
        # Seed index
        orch.index = populated_pipeline.index
        orch.rag_pipeline.index = populated_pipeline.index

        # Query once: cache miss then fill
        q = AgentQuery(query_str="net sales 2025", top_k=2)
        resp1 = orch.run_query(q, use_cache=True)
        # Query second time: cache hit
        resp2 = orch.run_query(q, use_cache=True)

        assert resp1.answer == resp2.answer
        assert orch.query_cache is not None
        assert orch.query_cache.hits == 1
        assert len(orch.query_cache._cache) == 1

    def test_pipeline_ingest_file_pdf(self, tmp_path: Path):
        import pypdf

        writer = pypdf.PdfWriter()
        writer.add_blank_page(width=72, height=72)
        pdf_path = tmp_path / "filing.pdf"
        with open(pdf_path, "wb") as f:
            writer.write(f)

        pipeline = FinancialRAGPipeline()
        doc = pipeline.ingest_file(pdf_path, ticker="AAPL")
        assert doc.metadata.ticker == "AAPL"
        assert pipeline.index.total_documents() == 1

    def test_pipeline_ingest_directory_corrupt_file(self, tmp_path: Path, monkeypatch):
        pipeline = FinancialRAGPipeline()
        test_file = tmp_path / "valid.txt"
        test_file.write_text("Revenue was $500M.", encoding="utf-8")

        # Mock ingest_file to raise on second attempt to verify error skipping
        original_ingest_file = pipeline.ingest_file
        call_count = 0

        def flaky_ingest(file_path, ticker=None):
            nonlocal call_count
            call_count += 1
            if call_count > 1:
                raise ValueError("Simulated parse failure")
            return original_ingest_file(file_path, ticker=ticker)

        (tmp_path / "f2.txt").write_text("Failing document", encoding="utf-8")
        monkeypatch.setattr(pipeline, "ingest_file", flaky_ingest)

        docs = pipeline.ingest_directory(tmp_path)
        assert len(docs) == 1


class TestAdditionalParserAndAnalyticsEdgeCases:
    """Additional edge cases for table parser, doc extractor, and analytics."""

    def test_table_parser_additional_currencies(self):
        chf_text = "Operating results in CHF thousands."
        s_chf, u_chf = TableParser.detect_scale_and_unit(chf_text)
        assert u_chf == "CHF"

        cad_text = "Figures expressed in CAD millions."
        s_cad, u_cad = TableParser.detect_scale_and_unit(cad_text)
        assert u_cad == "CAD"

        aud_text = "Revenue in AUD billions."
        s_aud, u_aud = TableParser.detect_scale_and_unit(aud_text)
        assert u_aud == "AUD"

    def test_table_parser_title_above_table(self):
        raw_table = """
        Consolidated Statement of Operations
        | Metric | 2025 |
        | --- | --- |
        | Net Sales | $5,000 |
        """
        tbl = TableParser.parse_markdown_table(raw_table)
        assert tbl.title == "Consolidated Statement of Operations"
        assert len(tbl.rows) == 1

    def test_table_parser_empty_csv(self):
        tbl = TableParser.parse_delimited_table("", title="Empty CSV")
        assert tbl.title == "Empty CSV"
        assert tbl.headers == []
        assert tbl.rows == []

    def test_extractor_1900s_and_doc_types(self):
        parser = FinancialDocumentParser()

        text_8k = "Item 2.02 Results of Operations. Form 8-K Current Report 1999."
        doc_8k = parser.parse_text(text_8k, metadata=None)
        assert doc_8k.metadata.doc_type == "8-K"
        assert doc_8k.metadata.year == 1999

        text_er = "Press Release: ACME Reports Record Earnings Release for FY 2025."
        doc_er = parser.parse_text(text_er, metadata=None)
        assert doc_er.metadata.doc_type == "earnings_release"

    def test_comparator_nan_and_ragged_rows(self):

        m_nan = FinancialMetric(name="Missing", raw_value="NaN", value=float("nan"))
        m_valid = FinancialMetric(name="Missing", raw_value="$100", value=100.0)

        # Base is NaN
        res1 = PeriodComparator.compare_metrics([m_nan], [m_valid])
        assert res1 == []

        # Compare is NaN
        res2 = PeriodComparator.compare_metrics([m_valid], [m_nan])
        assert res2 == []

        # Ragged rows in compare_table_columns
        tbl = TableData(headers=["Metric", "2024", "2025"], rows=[["Short"]])
        res_ragged = PeriodComparator.compare_table_columns(tbl, base_col=1, compare_col=2)
        assert res_ragged == []

    def test_ratios_table_substring_asset_exclusions(self):
        tbl = TableData(
            headers=["Category", "2025"],
            rows=[
                ["Total Current Assets", "$500"],
                ["Total Assets", "$2,000"],
                ["Total Current Liabilities", "$300"],
                ["Total Liabilities", "$800"],
                ["Total Stockholders' Equity", "$1,200"],
                ["Net Income", "$240"],
            ],
        )
        summary = FinancialRatioCalculator.compute_from_table(tbl, col_idx=1)
        assert summary.current_ratio == round(500 / 300, 3)
        assert summary.debt_to_equity is not None or summary.return_on_equity_pct == 20.0
        assert summary.return_on_equity_pct == 20.0
