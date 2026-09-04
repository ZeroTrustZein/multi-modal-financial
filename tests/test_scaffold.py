"""Scaffold and unit tests for multi-modal financial RAG pipeline."""

from click.testing import CliRunner

import multi_modal_financial
from multi_modal_financial import (
    AgentQuery,
    AgentResponse,
    Chunk,
    Document,
    DocumentMetadata,
    HybridIndex,
    HybridRetriever,
    ModalType,
    TableData,
    __version__,
)
from multi_modal_financial.agent.pipeline import FinancialRAGPipeline
from multi_modal_financial.agent.router import QueryIntent, QueryRouter
from multi_modal_financial.cli.main import cli
from multi_modal_financial.grounding.verifier import GroundingVerifier
from multi_modal_financial.indexing.bm25 import BM25Index
from multi_modal_financial.indexing.vector import DenseVectorIndex
from multi_modal_financial.parsing.extractor import FinancialDocumentParser
from multi_modal_financial.parsing.figure_parser import FigureParser
from multi_modal_financial.parsing.table_parser import TableParser
from multi_modal_financial.retrieval.reranker import FinancialReranker


def test_package_metadata():
    """Verify package version and export manifests."""
    assert __version__ == "0.1.0"
    assert multi_modal_financial.__version__ == "0.1.0"
    assert issubclass(ModalType, str)


def test_types_and_table_markdown(sample_metadata: DocumentMetadata, sample_table: TableData):
    """Test domain data types and markdown table serialization."""
    md = sample_table.to_markdown()
    assert "| Category | 2024 | 2025 |" in md
    assert "Operating Income" in md
    assert "**Table: Consolidated Statements of Operations**" in md

    chunk = Chunk(
        chunk_id="chk_01",
        doc_id=sample_metadata.doc_id,
        modal_type=ModalType.TABLE,
        content=md,
        table_data=sample_table,
    )
    assert chunk.modal_type == ModalType.TABLE
    assert chunk.table_data is not None
    assert chunk.table_data.scale == "millions"


def test_parsers():
    """Test table parser, figure parser, and financial document parser."""
    table_raw = """
    | Metric | Q1 | Q2 |
    | --- | --- | --- |
    | Revenue | $500M | $600M |
    """
    tbl = TableParser.parse_markdown_table(table_raw, title="Quarterly Growth")
    assert tbl.headers == ["Metric", "Q1", "Q2"]
    assert len(tbl.rows) == 1
    assert tbl.rows[0][0] == "Revenue"

    fig_raw = "Figure 1: Revenue by Region. Cloud: 450.0, Hardware: 250.0"
    fig = FigureParser.parse_figure_block(fig_raw, figure_id="fig_1")
    assert fig.caption == "Revenue by Region. Cloud: 450.0, Hardware: 250.0"
    assert fig.data_points.get("Cloud") == 450.0

    raw_doc = """
    In fiscal year 2025, operating cash flows expanded substantially.

    | Metric | Amount |
    | Operating Cash Flow | $120B |

    Figure 2: Cash Allocation
    Capex: 35.0, Dividends: 15.0

    Diluted EPS: $4.50
    """
    parser = FinancialDocumentParser(chunk_size=100)
    doc = parser.parse_text(raw_doc)
    assert len(doc.chunks) >= 4
    modalities = {c.modal_type for c in doc.chunks}
    assert ModalType.TEXT in modalities
    assert ModalType.TABLE in modalities
    assert ModalType.FIGURE in modalities


def test_bm25_index():
    """Verify BM25 Okapi with financial numbers and terms."""
    bm25 = BM25Index()
    docs = [
        "Revenue increased 14.5% to $12,500 million in Q3.",
        "Operating expenses remained flat at $4,200 million.",
        "Net income grew to $2,300 million with EPS of $1.85.",
    ]
    bm25.fit(["d1", "d2", "d3"], docs)

    # Search for percentage & revenue
    res = bm25.search("revenue 14.5%")
    assert len(res) > 0
    assert res[0][0] == "d1"

    # Search for EPS
    res_eps = bm25.search("EPS $1.85")
    assert len(res_eps) > 0
    assert res_eps[0][0] == "d3"


def test_vector_and_hybrid_retrieval(sample_document: Document):
    """Test vector embedding search, hybrid index, and retriever."""
    v_index = DenseVectorIndex(dimension=64)
    v_index.fit(
        ["c1", "c2"], ["Apple smartphones and computers", "Operating income statement table"]
    )
    v_results = v_index.search("computers hardware")
    assert len(v_results) > 0
    assert v_results[0][0] == "c1"

    # Hybrid index integration
    h_index = HybridIndex(dimension=64)
    h_index.index_document(sample_document)

    retriever = HybridRetriever(h_index)
    q = AgentQuery(query_str="What was operating income for fiscal 2025?", top_k=2)
    hits = retriever.retrieve(q, use_rrf=True)
    assert len(hits) > 0
    assert any(h.chunk.chunk_id == "aapl_c2" for h in hits)


def test_reranker_and_grounding(sample_document: Document):
    """Test domain-specific reranking and citation grounding."""
    h_index = HybridIndex(dimension=64)
    h_index.index_document(sample_document)
    retriever = HybridRetriever(h_index)

    hits = retriever.retrieve(AgentQuery(query_str="operating income 2025", top_k=3))
    reranker = FinancialReranker()
    reranked = reranker.rerank("Operating Income 2025", hits)
    assert len(reranked) > 0
    assert reranked[0].rank == 1

    # Citation grounding & verification
    verifier = GroundingVerifier(confidence_threshold=0.5)
    source_chunks = [sc.chunk for sc in reranked]

    # Valid claim with existing numbers
    valid_claim = "Operating Income reached $123,216 in 2025."
    verdict_valid = verifier.verify_claim(valid_claim, source_chunks)
    assert verdict_valid.is_supported is True
    assert verdict_valid.support_score > 0.5

    # Hallucinated claim with missing numbers
    hallucinated_claim = "Operating Income was $999,999 in 2025."
    verdict_fake = verifier.verify_claim(hallucinated_claim, source_chunks)
    assert "Warning: Numbers" in verdict_fake.reasoning


def test_router():
    """Verify financial query intent routing."""
    router = QueryRouter()
    r1 = router.route("Compare operating income vs net sales in FY 2025 for AAPL")
    assert r1["intent"] == QueryIntent.COMPARATIVE_ANALYSIS
    assert r1["ticker"] == "AAPL"
    assert r1["year"] == 2025

    r2 = router.route("What are the key supply chain risks and headwinds?")
    assert r2["intent"] == QueryIntent.QUALITATIVE_RISK
    assert r2["preferred_modal"] == ModalType.TEXT


def test_pipeline_end_to_end(populated_pipeline: FinancialRAGPipeline):
    """Test full RAG pipeline query response."""
    resp = populated_pipeline.query("What was total net sales in 2025 for AAPL?")
    assert isinstance(resp, AgentResponse)
    assert len(resp.retrieved_chunks) > 0
    assert len(resp.citations) > 0
    assert len(resp.grounding_verdicts) > 0
    assert resp.execution_time_ms > 0


def test_cli_execution():
    """Verify Click CLI commands."""
    runner = CliRunner()

    res_info = runner.invoke(cli, ["info"])
    assert res_info.exit_code == 0
    assert "FinancialDocumentParser" in res_info.output

    res_query = runner.invoke(cli, ["query", "What was revenue for ACM in Q3?"])
    assert res_query.exit_code == 0
    assert "Citations & Grounding" in res_query.output

    res_bench = runner.invoke(cli, ["benchmark"])
    assert res_bench.exit_code == 0
    assert "Benchmark Results" in res_bench.output
