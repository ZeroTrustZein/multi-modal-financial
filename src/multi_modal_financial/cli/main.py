"""CLI Interface for Multi-Modal Financial RAG Pipeline."""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console

from multi_modal_financial import __version__
from multi_modal_financial.agent.pipeline import FinancialRAGPipeline
from multi_modal_financial.cli.formatters import (
    format_architecture_table,
    format_benchmark_table,
    format_citations_table,
    format_response_panel,
)
from multi_modal_financial.types import AgentQuery

console = Console()


@click.group()
@click.version_option(__version__, prog_name="multi-modal-financial")
def cli():
    """Multi-Modal Financial Document RAG CLI."""
    pass


@cli.command()
def info():
    """Display pipeline configuration and environment details."""
    table = format_architecture_table()
    console.print(table)


@cli.command()
@click.argument("file_path", type=click.Path(exists=True))
@click.option("--ticker", help="Stock ticker symbol (e.g. AAPL, MSFT)", default=None)
def ingest(file_path: str, ticker: str | None):
    """Ingest a financial document (text or PDF) into the pipeline."""
    pipeline = FinancialRAGPipeline()
    path = Path(file_path)

    if path.is_dir():
        count = 0
        for f in path.glob("*.*"):
            if f.suffix.lower() in [".txt", ".md", ".pdf"]:
                pipeline.ingest_file(f, ticker=ticker)
                count += 1
        console.print(
            f"[bold green]Successfully ingested {count} documents from {file_path}[/bold green]"
        )
    else:
        doc = pipeline.ingest_file(path, ticker=ticker)
        console.print(
            f"[bold green]Ingested '{doc.doc_id}' with {len(doc.chunks)} chunks.[/bold green]"
        )


@cli.command()
@click.argument("query_text")
@click.option("--ticker", help="Filter by ticker symbol", default=None)
@click.option("--top-k", default=3, help="Number of chunks to retrieve")
@click.option("--alpha", default=0.5, help="Hybrid weight (0.0 BM25 only, 1.0 dense only)")
def query(query_text: str, ticker: str | None, top_k: int, alpha: float):
    """Query the financial RAG pipeline."""
    pipeline = FinancialRAGPipeline()

    # Pre-seed with sample financial disclosure for standalone testing
    sample_text = """
    Q3 2025 Financial Summary for ACME Corp (Ticker: ACM):
    Total revenue for the third quarter was $1,250 million, up 15% year-over-year.
    Gross profit reached $520 million, yielding a gross margin of 41.6%.
    Operating income was $280 million compared to $240 million in Q3 2024.

    | Metric | Q3 2024 | Q3 2025 | YoY Change |
    | Revenue | $1,087M | $1,250M | +15.0% |
    | Net Income | $185M | $215M | +16.2% |
    | Free Cash Flow | $140M | $175M | +25.0% |

    Figure 1: Quarterly Revenue Breakdown by Segment
    Cloud: 650.0, Enterprise: 400.0, Consumer: 200.0
    """
    pipeline.ingest_text(sample_text, doc_id="acme_q3_2025", ticker="ACM", period="Q3", year=2025)

    agent_q = AgentQuery(
        query_str=query_text,
        ticker_filter=ticker,
        top_k=top_k,
        alpha=alpha,
    )

    resp = pipeline.query(agent_q)

    console.print(format_response_panel(resp.answer, resp.execution_time_ms))
    console.print(format_citations_table(resp.citations))


@cli.command()
def benchmark():
    """Run baseline retrieval benchmark across BM25, Dense, and Hybrid modes."""
    pipeline = FinancialRAGPipeline()

    # Ingest synthetic benchmark corpus
    docs = [
        (
            "doc_rev",
            "Apple reported quarterly revenue of $94.9 billion, up 6 percent year-over-year.",
            "AAPL",
        ),
        (
            "doc_margin",
            "Services gross margin reached 74.0 percent, an all-time record for the segment.",
            "AAPL",
        ),
        (
            "doc_msft_cloud",
            "Microsoft Cloud revenue exceeded $38.9 billion, driven by Azure AI expansion.",
            "MSFT",
        ),
        (
            "doc_nvda_dc",
            "NVIDIA Data Center revenue was $30.8 billion, representing 112 percent growth YoY.",
            "NVDA",
        ),
        (
            "doc_cf",
            "Cash generated from operating activities totaled $26.8 billion for the quarter.",
            "AAPL",
        ),
    ]
    for did, txt, tkr in docs:
        pipeline.ingest_text(txt, doc_id=did, ticker=tkr)

    test_queries = [
        ("What was Services gross margin for Apple?", "doc_margin"),
        ("How much was NVIDIA Data Center revenue?", "doc_nvda_dc"),
        ("Microsoft Cloud quarterly performance", "doc_msft_cloud"),
    ]

    console.print("\n[bold]Running Retrieval Benchmark...[/bold]\n")

    # Evaluate BM25
    bm25_correct = 0
    mrr_bm25 = 0.0
    for q, target in test_queries:
        bm25_res = pipeline.retriever.retrieve(
            AgentQuery(query_str=q, top_k=3, alpha=0.0), use_rrf=False
        )
        rank = next((i + 1 for i, item in enumerate(bm25_res) if item.chunk.doc_id == target), 0)
        if rank == 1:
            bm25_correct += 1
        if rank > 0:
            mrr_bm25 += 1.0 / rank

    # Evaluate Hybrid RRF
    hybrid_correct = 0
    mrr_hybrid = 0.0
    for q, target in test_queries:
        hybrid_resp = pipeline.query(q, top_k=3)
        rank = next(
            (
                i + 1
                for i, item in enumerate(hybrid_resp.retrieved_chunks)
                if item.chunk.doc_id == target
            ),
            0,
        )
        if rank == 1:
            hybrid_correct += 1
        if rank > 0:
            mrr_hybrid += 1.0 / rank

    total = len(test_queries)
    table = format_benchmark_table(
        bm25_r1=bm25_correct / total,
        bm25_mrr=mrr_bm25 / total,
        hybrid_r1=hybrid_correct / total,
        hybrid_mrr=mrr_hybrid / total,
    )
    console.print(table)


if __name__ == "__main__":
    cli()
