"""CLI Interface for Multi-Modal Financial RAG Pipeline."""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console

from multi_modal_financial import (
    FinancialPipelineOrchestrator,
    FinancialRAGPipeline,
    OrchestratorConfig,
    __version__,
)
from multi_modal_financial.cli.formatters import (
    format_architecture_table,
    format_audit_table,
    format_benchmark_table,
    format_cache_stats_table,
    format_citations_table,
    format_comparison_table,
    format_ratios_table,
    format_rerank_explanations_table,
    format_response_panel,
)
from multi_modal_financial.types import (
    AgentQuery,
    RerankerStrategy,
    RetrievalStrategy,
)

console = Console()


@click.group()
@click.version_option(__version__, prog_name="multi-modal-financial")
def cli():
    """Multi-Modal Financial Document RAG CLI."""
    pass


@cli.command()
@click.option(
    "--status",
    is_flag=True,
    default=False,
    help="Show live subsystem status and index diagnostics.",
)
def info(status: bool):
    """Display pipeline configuration and environment details."""
    table = format_architecture_table()
    console.print(table)
    if status:
        orchestrator = FinancialPipelineOrchestrator()
        stat_dict = orchestrator.status()
        sem_cache = stat_dict.get("semantic_cache")
        if sem_cache:
            console.print(format_cache_stats_table(sem_cache))


@cli.command()
@click.argument("file_path", type=click.Path(exists=True))
@click.option("--ticker", help="Stock ticker symbol (e.g. AAPL, MSFT)", default=None)
@click.option("--clean/--no-clean", default=True, help="Clean and normalize text")
@click.option("--validate/--no-validate", default=True, help="Validate financial tables")
def ingest(file_path: str, ticker: str | None, clean: bool, validate: bool):
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
@click.option(
    "--strategy",
    "--retrieval-strategy",
    "retrieval_strategy",
    type=click.Choice(["hybrid_rrf", "hybrid_convex", "dense", "sparse"], case_sensitive=False),
    default="hybrid_rrf",
    help="Retrieval fusion strategy",
)
@click.option(
    "--reranker-strategy",
    type=click.Choice(["heuristic", "cross_encoder", "hybrid"], case_sensitive=False),
    default="hybrid",
    help="Reranking model strategy",
)
@click.option("--rerank/--no-rerank", default=True, help="Enable or disable neural reranking")
@click.option(
    "--reranker-top-k", default=None, type=int, help="Number of chunks to retain after reranking"
)
@click.option(
    "--reranker-threshold", default=0.0, type=float, help="Minimum score threshold for reranker"
)
@click.option(
    "--semantic-cache/--no-semantic-cache",
    default=True,
    help="Enable or disable semantic vector caching",
)
@click.option(
    "--explain",
    is_flag=True,
    default=False,
    help="Display reranking score and explanation breakdown",
)
@click.option("--period", default=None, help="Filter by financial period (e.g. Q1, Q2, Q3, Q4, FY)")
@click.option("--year", default=None, type=int, help="Filter by reporting fiscal year")
def query(
    query_text: str,
    ticker: str | None,
    top_k: int,
    alpha: float,
    retrieval_strategy: str,
    reranker_strategy: str,
    rerank: bool,
    reranker_top_k: int | None,
    reranker_threshold: float,
    semantic_cache: bool,
    explain: bool,
    period: str | None,
    year: int | None,
):
    """Query the financial RAG pipeline with hybrid search, reranking, and grounding."""
    orchestrator = FinancialPipelineOrchestrator(
        config=OrchestratorConfig(enable_semantic_cache=semantic_cache)
    )

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
    orchestrator.rag_pipeline.ingest_text(
        sample_text, doc_id="acme_q3_2025", ticker="ACM", period="Q3", year=2025
    )

    strat_enum = RetrievalStrategy(retrieval_strategy.lower())
    rerank_enum = RerankerStrategy(reranker_strategy.lower())

    agent_q = AgentQuery(
        query_str=query_text,
        ticker_filter=ticker,
        period_filter=period,
        year_filter=year,
        top_k=top_k,
        alpha=alpha,
        retrieval_strategy=strat_enum,
        reranker_strategy=rerank_enum,
        use_reranker=rerank,
        reranker_top_k=reranker_top_k,
        reranker_threshold=reranker_threshold,
        use_semantic_cache=semantic_cache,
    )

    resp = orchestrator.run_query(agent_q, top_k=top_k, alpha=alpha, use_cache=semantic_cache)

    console.print(
        format_response_panel(
            answer=resp.answer,
            execution_time_ms=resp.execution_time_ms,
            cache_hit=resp.cache_hit,
            cache_type=resp.cache_type,
            confidence=resp.overall_confidence,
            reranker_strategy=reranker_strategy,
            retrieval_strategy=retrieval_strategy,
        )
    )
    console.print(format_citations_table(resp.citations))

    if explain:
        console.print(format_rerank_explanations_table(resp.rerank_explanations))


@cli.command()
@click.option("--top-k", default=3, help="Number of candidates to evaluate")
def benchmark(top_k: int):
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
            AgentQuery(
                query_str=q, top_k=top_k, alpha=0.0, retrieval_strategy=RetrievalStrategy.SPARSE
            ),
            use_rrf=False,
        )
        rank = next((i + 1 for i, item in enumerate(bm25_res) if item.chunk.doc_id == target), 0)
        if rank == 1:
            bm25_correct += 1
        if rank > 0:
            mrr_bm25 += 1.0 / rank

    # Evaluate Dense
    dense_correct = 0
    mrr_dense = 0.0
    for q, target in test_queries:
        dense_res = pipeline.retriever.retrieve(
            AgentQuery(
                query_str=q, top_k=top_k, alpha=1.0, retrieval_strategy=RetrievalStrategy.DENSE
            ),
            use_rrf=False,
        )
        rank = next((i + 1 for i, item in enumerate(dense_res) if item.chunk.doc_id == target), 0)
        if rank == 1:
            dense_correct += 1
        if rank > 0:
            mrr_dense += 1.0 / rank

    # Evaluate Hybrid RRF + Reranker
    hybrid_correct = 0
    mrr_hybrid = 0.0
    for q, target in test_queries:
        hybrid_resp = pipeline.query(q, top_k=top_k)
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
        dense_r1=dense_correct / total,
        dense_mrr=mrr_dense / total,
    )
    console.print(table)


@cli.command("cache-stats")
def cache_stats():
    """Display semantic and query cache performance statistics."""
    orchestrator = FinancialPipelineOrchestrator()
    stats = orchestrator.status()
    sem_stats = stats.get("semantic_cache") or {}
    console.print(format_cache_stats_table(sem_stats))


@cli.command("ratios")
@click.argument("file_path", type=click.Path(exists=True))
def ratios(file_path: str):
    """Compute financial ratios (margins, liquidity, solvency) from a document."""
    orchestrator = FinancialPipelineOrchestrator()
    docs = orchestrator.ingest_files(file_path)
    if not docs:
        console.print("[bold red]No valid document parsed from path.[/bold red]")
        return
    doc = docs[0]
    r_summary = orchestrator.analyze_document_ratios(doc.doc_id)
    console.print(format_ratios_table(r_summary, doc_id=doc.doc_id))


@cli.command("compare")
@click.argument("file_path_base", type=click.Path(exists=True))
@click.argument("file_path_compare", type=click.Path(exists=True))
def compare(file_path_base: str, file_path_compare: str):
    """Compare financial metrics and calculate period-over-period variance."""
    orchestrator = FinancialPipelineOrchestrator()
    docs_base = orchestrator.ingest_files(file_path_base)
    docs_comp = orchestrator.ingest_files(file_path_compare)
    if not docs_base or not docs_comp:
        console.print("[bold red]Failed to load one or both documents for comparison.[/bold red]")
        return
    doc1 = docs_base[0]
    doc2 = docs_comp[0]
    variances = orchestrator.compare_documents(doc1.doc_id, doc2.doc_id)
    console.print(format_comparison_table(variances, base_id=doc1.doc_id, comp_id=doc2.doc_id))


@cli.command("audit")
@click.argument("queries", nargs=-1)
@click.option(
    "--max-hallucination-rate", default=0.05, type=float, help="Max tolerable hallucination rate"
)
def audit(queries: tuple[str, ...], max_hallucination_rate: float):
    """Run factual grounding compliance audit across a batch of financial queries."""
    orchestrator = FinancialPipelineOrchestrator(
        config=OrchestratorConfig(max_hallucination_rate=max_hallucination_rate)
    )
    # Pre-seed sample document
    sample_text = """
    Q3 2025 Financial Summary for ACME Corp (Ticker: ACM):
    Total revenue for the third quarter was $1,250 million, up 15% year-over-year.
    Gross profit reached $520 million, yielding a gross margin of 41.6%.
    Operating income was $280 million compared to $240 million in Q3 2024.
    """
    orchestrator.rag_pipeline.ingest_text(
        sample_text, doc_id="acme_q3_2025", ticker="ACM", period="Q3", year=2025
    )
    query_list = (
        list(queries)
        if queries
        else [
            "What was revenue for ACM in Q3?",
            "What was gross profit and operating income?",
        ]
    )
    report = orchestrator.audit_queries(query_list)
    console.print(format_audit_table(report))


if __name__ == "__main__":
    cli()
