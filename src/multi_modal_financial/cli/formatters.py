"""CLI presentation formatters and Rich visual components."""

from __future__ import annotations

from collections.abc import Sequence

from rich.panel import Panel
from rich.table import Table

from multi_modal_financial.types import Citation


def format_architecture_table() -> Table:
    """Build Rich Table displaying pipeline architecture and subsystems."""
    table = Table(title="Multi-Modal Financial RAG - Architecture")
    table.add_column("Component", style="cyan", no_wrap=True)
    table.add_column("Implementation", style="green")
    table.add_column("Description", style="white")

    table.add_row(
        "Parser",
        "FinancialDocumentParser",
        "Extracts text, markdown tables, and figures from PDFs/text",
    )
    table.add_row(
        "Sparse Index",
        "BM25 Okapi",
        "Financial-tuned tokenization preserving tickers, currencies, %",
    )
    table.add_row(
        "Dense Index",
        "DenseVectorIndex",
        "Deterministic projection / embedding cosine similarity search",
    )
    table.add_row(
        "Fusion",
        "HybridRetriever",
        "Reciprocal Rank Fusion (RRF) & convex alpha score blending",
    )
    table.add_row(
        "Reranker",
        "FinancialReranker",
        "Financial term overlap, number matching, and table weighting",
    )
    table.add_row(
        "Grounding",
        "GroundingVerifier",
        "Numerical verification, claim attribution, and citation checks",
    )
    return table


def format_response_panel(answer: str, execution_time_ms: float) -> Panel:
    """Format synthesized agent response inside a styled Rich Panel."""
    return Panel(answer, title=f"Synthesized Response ({execution_time_ms} ms)")


def format_citations_table(citations: Sequence[Citation]) -> Table:
    """Format retrieved citations and provenance badges into a structured Table."""
    table = Table(title="Citations & Grounding")
    table.add_column("Citation ID", style="cyan")
    table.add_column("Doc ID", style="magenta")
    table.add_column("Modality", style="green")
    table.add_column("Confidence", style="yellow")

    for cite in citations:
        table.add_row(
            cite.citation_id,
            cite.doc_id,
            str(cite.modal_type),
            f"{cite.confidence:.2f}",
        )
    return table


def format_benchmark_table(
    bm25_r1: float,
    bm25_mrr: float,
    hybrid_r1: float,
    hybrid_mrr: float,
) -> Table:
    """Format retrieval benchmark recall and MRR comparison table."""
    table = Table(title="Benchmark Results (Recall@1 & MRR)")
    table.add_column("Method", style="cyan")
    table.add_column("Recall@1", style="green")
    table.add_column("MRR", style="yellow")

    table.add_row("BM25 (Sparse)", f"{bm25_r1 * 100:.1f}%", f"{bm25_mrr:.3f}")
    table.add_row(
        "Hybrid RRF + Reranker",
        f"{hybrid_r1 * 100:.1f}%",
        f"{hybrid_mrr:.3f}",
    )
    return table
