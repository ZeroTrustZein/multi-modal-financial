"""CLI presentation formatters and Rich visual components."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from rich.panel import Panel
from rich.table import Table

from multi_modal_financial.analytics.comparator import VarianceResult
from multi_modal_financial.analytics.ratios import RatioSummary
from multi_modal_financial.grounding.audit import GroundingAuditReport
from multi_modal_financial.types import (
    CacheHitType,
    Citation,
    RerankExplanation,
    SemanticCacheStats,
)


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
        "Cross-Encoder",
        "FinancialCrossEncoder",
        "Neural text pair relevance scoring & score calibration",
    )
    table.add_row(
        "Semantic Cache",
        "SemanticCache",
        "Exact & cosine similarity vector caching with LRU/LFU/FIFO/TTL",
    )
    table.add_row(
        "Analytics",
        "FinancialRatioCalculator",
        "Margins, liquidity, solvency ratios & multi-period variance",
    )
    table.add_row(
        "Grounding",
        "GroundingVerifier",
        "Numerical verification, claim attribution, and citation checks",
    )
    table.add_row(
        "Audit Ledger",
        "GroundingAuditor",
        "Automated compliance auditing and hallucination rate tracking",
    )
    return table


def format_response_panel(
    answer: str,
    execution_time_ms: float = 0.0,
    cache_hit: bool | str = False,
    cache_type: CacheHitType | str = CacheHitType.NONE,
    confidence: float | None = None,
    reranker_strategy: str | None = None,
    retrieval_strategy: str | None = None,
) -> Panel:
    """Format synthesized agent response inside a styled Rich Panel."""
    subtitle_parts: list[str] = []
    if confidence is not None:
        subtitle_parts.append(f"Confidence: {confidence:.2f}")

    if cache_hit:
        c_type = cache_type.value if isinstance(cache_type, CacheHitType) else str(cache_type)
        subtitle_parts.append(f"Cache: {c_type}")

    if retrieval_strategy:
        subtitle_parts.append(f"Retrieval: {retrieval_strategy}")

    if reranker_strategy:
        subtitle_parts.append(f"Reranker: {reranker_strategy}")

    subtitle = " | ".join(subtitle_parts) if subtitle_parts else None
    return Panel(
        answer,
        title=f"Synthesized Response ({execution_time_ms} ms)",
        subtitle=subtitle,
    )


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


def format_rerank_explanations_table(explanations: Sequence[RerankExplanation]) -> Table:
    """Format reranker score breakdowns, rank transitions, and explanations."""
    table = Table(title="Reranker Scoring & Explanations")
    table.add_column("Chunk ID", style="cyan", no_wrap=True)
    table.add_column("Rank Shift", style="blue", justify="center")
    table.add_column("Initial Score", style="white", justify="right")
    table.add_column("Final Score", style="green", justify="right")
    table.add_column("Delta", style="yellow", justify="right")
    table.add_column("Factors & Explanations", style="white")

    if not explanations:
        table.add_row("-", "-", "-", "-", "-", "(No rerank adjustments recorded)")
        return table

    for exp in explanations:
        delta = exp.final_score - exp.initial_score
        delta_str = f"{delta:+.4f}"
        delta_style = "green" if delta > 0 else ("red" if delta < 0 else "white")
        rank_shift = f"#{exp.initial_rank} -> #{exp.final_rank}"
        factors = "; ".join(exp.reasons) if exp.reasons else "Score threshold passed"
        table.add_row(
            exp.chunk_id,
            rank_shift,
            f"{exp.initial_score:.4f}",
            f"{exp.final_score:.4f}",
            f"[{delta_style}]{delta_str}[/{delta_style}]",
            factors,
        )
    return table


def format_cache_stats_table(stats: SemanticCacheStats | dict[str, Any]) -> Table:
    """Format semantic cache telemetry, hit rates, and capacity into a Table."""
    table = Table(title="Semantic Cache Telemetry")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green", justify="right")
    table.add_column("Description", style="white")

    data = stats.to_dict() if isinstance(stats, SemanticCacheStats) else stats

    lookups = data.get("total_lookups", 0)
    exact = data.get("exact_hits", 0)
    semantic = data.get("semantic_hits", 0)
    misses = data.get("misses", 0)
    hit_rate = data.get("hit_rate_pct", 0.0)
    entries = data.get("total_entries", 0)
    max_entries = data.get("max_entries", 0)
    evictions = data.get("evictions", 0)
    expired = data.get("expired_count", 0)

    table.add_row("Total Lookups", str(lookups), "Cumulative queries checked against cache")
    table.add_row("Exact Hits", str(exact), "Direct hash matches (zero latency)")
    table.add_row("Semantic Hits", str(semantic), "Vector similarity threshold matches")
    table.add_row("Cache Misses", str(misses), "Queries forwarded to full retrieval pipeline")
    table.add_row("Hit Rate", f"{hit_rate:.1f}%", "Overall cache efficiency (hits / lookups)")
    table.add_row("Active Entries", f"{entries} / {max_entries}", "Current occupancy vs maximum capacity")
    table.add_row("Evictions", str(evictions), "Entries evicted under configured policy")
    table.add_row("Expired", str(expired), "Entries invalidated by TTL expiration")
    return table


def format_ratios_table(ratios: RatioSummary, doc_id: str = "") -> Table:
    """Format calculated financial ratios into categorized Rich Table."""
    title = f"Financial Ratios Summary - {doc_id}" if doc_id else "Financial Ratios Summary"
    table = Table(title=title)
    table.add_column("Category", style="cyan")
    table.add_column("Ratio / Metric", style="white")
    table.add_column("Value", style="green", justify="right")

    # Profitability / Margins
    table.add_row(
        "Profitability",
        "Gross Margin",
        f"{ratios.gross_margin_pct:.1f}%" if ratios.gross_margin_pct is not None else "N/A",
    )
    table.add_row(
        "Profitability",
        "Operating Margin",
        f"{ratios.operating_margin_pct:.1f}%" if ratios.operating_margin_pct is not None else "N/A",
    )
    table.add_row(
        "Profitability",
        "Net Profit Margin",
        f"{ratios.net_margin_pct:.1f}%" if ratios.net_margin_pct is not None else "N/A",
    )

    # Liquidity
    table.add_row(
        "Liquidity",
        "Current Ratio",
        f"{ratios.current_ratio:.2f}x" if ratios.current_ratio is not None else "N/A",
    )
    table.add_row(
        "Liquidity",
        "Quick Ratio",
        f"{ratios.quick_ratio:.2f}x" if ratios.quick_ratio is not None else "N/A",
    )

    # Solvency & Returns
    table.add_row(
        "Solvency",
        "Debt to Equity",
        f"{ratios.debt_to_equity:.2f}x" if ratios.debt_to_equity is not None else "N/A",
    )
    table.add_row(
        "Solvency",
        "Debt to Assets",
        f"{ratios.debt_to_assets:.2f}x" if ratios.debt_to_assets is not None else "N/A",
    )
    table.add_row(
        "Performance",
        "Return on Equity (ROE)",
        f"{ratios.return_on_equity_pct:.1f}%" if ratios.return_on_equity_pct is not None else "N/A",
    )
    return table


def format_comparison_table(
    variances: Sequence[VarianceResult],
    base_id: str = "",
    comp_id: str = "",
) -> Table:
    """Format period-over-period variance analysis table."""
    title_suffix = f" ({base_id} vs {comp_id})" if (base_id and comp_id) else ""
    table = Table(title=f"Financial Period Comparison{title_suffix}")
    table.add_column("Metric", style="cyan")
    table.add_column("Base Value", style="white", justify="right")
    table.add_column("Comparison Value", style="white", justify="right")
    table.add_column("Variance ($)", style="yellow", justify="right")
    table.add_column("Change (%)", style="green", justify="right")
    table.add_column("Trend", style="magenta")

    for v in variances:
        pct_str = f"{v.percentage_change:+.1f}%" if v.percentage_change is not None else "N/A"
        trend_style = "green" if v.trend == "UP" else ("red" if v.trend == "DOWN" else "white")
        table.add_row(
            v.metric_name,
            f"{v.base_value:,.1f}",
            f"{v.compare_value:,.1f}",
            f"{v.absolute_change:+,.1f}",
            f"[{trend_style}]{pct_str}[/{trend_style}]",
            v.trend,
        )
    return table


def format_audit_table(report: GroundingAuditReport) -> Table:
    """Format batch compliance audit report."""
    status_style = "bold green" if report.compliance_status == "PASS" else "bold red"
    status_text = f"[{status_style}]{report.compliance_status}[/{status_style}]"

    table = Table(title="Grounding Compliance Audit Report")
    table.add_column("Audit Parameter", style="cyan")
    table.add_column("Value", style="white", justify="right")
    table.add_column("Status", style="yellow")

    table.add_row("Total Queries Audited", str(report.total_queries), "Complete batch")
    table.add_row("Total Claims Evaluated", str(report.total_claims), "Extracted propositions")
    table.add_row("Supported Claims", str(report.supported_claims), "Fully grounded")
    table.add_row("Partially Supported", str(report.partially_supported_claims), "Contains ungrounded elements")
    table.add_row("Unsupported Claims", str(report.unsupported_claims), "Hallucinations detected")
    table.add_row(
        "Hallucination Rate",
        f"{report.hallucination_rate * 100:.1f}%",
        "Ratio of unsupported claims",
    )
    table.add_row("Average Confidence", f"{report.mean_confidence:.2f}", "Across all verdicts")
    table.add_row("Compliance Verdict", status_text, "Overall compliance")
    return table


def format_benchmark_table(
    bm25_r1: float,
    bm25_mrr: float,
    hybrid_r1: float,
    hybrid_mrr: float,
    dense_r1: float | None = None,
    dense_mrr: float | None = None,
    convex_r1: float | None = None,
    convex_mrr: float | None = None,
) -> Table:
    """Format retrieval benchmark recall and MRR comparison table."""
    table = Table(title="Benchmark Results (Recall@1 & MRR)")
    table.add_column("Method", style="cyan")
    table.add_column("Recall@1", style="green")
    table.add_column("MRR", style="yellow")

    table.add_row("BM25 (Sparse)", f"{bm25_r1 * 100:.1f}%", f"{bm25_mrr:.3f}")

    if dense_r1 is not None and dense_mrr is not None:
        table.add_row("Dense Vector", f"{dense_r1 * 100:.1f}%", f"{dense_mrr:.3f}")

    if convex_r1 is not None and convex_mrr is not None:
        table.add_row("Hybrid (Convex Blend)", f"{convex_r1 * 100:.1f}%", f"{convex_mrr:.3f}")

    table.add_row(
        "Hybrid RRF + Reranker",
        f"{hybrid_r1 * 100:.1f}%",
        f"{hybrid_mrr:.3f}",
    )
    return table

