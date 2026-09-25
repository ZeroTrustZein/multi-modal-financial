"""CLI module for multi-modal financial RAG."""

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
from multi_modal_financial.cli.main import cli

__all__ = [
    "cli",
    "format_architecture_table",
    "format_audit_table",
    "format_benchmark_table",
    "format_cache_stats_table",
    "format_citations_table",
    "format_comparison_table",
    "format_ratios_table",
    "format_rerank_explanations_table",
    "format_response_panel",
]
