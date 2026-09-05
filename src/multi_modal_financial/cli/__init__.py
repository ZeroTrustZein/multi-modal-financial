"""CLI module for multi-modal financial RAG."""

from multi_modal_financial.cli.formatters import (
    format_architecture_table,
    format_benchmark_table,
    format_citations_table,
    format_response_panel,
)
from multi_modal_financial.cli.main import cli

__all__ = [
    "cli",
    "format_architecture_table",
    "format_benchmark_table",
    "format_citations_table",
    "format_response_panel",
]
