"""Figure and chart parser for multi-modal financial documents."""

from __future__ import annotations

import re

from multi_modal_financial.types import FigureData


class FigureParser:
    """Parses chart captions, data labels, and figure annotations."""

    @staticmethod
    def parse_figure_block(figure_text: str, figure_id: str) -> FigureData:
        """Parse structured figure description into FigureData."""
        caption: str | None = None
        chart_type: str | None = "chart"
        data_points: dict[str, float] = {}

        # Look for caption
        caption_match = re.search(r"(?:Figure|Chart)\s*\d*:\s*(.+)", figure_text, re.IGNORECASE)
        if caption_match:
            caption = caption_match.group(1).strip()

        # Look for chart type
        for ctype in ["bar", "line", "pie", "waterfall", "scatter", "histogram"]:
            if ctype in figure_text.lower():
                chart_type = ctype
                break

        # Extract numerical key-value pairs (e.g. Q1: 45.2, Revenue: 1200)
        kv_matches = re.findall(r"([A-Za-z0-9_\s]+):\s*\$?([0-9]+(?:\.[0-9]+)?)", figure_text)
        for k, v in kv_matches:
            try:
                data_points[k.strip()] = float(v)
            except ValueError:
                continue

        return FigureData(
            figure_id=figure_id,
            caption=caption,
            chart_type=chart_type,
            summary_text=figure_text.strip(),
            data_points=data_points,
        )

    @staticmethod
    def extract_figures_from_text(text: str) -> list[FigureData]:
        """Find figure / chart annotations in markdown/text."""
        pattern = re.compile(r"(\[Figure[^\]]+\]|\*\*Figure[^\*]+\*\*|Figure\s+\d+:[^\n]+(?:\n[ \t]+[^\n]+)*)", re.IGNORECASE)
        figures: list[FigureData] = []
        for idx, match in enumerate(pattern.finditer(text)):
            block = match.group(0)
            figures.append(FigureParser.parse_figure_block(block, figure_id=f"fig_{idx+1}"))
        return figures
