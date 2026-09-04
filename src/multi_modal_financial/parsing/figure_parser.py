"""Figure and chart parser for multi-modal financial documents."""

from __future__ import annotations

import re

from multi_modal_financial.types import FigureData


class FigureParser:
    """Parses chart captions, data labels, and figure annotations."""

    @staticmethod
    def detect_chart_type(text: str) -> str:
        """Infer chart archetype from text descriptions."""
        t_lower = text.lower()
        if "waterfall" in t_lower or "bridge" in t_lower or "walk" in t_lower:
            return "waterfall"
        elif "pie" in t_lower or "donut" in t_lower or "doughnut" in t_lower:
            return "pie"
        elif "bar" in t_lower or "column" in t_lower or "histogram" in t_lower:
            return "bar"
        elif "line" in t_lower or "trend" in t_lower or "trajectory" in t_lower:
            return "line"
        elif "scatter" in t_lower or "bubble" in t_lower:
            return "scatter"
        elif "area" in t_lower:
            return "area"
        return "chart"

    @staticmethod
    def detect_scale_and_unit(text: str) -> tuple[str | None, str | None]:
        """Extract units ($ USD, %, EUR) and scale (millions, billions)."""
        scale: str | None = None
        unit: str | None = None
        t_lower = text.lower()

        if "billion" in t_lower or "$b" in t_lower:
            scale = "billions"
        elif "million" in t_lower or "$m" in t_lower:
            scale = "millions"
        elif "thousand" in t_lower or "$k" in t_lower:
            scale = "thousands"
        elif "%" in t_lower or "percent" in t_lower:
            scale = "percent"

        if "$" in text or "usd" in t_lower:
            unit = "USD"
        elif "€" in text or "eur" in t_lower:
            unit = "EUR"
        elif "£" in text or "gbp" in t_lower:
            unit = "GBP"
        elif "¥" in text or "jpy" in t_lower:
            unit = "JPY"

        return scale, unit

    @staticmethod
    def extract_axis_labels(text: str) -> tuple[str | None, str | None]:
        """Extract X and Y axis descriptions from figure annotations."""
        x_match = re.search(
            r"(?:x-axis|horizontal\s+axis|x)\s*[:=]\s*([^\n,;]+)", text, re.IGNORECASE
        )
        y_match = re.search(
            r"(?:y-axis|vertical\s+axis|y)\s*[:=]\s*([^\n,;]+)", text, re.IGNORECASE
        )
        x_label = x_match.group(1).strip() if x_match else None
        y_label = y_match.group(1).strip() if y_match else None
        return x_label, y_label

    @staticmethod
    def parse_figure_block(figure_text: str, figure_id: str = "fig_1") -> FigureData:
        """Parse structured figure description into FigureData with analytics."""
        caption: str | None = None
        chart_type = FigureParser.detect_chart_type(figure_text)
        scale, unit = FigureParser.detect_scale_and_unit(figure_text)
        x_label, y_label = FigureParser.extract_axis_labels(figure_text)
        data_points: dict[str, float] = {}

        # Look for explicit caption
        caption_match = re.search(
            r"(?:Figure|Chart)\s*\d*[:\-–]\s*([^\n]+)", figure_text, re.IGNORECASE
        )
        if caption_match:
            caption = caption_match.group(1).strip()
        elif "[Figure" in figure_text:
            bracket_match = re.search(r"\[Figure[^:]*:\s*([^\]]+)\]", figure_text, re.IGNORECASE)
            if bracket_match:
                caption = bracket_match.group(1).strip()

        # Extract data points from bullet items or key-value declarations
        # Examples: "Cloud: 450.0", "- Hardware: $250.0M", "Americas: 150"
        lines = figure_text.splitlines()
        for line in lines:
            line_clean = line.strip().lstrip("-*• ")
            kv_match = re.search(
                r"^([A-Za-z0-9_\s\-/&]+)[:=]\s*\$?([0-9]+(?:,[0-9]{3})*(?:\.[0-9]+)?)\s*(?:M|B|K|%)?",
                line_clean,
            )
            if kv_match:
                k = kv_match.group(1).strip()
                # Exclude axis or structural labels
                if k.lower() in {"figure", "chart", "x-axis", "y-axis", "x", "y", "source", "note"}:
                    continue
                v_str = kv_match.group(2).replace(",", "")
                try:
                    data_points[k] = float(v_str)
                except ValueError:
                    continue

        # Fallback regex over whole text if no lines matched
        if not data_points:
            kv_matches = re.findall(r"([A-Za-z0-9_\s]+):\s*\$?([0-9]+(?:\.[0-9]+)?)", figure_text)
            for k, v in kv_matches:
                k_clean = k.strip()
                if k_clean.lower() in {"figure", "chart", "x-axis", "y-axis", "x", "y"}:
                    continue
                try:
                    data_points[k_clean] = float(v)
                except ValueError:
                    continue

        # Construct or enhance summary text
        summary = figure_text.strip()
        if data_points and (not caption or len(summary) < 50):
            pt_summary = ", ".join(f"{k}: {v:g}" for k, v in list(data_points.items())[:5])
            summary = f"{caption or 'Figure'} showing {chart_type} distribution: {pt_summary}"

        return FigureData(
            figure_id=figure_id,
            caption=caption or (figure_text.splitlines()[0] if figure_text else None),
            chart_type=chart_type,
            summary_text=summary,
            data_points=data_points,
            x_label=x_label,
            y_label=y_label,
            unit=unit or "USD",
            scale=scale or "millions",
        )

    @staticmethod
    def extract_figures_from_text(text: str) -> list[FigureData]:
        """Find figure / chart annotations in markdown/text."""
        pattern = re.compile(
            r"((?:\[Figure[^\]]+\]|\*\*Figure[^\*]+\*\*|Figure\s+\d*:[^\n]+(?:\n[ \t]+[^\n]+)*))",
            re.IGNORECASE,
        )
        figures: list[FigureData] = []
        for idx, match in enumerate(pattern.finditer(text)):
            block = match.group(0)
            figures.append(FigureParser.parse_figure_block(block, figure_id=f"fig_{idx + 1}"))
        return figures
