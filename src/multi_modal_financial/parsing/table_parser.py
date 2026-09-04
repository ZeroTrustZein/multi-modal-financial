"""Table parser for structured financial tables."""

from __future__ import annotations

import re

from multi_modal_financial.types import TableData


class TableParser:
    """Parses markdown or delimiter-separated text into structured TableData."""

    @staticmethod
    def parse_markdown_table(table_text: str, title: str | None = None) -> TableData:
        """Parse markdown table string into TableData."""
        lines = [line.strip() for line in table_text.strip().splitlines() if line.strip()]
        if not lines:
            return TableData(title=title)

        headers: list[str] = []
        rows: list[list[str]] = []

        for line in lines:
            if line.startswith("**Table:") or line.startswith("### Table"):
                if not title:
                    extracted_title = line.strip("*#").replace("Table:", "").strip()
                    title = extracted_title
                continue

            # Check if separator row like | --- | --- |
            if re.match(r"^\|?(\s*:?-+:?\s*\|)+\s*:?-+:?\s*\|?$", line):
                continue

            cells = [c.strip() for c in line.strip("|").split("|")]
            if not headers:
                headers = cells
            else:
                rows.append(cells)

        return TableData(
            headers=headers,
            rows=rows,
            title=title,
        )

    @staticmethod
    def extract_tables_from_text(text: str) -> list[TableData]:
        """Detect and extract all markdown tables in text block."""
        table_regex = re.compile(
            r"((?:\|[^\n]+\|\r?\n)+)",
            re.MULTILINE,
        )
        tables: list[TableData] = []
        for match in table_regex.finditer(text):
            chunk = match.group(1)
            parsed = TableParser.parse_markdown_table(chunk)
            if parsed.headers or parsed.rows:
                tables.append(parsed)
        return tables
