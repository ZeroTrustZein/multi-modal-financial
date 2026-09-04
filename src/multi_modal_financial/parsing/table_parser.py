"""Table parser for structured financial tables."""

from __future__ import annotations

import csv
import io
import re

from multi_modal_financial.types import FinancialStatementType, TableData


class TableParser:
    """Parses markdown or delimiter-separated text into structured TableData."""

    @staticmethod
    def detect_scale_and_unit(text: str) -> tuple[str | None, str | None]:
        """Detect reporting scale and currency unit from financial table annotations."""
        scale: str | None = None
        unit: str | None = None
        t_lower = text.lower()

        # Scale detection
        if "billion" in t_lower or "in billions" in t_lower or "$b" in t_lower:
            scale = "billions"
        elif "million" in t_lower or "in millions" in t_lower or "$m" in t_lower:
            scale = "millions"
        elif "thousand" in t_lower or "in thousands" in t_lower or "$k" in t_lower:
            scale = "thousands"
        elif "%" in t_lower or "percentage" in t_lower or "percent" in t_lower:
            scale = "percent"

        # Currency detection
        if "$" in text or "usd" in t_lower or "dollar" in t_lower:
            unit = "USD"
        elif "€" in text or "eur" in t_lower or "euro" in t_lower:
            unit = "EUR"
        elif "£" in text or "gbp" in t_lower or "pound" in t_lower:
            unit = "GBP"
        elif "¥" in text or "jpy" in t_lower or "yen" in t_lower or "cny" in t_lower:
            unit = "JPY"
        elif "chf" in t_lower:
            unit = "CHF"
        elif "cad" in t_lower:
            unit = "CAD"
        elif "aud" in t_lower:
            unit = "AUD"

        return scale, unit

    @staticmethod
    def detect_statement_type(
        headers: list[str],
        rows: list[list[str]],
        title: str | None = None,
    ) -> FinancialStatementType:
        """Heuristically identify the financial statement category."""
        corpus = (title or "").lower() + " "
        corpus += " ".join(headers).lower() + " "
        row_corpus = " ".join(r[0].lower() for r in rows if r)
        combined = corpus + " " + row_corpus

        if any(
            w in combined
            for w in [
                "operations",
                "income statement",
                "statement of earnings",
                "operating income",
                "gross profit",
                "cost of sales",
                "diluted earnings per share",
            ]
        ):
            return FinancialStatementType.INCOME_STATEMENT

        if any(
            w in combined
            for w in [
                "comprehensive income",
                "other comprehensive",
                "foreign currency translation",
            ]
        ):
            return FinancialStatementType.COMPREHENSIVE_INCOME

        if any(
            w in combined
            for w in [
                "statement of equity",
                "stockholders' equity",
                "shareholders' equity",
                "common stock",
                "treasury stock",
            ]
        ):
            return FinancialStatementType.STOCKHOLDERS_EQUITY

        if any(
            w in combined
            for w in [
                "cash flow",
                "operating activities",
                "investing activities",
                "financing activities",
                "capital expenditures",
                "depreciation and amortization",
            ]
        ):
            return FinancialStatementType.CASH_FLOW

        if any(
            w in combined
            for w in [
                "balance sheet",
                "financial position",
                "total assets",
                "current liabilities",
                "retained earnings",
                "accounts receivable",
            ]
        ):
            return FinancialStatementType.BALANCE_SHEET

        if any(
            w in combined
            for w in [
                "segment",
                "geographic",
                "americas",
                "europe",
                "asia pacific",
                "greater china",
            ]
        ):
            return FinancialStatementType.SEGMENT_METRICS

        if "note" in combined or "accounting policies" in combined:
            return FinancialStatementType.NOTES

        return FinancialStatementType.UNKNOWN

    @staticmethod
    def parse_markdown_table(
        table_text: str,
        title: str | None = None,
        statement_type: FinancialStatementType | None = None,
    ) -> TableData:
        """Parse markdown table string into structured TableData."""
        lines = [line.strip() for line in table_text.strip().splitlines() if line.strip()]
        if not lines:
            return TableData(title=title)

        headers: list[str] = []
        rows: list[list[str]] = []
        footnotes: list[str] = []

        # Detect scale and unit from table preamble or raw text
        detected_scale, detected_unit = TableParser.detect_scale_and_unit(table_text)

        for line in lines:
            # Table title markers
            if (
                line.startswith("**Table:")
                or line.startswith("### Table")
                or line.startswith("# Table")
            ):
                if not title:
                    extracted_title = line.strip("*#").replace("Table:", "").strip()
                    title = extracted_title
                continue

            # Footnote markers below table
            if (
                line.startswith("*")
                or line.startswith("(1)")
                or line.startswith("(2)")
                or line.startswith("Note:")
            ):
                # Make sure it's not a markdown bullet row or table row
                if not line.startswith("|"):
                    footnotes.append(line.strip())
                    continue

            # Check if separator row like | --- | --- |
            if re.match(r"^\|?(\s*:?-+:?\s*\|)+\s*:?-+:?\s*\|?$", line):
                continue

            # Table row with pipe delimiter
            if "|" in line:
                cells = [c.strip() for c in line.strip("|").split("|")]
                if not headers:
                    headers = cells
                else:
                    rows.append(cells)
            elif not headers and not title:
                # Potential title line above table
                title = line.strip()

        st_type = statement_type or TableParser.detect_statement_type(headers, rows, title)

        return TableData(
            headers=headers,
            rows=rows,
            title=title,
            unit=detected_unit or "USD",
            scale=detected_scale or "thousands",
            statement_type=st_type,
            footnotes=footnotes,
        )

    @staticmethod
    def parse_delimited_table(
        csv_text: str,
        delimiter: str = ",",
        title: str | None = None,
    ) -> TableData:
        """Parse CSV/TSV table into TableData."""
        f = io.StringIO(csv_text.strip())
        reader = csv.reader(f, delimiter=delimiter)
        raw_rows = [row for row in reader if row and any(cell.strip() for cell in row)]
        if not raw_rows:
            return TableData(title=title)

        headers = [c.strip() for c in raw_rows[0]]
        rows = [[c.strip() for c in r] for r in raw_rows[1:]]

        detected_scale, detected_unit = TableParser.detect_scale_and_unit(csv_text)
        st_type = TableParser.detect_statement_type(headers, rows, title)

        return TableData(
            headers=headers,
            rows=rows,
            title=title,
            unit=detected_unit or "USD",
            scale=detected_scale or "thousands",
            statement_type=st_type,
        )

    @staticmethod
    def extract_tables_from_text(text: str) -> list[TableData]:
        """Detect and extract all markdown tables in text block."""
        table_regex = re.compile(
            r"((?:(?:\*\*Table:[^\n]+\*\*|###\s*Table[^\n]+)\r?\n)?(?:[ \t]*\|[^\n]+\|\r?\n)+)",
            re.MULTILINE,
        )
        tables: list[TableData] = []
        for match in table_regex.finditer(text):
            chunk = match.group(1)
            parsed = TableParser.parse_markdown_table(chunk)
            if parsed.headers or parsed.rows:
                tables.append(parsed)
        return tables
