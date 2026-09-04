"""Financial document text cleaning and normalization subsystem."""

from __future__ import annotations

import re


class FinancialDataCleaner:
    """Preprocesses and normalizes financial documents, SEC filings, and tables."""

    # Unicode dashes and minuses
    UNICODE_DASH_PATTERN = re.compile(
        r"[\u2010\u2011\u2012\u2013\u2014\u2015\u2212\uFE58\uFE63\uFF0D]"
    )

    # SEC Edgar line-level header tags like <TYPE>10-K, <SEQUENCE>1, etc.
    SEC_HEADER_LINE_TAGS = re.compile(
        r"^<(?:TYPE|SEQUENCE|FILENAME|DESCRIPTION)[^>]*>.*$",
        re.IGNORECASE | re.MULTILINE,
    )
    # Structural SEC wrapper tags
    SEC_WRAPPER_TAGS = re.compile(
        r"</?(?:DOCUMENT|TEXT|SEC-HEADER)[^>]*>",
        re.IGNORECASE,
    )
    HTML_TAGS = re.compile(r"<[^>]+>")
    HTML_ENTITIES = {
        "&nbsp;": " ",
        "&#160;": " ",
        "&amp;": "&",
        "&lt;": "<",
        "&gt;": ">",
        "&quot;": '"',
        "&#39;": "'",
        "&mdash;": "-",
        "&ndash;": "-",
        "&minus;": "-",
    }

    # Accounting negative numbers e.g. ($1,234), (450), ( 12.5% )
    ACCOUNTING_NEG_CURRENCY = re.compile(
        r"\(\s*([$€£¥])\s*([0-9]+(?:,[0-9]{3})*(?:\.[0-9]+)?)\s*\)"
    )
    ACCOUNTING_NEG_PERCENT = re.compile(r"\(\s*([0-9]+(?:,[0-9]{3})*(?:\.[0-9]+)?)\s*%\s*\)")
    ACCOUNTING_NEG_RAW = re.compile(r"\(\s*([0-9]+(?:,[0-9]{3})*(?:\.[0-9]+)?)\s*\)")

    # Page headers and footers typical in 10-K / 10-Q filings
    PAGE_NUMBER_FOOTER = re.compile(
        r"^\s*(?:Page\s+)?\d+\s+(?:of\s+\d+\s+)?$", re.IGNORECASE | re.MULTILINE
    )
    TOC_HEADER = re.compile(r"^\s*Table of Contents\s*$", re.IGNORECASE | re.MULTILINE)

    # Footnote patterns
    FOOTNOTE_BRACKET = re.compile(r"\[\w+\]$")
    FOOTNOTE_PAREN = re.compile(r"\([a-zA-Z]\)$")
    FOOTNOTE_SYMBOLS = re.compile(r"[\*†‡]+$")

    @classmethod
    def normalize_dashes(cls, text: str) -> str:
        """Replace Unicode dashes, hyphens, and minus symbols with ASCII hyphen."""
        return cls.UNICODE_DASH_PATTERN.sub("-", text)

    @classmethod
    def strip_html_and_sec_tags(cls, text: str) -> str:
        """Remove SEC SGML/XML wrappers and HTML tags, decoding entities."""
        cleaned = cls.SEC_HEADER_LINE_TAGS.sub("", text)
        cleaned = cls.SEC_WRAPPER_TAGS.sub("", cleaned)
        for entity, replacement in cls.HTML_ENTITIES.items():
            cleaned = cleaned.replace(entity, replacement)
        cleaned = cls.HTML_TAGS.sub("", cleaned)
        return cleaned

    @classmethod
    def normalize_accounting_negatives(cls, text: str) -> str:
        """Convert parenthetical accounting negative representations to standard negative numbers."""
        # ($1,234.5) -> -$1,234.5
        res = cls.ACCOUNTING_NEG_CURRENCY.sub(r"-\1\2", text)
        # (12.5%) -> -12.5%
        res = cls.ACCOUNTING_NEG_PERCENT.sub(r"-\1%", res)
        # (500) -> -500
        res = cls.ACCOUNTING_NEG_RAW.sub(r"-\1", res)
        return res

    @classmethod
    def clean_cell_footnotes(cls, cell_text: str) -> str:
        """Strip footnote superscripts/references like [1], (a), *, † from cell contents."""
        cell = cell_text.strip()
        # Remove trailing footnote markers like [1], [a], (1), (a), *, **, †
        cell = cls.FOOTNOTE_BRACKET.sub("", cell).strip()
        cell = cls.FOOTNOTE_PAREN.sub("", cell).strip()
        cell = cls.FOOTNOTE_SYMBOLS.sub("", cell).strip()
        return cell

    @classmethod
    def repair_markdown_table(cls, table_text: str) -> str:
        """Clean and repair ragged or malformed Markdown table rows."""
        lines = [line.strip() for line in table_text.strip().splitlines() if line.strip()]
        if not lines:
            return ""

        repaired_lines: list[str] = []
        max_cols = 0

        # First pass: determine max column count
        for line in lines:
            if "|" in line:
                parts = [p.strip() for p in line.split("|")]
                if parts and parts[0] == "":
                    parts = parts[1:]
                if parts and parts[-1] == "":
                    parts = parts[:-1]
                max_cols = max(max_cols, len(parts))

        if max_cols == 0:
            return table_text

        for idx, line in enumerate(lines):
            if not line.startswith("|") and not line.endswith("|") and "|" not in line:
                # Non-table line, retain as-is
                repaired_lines.append(line)
                continue

            # Strip external edges and split
            content = line
            if content.startswith("|"):
                content = content[1:]
            if content.endswith("|"):
                content = content[:-1]

            parts = [cls.clean_cell_footnotes(p.strip()) for p in content.split("|")]

            # Pad parts to max_cols if short
            while len(parts) < max_cols:
                parts.append("")

            # If this is the separator line (or second line without dashes), format standard dashes
            if idx == 1 and all(set(p).issubset({"-", ":", " "}) for p in parts if p):
                sep_cells = [":---" if p.startswith(":") else "---" for p in parts]
                repaired_lines.append("| " + " | ".join(sep_cells) + " |")
            else:
                repaired_lines.append("| " + " | ".join(parts) + " |")

        return "\n".join(repaired_lines)

    @classmethod
    def clean_document_text(cls, text: str, strip_sec_headers: bool = True) -> str:
        """End-to-end cleaning pipeline for financial document text."""
        if not text:
            return ""

        cleaned = text

        # 1. Strip SEC Edgar SGML and HTML tags
        if strip_sec_headers:
            cleaned = cls.strip_html_and_sec_tags(cleaned)

        # 2. Normalize Unicode dashes & quotes
        cleaned = cls.normalize_dashes(cleaned)
        cleaned = cleaned.replace("\u2018", "'").replace("\u2019", "'")
        cleaned = cleaned.replace("\u201c", '"').replace("\u201d", '"')

        # 3. Strip recurring boilerplate like page numbers and Table of Contents
        cleaned = cls.PAGE_NUMBER_FOOTER.sub("", cleaned)
        cleaned = cls.TOC_HEADER.sub("", cleaned)

        # 4. Normalize accounting parenthetical negatives
        cleaned = cls.normalize_accounting_negatives(cleaned)

        # 5. Normalize extra blank lines and trailing spaces
        lines = [line.rstrip() for line in cleaned.splitlines()]
        result_lines: list[str] = []
        blank_counter = 0

        for line in lines:
            if not line.strip():
                blank_counter += 1
                if blank_counter <= 2:
                    result_lines.append("")
            else:
                blank_counter = 0
                result_lines.append(line)

        return "\n".join(result_lines).strip()
