"""Financial query router and intent classifier."""

from __future__ import annotations

import re
from typing import Any

from multi_modal_financial.types import AgentQuery, ModalType, QueryIntent


class QueryRouter:
    """Extracts financial entities and routes query to optimal retrieval modality."""

    TICKER_PATTERN = re.compile(r"\b[A-Z]{1,5}\b")
    PERIOD_PATTERN = re.compile(
        r"\b(Q[1-4]|FY|FY\d{2,4}|10-[KQ]|annual|quarterly)\b", re.IGNORECASE
    )
    YEAR_PATTERN = re.compile(r"\b(20\d{2}|19\d{2})\b")

    KNOWN_FINANCIAL_WORDS = {
        "revenue",
        "ebitda",
        "margin",
        "income",
        "profit",
        "eps",
        "diluted",
        "cash",
        "flow",
        "capex",
        "debt",
        "equity",
        "guidance",
        "dividend",
        "operating",
        "gross",
        "net",
        "sales",
        "earnings",
    }

    def route(self, query_str: str) -> dict[str, Any]:
        """Analyze query string and return intent, extracted filters, and preferred modality."""
        q_lower = query_str.lower()

        # Intent detection
        if any(w in q_lower for w in ["compare", "versus", "vs", "difference between"]):
            intent = QueryIntent.COMPARATIVE_ANALYSIS
            preferred_modal = ModalType.TABLE
        elif any(w in q_lower for w in ["growth", "trend", "cagr", "rate", "change"]):
            intent = QueryIntent.TREND_CALCULATION
            preferred_modal = ModalType.TABLE
        elif any(w in q_lower for w in ["risk", "litigation", "headwind", "strategy", "outlook"]):
            intent = QueryIntent.QUALITATIVE_RISK
            preferred_modal = ModalType.TEXT
        elif any(w in q_lower for w in self.KNOWN_FINANCIAL_WORDS):
            intent = QueryIntent.METRIC_LOOKUP
            preferred_modal = ModalType.METRIC
        else:
            intent = QueryIntent.GENERAL
            preferred_modal = ModalType.TEXT

        # Entity extraction
        year_match = self.YEAR_PATTERN.search(query_str)
        year = int(year_match.group(1)) if year_match else None

        period_match = self.PERIOD_PATTERN.search(query_str)
        period = period_match.group(1).upper() if period_match else None

        # Ticker search (filter out common English words in all caps)
        tokens = query_str.split()
        ticker: str | None = None
        for tok in tokens:
            cleaned = re.sub(r"[^A-Za-z]", "", tok)
            if (
                cleaned.isupper()
                and 1 <= len(cleaned) <= 5
                and cleaned not in {"A", "I", "IN", "ON", "OF", "THE", "Q1", "Q2", "Q3", "Q4", "FY"}
            ):
                ticker = cleaned
                break

        return {
            "intent": intent,
            "preferred_modal": preferred_modal,
            "ticker": ticker,
            "period": period,
            "year": year,
        }

    def build_agent_query(self, query_str: str, top_k: int = 5) -> AgentQuery:
        """Construct AgentQuery from raw string with automatically parsed filters."""
        routing = self.route(query_str)
        return AgentQuery(
            query_str=query_str,
            ticker_filter=routing["ticker"],
            period_filter=routing["period"],
            year_filter=routing["year"],
            top_k=top_k,
            alpha=0.6 if routing["intent"] == QueryIntent.QUALITATIVE_RISK else 0.4,
        )
