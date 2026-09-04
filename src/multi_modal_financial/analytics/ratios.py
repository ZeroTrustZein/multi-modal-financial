"""Financial ratio and margin calculation subsystem."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from multi_modal_financial.types import Document, TableData


@dataclass
class RatioSummary:
    """Consolidated financial ratios and profitability metrics."""

    gross_margin_pct: float | None = None
    operating_margin_pct: float | None = None
    net_margin_pct: float | None = None
    current_ratio: float | None = None
    quick_ratio: float | None = None
    debt_to_equity: float | None = None
    debt_to_assets: float | None = None
    return_on_equity_pct: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class FinancialRatioCalculator:
    """Calculates profitability, liquidity, solvency, and efficiency ratios."""

    @staticmethod
    def compute_margins(
        revenue: float,
        gross_profit: float | None = None,
        operating_income: float | None = None,
        net_income: float | None = None,
    ) -> dict[str, float]:
        """Compute profit margins as percentages of total revenue."""
        results: dict[str, float] = {}
        if revenue <= 0:
            return results

        if gross_profit is not None:
            results["gross_margin"] = round((gross_profit / revenue) * 100.0, 2)
        if operating_income is not None:
            results["operating_margin"] = round((operating_income / revenue) * 100.0, 2)
        if net_income is not None:
            results["net_margin"] = round((net_income / revenue) * 100.0, 2)

        return results

    @staticmethod
    def compute_liquidity(
        current_assets: float,
        current_liabilities: float,
        cash_and_equivalents: float | None = None,
        marketable_securities: float | None = None,
        accounts_receivable: float | None = None,
    ) -> dict[str, float]:
        """Compute current ratio and quick ratio."""
        results: dict[str, float] = {}
        if current_liabilities <= 0:
            return results

        results["current_ratio"] = round(current_assets / current_liabilities, 3)

        quick_assets = 0.0
        has_quick = False
        if cash_and_equivalents is not None:
            quick_assets += cash_and_equivalents
            has_quick = True
        if marketable_securities is not None:
            quick_assets += marketable_securities
            has_quick = True
        if accounts_receivable is not None:
            quick_assets += accounts_receivable
            has_quick = True

        if has_quick:
            results["quick_ratio"] = round(quick_assets / current_liabilities, 3)

        return results

    @staticmethod
    def compute_solvency(
        total_debt: float,
        total_equity: float,
        total_assets: float | None = None,
    ) -> dict[str, float]:
        """Compute debt-to-equity and debt-to-assets leverage metrics."""
        results: dict[str, float] = {}
        if total_equity > 0:
            results["debt_to_equity"] = round(total_debt / total_equity, 3)

        if total_assets is not None and total_assets > 0:
            results["debt_to_assets"] = round(total_debt / total_assets, 3)

        return results

    @classmethod
    def _extract_number_from_table(
        cls, table: TableData, col_idx: int, *keywords: str
    ) -> float | None:
        """Helper to retrieve metric value prioritizing exact label matches."""
        return table.find_metric_value(col_idx, *keywords)

    @classmethod
    def compute_from_table(cls, table: TableData, col_idx: int = 1) -> RatioSummary:
        """Derive standard financial ratios from a single parsed table."""
        summary = RatioSummary()

        # Profitability
        rev = cls._extract_number_from_table(
            table, col_idx, "total revenue", "revenue", "net sales"
        )
        gp = cls._extract_number_from_table(table, col_idx, "gross profit", "gross margin")
        op_inc = cls._extract_number_from_table(
            table, col_idx, "operating income", "operating profit"
        )
        net_inc = cls._extract_number_from_table(table, col_idx, "net income", "net earnings")

        if rev is not None and rev > 0:
            margins = cls.compute_margins(rev, gp, op_inc, net_inc)
            summary.gross_margin_pct = margins.get("gross_margin")
            summary.operating_margin_pct = margins.get("operating_margin")
            summary.net_margin_pct = margins.get("net_margin")

        # Liquidity
        curr_assets = cls._extract_number_from_table(
            table, col_idx, "total current assets", "current assets"
        )
        curr_liab = cls._extract_number_from_table(
            table, col_idx, "total current liabilities", "current liabilities"
        )
        cash = cls._extract_number_from_table(table, col_idx, "cash and cash equivalents", "cash")
        mkt_sec = cls._extract_number_from_table(
            table, col_idx, "marketable securities", "short-term investments"
        )
        ar = cls._extract_number_from_table(table, col_idx, "accounts receivable", "receivables")

        if curr_assets is not None and curr_liab is not None and curr_liab > 0:
            liq = cls.compute_liquidity(curr_assets, curr_liab, cash, mkt_sec, ar)
            summary.current_ratio = liq.get("current_ratio")
            summary.quick_ratio = liq.get("quick_ratio")

        # Solvency
        tot_debt = cls._extract_number_from_table(table, col_idx, "total debt", "long-term debt")
        equity = cls._extract_number_from_table(
            table,
            col_idx,
            "total stockholders' equity",
            "total shareholders' equity",
            "total equity",
        )
        assets = cls._extract_number_from_table(table, col_idx, "total assets")

        if tot_debt is not None and equity is not None and equity > 0:
            solv = cls.compute_solvency(tot_debt, equity, assets)
            summary.debt_to_equity = solv.get("debt_to_equity")
            summary.debt_to_assets = solv.get("debt_to_assets")

        if net_inc is not None and equity is not None and equity > 0:
            summary.return_on_equity_pct = round((net_inc / equity) * 100.0, 2)

        return summary

    @classmethod
    def compute_from_document(cls, document: Document) -> RatioSummary:
        """Combine metrics across all tables in a document to produce consolidated ratios."""
        combined = RatioSummary()
        for chunk in document.chunks:
            if chunk.table_data:
                table_ratios = cls.compute_from_table(chunk.table_data, col_idx=1)
                for f_name, val in asdict(table_ratios).items():
                    if val is not None and getattr(combined, f_name) is None:
                        setattr(combined, f_name, val)
        return combined
