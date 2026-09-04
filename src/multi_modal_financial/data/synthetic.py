"""Synthetic financial filing and multi-modal document generator."""

from __future__ import annotations

import random
from typing import Any

from multi_modal_financial.types import (
    Currency,
    Document,
    DocumentMetadata,
    DocumentType,
    FinancialStatementType,
    TableData,
    UnitScale,
)


class SyntheticFilingGenerator:
    """Generates synthetic SEC filings, earnings releases, and financial statements."""

    COMPANIES: dict[str, dict[str, Any]] = {
        "AAPL": {"name": "Apple Inc.", "industry": "Consumer Technology", "base_rev": 95000.0},
        "MSFT": {
            "name": "Microsoft Corporation",
            "industry": "Enterprise Software & Cloud",
            "base_rev": 65000.0,
        },
        "NVDA": {
            "name": "NVIDIA Corporation",
            "industry": "Semiconductors & AI Hardware",
            "base_rev": 35000.0,
        },
        "AMZN": {
            "name": "Amazon.com Inc.",
            "industry": "E-Commerce & Cloud Infrastructure",
            "base_rev": 145000.0,
        },
        "GOOGL": {
            "name": "Alphabet Inc.",
            "industry": "Digital Advertising & Cloud",
            "base_rev": 82000.0,
        },
    }

    def __init__(self, seed: int = 42):
        self.random = random.Random(seed)

    def generate_income_statement(
        self,
        ticker: str = "AAPL",
        year: int = 2025,
        period: str = "Q3",
        scale: UnitScale = UnitScale.MILLIONS,
        currency: Currency = Currency.USD,
    ) -> TableData:
        """Generate a mathematically reconciled Income Statement table."""
        comp = self.COMPANIES.get(ticker, {"name": f"{ticker} Corp", "base_rev": 50000.0})
        base_rev = float(comp["base_rev"]) * (1.0 + self.random.uniform(-0.05, 0.15))

        rev = round(base_rev, 1)
        cogs = round(rev * self.random.uniform(0.52, 0.58), 1)
        gross_profit = round(rev - cogs, 1)

        rd_exp = round(rev * self.random.uniform(0.08, 0.14), 1)
        sga_exp = round(rev * self.random.uniform(0.07, 0.12), 1)
        total_opex = round(rd_exp + sga_exp, 1)
        operating_income = round(gross_profit - total_opex, 1)

        tax = round(operating_income * 0.16, 1)
        net_income = round(operating_income - tax, 1)

        headers = [
            "Consolidated Statements of Operations",
            f"{period} {year}",
            f"{period} {year - 1}",
        ]
        # Previous year comparison with -8% to +10%
        factor = self.random.uniform(0.88, 0.95)
        prev_rev = round(rev * factor, 1)
        prev_cogs = round(cogs * factor, 1)
        prev_gp = round(prev_rev - prev_cogs, 1)
        prev_rd = round(rd_exp * factor, 1)
        prev_sga = round(sga_exp * factor, 1)
        prev_opex = round(prev_rd + prev_sga, 1)
        prev_op_inc = round(prev_gp - prev_opex, 1)
        prev_tax = round(prev_op_inc * 0.16, 1)
        prev_net = round(prev_op_inc - prev_tax, 1)

        curr_symbol = "$" if currency == Currency.USD else currency.value + " "

        rows = [
            ["Total Revenue", f"{curr_symbol}{rev:,.1f}", f"{curr_symbol}{prev_rev:,.1f}"],
            ["Cost of Goods Sold", f"{curr_symbol}{cogs:,.1f}", f"{curr_symbol}{prev_cogs:,.1f}"],
            ["Gross Profit", f"{curr_symbol}{gross_profit:,.1f}", f"{curr_symbol}{prev_gp:,.1f}"],
            [
                "Research and Development",
                f"{curr_symbol}{rd_exp:,.1f}",
                f"{curr_symbol}{prev_rd:,.1f}",
            ],
            [
                "Selling, General and Administrative",
                f"{curr_symbol}{sga_exp:,.1f}",
                f"{curr_symbol}{prev_sga:,.1f}",
            ],
            [
                "Total Operating Expenses",
                f"{curr_symbol}{total_opex:,.1f}",
                f"{curr_symbol}{prev_opex:,.1f}",
            ],
            [
                "Operating Income",
                f"{curr_symbol}{operating_income:,.1f}",
                f"{curr_symbol}{prev_op_inc:,.1f}",
            ],
            [
                "Provision for Income Taxes",
                f"{curr_symbol}{tax:,.1f}",
                f"{curr_symbol}{prev_tax:,.1f}",
            ],
            ["Net Income", f"{curr_symbol}{net_income:,.1f}", f"{curr_symbol}{prev_net:,.1f}"],
        ]

        return TableData(
            title=f"{comp['name']} Condensed Consolidated Statements of Operations",
            headers=headers,
            rows=rows,
            statement_type=FinancialStatementType.INCOME_STATEMENT,
            unit=currency.value,
            scale=scale.value,
            footnotes=[f"(1) Expressed in {scale.value.lower()} of {currency.value}."],
        )

    def generate_balance_sheet(
        self,
        ticker: str = "AAPL",
        year: int = 2025,
        period: str = "Q3",
        scale: UnitScale = UnitScale.MILLIONS,
        currency: Currency = Currency.USD,
    ) -> TableData:
        """Generate a mathematically reconciled Balance Sheet table."""
        comp = self.COMPANIES.get(ticker, {"name": f"{ticker} Corp", "base_rev": 50000.0})
        total_assets = round(float(comp["base_rev"]) * self.random.uniform(3.0, 4.5), 1)

        cash = round(total_assets * 0.18, 1)
        marketable_sec = round(total_assets * 0.12, 1)
        receivables = round(total_assets * 0.15, 1)
        current_assets = round(cash + marketable_sec + receivables, 1)
        ppe = round(total_assets * 0.35, 1)
        other_assets = round(total_assets - (current_assets + ppe), 1)

        # Liabilities
        current_liab = round(total_assets * 0.28, 1)
        long_term_debt = round(total_assets * 0.32, 1)
        total_liabilities = round(current_liab + long_term_debt, 1)

        # Equity must equal Total Assets - Total Liabilities
        equity = round(total_assets - total_liabilities, 1)
        total_liab_and_equity = round(total_liabilities + equity, 1)

        curr_symbol = "$" if currency == Currency.USD else currency.value + " "

        headers = ["Condensed Consolidated Balance Sheets", f"As of {period} {year}"]
        rows = [
            ["Cash and Cash Equivalents", f"{curr_symbol}{cash:,.1f}"],
            ["Marketable Securities", f"{curr_symbol}{marketable_sec:,.1f}"],
            ["Accounts Receivable", f"{curr_symbol}{receivables:,.1f}"],
            ["Total Current Assets", f"{curr_symbol}{current_assets:,.1f}"],
            ["Property, Plant and Equipment, Net", f"{curr_symbol}{ppe:,.1f}"],
            ["Other Non-Current Assets", f"{curr_symbol}{other_assets:,.1f}"],
            ["Total Assets", f"{curr_symbol}{total_assets:,.1f}"],
            ["Current Liabilities", f"{curr_symbol}{current_liab:,.1f}"],
            ["Long-Term Debt", f"{curr_symbol}{long_term_debt:,.1f}"],
            ["Total Liabilities", f"{curr_symbol}{total_liabilities:,.1f}"],
            ["Total Stockholders' Equity", f"{curr_symbol}{equity:,.1f}"],
            [
                "Total Liabilities and Stockholders' Equity",
                f"{curr_symbol}{total_liab_and_equity:,.1f}",
            ],
        ]

        return TableData(
            title=f"{comp['name']} Condensed Consolidated Balance Sheets",
            headers=headers,
            rows=rows,
            statement_type=FinancialStatementType.BALANCE_SHEET,
            unit=currency.value,
            scale=scale.value,
            footnotes=["Unaudited interim financial disclosure."],
        )

    def generate_filing_document(
        self,
        ticker: str = "AAPL",
        year: int = 2025,
        period: str = "Q3",
        doc_type: DocumentType = DocumentType.TEN_Q,
    ) -> Document:
        """Generate a complete multi-modal Document containing narrative, tables, and figures."""
        from multi_modal_financial.parsing.extractor import FinancialDocumentParser

        comp = self.COMPANIES.get(
            ticker, {"name": f"{ticker} Corp", "industry": "General Technology"}
        )
        is_table = self.generate_income_statement(ticker=ticker, year=year, period=period)
        bs_table = self.generate_balance_sheet(ticker=ticker, year=year, period=period)

        doc_text = f"""
# {comp["name"]} - Form {doc_type.value}
Ticker: {ticker} | Period: {period} {year} | Industry: {comp["industry"]}

## Management's Discussion and Analysis of Financial Condition and Results of Operations
During the quarter ended {period} {year}, {comp["name"]} continued its investments across core and emerging markets.
Total revenue expanded significantly driven by accelerated adoption in cloud infrastructure and customer demand.
Operating cash flow generated sufficient liquidity to fund continuous research, capital expenditures, and share repurchases.

### Statements of Operations
The following table summarizes operations for {period} {year}:

{is_table.to_markdown()}

Gross margins expanded favorably due to operational efficiencies and disciplined cost controls.
Operating income showed resilient year-over-year gains reflecting scalable infrastructure.

### Financial Position and Liquidity
Cash and marketable securities provide extensive flexibility to support ongoing operational requirements.

{bs_table.to_markdown()}

Total assets reached balance sheet targets, with stockholders' equity sustaining conservative leverage ratios.

Figure 1: Segment Revenue Distribution
Cloud: 45.0, Enterprise: 30.0, Consumer: 25.0
        """

        doc_id = f"{ticker.lower()}_{doc_type.value.lower()}_{year}_{period.lower()}"
        meta = DocumentMetadata(
            doc_id=doc_id,
            filename=f"{doc_id}.txt",
            ticker=ticker,
            period=period,
            year=year,
            doc_type=doc_type.value,
            company_name=str(comp["name"]),
        )

        parser = FinancialDocumentParser()
        return parser.parse_text(doc_text, metadata=meta)
